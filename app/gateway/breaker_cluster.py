"""Circuit-breaker state shared by every API replica.

The in-process breakers (``app/gateway/circuit_breaker.py``) are still the hot path: the decision to
skip a deployment is a dictionary lookup, not a Redis round-trip. What this adds is agreement. With
two replicas and a failure threshold of 3, a provider failing every other request has each replica
counting 2 failures and neither opening its circuit, while six users get errors. Pooled, the third
failure opens it for everyone.

Three keys per deployment, all of which expire on their own so nothing has to be cleaned up:

    breaker:fails:<dep>   failures counted across replicas, within a window
    breaker:open:<dep>    exists while the circuit is open; its TTL *is* the cooldown
    breaker:probe:<dep>   the right to send the one probe request after a cooldown

Each replica reads the open and failure keys at most once per ``refresh_seconds`` (one pipelined
read covering the whole chain, the same trade as the fault table), and writes only when a request
succeeds or fails.

**When the cooldown expires, one replica probes, not all of them.** Whoever wins ``SET NX`` on the
probe key moves its local breaker to half-open and sends the request; the others stay open until the
next refresh tells them the answer. Without that, every replica would probe a provider that is
probably still down.

**Redis is never in the way of a request.** Every call here is best-effort: if Redis cannot be
reached the local breakers keep their own counts and timers, which is exactly how the gateway
behaved before any of this existed — per-replica circuit breaking, no sharing.
"""

import logging
import time
from collections.abc import Callable, Iterable, Mapping

from redis.asyncio import Redis

from app.gateway.circuit_breaker import BreakerState, CircuitBreaker
from app.observability import metrics

log = logging.getLogger(__name__)

FAILS = "breaker:fails:{}"
OPEN = "breaker:open:{}"
PROBE = "breaker:probe:{}"


class ClusterBreakers:
    def __init__(
        self,
        redis: Redis,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        failure_window_seconds: float = 60.0,
        refresh_seconds: float = 1.0,
        probe_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._redis = redis
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_window_seconds = failure_window_seconds
        self._refresh = refresh_seconds
        self._probe_seconds = probe_seconds
        self._clock = clock
        self._synced_at = float("-inf")

    # ------------------------------------------------------------------ reading

    async def sync(self, breakers: Mapping[str, CircuitBreaker], names: Iterable[str]) -> None:
        """Bring these deployments' local breakers in line with what the cluster knows."""
        now = self._clock()
        if now - self._synced_at < self._refresh:
            return
        self._synced_at = now
        names = list(names)
        try:
            pipe = self._redis.pipeline(transaction=False)
            for name in names:
                pipe.get(OPEN.format(name))
                pipe.get(FAILS.format(name))
            results = await pipe.execute()
        except Exception:
            # Keep the local state and its own timers: per-replica breaking, as before.
            log.warning("could not read shared breaker state; using local only", exc_info=True)
            return

        for name, open_flag, fails in zip(names, results[::2], results[1::2], strict=True):
            breaker = breakers.get(name)
            if breaker is None:
                continue
            if open_flag:
                await self._adopt_open(name, breaker)
            elif breaker.state is BreakerState.OPEN:
                # The cooldown has expired cluster-wide. One replica probes; the rest wait.
                await self._claim_probe(name, breaker)
            elif fails is not None:
                breaker.adopt(state=breaker.state, consecutive_failures=int(fails))

    async def _adopt_open(self, name: str, breaker: CircuitBreaker) -> None:
        if breaker.state is not BreakerState.OPEN:
            metrics.BREAKER_ADOPTIONS.labels(name, "open").inc()
            log.info("adopting an open circuit from another replica", extra={"deployment": name})
        # Re-stamping the local timer on every refresh keeps this replica open for as long as the
        # shared key lives, so the cooldown is the cluster's, not this process's.
        breaker.adopt(state=BreakerState.OPEN, consecutive_failures=self.failure_threshold)

    async def _claim_probe(self, name: str, breaker: CircuitBreaker) -> None:
        try:
            won = await self._redis.set(
                PROBE.format(name), "1", nx=True, ex=max(int(self._probe_seconds), 1)
            )
        except Exception:
            log.warning("could not claim the breaker probe; probing anyway", exc_info=True)
            won = True  # degrade to the old behaviour rather than never recovering
        if won:
            metrics.BREAKER_PROBES.labels(name, "claimed").inc()
            breaker.adopt(state=BreakerState.HALF_OPEN, consecutive_failures=0)
        else:
            metrics.BREAKER_PROBES.labels(name, "yielded").inc()

    # ------------------------------------------------------------------ writing

    async def record_failure(self, name: str, breaker: CircuitBreaker) -> bool:
        """Count a failure locally and cluster-wide. Returns True if the circuit is now open."""
        opened = breaker.record_failure()
        try:
            if opened:
                # This replica has seen enough on its own (or a probe just failed).
                await self._open(name)
                return True
            pipe = self._redis.pipeline(transaction=False)
            pipe.incr(FAILS.format(name))
            pipe.expire(FAILS.format(name), max(int(self.failure_window_seconds), 1))
            count = (await pipe.execute())[0]
        except Exception:
            log.warning("could not record a shared breaker failure", exc_info=True)
            return opened
        if int(count) >= self.failure_threshold:
            # No single replica saw enough failures, but together they did.
            log.warning(
                "circuit opened on pooled failures",
                extra={"deployment": name, "failures": int(count)},
            )
            await self._open(name)
            breaker.adopt(state=BreakerState.OPEN, consecutive_failures=int(count))
            return True
        return False

    async def record_success(self, name: str, breaker: CircuitBreaker) -> None:
        """A success closes the circuit for every replica, not just this one."""
        breaker.record_success()
        try:
            await self._redis.delete(FAILS.format(name), OPEN.format(name), PROBE.format(name))
        except Exception:
            log.warning("could not clear shared breaker state", exc_info=True)

    async def _open(self, name: str) -> None:
        """The open key's TTL is the cooldown: when it expires, the circuit is eligible to probe."""
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.set(OPEN.format(name), "1", ex=max(int(self.cooldown_seconds), 1))
            pipe.delete(FAILS.format(name), PROBE.format(name))
            await pipe.execute()
        except Exception:
            log.warning("could not publish an open circuit", exc_info=True)

    async def states(self, names: Iterable[str]) -> dict[str, dict]:
        """What the cluster holds, for the admin view: open circuits and how long they have left."""
        names = list(names)
        try:
            pipe = self._redis.pipeline(transaction=False)
            for name in names:
                pipe.ttl(OPEN.format(name))
                pipe.get(FAILS.format(name))
            results = await pipe.execute()
        except Exception:
            log.warning("could not read shared breaker state", exc_info=True)
            return {}
        shared = {}
        for name, ttl, fails in zip(names, results[::2], results[1::2], strict=True):
            shared[name] = {
                "open": int(ttl) > 0,
                "cooldown_remaining_s": max(int(ttl), 0),
                "pooled_failures": int(fails or 0),
            }
        return shared
