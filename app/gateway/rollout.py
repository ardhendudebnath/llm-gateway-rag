"""Canary rollouts for the route table: ship a provider change to a slice of traffic first.

A route change — a new provider, a different fallback order, a cheaper model in front — is a
deployment in every sense except that nothing gets rebuilt. This lets one go out to a percentage
of requests, watched, and then promoted or dropped, with no restart and no redeploy:

    PUT    /v1/admin/routes/canary   {"config": "<yaml>", "weight": 10}   10% of traffic
    GET    /v1/admin/routes                                               how it is doing
    POST   /v1/admin/routes/canary/promote                                make it the route table
    DELETE /v1/admin/routes/canary                                        drop it

Both versions live in Redis because the API runs several replicas: a canary published on one pod
must reach all of them, and the counters that decide whether it is healthy have to be pooled, or
each replica would judge the canary on its own handful of requests. Each replica caches the pair
for a second, the same trade as the fault table.

**It rolls itself back.** Once the canary has served ``min_requests``, if its error rate is worse
than ``max_error_rate`` the weight drops to 0 automatically and the reason is recorded. That is the
point of a canary: the bad version stops taking traffic without anyone watching a dashboard.

Failure of this machinery is never failure of a request: if Redis can't be read, the stable table
compiled at start-up is used, which is the same table the gateway would have had without any of
this.
"""

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from redis.asyncio import Redis

from app.gateway.routing_config import RoutingConfig
from app.observability import metrics

log = logging.getLogger(__name__)

STABLE_KEY = "rollout:stable"
CANARY_KEY = "rollout:canary"
STATS_KEY = "rollout:stats"

STABLE = "stable"
CANARY = "canary"


@dataclass(frozen=True)
class Variant:
    """Which route table served a request, and which version of it."""

    name: str
    version: str


@dataclass
class CanaryState:
    version: str
    weight: int
    note: str | None = None
    created_at: str = ""
    rolled_back_at: str | None = None
    rollback_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.weight > 0 and self.rolled_back_at is None


@dataclass
class RolloutSettings:
    min_requests: int = 20
    max_error_rate: float = 0.10
    refresh_seconds: float = 1.0


@dataclass
class _Cached:
    stable: tuple[str, RoutingConfig] | None = None
    canary: tuple[CanaryState, RoutingConfig] | None = None
    fetched_at: float = field(default=float("-inf"))


class RolloutManager:
    """Chooses a route table per request, counts how each version fares, and pulls the cord."""

    def __init__(
        self,
        redis: Redis,
        fallback: RoutingConfig,
        *,
        settings: RolloutSettings | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._redis = redis
        self._fallback = (fallback.fingerprint(), fallback)
        self.settings = settings or RolloutSettings()
        self._rng = rng or random.Random()
        self._clock = clock
        self._cache = _Cached()

    # ---------------------------------------------------------------- reading

    async def _load(self) -> _Cached:
        now = self._clock()
        if now - self._cache.fetched_at < self.settings.refresh_seconds:
            return self._cache
        cache = _Cached(fetched_at=now)
        try:
            stable_raw, canary_raw = await self._redis.mget(STABLE_KEY, CANARY_KEY)
            if stable_raw:
                stored = json.loads(stable_raw)
                cache.stable = (stored["version"], RoutingConfig.from_text(stored["config"]))
            if canary_raw:
                stored = json.loads(canary_raw)
                config = RoutingConfig.from_text(stored.pop("config"))
                cache.canary = (CanaryState(**stored), config)
        except Exception:
            # Never fail a request over rollout bookkeeping: fall back to the table on disk.
            log.warning(
                "could not read the rollout state; using the start-up routes", exc_info=True
            )
            cache = _Cached(fetched_at=now)
        self._cache = cache
        return cache

    async def stable(self) -> tuple[str, RoutingConfig]:
        return (await self._load()).stable or self._fallback

    async def canary(self) -> tuple[CanaryState, RoutingConfig] | None:
        return (await self._load()).canary

    async def select(self) -> tuple[RoutingConfig, Variant]:
        """Pick the table for one request: the canary with probability weight/100."""
        cache = await self._load()
        stable_version, stable_config = cache.stable or self._fallback
        canary = cache.canary
        if canary and canary[0].active and self._rng.random() * 100 < canary[0].weight:
            return canary[1], Variant(CANARY, canary[0].version)
        return stable_config, Variant(STABLE, stable_version)

    # ---------------------------------------------------------------- writing

    async def record(self, variant: Variant, *, ok: bool) -> None:
        """Count the outcome, and roll the canary back if it has earned it."""
        metrics.ROLLOUT_REQUESTS.labels(variant.name, "success" if ok else "error").inc()
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hincrby(STATS_KEY, f"{variant.version}:requests", 1)
            if not ok:
                pipe.hincrby(STATS_KEY, f"{variant.version}:errors", 1)
            await pipe.execute()
        except Exception:
            log.warning(
                "could not record a rollout outcome",
                exc_info=True,
                extra={"version": variant.version},
            )
            return
        if variant.name == CANARY and not ok:
            await self._maybe_roll_back(variant.version)

    async def _maybe_roll_back(self, version: str) -> None:
        canary = await self.canary()
        if canary is None or canary[0].version != version or not canary[0].active:
            return
        stats = await self.stats(version)
        requests, errors = stats["requests"], stats["errors"]
        if requests < self.settings.min_requests:
            return
        rate = errors / requests
        if rate <= self.settings.max_error_rate:
            return
        reason = (
            f"error rate {rate:.0%} over {requests} requests, above the "
            f"{self.settings.max_error_rate:.0%} the canary is allowed"
        )
        await self.roll_back(reason)
        metrics.ROLLOUT_ROLLBACKS.inc()
        log.error("canary rolled back automatically", extra={"version": version, "reason": reason})

    async def publish(self, config_text: str, weight: int, note: str | None = None) -> CanaryState:
        """Put a candidate route table in front of `weight` percent of traffic."""
        config = RoutingConfig.from_text(config_text)  # raises before anything is stored
        state = CanaryState(
            version=config.fingerprint(),
            weight=weight,
            note=note,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        await self._write_canary(state, config_text)
        await self._redis.hdel(
            STATS_KEY, f"{state.version}:requests", f"{state.version}:errors"
        )  # a republished canary is judged on its own traffic, not the last attempt's
        log.info(
            "canary published", extra={"version": state.version, "weight": weight, "note": note}
        )
        return state

    async def set_weight(self, weight: int) -> CanaryState | None:
        canary = await self.canary()
        if canary is None:
            return None
        state, config = canary
        state.weight = weight
        state.rolled_back_at = None if weight else state.rolled_back_at
        await self._write_canary(state, config_text=None, config=config)
        return state

    async def roll_back(self, reason: str) -> CanaryState | None:
        canary = await self.canary()
        if canary is None:
            return None
        state, config = canary
        state.weight = 0
        state.rolled_back_at = datetime.now(UTC).isoformat(timespec="seconds")
        state.rollback_reason = reason
        await self._write_canary(state, config_text=None, config=config)
        return state

    async def promote(self) -> tuple[str, RoutingConfig] | None:
        """The canary becomes the route table for everything, and stops being a canary."""
        canary = await self.canary()
        if canary is None:
            return None
        state, config = canary
        await self._redis.set(
            STABLE_KEY,
            json.dumps(
                {
                    "version": state.version,
                    "config": config.model_dump_json(),
                    "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            ),
        )
        await self._redis.delete(CANARY_KEY)
        self._invalidate()
        log.info("canary promoted to stable", extra={"version": state.version})
        return state.version, config

    async def discard(self) -> bool:
        removed = await self._redis.delete(CANARY_KEY)
        self._invalidate()
        return bool(removed)

    async def stats(self, version: str) -> dict[str, int]:
        raw = await self._redis.hmget(STATS_KEY, f"{version}:requests", f"{version}:errors")
        requests, errors = (int(v or 0) for v in raw)
        return {"requests": requests, "errors": errors}

    async def _write_canary(
        self, state: CanaryState, config_text: str | None, config: RoutingConfig | None = None
    ) -> None:
        payload = {**vars(state), "config": config_text or config.model_dump_json()}
        await self._redis.set(CANARY_KEY, json.dumps(payload))
        self._invalidate()
        metrics.ROLLOUT_CANARY_WEIGHT.set(state.weight)

    def _invalidate(self) -> None:
        """This replica sees the change at once; the others within their refresh window."""
        self._cache.fetched_at = float("-inf")
