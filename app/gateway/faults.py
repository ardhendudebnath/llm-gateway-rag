"""Runtime fault injection, for chaos tests: make a deployment fail on demand, on every replica.

    PUT /v1/admin/faults/mock-primary  {"failure_rate": 1.0}   -> mock-primary starts failing
    DELETE /v1/admin/faults/mock-primary                       -> and recovers

An injected failure is raised exactly where a real provider failure would be, so it exercises the
real retry, fallback and circuit-breaker paths.

The fault table lives in Redis because the API runs several replicas: a fault set on one pod must
break the deployment for all of them. Each replica caches the table for a second, so this costs one
Redis read per second rather than one per request. It is disabled unless
``NEXUSGATE_FAULT_INJECTION_ENABLED`` is set, and it fails open: if Redis can't be read, no fault is
injected.
"""

import logging
import random
import time
from collections.abc import Callable

from redis.asyncio import Redis

from app.gateway.providers import ProviderError

log = logging.getLogger(__name__)

FAULTS_KEY = "faults:deployments"


class FaultInjector:
    def __init__(
        self,
        redis: Redis,
        *,
        refresh_seconds: float = 1.0,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._redis = redis
        self._refresh = refresh_seconds
        self._rng = rng or random.Random()
        self._clock = clock
        self._rates: dict[str, float] = {}
        self._fetched_at = float("-inf")

    async def rates(self) -> dict[str, float]:
        now = self._clock()
        if now - self._fetched_at >= self._refresh:
            try:
                raw = await self._redis.hgetall(FAULTS_KEY)
                self._rates = {name: float(rate) for name, rate in raw.items()}
            except Exception:
                log.warning("could not read injected faults; injecting none", exc_info=True)
                self._rates = {}
            self._fetched_at = now
        return self._rates

    async def maybe_fail(self, deployment: str) -> None:
        rate = (await self.rates()).get(deployment, 0.0)
        if rate > 0 and self._rng.random() < rate:
            raise ProviderError(f"{deployment}: injected fault", retryable=True, status_code=503)

    async def set(self, deployment: str, failure_rate: float) -> None:
        await self._redis.hset(FAULTS_KEY, deployment, failure_rate)
        self._fetched_at = float("-inf")  # this replica sees it at once; others within a second

    async def clear(self, deployment: str) -> bool:
        removed = await self._redis.hdel(FAULTS_KEY, deployment)
        self._fetched_at = float("-inf")
        return bool(removed)
