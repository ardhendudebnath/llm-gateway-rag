"""Fault injection: shared through Redis, cached per replica, and failing open."""

import random

import pytest

from app.gateway.faults import FAULTS_KEY, FaultInjector
from app.gateway.providers import ProviderError


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


def injector(redis, clock, rng=None) -> FaultInjector:
    return FaultInjector(redis, refresh_seconds=1.0, rng=rng or random.Random(0), clock=clock)


async def test_no_faults_means_no_failures(redis_pair, clock):
    await injector(redis_pair[0], clock).maybe_fail("primary")  # does not raise


async def test_a_full_fault_always_fails_like_a_provider_would(redis_pair, clock):
    faults = injector(redis_pair[0], clock)
    await faults.set("primary", 1.0)
    with pytest.raises(ProviderError, match="injected fault") as err:
        await faults.maybe_fail("primary")
    assert err.value.retryable is True and err.value.status_code == 503
    await faults.maybe_fail("secondary")  # other deployments are unaffected


async def test_a_partial_fault_fails_about_that_share_of_calls(redis_pair, clock):
    faults = injector(redis_pair[0], clock, rng=random.Random(42))
    await faults.set("primary", 0.3)
    failures = 0
    for _ in range(1000):
        try:
            await faults.maybe_fail("primary")
        except ProviderError:
            failures += 1
    assert 250 < failures < 350


async def test_a_fault_set_by_another_replica_is_seen_within_the_refresh(redis_pair, clock):
    here = injector(redis_pair[0], clock)
    other_replica = injector(redis_pair[0], clock)
    await here.rates()  # caches "no faults"

    await other_replica.set("primary", 1.0)
    assert await here.rates() == {}  # still cached
    clock.now += 1.0
    assert await here.rates() == {"primary": 1.0}


async def test_clearing_stops_the_fault(redis_pair, clock):
    faults = injector(redis_pair[0], clock)
    await faults.set("primary", 1.0)
    assert await faults.clear("primary") is True
    await faults.maybe_fail("primary")  # recovered
    assert await faults.clear("primary") is False


async def test_it_fails_open_when_redis_is_unreadable(redis_pair, clock):
    class BrokenRedis:
        async def hgetall(self, key):
            raise ConnectionError("redis is down")

    await injector(BrokenRedis(), clock).maybe_fail("primary")  # no fault, no crash


async def test_fault_table_lives_in_one_redis_hash(redis_pair, clock):
    await injector(redis_pair[0], clock).set("primary", 0.5)
    assert await redis_pair[0].hgetall(FAULTS_KEY) == {"primary": "0.5"}
