import asyncio

import pytest

from app.core.rate_limit import TokenBucketLimiter


class FakeClock:
    def __init__(self):
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(redis_pair, clock):
    text, _ = redis_pair
    return TokenBucketLimiter(text, clock=clock)


async def test_allows_burst_up_to_capacity_then_rejects(limiter):
    decisions = [await limiter.hit("k", capacity=5, refill_per_sec=1) for _ in range(6)]
    assert [d.allowed for d in decisions] == [True] * 5 + [False]
    assert decisions[4].remaining == 0
    assert decisions[5].retry_after_seconds == 1


async def test_refills_over_time(limiter, clock):
    for _ in range(5):
        await limiter.hit("k", capacity=5, refill_per_sec=2)
    assert not (await limiter.hit("k", capacity=5, refill_per_sec=2)).allowed
    clock.now += 1.0  # 2 tokens back
    assert (await limiter.hit("k", capacity=5, refill_per_sec=2)).allowed
    assert (await limiter.hit("k", capacity=5, refill_per_sec=2)).allowed
    assert not (await limiter.hit("k", capacity=5, refill_per_sec=2)).allowed


async def test_refill_never_exceeds_capacity(limiter, clock):
    await limiter.hit("k", capacity=3, refill_per_sec=1)
    clock.now += 3600
    decision = await limiter.hit("k", capacity=3, refill_per_sec=1)
    assert decision.remaining == 2


async def test_retry_after_reflects_refill_rate(limiter):
    for _ in range(2):
        await limiter.hit("k", capacity=2, refill_per_sec=0.1)
    decision = await limiter.hit("k", capacity=2, refill_per_sec=0.1)
    assert not decision.allowed
    assert decision.retry_after_seconds == 10


async def test_buckets_are_independent_per_key(limiter):
    await limiter.hit("a", capacity=1, refill_per_sec=0.01)
    assert not (await limiter.hit("a", capacity=1, refill_per_sec=0.01)).allowed
    assert (await limiter.hit("b", capacity=1, refill_per_sec=0.01)).allowed


async def test_concurrent_hits_never_overspend(limiter):
    decisions = await asyncio.gather(
        *[limiter.hit("k", capacity=10, refill_per_sec=0.001) for _ in range(50)]
    )
    assert sum(d.allowed for d in decisions) == 10
