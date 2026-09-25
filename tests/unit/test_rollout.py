"""Canary rollouts: traffic splitting, pooled counters, automatic rollback, promotion."""

import random

import pytest

from app.gateway.rollout import CANARY, STABLE, RolloutManager, RolloutSettings
from app.gateway.routing_config import RoutingConfig
from tests.conftest import make_routing

CANDIDATE = """
routes:
  default:
    - name: candidate
      provider: fake
      model: fake/candidate
      pricing: { input_per_mtok: 0.1, output_per_mtok: 0.2 }
"""


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def manager(redis_pair):
    def build(seed: int = 0, clock=None, **settings) -> RolloutManager:
        return RolloutManager(
            redis_pair[0],
            make_routing(),
            settings=RolloutSettings(**settings),
            rng=random.Random(seed),
            clock=clock or Clock(),
        )

    return build


async def test_without_a_canary_everything_is_served_by_the_start_up_table(manager):
    rollout = manager()

    config, variant = await rollout.select()

    assert variant.name == STABLE
    assert set(config.routes) == {"default", "single", "selfhosted"}
    assert variant.version == make_routing().fingerprint()


async def test_the_weight_decides_how_much_traffic_the_canary_takes(manager):
    rollout = manager(seed=7)
    await rollout.publish(CANDIDATE, weight=25)

    picks = [(await rollout.select())[1].name for _ in range(400)]

    share = picks.count(CANARY) / len(picks)
    assert 0.18 < share < 0.32  # 25%, allowing for the sample size
    assert all(p in (STABLE, CANARY) for p in picks)


@pytest.mark.parametrize(("weight", "expected"), [(100, CANARY), (0, STABLE)])
async def test_the_extremes_are_absolute(manager, weight, expected):
    rollout = manager()
    await rollout.publish(CANDIDATE, weight=weight)

    assert {(await rollout.select())[1].name for _ in range(50)} == {expected}


async def test_a_malformed_route_table_is_rejected_before_it_is_stored(manager):
    rollout = manager()

    with pytest.raises(ValueError):
        await rollout.publish("routes: {default: []}", weight=50)  # a route with no deployments

    assert await rollout.canary() is None  # nothing was published


async def test_a_failing_canary_withdraws_itself(manager):
    rollout = manager(min_requests=10, max_error_rate=0.2)
    state = await rollout.publish(CANDIDATE, weight=50)
    variant = (await rollout.select())[1]

    for i in range(10):  # 10 requests, half of them failures: well over the 20% allowed
        await rollout.record(type(variant)(CANARY, state.version), ok=i % 2 == 0)

    canary = await rollout.canary()
    assert canary[0].weight == 0 and not canary[0].active
    assert "error rate 50%" in canary[0].rollback_reason
    assert (await rollout.select())[1].name == STABLE  # traffic is back on stable


async def test_a_canary_is_not_judged_before_it_has_served_enough(manager):
    rollout = manager(min_requests=20, max_error_rate=0.01)
    state = await rollout.publish(CANDIDATE, weight=100)
    variant = (await rollout.select())[1]

    for _ in range(5):  # every one a failure, but only five of them
        await rollout.record(variant, ok=False)

    assert (await rollout.canary())[0].active
    assert await rollout.stats(state.version) == {"requests": 5, "errors": 5}


async def test_a_healthy_canary_keeps_its_traffic(manager):
    rollout = manager(min_requests=5, max_error_rate=0.2)
    await rollout.publish(CANDIDATE, weight=100)
    variant = (await rollout.select())[1]

    for i in range(50):
        await rollout.record(variant, ok=i != 0)  # one failure in fifty

    assert (await rollout.canary())[0].active


async def test_a_canary_with_errors_under_the_limit_keeps_serving(manager):
    # The threshold is a budget, not zero tolerance: a provider blip should not end a rollout.
    rollout = manager(min_requests=5, max_error_rate=0.25)
    state = await rollout.publish(CANDIDATE, weight=100)
    variant = (await rollout.select())[1]
    for _ in range(9):
        await rollout.record(variant, ok=True)

    await rollout.record(variant, ok=False)  # 1 in 10, inside the 25% budget

    assert (await rollout.canary())[0].active
    assert await rollout.stats(state.version) == {"requests": 10, "errors": 1}


async def test_an_outcome_from_a_replaced_canary_cannot_roll_back_the_new_one(manager):
    rollout = manager(min_requests=1, max_error_rate=0.01)
    await rollout.publish(CANDIDATE, weight=100)
    stale = type((await rollout.select())[1])(CANARY, "0123456789ab")  # a version long gone

    await rollout.record(stale, ok=False)

    assert (await rollout.canary())[0].active


async def test_counters_that_cannot_be_written_do_not_break_the_request(manager):
    rollout = manager()
    await rollout.publish(CANDIDATE, weight=100)
    variant = (await rollout.select())[1]

    def broken_pipeline(*_args, **_kwargs):
        raise ConnectionError("redis gone")

    rollout._redis.pipeline = broken_pipeline

    await rollout.record(variant, ok=False)  # the point is that this returns rather than raises

    assert (await rollout.canary())[0].active  # and nothing was concluded from missing data


@pytest.mark.parametrize("action", ["set_weight", "roll_back"])
async def test_changing_a_canary_that_does_not_exist_reports_nothing_to_change(manager, action):
    rollout = manager()

    argument = 50 if action == "set_weight" else "because"
    assert await getattr(rollout, action)(argument) is None


async def test_promoting_makes_the_canary_the_only_table(manager):
    rollout = manager()
    await rollout.publish(CANDIDATE, weight=10)

    version, config = await rollout.promote()

    assert list(config.routes) == ["default"]
    assert [d.name for d in config.routes["default"]] == ["candidate"]
    assert await rollout.canary() is None  # nothing left to roll out
    stable_version, stable_config = await rollout.stable()
    assert stable_version == version and stable_config.routes == config.routes
    assert (await rollout.select())[1].name == STABLE  # and it is what everything now uses


async def test_discarding_returns_every_request_to_stable(manager):
    rollout = manager()
    await rollout.publish(CANDIDATE, weight=100)

    assert await rollout.discard() is True

    assert await rollout.canary() is None
    assert (await rollout.select())[1].name == STABLE
    assert await rollout.discard() is False  # already gone


async def test_republishing_a_canary_starts_its_score_from_zero(manager):
    rollout = manager(min_requests=4, max_error_rate=0.1)
    first = await rollout.publish(CANDIDATE, weight=100)
    variant = (await rollout.select())[1]
    for _ in range(3):
        await rollout.record(variant, ok=False)

    again = await rollout.publish(CANDIDATE, weight=100)

    assert again.version == first.version  # same table, same fingerprint
    assert await rollout.stats(again.version) == {"requests": 0, "errors": 0}


async def test_another_replica_sees_a_canary_within_the_refresh_window(redis_pair):
    # Two managers over one Redis: the publisher and a second pod that never published anything.
    clock = Clock()
    publisher, other = (
        RolloutManager(
            redis_pair[0],
            make_routing(),
            settings=RolloutSettings(refresh_seconds=1.0),
            rng=random.Random(1),
            clock=clock,
        )
        for _ in range(2)
    )
    assert (await other.select())[1].name == STABLE  # warms the other replica's cache

    await publisher.publish(CANDIDATE, weight=100)

    assert (await other.select())[1].name == STABLE, "not before its cache expires"
    clock.now += 1.5
    assert (await other.select())[1].name == CANARY, "and within a second of it expiring"


async def test_unreadable_redis_serves_the_start_up_table_rather_than_failing(manager):
    rollout = manager()
    await rollout.publish(CANDIDATE, weight=100)

    async def broken(*_args, **_kwargs):
        raise ConnectionError("redis gone")

    rollout._redis.mget = broken
    rollout._cache.fetched_at = float("-inf")

    config, variant = await rollout.select()

    assert variant.name == STABLE
    assert config.routes == RoutingConfig.model_validate(make_routing().model_dump()).routes
