"""Circuit state shared across replicas: pooled failures, adopted state, one probe between them."""

import asyncio

import pytest

from app.gateway.breaker_cluster import FAILS, OPEN, PROBE, ClusterBreakers
from app.gateway.circuit_breaker import BreakerState, CircuitBreaker
from app.gateway.providers import ProviderError
from app.gateway.router import LLMRouter
from app.gateway.schemas import ChatRequest
from tests.conftest import ScriptedProvider, make_routing

THRESHOLD = 3
COOLDOWN = 30.0


class Clock:
    """One clock for every replica, so a refresh window can be stepped over deliberately."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 1.5) -> None:
        self.now += seconds


class Replica:
    """One API pod: its own in-process breakers, the same Redis as the others."""

    def __init__(self, redis, clock: Clock, provider: ScriptedProvider | None = None):
        self.provider = provider or ScriptedProvider()
        self.cluster = ClusterBreakers(
            redis,
            failure_threshold=THRESHOLD,
            cooldown_seconds=COOLDOWN,
            refresh_seconds=1.0,
            probe_seconds=5.0,
            clock=clock,
        )
        self.router = LLMRouter(
            make_routing(),
            {"fake": self.provider},
            default_timeout=5.0,
            breaker_factory=lambda: CircuitBreaker(THRESHOLD, COOLDOWN, clock=clock),
            sleep=lambda _: asyncio.sleep(0),
            cluster=self.cluster,
        )

    @property
    def primary(self) -> CircuitBreaker:
        return self.router.breakers["primary"]

    async def ask(self, content: str = "hello"):
        return await self.router.complete(
            ChatRequest(model="default", messages=[{"role": "user", "content": content}])
        )


@pytest.fixture
def cluster_of(redis_pair):
    clock = Clock()

    def build(count: int = 2) -> tuple[list[Replica], Clock]:
        return [Replica(redis_pair[0], clock) for _ in range(count)], clock

    return build


async def test_failures_split_across_replicas_still_open_the_circuit(cluster_of):
    """The point of sharing: with one failure each, no replica could ever have opened it alone."""
    (a, b, c), clock = cluster_of(3)
    for replica in (a, b, c):
        replica.provider.always_fail("primary", n=1)

    await a.ask()
    clock.tick()
    await b.ask()
    clock.tick()
    assert [r.provider.calls.count("primary") for r in (a, b)] == [1, 1], "one failure each"
    # Each pod counts what the cluster has seen, not what it has seen: that is the whole mechanism.
    assert b.primary.consecutive_failures == 2
    assert {r.primary.state for r in (a, b)} == {BreakerState.CLOSED}, "still under the threshold"

    await c.ask()  # the third failure anywhere in the cluster

    assert c.primary.state is BreakerState.OPEN
    assert c.provider.calls.count("primary") == 1, "it opened a circuit on one failure of its own"
    clock.tick()
    await a.ask()
    assert a.primary.state is BreakerState.OPEN, "and the others learn it on their next read"


async def test_failures_landing_in_the_same_window_still_add_up(cluster_of):
    """The concurrent case: three pods fail at once, before any of them has re-read the count.

    Each pod's local breaker sees a single failure and stays closed on its own reckoning; the shared
    counter is what notices that the deployment has failed three times.
    """
    (a, b, c), _clock = cluster_of(3)  # the clock never moves here: that is the point
    for replica in (a, b, c):
        await replica.ask()  # a healthy request each: everyone starts from zero
    for replica in (a, b, c):
        replica.provider.always_fail("primary", n=1)

    await a.ask()  # no clock.tick(): all three land inside one refresh window
    await b.ask()
    assert [r.primary.consecutive_failures for r in (a, b)] == [1, 1], "no pod has re-read yet"
    await c.ask()

    assert c.primary.state is BreakerState.OPEN
    assert c.primary.consecutive_failures == THRESHOLD, "opened on the pooled count"


async def test_a_replica_adopts_an_open_circuit_it_never_saw_fail(cluster_of):
    (a, b), clock = cluster_of()
    a.provider.always_fail("primary")
    for _ in range(THRESHOLD):  # a alone trips it
        await a.ask()
        clock.tick()
    assert a.primary.state is BreakerState.OPEN

    await b.ask()  # b has never seen primary fail

    assert b.primary.state is BreakerState.OPEN
    assert b.provider.calls == ["secondary"], "it skipped straight to the fallback"


async def test_a_success_on_one_replica_clears_the_circuit_everywhere(cluster_of, redis_pair):
    (a, b), clock = cluster_of()
    a.provider.always_fail("primary", n=THRESHOLD)
    for _ in range(THRESHOLD):
        await a.ask()
        clock.tick()
    await redis_pair[0].delete(OPEN.format("primary"))  # what the cooldown expiring looks like
    clock.tick()

    await a.ask()  # a probes, and primary answers this time
    assert a.primary.state is BreakerState.CLOSED
    clock.tick()
    await b.ask()

    assert b.primary.state is BreakerState.CLOSED
    assert await redis_pair[0].get(FAILS.format("primary")) is None


async def test_only_one_replica_probes_when_the_cooldown_expires(cluster_of, redis_pair):
    (a, b, c), clock = cluster_of(3)
    a.provider.always_fail("primary")
    for _ in range(THRESHOLD):
        await a.ask()
        clock.tick()
    for replica in (b, c):  # everyone agrees it is open
        await replica.ask()
        clock.tick()
    assert {r.primary.state for r in (a, b, c)} == {BreakerState.OPEN}

    await redis_pair[0].delete(OPEN.format("primary"))  # cooldown over
    clock.tick()
    states = []
    for replica in (b, c):
        await replica.cluster.sync(replica.router.breakers, ["primary"])
        states.append(replica.primary.state)

    assert sorted(s.value for s in states) == ["half_open", "open"], "one probes, one waits"
    assert await redis_pair[0].get(PROBE.format("primary")) == "1"


async def test_a_failed_probe_re_opens_the_circuit_for_everyone(cluster_of, redis_pair):
    (a, b), clock = cluster_of()
    a.provider.always_fail("primary")
    for _ in range(THRESHOLD):
        await a.ask()
        clock.tick()
    await redis_pair[0].delete(OPEN.format("primary"))
    clock.tick()

    await a.ask()  # a claims the probe; primary is still failing

    assert a.primary.state is BreakerState.OPEN
    assert await redis_pair[0].get(OPEN.format("primary")) == "1"
    assert 0 < await redis_pair[0].ttl(OPEN.format("primary")) <= COOLDOWN
    clock.tick()
    await b.ask()
    assert b.primary.state is BreakerState.OPEN


async def test_pooled_failures_expire_so_they_cannot_accumulate_forever(cluster_of, redis_pair):
    (a, _), _ = cluster_of()
    a.provider.always_fail("primary", n=1)

    await a.ask()

    ttl = await redis_pair[0].ttl(FAILS.format("primary"))
    assert 0 < ttl <= 60, "a failure from an idle hour ago must not count toward the threshold"


async def test_the_shared_state_is_read_once_per_refresh_window(cluster_of, redis_pair):
    (a, _), clock = cluster_of()
    reads = 0
    real = redis_pair[0].pipeline

    def counting(*args, **kwargs):
        nonlocal reads
        reads += 1
        return real(*args, **kwargs)

    redis_pair[0].pipeline = counting

    for _ in range(5):
        await a.ask()  # five requests inside one window
    assert reads == 1, "one pipelined read, not one per request"

    clock.tick()
    await a.ask()
    assert reads == 2


async def test_an_unreachable_redis_leaves_the_local_breakers_in_charge(cluster_of):
    (a, _), clock = cluster_of()

    async def broken(*_args, **_kwargs):
        raise ConnectionError("redis gone")

    a.cluster._redis.pipeline = lambda *_a, **_k: (_ for _ in ()).throw(ConnectionError("gone"))
    a.cluster._redis.set = broken
    a.cluster._redis.delete = broken
    a.provider.always_fail("primary")

    for _ in range(THRESHOLD):  # exactly the behaviour from before any sharing existed
        await a.ask()
        clock.tick()

    assert a.primary.state is BreakerState.OPEN, "it still protects this replica"


async def test_a_deployment_this_replica_has_no_breaker_for_is_skipped(cluster_of):
    (a, _), _ = cluster_of()

    await a.cluster.sync(a.router.breakers, ["primary", "not-here"])  # must not raise

    assert "not-here" not in a.router.breakers


async def test_a_probe_is_taken_rather_than_missed_when_redis_cannot_answer(cluster_of, redis_pair):
    """Recovery must not depend on Redis: better two probes than a circuit that never closes."""
    (a, _), clock = cluster_of()
    a.provider.always_fail("primary")
    for _ in range(THRESHOLD):
        await a.ask()
        clock.tick()
    await redis_pair[0].delete(OPEN.format("primary"))

    async def broken(*_args, **_kwargs):
        raise ConnectionError("redis gone")

    a.cluster._redis.set = broken
    clock.tick()
    await a.cluster.sync(a.router.breakers, ["primary"])

    assert a.primary.state is BreakerState.HALF_OPEN


async def test_the_admin_view_survives_an_unreadable_redis(cluster_of):
    (a, _), _ = cluster_of()

    def broken(*_args, **_kwargs):
        raise ConnectionError("redis gone")

    a.cluster._redis.pipeline = broken

    assert await a.cluster.states(["primary"]) == {}


async def test_the_admin_view_reports_what_the_cluster_holds(cluster_of):
    (a, _), clock = cluster_of()
    a.provider.always_fail("primary")
    for _ in range(THRESHOLD):
        await a.ask()
        clock.tick()

    shared = await a.cluster.states(["primary", "secondary"])

    assert shared["primary"]["open"] is True
    assert 0 < shared["primary"]["cooldown_remaining_s"] <= COOLDOWN
    assert shared["secondary"] == {
        "open": False,
        "cooldown_remaining_s": 0,
        "pooled_failures": 0,
    }


async def test_a_client_error_clears_the_pooled_failures(cluster_of, redis_pair):
    # A 400 is the request's fault, not the provider's: it must not leave failures behind that
    # push an otherwise healthy deployment over the threshold later.
    (a, _), _ = cluster_of()
    a.provider.script("primary", ProviderError("blip", retryable=True, status_code=503))
    await a.ask()
    assert await redis_pair[0].get(FAILS.format("primary")) == "1"

    a.provider.script("primary", ProviderError("bad prompt", retryable=False, status_code=400))
    with pytest.raises(Exception, match="bad prompt"):
        await a.ask()

    assert await redis_pair[0].get(FAILS.format("primary")) is None
