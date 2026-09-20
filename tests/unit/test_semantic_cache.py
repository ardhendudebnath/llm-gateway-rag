import numpy as np
import pytest
from redis.exceptions import ResponseError

from app.cache.semantic_cache import (
    BruteForceIndex,
    CachedCompletion,
    RediSearchIndex,
    SemanticCache,
    namespace_for,
)
from app.core.embeddings import HashingEmbedder
from app.gateway.schemas import ChatMessage, ChatRequest


def req(content: str, system: str | None = None, **kw) -> ChatRequest:
    messages = [ChatMessage(role="system", content=system)] if system else []
    messages.append(ChatMessage(role="user", content=content))
    return ChatRequest(model="default", messages=messages, **kw)


def completion(content: str = "Paris.", cost: float = 0.002) -> CachedCompletion:
    return CachedCompletion(
        content=content,
        model="m",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=2,
        deployment="primary",
        cost_usd=cost,
    )


class TickingClock:
    """Advances 1s per read, so every store gets a distinct, ordered timestamp."""

    def __init__(self):
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        self.now += 1
        return self.now


def make_cache(raw_redis, threshold=0.9, ttl=3600, max_entries=100) -> SemanticCache:
    return SemanticCache(
        raw_redis,
        HashingEmbedder(),
        BruteForceIndex(raw_redis, max_entries),
        threshold=threshold,
        ttl_seconds=ttl,
        max_entries=max_entries,
        clock=TickingClock(),
    )


@pytest.fixture
def raw(redis_pair):
    return redis_pair[1]


async def test_miss_on_empty_cache(raw):
    assert await make_cache(raw).lookup("t1", req("What is the capital of France?")) is None


async def test_exact_repeat_hits(raw):
    cache = make_cache(raw)
    await cache.store("t1", req("What is the capital of France?"), completion())
    hit = await cache.lookup("t1", req("What is the capital of France?"))
    assert hit is not None
    assert hit.completion.content == "Paris."
    assert hit.similarity == pytest.approx(1.0)


async def test_near_duplicate_hits(raw):
    cache = make_cache(raw, threshold=0.8)
    await cache.store("t1", req("What is the capital of France?"), completion())
    assert await cache.lookup("t1", req("what is the capital of france")) is not None


async def test_unrelated_query_misses(raw):
    cache = make_cache(raw)
    await cache.store("t1", req("What is the capital of France?"), completion())
    assert await cache.lookup("t1", req("How do I reverse a linked list in Rust?")) is None


async def test_tenants_are_isolated(raw):
    cache = make_cache(raw)
    await cache.store("tenant-a", req("What is the capital of France?"), completion())
    assert await cache.lookup("tenant-b", req("What is the capital of France?")) is None


async def test_different_system_prompt_never_collides(raw):
    cache = make_cache(raw)
    await cache.store("t1", req("Capital of France?", system="Answer in French."), completion())
    assert await cache.lookup("t1", req("Capital of France?", system="Answer in English.")) is None


async def test_generation_params_are_part_of_the_namespace(raw):
    cache = make_cache(raw)
    await cache.store("t1", req("Capital of France?", temperature=0), completion())
    assert await cache.lookup("t1", req("Capital of France?", temperature=1.2)) is None


async def test_entries_expire_via_ttl(raw):
    cache = make_cache(raw, ttl=60)
    await cache.store("t1", req("Capital of France?"), completion())
    ttls = [await raw.ttl(k) async for k in raw.scan_iter(match="semcache:e:*")]
    assert ttls and all(0 < t <= 60 for t in ttls)


async def test_expired_entries_are_skipped_and_pruned(raw):
    cache = make_cache(raw)
    await cache.store("t1", req("Capital of France?"), completion())
    async for k in raw.scan_iter(match="semcache:e:*"):
        await raw.delete(k)  # simulate TTL expiry
    assert await cache.lookup("t1", req("Capital of France?")) is None
    ns = namespace_for("t1", req("Capital of France?"))
    assert await raw.zcard(f"semcache:z:{ns}") == 0


async def test_namespace_is_capped_oldest_evicted(raw):
    cache = make_cache(raw, max_entries=3)
    for i in range(5):
        await cache.store("t1", req(f"question number {i} about topic {i}"), completion(f"a{i}"))
    ns = namespace_for("t1", req("anything"))
    assert await raw.zcard(f"semcache:z:{ns}") == 3
    keys = [k async for k in raw.scan_iter(match=f"semcache:e:{ns}:*")]
    assert len(keys) == 3
    assert await cache.lookup("t1", req("question number 0 about topic 0")) is None
    assert await cache.lookup("t1", req("question number 4 about topic 4")) is not None


async def test_purge_tenant_only_removes_that_tenant(raw):
    cache = make_cache(raw)
    await cache.store("a", req("Capital of France?"), completion())
    await cache.store("b", req("Capital of France?"), completion())
    assert await cache.purge_tenant("a") == 1
    assert await cache.lookup("a", req("Capital of France?")) is None
    assert await cache.lookup("b", req("Capital of France?")) is not None


async def test_purge_all(raw):
    cache = make_cache(raw)
    await cache.store("a", req("Capital of France?"), completion())
    await cache.store("b", req("Capital of Spain?"), completion())
    assert await cache.purge_all() == 2
    assert [k async for k in raw.scan_iter(match="semcache:*")] == []


def test_threshold_must_be_valid(raw):
    with pytest.raises(ValueError):
        make_cache(raw, threshold=0)


async def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = await emb.embed("Hello world"), await emb.embed("hello, world!")
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(await emb.embed(""))) == 0.0


class FakeSearch:
    """Stands in for ``redis.ft(...)``: no index exists yet; create_index fails as configured."""

    def __init__(self, create_error: Exception | None):
        self._create_error = create_error
        self.create_calls = 0

    async def info(self):
        raise ResponseError("Unknown index name")

    async def create_index(self, *args, **kwargs):
        self.create_calls += 1
        if self._create_error is not None:
            raise self._create_error


class FakeSearchRedis:
    def __init__(self, search: FakeSearch):
        self._search = search

    def ft(self, name: str) -> FakeSearch:
        return self._search


async def test_redisearch_setup_tolerates_losing_the_create_race():
    # Two API replicas start together: both see no index, and the other one creates it first.
    search = FakeSearch(ResponseError("Index already exists"))
    await RediSearchIndex(FakeSearchRedis(search)).setup(dim=8)
    assert search.create_calls == 1


async def test_redisearch_setup_still_raises_other_errors():
    search = FakeSearch(ResponseError("Invalid field type"))
    with pytest.raises(ResponseError, match="Invalid field type"):
        await RediSearchIndex(FakeSearchRedis(search)).setup(dim=8)
