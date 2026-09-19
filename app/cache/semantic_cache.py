"""Redis-backed semantic response cache.

Design
------
* **What is matched semantically:** only the *final user message*. Everything that changes what a
  correct answer looks like — tenant, route, temperature, max_tokens, system prompt, earlier turns —
  is hashed into an exact-match *namespace*. "Answer in French" vs "Answer in English" with the same
  question therefore can never collide, however similar the embeddings are.
* **Tenant isolation:** the tenant id is part of the namespace; tenant A can never be served
  tenant B's cached answer.
* **Invalidation:** every entry has a TTL; ``purge_tenant`` / ``purge_all`` back the explicit
  purge endpoints. Each namespace is capped at ``max_entries`` (oldest evicted first).

Storage layout (shared by both index backends)::

    semcache:e:{ns}:{id}   HASH   ns, vec (float32 bytes), payload (json), created
    semcache:z:{ns}        ZSET   id -> created   (recency order, capping, brute-force scan)
    semcache:t:{tenant}    SET    namespaces owned by the tenant (for purge)

Index backends
--------------
* ``BruteForceIndex`` — numpy dot product over the namespace's entries. Zero extra infrastructure;
  fine for dev, tests and small namespaces.
* ``RediSearchIndex`` — HNSW vector index via the Redis Query Engine (redis-stack / Redis 8).
  O(log n) lookups; what ``docker compose`` runs.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.cache.embeddings import Embedder
from app.gateway.schemas import ChatRequest

PREFIX = "semcache"
INDEX_NAME = "semcache_idx"


class CachedCompletion(BaseModel):
    content: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    deployment: str
    cost_usd: float  # what the original call cost — i.e. what each hit saves


@dataclass
class CacheHit:
    completion: CachedCompletion
    similarity: float


def _entry_key(ns: str, entry_id: str) -> str:
    return f"{PREFIX}:e:{ns}:{entry_id}"


def _zset_key(ns: str) -> str:
    return f"{PREFIX}:z:{ns}"


def _tenant_key(tenant_id: str) -> str:
    return f"{PREFIX}:t:{tenant_id}"


def namespace_for(tenant_id: str, request: ChatRequest) -> str:
    *context, _last = request.messages
    material = json.dumps(
        {
            "tenant": tenant_id,
            "route": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "context": [m.model_dump() for m in context],
            "last_role": _last.role,
        },
        sort_keys=True,
    )
    # Hex only: safe to use unescaped as a RediSearch TAG value.
    return hashlib.sha256(material.encode()).hexdigest()[:32]


class VectorIndex(Protocol):
    async def setup(self, dim: int) -> None: ...

    async def nearest(self, ns: str, vec: np.ndarray) -> tuple[str, float] | None:
        """Return (entry_key, cosine_similarity) of the closest entry in ``ns``, if any."""
        ...


class BruteForceIndex:
    def __init__(self, redis: Redis, max_entries: int):
        self._redis = redis
        self._max_entries = max_entries

    async def setup(self, dim: int) -> None:
        return None

    async def nearest(self, ns: str, vec: np.ndarray) -> tuple[str, float] | None:
        ids = [
            i.decode() for i in await self._redis.zrevrange(_zset_key(ns), 0, self._max_entries - 1)
        ]
        if not ids:
            return None
        pipe = self._redis.pipeline(transaction=False)
        for entry_id in ids:
            pipe.hget(_entry_key(ns, entry_id), "vec")
        raw = await pipe.execute()

        live = [(i, v) for i, v in zip(ids, raw, strict=True) if v is not None]
        if expired := [i for i, v in zip(ids, raw, strict=True) if v is None]:
            await self._redis.zrem(_zset_key(ns), *expired)
        if not live:
            return None

        matrix = np.frombuffer(b"".join(v for _, v in live), dtype=np.float32).reshape(
            len(live), -1
        )
        sims = matrix @ vec
        best = int(np.argmax(sims))
        return _entry_key(ns, live[best][0]), float(sims[best])


class RediSearchIndex:
    def __init__(self, redis: Redis):
        self._redis = redis

    async def setup(self, dim: int) -> None:
        from redis.commands.search.field import TagField, VectorField

        try:
            from redis.commands.search.index_definition import IndexDefinition, IndexType
        except ImportError:  # redis-py < 6
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        try:
            await self._redis.ft(INDEX_NAME).info()
            return
        except ResponseError:
            pass
        try:
            await self._redis.ft(INDEX_NAME).create_index(
                [
                    TagField("ns"),
                    VectorField(
                        "vec", "HNSW", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"}
                    ),
                ],
                definition=IndexDefinition(prefix=[f"{PREFIX}:e:"], index_type=IndexType.HASH),
            )
        except ResponseError as e:
            # Replicas starting together all see "no index" and race to create it; losing that
            # race is success, not a reason to fail startup.
            if "already exists" not in str(e).lower():
                raise

    async def nearest(self, ns: str, vec: np.ndarray) -> tuple[str, float] | None:
        from redis.commands.search.query import Query

        query = (
            Query(f"(@ns:{{{ns}}})=>[KNN 1 @vec $vec AS dist]")
            .sort_by("dist")
            .return_fields("dist")
            .dialect(2)
        )
        result = await self._redis.ft(INDEX_NAME).search(
            query, query_params={"vec": vec.astype(np.float32).tobytes()}
        )
        if not result.docs:
            return None
        doc = result.docs[0]
        return doc.id, 1.0 - float(doc.dist)  # COSINE distance = 1 - similarity


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        embedder: Embedder,
        index: VectorIndex,
        *,
        threshold: float,
        ttl_seconds: int,
        max_entries: int,
        clock: Callable[[], float] = time.time,
    ):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self._clock = clock
        self._redis = redis
        self._embedder = embedder
        self._index = index
        self.threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    async def setup(self) -> None:
        await self._index.setup(self._embedder.dim)

    async def lookup(self, tenant_id: str, request: ChatRequest) -> CacheHit | None:
        ns = namespace_for(tenant_id, request)
        vec = await self._embedder.embed(request.messages[-1].content)
        found = await self._index.nearest(ns, vec)
        if found is None or found[1] < self.threshold:
            return None
        key, similarity = found
        payload = await self._redis.hget(key, "payload")
        if payload is None:  # expired between search and fetch
            return None
        return CacheHit(CachedCompletion.model_validate_json(payload), round(similarity, 4))

    async def store(
        self, tenant_id: str, request: ChatRequest, completion: CachedCompletion
    ) -> None:
        ns = namespace_for(tenant_id, request)
        vec = await self._embedder.embed(request.messages[-1].content)
        entry_id = uuid.uuid4().hex
        now = self._clock()

        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(
            _entry_key(ns, entry_id),
            mapping={
                "ns": ns,
                "vec": vec.astype(np.float32).tobytes(),
                "payload": completion.model_dump_json(),
                "created": now,
            },
        )
        pipe.expire(_entry_key(ns, entry_id), self._ttl)
        pipe.zadd(_zset_key(ns), {entry_id: now})
        pipe.zremrangebyscore(_zset_key(ns), "-inf", now - self._ttl)
        pipe.zrange(_zset_key(ns), 0, -(self._max_entries + 1))  # ids beyond the cap
        pipe.zremrangebyrank(_zset_key(ns), 0, -(self._max_entries + 1))
        pipe.expire(_zset_key(ns), self._ttl)
        pipe.sadd(_tenant_key(tenant_id), ns)
        pipe.expire(_tenant_key(tenant_id), self._ttl)
        results = await pipe.execute()

        if evicted := results[4]:
            await self._redis.delete(*(_entry_key(ns, i.decode()) for i in evicted))

    async def purge_tenant(self, tenant_id: str) -> int:
        namespaces = [n.decode() for n in await self._redis.smembers(_tenant_key(tenant_id))]
        deleted = 0
        for ns in namespaces:
            deleted += await self._delete_matching(f"{PREFIX}:e:{ns}:*")
            await self._redis.delete(_zset_key(ns))
        await self._redis.delete(_tenant_key(tenant_id))
        return deleted

    async def purge_all(self) -> int:
        deleted = await self._delete_matching(f"{PREFIX}:e:*")
        await self._delete_matching(f"{PREFIX}:z:*")
        await self._delete_matching(f"{PREFIX}:t:*")
        return deleted

    async def _delete_matching(self, pattern: str) -> int:
        deleted = 0
        batch: list[bytes] = []
        async for key in self._redis.scan_iter(match=pattern, count=500):
            batch.append(key)
            if len(batch) >= 500:
                deleted += await self._redis.delete(*batch)
                batch.clear()
        if batch:
            deleted += await self._redis.delete(*batch)
        return deleted
