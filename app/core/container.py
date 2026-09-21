"""Composition root: builds every long-lived service once per process.

Everything is injectable so tests can swap in fakeredis, fake providers, an in-memory Qdrant and a
deterministic embedder without monkeypatching.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app import __version__
from app.cache.semantic_cache import BruteForceIndex, RediSearchIndex, SemanticCache
from app.core.concurrency import InferenceGate
from app.core.config import Settings
from app.core.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from app.core.metering import UsageMeter
from app.core.rate_limit import TokenBucketLimiter
from app.core.security import ApiKeyStore, TokenService
from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.faults import FaultInjector
from app.gateway.providers import LiteLLMProvider, MockProvider, Provider
from app.gateway.router import LLMRouter
from app.gateway.routing_config import RoutingConfig
from app.gateway.service import ChatService
from app.observability.tracing import LangfuseTracer, NoopTracer, Tracer
from app.rag.chunking import build_chunker
from app.rag.documents import DocumentRegistry
from app.rag.ingestion import IngestionService
from app.rag.reranking import CrossEncoderReranker, Reranker
from app.rag.retrieval import Retriever
from app.rag.service import RagService
from app.rag.vector_store import QdrantChunkStore
from app.workers.jobs import DeadLetterQueue, JobStore
from app.workers.payloads import PayloadStore
from app.workers.queue import CeleryJobQueue, InlineJobQueue, JobQueue
from app.workers.service import IngestionJobService

log = logging.getLogger(__name__)

_INSECURE_DEFAULTS = {
    "admin_token": Settings.model_fields["admin_token"].default.get_secret_value(),
    "jwt_secret": Settings.model_fields["jwt_secret"].default.get_secret_value(),
}


@dataclass
class RagComponents:
    store: QdrantChunkStore
    documents: DocumentRegistry
    ingestion: IngestionService
    retriever: Retriever
    answers: RagService
    jobs: IngestionJobService
    job_store: JobStore
    dead_letters: DeadLetterQueue


@dataclass
class Services:
    settings: Settings
    redis: Redis
    cache_redis: Redis
    router: LLMRouter
    cache: SemanticCache | None
    keys: ApiKeyStore
    tokens: TokenService
    limiter: TokenBucketLimiter
    meter: UsageMeter
    chat: ChatService
    tracer: Tracer
    rag: RagComponents

    async def aclose(self) -> None:
        self.tracer.shutdown()  # flush buffered traces before the process exits
        await self.rag.store.aclose()
        await self.redis.aclose()
        if self.cache_redis is not self.redis:
            await self.cache_redis.aclose()


def check_production_secrets(settings: Settings) -> None:
    if not settings.is_prod:
        return
    for field, default in _INSECURE_DEFAULTS.items():
        if getattr(settings, field).get_secret_value() == default:
            raise RuntimeError(f"NEXUSGATE_{field.upper()} must be set in production")


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "fastembed":
        return FastEmbedEmbedder(
            settings.embedding_model,
            settings.model_cache_dir,
            settings.model_threads,
            gate=InferenceGate(
                "embedder",
                max_concurrency=settings.embed_max_concurrency,
                max_queue=settings.embed_max_queue,
            ),
        )
    return HashingEmbedder()


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.rag_reranker == "cross-encoder":
        return CrossEncoderReranker(
            settings.rag_reranker_model,
            settings.model_cache_dir,
            settings.model_threads,
            gate=InferenceGate(
                "reranker",
                max_concurrency=settings.rerank_max_concurrency,
                max_queue=settings.rerank_max_queue,
            ),
            max_wait_seconds=settings.rerank_max_wait_ms / 1000 or None,
        )
    return None


def build_tracer(settings: Settings) -> Tracer:
    if not settings.tracing_configured:
        return NoopTracer()
    try:
        from langfuse import Langfuse, propagate_attributes  # heavy import; only when configured

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            environment=settings.langfuse_environment or settings.env,
            release=__version__,
        )
        log.info("LLM tracing enabled", extra={"langfuse_host": settings.langfuse_host})
        return LangfuseTracer(client, propagate_attributes)
    except Exception:
        log.exception("could not start Langfuse tracing; continuing without it")
        return NoopTracer()


def build_job_queue(settings: Settings) -> JobQueue:
    if settings.worker_mode == "celery":
        from app.workers.celery_app import celery_app  # imported lazily: only workers need Celery

        return CeleryJobQueue(celery_app, settings.ingest_queue)
    return InlineJobQueue()


def build_qdrant(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_url == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=10)


async def build_rag(
    settings: Settings,
    redis: Redis,
    cache_redis: Redis,
    embedder: Embedder,
    chat: ChatService,
    tracer: Tracer,
    *,
    qdrant: AsyncQdrantClient | None = None,
    reranker: Reranker | None = None,
    job_queue: JobQueue | None = None,
) -> RagComponents:
    store = QdrantChunkStore(qdrant or build_qdrant(settings), settings.rag_collection)
    await store.setup(embedder.dim)
    documents = DocumentRegistry(redis)
    retriever = Retriever(
        store,
        embedder,
        reranker if reranker is not None else build_reranker(settings),
        candidates=settings.rag_rerank_candidates,
    )
    ingestion = IngestionService(
        store,
        documents,
        embedder,
        build_chunker(
            settings.rag_chunker, settings.rag_chunk_max_words, settings.rag_chunk_overlap_words
        ),
        chunker_name=(
            f"{settings.rag_chunker}/{settings.rag_chunk_max_words}"
            f"/{settings.rag_chunk_overlap_words}"
        ),
        max_bytes=settings.rag_max_upload_bytes,
    )
    job_store = JobStore(redis, settings.job_retention_seconds)
    dead_letters = DeadLetterQueue(redis)
    queue = job_queue or build_job_queue(settings)
    jobs = IngestionJobService(
        job_store,
        PayloadStore(cache_redis, settings.upload_payload_ttl_seconds),
        dead_letters,
        ingestion,
        queue,
        max_attempts=settings.job_max_attempts,
    )
    if isinstance(queue, InlineJobQueue):
        queue.bind(jobs)  # the inline queue runs jobs through the service that owns it
    return RagComponents(
        store=store,
        documents=documents,
        ingestion=ingestion,
        retriever=retriever,
        answers=RagService(retriever, chat, tracer),
        jobs=jobs,
        job_store=job_store,
        dead_letters=dead_letters,
    )


async def build_services(
    settings: Settings,
    *,
    redis: Redis | None = None,
    cache_redis: Redis | None = None,
    providers: Mapping[str, Provider] | None = None,
    routing: RoutingConfig | None = None,
    embedder: Embedder | None = None,
    qdrant: AsyncQdrantClient | None = None,
    reranker: Reranker | None = None,
    job_queue: JobQueue | None = None,
    tracer: Tracer | None = None,
) -> Services:
    check_production_secrets(settings)
    tracer = tracer or build_tracer(settings)

    # Two clients over the same server: string-decoding for app data, raw bytes for vectors.
    redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
    cache_redis = cache_redis or Redis.from_url(settings.redis_url, decode_responses=False)
    # One model instance serves both the semantic cache and RAG.
    embedder = embedder or build_embedder(settings)

    router = LLMRouter(
        routing or RoutingConfig.from_yaml(settings.routes_file),
        providers or {"litellm": LiteLLMProvider(), "mock": MockProvider()},
        default_timeout=settings.provider_timeout_seconds,
        breaker_factory=lambda: CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_cooldown_seconds
        ),
        faults=FaultInjector(redis) if settings.fault_injection_enabled else None,
    )

    cache: SemanticCache | None = None
    if settings.cache_enabled:
        index = (
            RediSearchIndex(cache_redis)
            if settings.cache_index_backend == "redisearch"
            else BruteForceIndex(cache_redis, settings.cache_max_entries_per_namespace)
        )
        cache = SemanticCache(
            cache_redis,
            embedder,
            index,
            threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries_per_namespace,
        )
        await cache.setup()

    meter = UsageMeter(redis, settings.usage_retention_days)
    chat = ChatService(router, cache, meter, tracer)
    return Services(
        settings=settings,
        redis=redis,
        cache_redis=cache_redis,
        router=router,
        cache=cache,
        keys=ApiKeyStore(redis),
        tokens=TokenService(
            settings.jwt_secret.get_secret_value(), settings.jwt_algorithm, settings.jwt_ttl_seconds
        ),
        limiter=TokenBucketLimiter(redis),
        meter=meter,
        chat=chat,
        tracer=tracer,
        rag=await build_rag(
            settings,
            redis,
            cache_redis,
            embedder,
            chat,
            tracer,
            qdrant=qdrant,
            reranker=reranker,
            job_queue=job_queue,
        ),
    )
