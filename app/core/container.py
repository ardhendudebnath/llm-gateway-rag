"""Composition root: builds every long-lived service once per process.

Everything is injectable so tests can swap in fakeredis, fake providers and a deterministic
embedder without monkeypatching.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from redis.asyncio import Redis

from app.cache.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from app.cache.semantic_cache import BruteForceIndex, RediSearchIndex, SemanticCache
from app.core.config import Settings
from app.core.metering import UsageMeter
from app.core.rate_limit import TokenBucketLimiter
from app.core.security import ApiKeyStore, TokenService
from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.providers import LiteLLMProvider, MockProvider, Provider
from app.gateway.router import LLMRouter
from app.gateway.routing_config import RoutingConfig
from app.gateway.service import ChatService

_INSECURE_DEFAULTS = {
    "admin_token": Settings.model_fields["admin_token"].default.get_secret_value(),
    "jwt_secret": Settings.model_fields["jwt_secret"].default.get_secret_value(),
}


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

    async def aclose(self) -> None:
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
        return FastEmbedEmbedder(settings.embedding_model)
    return HashingEmbedder()


async def build_services(
    settings: Settings,
    *,
    redis: Redis | None = None,
    cache_redis: Redis | None = None,
    providers: Mapping[str, Provider] | None = None,
    routing: RoutingConfig | None = None,
    embedder: Embedder | None = None,
) -> Services:
    check_production_secrets(settings)

    # Two clients over the same server: string-decoding for app data, raw bytes for vectors.
    redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
    cache_redis = cache_redis or Redis.from_url(settings.redis_url, decode_responses=False)

    router = LLMRouter(
        routing or RoutingConfig.from_yaml(settings.routes_file),
        providers or {"litellm": LiteLLMProvider(), "mock": MockProvider()},
        default_timeout=settings.provider_timeout_seconds,
        breaker_factory=lambda: CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_cooldown_seconds
        ),
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
            embedder or build_embedder(settings),
            index,
            threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries_per_namespace,
        )
        await cache.setup()

    meter = UsageMeter(redis, settings.usage_retention_days)
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
        chat=ChatService(router, cache, meter),
    )
