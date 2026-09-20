"""Chat orchestration: semantic cache -> router (fallback chain) -> cache fill -> metering.

Every completion, cache hit included, produces one trace (Langfuse when configured) and one
structured log line carrying the request id, so cost and latency are visible in both places.
"""

import logging
import time

from app.cache.semantic_cache import CachedCompletion, SemanticCache
from app.core.metering import UsageMeter
from app.core.security import Principal
from app.gateway.router import LLMRouter, UnknownModelError
from app.gateway.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    ChoiceMessage,
    GatewayMeta,
    UsageOut,
)
from app.observability import metrics
from app.observability.tracing import ChatTrace, NoopTracer, Tracer

log = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        router: LLMRouter,
        cache: SemanticCache | None,
        meter: UsageMeter,
        tracer: Tracer | None = None,
    ):
        self._router = router
        self._cache = cache
        self._meter = meter
        self._tracer = tracer or NoopTracer()

    async def complete(self, principal: Principal, request: ChatRequest) -> ChatResponse:
        if request.model not in self._router.config.routes:
            raise UnknownModelError(request.model)
        with self._tracer.chat(
            tenant_id=principal.tenant_id,
            key_id=principal.key_id,
            route=request.model,
            messages=[m.model_dump() for m in request.messages],
            model_parameters={"temperature": request.temperature, "max_tokens": request.max_tokens},
        ) as trace:
            response = await self._complete(principal, request, trace)
            _log_completion(principal, trace, response.nexusgate.latency_ms)
            return response

    async def _complete(
        self, principal: Principal, request: ChatRequest, trace: ChatTrace
    ) -> ChatResponse:
        start = time.perf_counter()
        use_cache = self._cache is not None and request.cache

        if use_cache:
            hit = await self._lookup(principal, request)
            if hit is not None:
                cached = hit.completion
                metrics.CACHE_LOOKUPS.labels("hit").inc()
                metrics.CACHE_COST_SAVED.inc(cached.cost_usd)
                trace.output = cached.content
                trace.model = cached.model
                trace.cached = True
                trace.cache_similarity = hit.similarity
                trace.prompt_tokens = cached.prompt_tokens
                trace.completion_tokens = cached.completion_tokens
                await self._meter.record(
                    principal.key_id,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    cached=True,
                    cost_saved_usd=cached.cost_usd,
                )
                return _build_response(
                    content=cached.content,
                    model=cached.model,
                    finish_reason=cached.finish_reason,
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    meta=GatewayMeta(
                        deployment=None,
                        cached=True,
                        cache_similarity=hit.similarity,
                        cost_usd=0.0,
                        latency_ms=_ms_since(start),
                    ),
                )
            metrics.CACHE_LOOKUPS.labels("miss").inc()

        result = await self._router.complete(request)
        resp = result.response
        trace.output = resp.content
        trace.model = resp.model
        trace.deployment = result.deployment.name
        trace.prompt_tokens = resp.usage.prompt_tokens
        trace.completion_tokens = resp.usage.completion_tokens
        trace.cost_usd = result.cost_usd
        trace.attempts = [a.model_dump() for a in result.attempts]

        if use_cache:
            await self._store(
                principal,
                request,
                CachedCompletion(
                    content=resp.content,
                    model=resp.model,
                    finish_reason=resp.finish_reason,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    deployment=result.deployment.name,
                    cost_usd=result.cost_usd,
                ),
            )
        await self._meter.record(
            principal.key_id,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            cost_usd=result.cost_usd,
            cached=False,
        )
        return _build_response(
            content=resp.content,
            model=resp.model,
            finish_reason=resp.finish_reason,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            meta=GatewayMeta(
                deployment=result.deployment.name,
                cached=False,
                cost_usd=result.cost_usd,
                latency_ms=_ms_since(start),
                attempts=result.attempts,
            ),
        )

    # The cache is an optimisation: if it is unavailable, serve uncached rather than fail.
    async def _lookup(self, principal: Principal, request: ChatRequest):
        try:
            return await self._cache.lookup(principal.tenant_id, request)
        except Exception:
            metrics.CACHE_LOOKUPS.labels("error").inc()
            log.exception("semantic cache lookup failed; serving uncached")
            return None

    async def _store(
        self, principal: Principal, request: ChatRequest, completion: CachedCompletion
    ):
        try:
            await self._cache.store(principal.tenant_id, request, completion)
        except Exception:
            log.exception("semantic cache store failed")


def _log_completion(principal: Principal, trace: ChatTrace, latency_ms: float) -> None:
    """One line per completion. The request id ties it to the access log and the Langfuse trace."""
    log.info(
        "chat completion",
        extra={
            "tenant_id": principal.tenant_id,
            "key_id": principal.key_id,
            "route": trace.route,
            "deployment": trace.deployment,
            "cached": trace.cached,
            "model": trace.model,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "cost_usd": round(trace.cost_usd, 8),
            "latency_ms": latency_ms,
        },
    )


def _build_response(
    *,
    content: str,
    model: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    meta: GatewayMeta,
) -> ChatResponse:
    return ChatResponse(
        model=model,
        choices=[Choice(message=ChoiceMessage(content=content), finish_reason=finish_reason)],
        usage=UsageOut(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        nexusgate=meta,
    )


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
