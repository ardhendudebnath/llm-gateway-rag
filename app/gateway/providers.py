"""Provider adapters. Each one turns a ``ChatRequest`` into a normalized ``ProviderResponse``
and maps vendor failures onto ``ProviderError`` so the router can decide whether to fall back.
"""

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Protocol

from app.gateway.routing_config import Deployment
from app.gateway.schemas import ChatRequest, ProviderResponse, StreamDelta, Usage


class ProviderError(Exception):
    """A provider call failed.

    ``retryable=True`` means the failure says something about the *provider* (timeout, 429, 5xx,
    auth misconfiguration) and the next deployment in the chain should be tried.
    ``retryable=False`` means the *request* is bad (400) — another provider would reject it too,
    so we fail fast instead of burning the whole chain.
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class Provider(Protocol):
    async def complete(self, deployment: Deployment, request: ChatRequest) -> ProviderResponse: ...


class StreamingProvider(Provider, Protocol):
    """A provider that can stream. Adapters without this are still usable for `stream=true`
    requests: the router chunks their complete answer instead (marked `synthesized`)."""

    def stream(
        self, deployment: Deployment, request: ChatRequest
    ) -> AsyncIterator[StreamDelta]: ...


class LiteLLMProvider:
    """Hosted providers (OpenAI, Anthropic, Mistral, ...) and self-hosted vLLM via LiteLLM's SDK.

    LiteLLM's own retries/fallbacks are disabled (``num_retries=0``): fallback and circuit breaking
    live in ``LLMRouter`` so the behaviour is explicit, observable, and unit-testable.
    """

    @staticmethod
    def _kwargs(deployment: Deployment, request: ChatRequest) -> dict:
        kwargs: dict = {
            "model": deployment.model,
            "messages": [m.model_dump() for m in request.messages],
            "num_retries": 0,
        }
        if deployment.api_base:
            kwargs["api_base"] = deployment.api_base
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        return kwargs

    async def complete(self, deployment: Deployment, request: ChatRequest) -> ProviderResponse:
        import litellm  # imported lazily: heavy module, not needed by tests or the mock provider

        kwargs = self._kwargs(deployment, request)
        try:
            resp = await litellm.acompletion(**kwargs)
        except litellm.exceptions.BadRequestError as e:  # includes context-window / content-policy
            raise ProviderError(str(e), retryable=False, status_code=400) from e
        except Exception as e:
            raise ProviderError(
                f"{type(e).__name__}: {e}",
                retryable=True,
                status_code=getattr(e, "status_code", None),
            ) from e

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            content=choice.message.content or "",
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=resp.model or deployment.model,
            finish_reason=choice.finish_reason or "stop",
        )

    async def stream(
        self, deployment: Deployment, request: ChatRequest
    ) -> AsyncIterator[StreamDelta]:
        """`include_usage` asks for a final chunk carrying token counts, so a streamed request is
        metered on the provider's numbers rather than on an estimate."""
        import litellm

        kwargs = self._kwargs(deployment, request) | {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            stream = await litellm.acompletion(**kwargs)
            async for part in stream:
                choices = getattr(part, "choices", None) or []
                choice = choices[0] if choices else None
                usage = getattr(part, "usage", None)
                yield StreamDelta(
                    content=(getattr(choice.delta, "content", None) or "") if choice else "",
                    finish_reason=getattr(choice, "finish_reason", None) if choice else None,
                    usage=(
                        Usage(
                            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        )
                        if usage
                        else None
                    ),
                    model=getattr(part, "model", None),
                )
        except litellm.exceptions.BadRequestError as e:
            raise ProviderError(str(e), retryable=False, status_code=400) from e
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"{type(e).__name__}: {e}",
                retryable=True,
                status_code=getattr(e, "status_code", None),
            ) from e


class MockProvider:
    """Offline provider for demos, load tests and chaos tests — no API keys, no cost.

    Options (per deployment, under ``options:`` in routes.yaml):
      latency_ms:   simulated response time (default 50)
      failure_rate: probability in [0, 1] of raising a retryable error (default 0)
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    async def complete(self, deployment: Deployment, request: ChatRequest) -> ProviderResponse:
        opts = deployment.options
        await asyncio.sleep(float(opts.get("latency_ms", 50)) / 1000)
        if self._rng.random() < float(opts.get("failure_rate", 0.0)):
            raise ProviderError(
                f"{deployment.name}: simulated 503", retryable=True, status_code=503
            )

        content = self._answer(deployment, request)
        return ProviderResponse(
            content=content,
            usage=self._usage(request, content),
            model=deployment.model,
        )

    async def stream(
        self, deployment: Deployment, request: ChatRequest
    ) -> AsyncIterator[StreamDelta]:
        """A word at a time, with the per-deployment latency spread across the stream, so the demo
        and the load tests exercise the real streaming path rather than one big chunk."""
        opts = deployment.options
        if self._rng.random() < float(opts.get("failure_rate", 0.0)):
            raise ProviderError(
                f"{deployment.name}: simulated 503", retryable=True, status_code=503
            )
        content = self._answer(deployment, request)
        words = content.split()
        per_word = float(opts.get("latency_ms", 50)) / 1000 / max(len(words), 1)
        for i, word in enumerate(words):
            await asyncio.sleep(per_word)
            if i and self._rng.random() < float(opts.get("stream_failure_rate", 0.0)):
                # Mid-stream failure, for exercising the path where fallback is no longer possible.
                raise ProviderError(
                    f"{deployment.name}: stream died mid-answer", retryable=True, status_code=503
                )
            yield StreamDelta(content=word if i == 0 else f" {word}", model=deployment.model)
        yield StreamDelta(
            finish_reason="stop", usage=self._usage(request, content), model=deployment.model
        )

    @staticmethod
    def _answer(deployment: Deployment, request: ChatRequest) -> str:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return f"[{deployment.name}] You said: {last_user}"

    @staticmethod
    def _usage(request: ChatRequest, content: str) -> Usage:
        return Usage(
            prompt_tokens=sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(content.split()),
        )
