"""Provider adapters. Each one turns a ``ChatRequest`` into a normalized ``ProviderResponse``
and maps vendor failures onto ``ProviderError`` so the router can decide whether to fall back.
"""

import asyncio
import random
from typing import Protocol

from app.gateway.routing_config import Deployment
from app.gateway.schemas import ChatRequest, ProviderResponse, Usage


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


class LiteLLMProvider:
    """Hosted providers (OpenAI, Anthropic, Mistral, ...) and self-hosted vLLM via LiteLLM's SDK.

    LiteLLM's own retries/fallbacks are disabled (``num_retries=0``): fallback and circuit breaking
    live in ``LLMRouter`` so the behaviour is explicit, observable, and unit-testable.
    """

    async def complete(self, deployment: Deployment, request: ChatRequest) -> ProviderResponse:
        import litellm  # imported lazily: heavy module, not needed by tests or the mock provider

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

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        content = f"[{deployment.name}] You said: {last_user}"
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return ProviderResponse(
            content=content,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(content.split())),
            model=deployment.model,
        )
