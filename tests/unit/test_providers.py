"""Provider adapter contracts: vendor failures map onto the router's retryable/fail-fast split."""

import random

import litellm
import pytest

from app.gateway.providers import LiteLLMProvider, MockProvider, ProviderError
from app.gateway.routing_config import Deployment
from app.gateway.schemas import ChatMessage, ChatRequest

DEP = Deployment(name="d", provider="litellm", model="openai/gpt-test", api_base="http://x/v1")
REQ = ChatRequest(
    messages=[ChatMessage(role="user", content="hello there")], temperature=0.2, max_tokens=64
)


async def test_litellm_success_is_normalised(monkeypatch):
    real = litellm.acompletion
    seen = {}

    async def offline(**kwargs):
        seen.update(kwargs)
        return await real(**kwargs, mock_response="general kenobi")  # LiteLLM's offline mode

    monkeypatch.setattr(litellm, "acompletion", offline)
    resp = await LiteLLMProvider().complete(DEP, REQ)

    assert resp.content == "general kenobi"
    assert resp.usage.prompt_tokens > 0 and resp.usage.completion_tokens > 0
    assert seen["num_retries"] == 0  # retries/fallback belong to our router, not LiteLLM
    assert seen["api_base"] == "http://x/v1"
    assert (seen["temperature"], seen["max_tokens"]) == (0.2, 64)


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (litellm.exceptions.RateLimitError("slow down", "openai", "gpt-test"), True),
        (litellm.exceptions.Timeout("timed out", "gpt-test", "openai"), True),
        (litellm.exceptions.ServiceUnavailableError("503", "openai", "gpt-test"), True),
        (litellm.exceptions.AuthenticationError("bad key", "openai", "gpt-test"), True),
        (litellm.exceptions.BadRequestError("bad", "gpt-test", "openai"), False),
        (litellm.exceptions.ContextWindowExceededError("too long", "gpt-test", "openai"), False),
    ],
)
async def test_litellm_error_classification(monkeypatch, exc, retryable):
    async def boom(**_):
        raise exc

    monkeypatch.setattr(litellm, "acompletion", boom)
    with pytest.raises(ProviderError) as info:
        await LiteLLMProvider().complete(DEP, REQ)
    assert info.value.retryable is retryable


async def test_mock_provider_echoes_and_counts_tokens():
    dep = Deployment(name="m", provider="mock", model="mock-1", options={"latency_ms": 0})
    resp = await MockProvider().complete(dep, REQ)
    assert resp.content == "[m] You said: hello there"
    assert resp.usage.prompt_tokens == 2


async def test_mock_provider_failure_rate():
    dep = Deployment(
        name="m", provider="mock", model="mock-1", options={"latency_ms": 0, "failure_rate": 1.0}
    )
    with pytest.raises(ProviderError) as info:
        await MockProvider(random.Random(0)).complete(dep, REQ)
    assert info.value.retryable
