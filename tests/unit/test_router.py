"""Fallback-chain contract: deterministic, no network, no API credits."""

import asyncio

import pytest

from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.providers import ProviderError
from app.gateway.router import (
    AllProvidersFailedError,
    ClientRequestError,
    LLMRouter,
    UnknownModelError,
)
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import ChatMessage, ChatRequest
from tests.conftest import ScriptedProvider, make_routing


def request(model: str = "default") -> ChatRequest:
    return ChatRequest(model=model, messages=[ChatMessage(role="user", content="hi")])


async def _no_sleep(_: float) -> None:
    return None


def build_router(provider, routing=None, threshold=3, timeout=1.0) -> LLMRouter:
    return LLMRouter(
        routing or make_routing(),
        {"fake": provider},
        default_timeout=timeout,
        breaker_factory=lambda: CircuitBreaker(threshold, 30),
        sleep=_no_sleep,
    )


async def test_primary_success_uses_primary_only():
    provider = ScriptedProvider()
    result = await build_router(provider).complete(request())
    assert result.deployment.name == "primary"
    assert provider.calls == ["primary"]
    assert [a.outcome for a in result.attempts] == ["success"]


async def test_cost_is_computed_from_deployment_pricing():
    result = await build_router(ScriptedProvider()).complete(request())
    # 100 prompt tokens @ $3/M + 50 completion tokens @ $15/M
    assert result.cost_usd == pytest.approx(100 * 3 / 1e6 + 50 * 15 / 1e6)


async def test_falls_back_on_retryable_error():
    provider = ScriptedProvider()
    provider.script("primary", ProviderError("429", retryable=True, status_code=429))
    result = await build_router(provider).complete(request())
    assert result.deployment.name == "secondary"
    assert provider.calls == ["primary", "secondary"]
    assert [a.outcome for a in result.attempts] == ["error", "success"]


async def test_falls_back_on_timeout():
    provider = ScriptedProvider()

    async def hang():
        await asyncio.sleep(10)

    provider.script("primary", hang)
    result = await build_router(provider, timeout=0.05).complete(request())
    assert result.deployment.name == "secondary"
    assert result.attempts[0].outcome == "timeout"


async def test_unexpected_adapter_exception_is_treated_as_retryable():
    provider = ScriptedProvider()
    provider.script("primary", RuntimeError("adapter bug"))
    result = await build_router(provider).complete(request())
    assert result.deployment.name == "secondary"


async def test_client_error_fails_fast_without_fallback():
    provider = ScriptedProvider()
    provider.script("primary", ProviderError("context too long", retryable=False, status_code=400))
    router = build_router(provider)
    with pytest.raises(ClientRequestError):
        await router.complete(request())
    assert provider.calls == ["primary"]
    assert router.breakers["primary"].consecutive_failures == 0  # a bad request isn't an outage


async def test_all_failed_raises_with_every_attempt_recorded():
    provider = ScriptedProvider()
    provider.always_fail("primary")
    provider.always_fail("secondary")
    with pytest.raises(AllProvidersFailedError) as exc:
        await build_router(provider).complete(request())
    assert [a.deployment for a in exc.value.attempts] == ["primary", "secondary"]


async def test_unknown_route():
    with pytest.raises(UnknownModelError):
        await build_router(ScriptedProvider()).complete(request("nope"))


async def test_open_breaker_skips_deployment_without_calling_it():
    provider = ScriptedProvider()
    provider.always_fail("primary")
    router = build_router(provider, threshold=2)
    for _ in range(2):
        await router.complete(request())
    provider.calls.clear()

    result = await router.complete(request())
    assert provider.calls == ["secondary"]
    assert result.attempts[0].outcome == "skipped_circuit_open"
    assert router.breaker_states()["primary"]["state"] == "open"


async def test_retries_same_deployment_before_falling_back():
    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(name="primary", provider="fake", model="p", max_retries=2),
                Deployment(name="secondary", provider="fake", model="s"),
            ]
        }
    )
    provider = ScriptedProvider()
    provider.script("primary", ProviderError("blip", retryable=True))
    result = await build_router(provider, routing=routing).complete(request())
    assert provider.calls == ["primary", "primary"]
    assert result.deployment.name == "primary"


async def test_retries_stop_once_breaker_opens():
    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(name="primary", provider="fake", model="p", max_retries=5),
                Deployment(name="secondary", provider="fake", model="s"),
            ]
        }
    )
    provider = ScriptedProvider()
    provider.always_fail("primary")
    result = await build_router(provider, routing=routing, threshold=2).complete(request())
    assert provider.calls == ["primary", "primary", "secondary"]
    assert result.deployment.name == "secondary"


def test_router_rejects_routes_with_unregistered_provider():
    with pytest.raises(ValueError, match="unregistered providers"):
        LLMRouter(
            make_routing(),
            {"other": ScriptedProvider()},
            default_timeout=1,
            breaker_factory=lambda: CircuitBreaker(1, 1),
        )
