"""Streaming through the router: fallback up to the first token, and never after it."""

import asyncio

import pytest

from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.providers import ProviderError
from app.gateway.router import (
    AllProvidersFailedError,
    ClientRequestError,
    LLMRouter,
    StreamInterruptedError,
    UnknownModelError,
)
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import ChatRequest
from tests.conftest import Stall, StreamBreak, WholeResponseProvider, make_routing


@pytest.fixture
def router_for(provider):
    def build(routing: RoutingConfig | None = None, *, timeout: float = 5.0, **providers):
        return LLMRouter(
            routing or make_routing(),
            {"fake": provider, **providers},
            default_timeout=timeout,
            breaker_factory=lambda: CircuitBreaker(failure_threshold=3, cooldown_seconds=30),
            sleep=lambda _: asyncio.sleep(0),
        )

    return build


def ask(route: str = "default", content: str = "hello there") -> ChatRequest:
    return ChatRequest(model=route, messages=[{"role": "user", "content": content}], stream=True)


async def drain(opened) -> str:
    """Everything the client would receive, first token included."""
    text = [opened.first.content]
    async for delta in opened.rest:
        text.append(delta.content)
    return "".join(text)


async def test_a_stream_is_assembled_from_its_deltas(router_for):
    router = router_for()

    opened = await router.open_stream(ask())

    assert await drain(opened) == "primary: hello there"
    assert opened.deployment.name == "primary"
    assert opened.finish_reason == "stop"
    assert opened.ttft_ms > 0


async def test_usage_and_cost_are_settled_when_the_stream_ends(router_for):
    router = router_for()

    opened = await router.open_stream(ask())
    assert opened.cost_usd == 0.0, "nothing is priced before the provider reports usage"
    await drain(opened)

    assert (opened.usage.prompt_tokens, opened.usage.completion_tokens) == (100, 50)
    assert opened.cost_usd == pytest.approx(100 / 1e6 * 3.0 + 50 / 1e6 * 15.0)
    assert opened.usage_estimated is False


async def test_a_deployment_that_fails_before_the_first_token_is_replaced(router_for, provider):
    provider.script("primary", ProviderError("down", retryable=True, status_code=503))
    router = router_for()

    opened = await router.open_stream(ask())

    assert await drain(opened) == "secondary: hello there", "one clean answer, from the fallback"
    assert opened.deployment.name == "secondary"
    assert [(a.deployment, a.outcome) for a in opened.attempts] == [
        ("primary", "error"),
        ("secondary", "success"),
    ]


async def test_a_deployment_that_never_sends_a_first_token_is_replaced(router_for, provider):
    provider.script("primary", Stall())  # connected, then silence
    router = router_for(timeout=0.05)

    opened = await router.open_stream(ask())

    assert opened.deployment.name == "secondary"
    assert opened.attempts[0].outcome == "timeout"
    assert "no first token" in opened.attempts[0].error


async def test_a_stream_that_breaks_after_the_first_token_is_not_replaced(router_for, provider):
    # The whole point: the client already has text, so a second answer would be worse than none.
    provider.script("primary", StreamBreak(after=2))
    router = router_for()

    opened = await router.open_stream(ask())
    with pytest.raises(StreamInterruptedError) as caught:
        await drain(opened)

    assert caught.value.deployment == "primary"
    assert caught.value.delivered == "primary: hello", "what the client got before the break"
    assert provider.calls == ["primary"], "no fallback was attempted"
    assert opened.attempts[-1].outcome == "stream_interrupted"


async def test_a_stream_that_stalls_mid_answer_is_cut_off(router_for, provider):
    provider.script("primary", Stall(after=2))
    router = router_for(timeout=0.05)

    opened = await router.open_stream(ask())
    with pytest.raises(StreamInterruptedError, match="stalled for more than"):
        await drain(opened)

    assert provider.calls == ["primary"]


async def test_a_deployment_is_only_called_healthy_once_its_stream_completes(router_for, provider):
    """Answering is not succeeding: a first token proves it is reachable, nothing more."""
    router = router_for()
    router.breakers["primary"].record_failure()
    router.breakers["primary"].record_failure()

    opened = await router.open_stream(ask())
    assert router.breakers["primary"].consecutive_failures == 2, "still on probation"

    await drain(opened)

    assert router.breakers["primary"].consecutive_failures == 0, "a full answer clears it"


async def test_an_interruption_counts_against_the_breaker(router_for, provider):
    provider.script("primary", StreamBreak(after=1))
    router = router_for()

    opened = await router.open_stream(ask())
    with pytest.raises(StreamInterruptedError):
        await drain(opened)

    assert router.breakers["primary"].consecutive_failures == 1, "it answered, then it failed"


async def test_an_open_breaker_is_skipped(router_for, provider):
    router = router_for()
    for _ in range(3):
        router.breakers["primary"].record_failure()

    opened = await router.open_stream(ask())

    assert opened.deployment.name == "secondary"
    assert opened.attempts[0].outcome == "skipped_circuit_open"
    assert provider.calls == ["secondary"]


async def test_a_rejected_request_does_not_burn_the_chain(router_for, provider):
    provider.script("primary", ProviderError("bad prompt", retryable=False, status_code=400))
    router = router_for()

    with pytest.raises(ClientRequestError):
        await router.open_stream(ask())

    assert provider.calls == ["primary"], "another vendor would reject it too"
    assert router.breakers["primary"].consecutive_failures == 0


async def test_when_nothing_can_start_a_stream_the_request_fails_outright(router_for, provider):
    provider.always_fail("primary")
    provider.always_fail("secondary")
    router = router_for()

    with pytest.raises(AllProvidersFailedError) as caught:
        await router.open_stream(ask())

    assert {a.outcome for a in caught.value.attempts} == {"error"}


async def test_an_unknown_route_is_rejected(router_for):
    with pytest.raises(UnknownModelError):
        await router_for().open_stream(ask(route="nope"))


async def test_a_provider_that_cannot_stream_is_chunked_instead_of_refused(router_for):
    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(
                    name="whole",
                    provider="whole",
                    model="whole/v1",
                    pricing={"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                )
            ]
        }
    )
    router = router_for(routing, whole=WholeResponseProvider())

    opened = await router.open_stream(ask())
    text = await drain(opened)

    assert text == "one two three four five six seven eight"
    assert opened.synthesized is True
    assert opened.usage.completion_tokens == 8 and opened.usage_estimated is False


async def test_a_stream_without_reported_usage_is_estimated_and_says_so(router_for):
    class Silent:
        async def complete(self, deployment, request):  # pragma: no cover - unused
            raise AssertionError

        async def stream(self, deployment, request):
            from app.gateway.schemas import StreamDelta

            for word in ["alpha", " beta", " gamma"]:
                yield StreamDelta(content=word)

    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(
                    name="silent",
                    provider="silent",
                    model="silent/v1",
                    pricing={"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                )
            ]
        }
    )
    router = router_for(routing, silent=Silent())

    opened = await router.open_stream(ask())
    await drain(opened)

    assert opened.usage_estimated is True
    assert opened.usage.completion_tokens == 3  # counted from the text
    assert opened.usage.prompt_tokens == 2  # "hello there"
    assert opened.cost_usd > 0


async def test_a_deployment_is_retried_before_the_chain_moves_on(router_for, provider):
    routing = make_routing()
    routing.routes["default"][0].max_retries = 2
    provider.script("primary", ProviderError("blip", retryable=True, status_code=503))
    router = router_for(routing)

    opened = await router.open_stream(ask())

    assert opened.deployment.name == "primary", "the retry succeeded; no fallback was needed"
    assert provider.calls == ["primary", "primary"]
    assert [a.outcome for a in opened.attempts] == ["error", "success"]


async def test_retries_stop_as_soon_as_the_breaker_opens(router_for, provider):
    routing = make_routing()
    routing.routes["default"][0].max_retries = 5
    provider.always_fail("primary")
    router = router_for(routing)

    opened = await router.open_stream(ask())

    assert opened.deployment.name == "secondary"
    assert provider.calls.count("primary") == 3, "it stopped at the failure threshold, not at 6"
    assert router.breakers["primary"].state.value == "open"


async def test_repeated_interruptions_open_the_breaker(router_for, provider):
    router = router_for()
    for _ in range(3):
        provider.script("primary", StreamBreak(after=1))
        opened = await router.open_stream(ask())
        with pytest.raises(StreamInterruptedError):
            await drain(opened)

    assert router.breakers["primary"].state.value == "open"


async def test_an_adapter_bug_is_treated_as_a_provider_failure(router_for, provider):
    """A broken adapter must not take the gateway down with it, or block the fallback."""

    class Buggy:
        async def complete(self, deployment, request):  # pragma: no cover - unused
            raise AssertionError

        async def stream(self, deployment, request):
            raise ValueError("adapter bug")
            yield  # pragma: no cover - unreachable, makes this a generator

    routing = make_routing()
    routing.routes["default"][0].provider = "buggy"
    router = router_for(routing, buggy=Buggy())

    opened = await router.open_stream(ask())

    assert opened.deployment.name == "secondary", "the chain carried on"
    assert opened.attempts[0].error == "ValueError: adapter bug"


async def test_a_route_naming_an_unregistered_adapter_is_rejected(router_for):
    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(
                    name="ghost",
                    provider="fake",
                    model="fake/ghost",
                    pricing={"input_per_mtok": 1.0, "output_per_mtok": 1.0},
                )
            ]
        }
    )
    router = router_for(routing)
    router.config.routes["default"][0].provider = "nobody"  # as a canary table could

    with pytest.raises(UnknownModelError, match="no provider adapter"):
        await router.open_stream(ask())


async def test_usage_reported_early_is_not_overwritten_by_the_estimate(router_for):
    class EagerUsage:
        async def complete(self, deployment, request):  # pragma: no cover - unused
            raise AssertionError

        async def stream(self, deployment, request):
            from app.gateway.schemas import StreamDelta, Usage

            yield StreamDelta(content="one", usage=Usage(prompt_tokens=11, completion_tokens=22))
            yield StreamDelta(content=" two three four")

    routing = RoutingConfig(
        routes={
            "default": [
                Deployment(
                    name="eager",
                    provider="eager",
                    model="eager/v1",
                    pricing={"input_per_mtok": 1.0, "output_per_mtok": 1.0},
                )
            ]
        }
    )
    router = router_for(routing, eager=EagerUsage())

    opened = await router.open_stream(ask())
    await drain(opened)

    assert (opened.usage.prompt_tokens, opened.usage.completion_tokens) == (11, 22)
    assert opened.usage_estimated is False, "the provider's own numbers win"


async def test_the_provider_stream_is_closed_when_the_client_stops_early(router_for, provider):
    router = router_for()

    opened = await router.open_stream(ask())
    await anext(opened.rest)
    await opened.rest.aclose()  # what a disconnected client amounts to

    assert provider.closed == ["primary"], "the provider connection was released"


async def test_a_stalled_first_token_does_not_leak_the_provider_stream(router_for, provider):
    provider.script("primary", Stall())
    router = router_for(timeout=0.05)

    await router.open_stream(ask())

    assert provider.closed == ["primary"]
