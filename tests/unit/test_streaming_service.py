"""Streaming orchestration: what happens when the client leaves, and what the footer reports."""

import asyncio

import pytest

from app.core.security import Principal
from app.gateway.schemas import ChatRequest
from tests.conftest import StreamBreak

PRINCIPAL = Principal(key_id="k1", tenant_id="acme")


def ask(content: str = "hello there", **extra) -> ChatRequest:
    return ChatRequest(
        model="default", messages=[{"role": "user", "content": content}], stream=True, **extra
    )


async def usage_of(services, key_id: str = "k1") -> dict:
    days = await services.meter.daily(key_id, days=1)
    return days[-1].model_dump()


@pytest.fixture
def stream(services):
    def start(request: ChatRequest | None = None):
        return services.chat.stream(PRINCIPAL, request or ask(cache=False))

    return start


async def test_a_client_that_hangs_up_is_still_billed_for_what_was_generated(services, stream):
    """The provider produced those tokens whether or not anyone read them."""
    gen = stream()
    await anext(gen)
    await anext(gen)

    await gen.aclose()  # the client goes away mid-answer
    await asyncio.sleep(0)  # let the metering task the generator handed off actually run

    usage = await usage_of(services)
    assert usage["requests"] == 1
    assert usage["completion_tokens"] > 0 and usage["cost_usd"] > 0


async def test_hanging_up_bills_the_words_that_were_sent_not_the_whole_answer(services, stream):
    # The provider would have reported 50 completion tokens for the full answer; two words went
    # out, so two are billed. The running estimate exists for exactly this moment.
    gen = stream()
    await anext(gen)
    await anext(gen)
    await gen.aclose()
    await asyncio.sleep(0)

    usage = await usage_of(services)
    assert usage["completion_tokens"] == 2


async def test_an_interrupted_stream_is_billed_from_its_partial_text(services, provider, stream):
    provider.script("primary", StreamBreak(after=2))

    chunks = [chunk async for chunk in stream()]

    meta = chunks[-1].nexusgate
    assert meta.interrupted and meta.usage_estimated is True
    assert chunks[-1].usage.completion_tokens == 2, "two words reached the client"
    usage = await usage_of(services)
    assert usage["completion_tokens"] == 2 and usage["cost_usd"] > 0


async def test_the_stream_reports_the_deployment_that_served_it(services, provider, stream):
    provider.always_fail("primary", n=1)

    chunks = [chunk async for chunk in stream()]

    meta = chunks[-1].nexusgate
    assert meta.deployment == "secondary"
    assert [a.outcome for a in meta.attempts] == ["error", "success"]
    assert meta.ttft_ms > 0 and meta.latency_ms >= meta.ttft_ms


async def test_a_streamed_answer_and_a_whole_one_agree(services, stream):
    chunks = [c async for c in stream(ask("same question", cache=False))]
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks)

    whole = await services.chat.complete(PRINCIPAL, ask("same question", cache=False))

    assert streamed == whole.choices[0].message.content
    assert chunks[-1].usage.model_dump() == whole.usage.model_dump()


async def test_an_unknown_route_fails_before_anything_is_streamed(services):
    gen = services.chat.stream(PRINCIPAL, ask())
    gen_model = ask()
    gen_model.model = "nope"

    from app.gateway.router import UnknownModelError

    with pytest.raises(UnknownModelError):
        await anext(services.chat.stream(PRINCIPAL, gen_model))
    await gen.aclose()
