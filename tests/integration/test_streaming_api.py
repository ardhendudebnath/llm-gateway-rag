"""`stream=true` over HTTP: the SSE wire format, and who learns about a failure how."""

import json

import pytest

from tests.conftest import Stall, StreamBreak, chat_body


async def collect(client, key: str, content: str = "hello there", **extra):
    """Read a streamed completion the way a client does. Returns (response, chunks, raw lines)."""
    chunks, lines = [], []
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_body(content, stream=True, **extra),
        headers={"Authorization": f"Bearer {key}"},
    ) as resp:
        assert resp.status_code == 200, (await resp.aread()).decode()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            lines.append(line)
            payload = line.removeprefix("data: ")
            if payload != "[DONE]":
                chunks.append(json.loads(payload))
        return resp, chunks, lines


def text_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


async def test_a_streamed_completion_arrives_as_openai_shaped_chunks(client, api_key):
    resp, chunks, lines = await collect(client, api_key)

    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert lines[-1] == "data: [DONE]", "clients stop on the sentinel, not on the socket closing"
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert len({c["id"] for c in chunks}) == 1, "one completion, one id"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert text_of(chunks) == "primary: hello there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_the_last_chunk_carries_usage_and_the_gateway_footer(client, api_key):
    _, chunks, _ = await collect(client, api_key, cache=False)

    last = chunks[-1]
    assert last["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    meta = last["nexusgate"]
    assert meta["deployment"] == "primary" and meta["cached"] is False
    assert meta["ttft_ms"] > 0 and meta["cost_usd"] > 0
    assert meta["synthesized"] is False and meta["usage_estimated"] is False
    assert "interrupted" not in meta, "excluded when the stream completed"


async def test_a_streamed_answer_is_cached_and_replayed_as_a_stream(client, api_key, provider):
    await collect(client, api_key, "what is cached?")

    _, chunks, _ = await collect(client, api_key, "what is cached?")

    assert text_of(chunks) == "primary: what is cached?", "identical to the live answer"
    meta = chunks[-1]["nexusgate"]
    assert meta["cached"] is True and meta["cost_usd"] == 0.0
    assert provider.calls == ["primary"], "the second request never reached a provider"


async def test_the_cache_is_shared_between_streamed_and_whole_responses(client, api_key, provider):
    whole = await client.post(
        "/v1/chat/completions",
        json=chat_body("shared question"),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert whole.status_code == 200

    _, chunks, _ = await collect(client, api_key, "shared question")

    assert text_of(chunks) == whole.json()["choices"][0]["message"]["content"]
    assert chunks[-1]["nexusgate"]["cached"] is True
    assert provider.calls == ["primary"]


async def test_a_failure_before_the_first_token_is_an_ordinary_http_error(
    client, api_key, provider
):
    # Nothing has been sent, so the status line is still ours to choose.
    provider.always_fail("primary")
    provider.always_fail("secondary")

    resp = await client.post(
        "/v1/chat/completions",
        json=chat_body("hello", stream=True),
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["retry-after"] == "5"
    error = resp.json()["error"]
    assert error["type"] == "all_providers_failed"
    assert [a["outcome"] for a in error["attempts"]] == ["error", "error"]


async def test_a_request_the_provider_rejects_is_a_400_not_a_stream(client, api_key, provider):
    from app.gateway.providers import ProviderError

    provider.script("primary", ProviderError("bad prompt", retryable=False, status_code=400))

    resp = await client.post(
        "/v1/chat/completions",
        json=chat_body("hello", stream=True),
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert resp.status_code == 400


async def test_fallback_is_invisible_to_a_streaming_client(client, api_key, provider):
    provider.always_fail("primary", n=1)

    _, chunks, _ = await collect(client, api_key, cache=False)

    assert text_of(chunks) == "secondary: hello there", "one answer, no sign of the first attempt"
    attempts = chunks[-1]["nexusgate"]["attempts"]
    assert [a["outcome"] for a in attempts] == ["error", "success"]


async def test_a_break_after_the_first_token_ends_the_stream_honestly(client, api_key, provider):
    provider.script("primary", StreamBreak(after=2))

    _, chunks, lines = await collect(client, api_key, cache=False)

    assert text_of(chunks) == "primary: hello", "the partial answer stands"
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "error"
    assert "died mid-stream" in last["nexusgate"]["interrupted"]
    assert lines[-1] == "data: [DONE]", "the stream still terminates cleanly"
    assert provider.calls == ["primary"], "no second answer was appended"


async def test_a_partial_answer_is_never_cached(client, api_key, provider):
    provider.script("primary", StreamBreak(after=2))
    await collect(client, api_key, "will this be cached?")

    _, chunks, _ = await collect(client, api_key, "will this be cached?")

    assert chunks[-1]["nexusgate"]["cached"] is False
    assert text_of(chunks) == "primary: will this be cached?", "answered afresh, in full"


async def test_a_streamed_request_is_metered_like_any_other(client, api_key):
    await collect(client, api_key, cache=False)

    auth = {"Authorization": f"Bearer {api_key}"}
    totals = (await client.get("/v1/usage", params={"days": 1}, headers=auth)).json()["totals"]

    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == 100 and totals["completion_tokens"] == 50
    assert totals["cost_usd"] > 0


async def test_an_interrupted_stream_is_still_metered(client, api_key, provider):
    """The provider generated those tokens whether or not the answer finished."""
    provider.script("primary", StreamBreak(after=2))
    await collect(client, api_key, cache=False)

    auth = {"Authorization": f"Bearer {api_key}"}
    totals = (await client.get("/v1/usage", params={"days": 1}, headers=auth)).json()["totals"]

    assert totals["requests"] == 1 and totals["cost_usd"] > 0


async def test_a_replayed_stream_is_metered_as_a_cache_hit(client, api_key):
    await collect(client, api_key, "meter me")
    await collect(client, api_key, "meter me")

    auth = {"Authorization": f"Bearer {api_key}"}
    totals = (await client.get("/v1/usage", params={"days": 1}, headers=auth)).json()["totals"]

    assert totals["requests"] == 2 and totals["cache_hits"] == 1
    assert totals["cost_saved_usd"] > 0


async def test_a_streamed_request_can_be_served_by_a_canary(client, api_key, admin_headers):
    canary = """
    routes:
      default:
        - name: candidate
          provider: fake
          model: fake/candidate
          pricing: { input_per_mtok: 0.5, output_per_mtok: 1.5 }
    """
    published = await client.put(
        "/v1/admin/routes/canary",
        json={"config": canary, "weight": 100},
        headers=admin_headers,
    )
    assert published.status_code == 200, published.text

    _, chunks, _ = await collect(client, api_key, cache=False)

    meta = chunks[-1]["nexusgate"]
    assert meta["deployment"] == "candidate" and meta["route_variant"] == "canary"
    assert text_of(chunks) == "candidate: hello there"
    view = (await client.get("/v1/admin/routes", headers=admin_headers)).json()
    assert view["canary"]["stats"] == {"requests": 1, "errors": 0, "error_rate": 0.0}


async def test_a_stream_that_never_starts_does_not_hang_the_request(client, api_key, provider):
    provider.script("primary", Stall(), Stall())
    provider.script("secondary", Stall(), Stall())

    resp = await client.post(
        "/v1/chat/completions",
        json=chat_body("hello", stream=True),
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert resp.status_code == 503, "the first-token timeout applies to every deployment in turn"


@pytest.mark.parametrize("stream", [True, False])
async def test_both_modes_need_a_key(client, stream):
    resp = await client.post("/v1/chat/completions", json=chat_body("hi", stream=stream))

    assert resp.status_code == 401
