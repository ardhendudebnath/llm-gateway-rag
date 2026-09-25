"""End-to-end tests against the real FastAPI app (fakeredis + scripted providers)."""

from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.container import build_services
from app.main import create_app
from tests.conftest import chat_body, make_routing

REPO_ROOT = Path(__file__).resolve().parents[2]


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------- ops


async def test_health_and_readiness(client):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"]["redis"] == "ok"


async def test_request_id_is_generated_or_propagated(client):
    generated = await client.get("/healthz")
    assert len(generated.headers["x-request-id"]) == 32
    propagated = await client.get("/healthz", headers={"X-Request-ID": "trace-abc.123"})
    assert propagated.headers["x-request-id"] == "trace-abc.123"
    unsafe = await client.get("/healthz", headers={"X-Request-ID": "bad id\nwith newline"})
    assert unsafe.headers["x-request-id"] != "bad id\nwith newline"


async def test_metrics_endpoint_exposes_gateway_metrics(client, api_key):
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    body = (await client.get("/metrics")).text
    assert "nexusgate_http_requests_total" in body
    assert 'nexusgate_llm_calls_total{deployment="primary",outcome="success"}' in body


# ---------------------------------------------------------------- auth


async def test_chat_requires_credentials(client):
    resp = await client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["error"]["type"] == "http_error"


async def test_invalid_key_rejected(client):
    resp = await client.post(
        "/v1/chat/completions", json=chat_body(), headers=bearer("ng_nope_nope")
    )
    assert resp.status_code == 401


@pytest.mark.parametrize("header", ["x-api-key", "authorization"])
async def test_api_key_accepted_in_either_header(client, api_key, header):
    headers = {"X-API-Key": api_key} if header == "x-api-key" else bearer(api_key)
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    assert resp.status_code == 200, resp.text


async def test_jwt_exchange_and_use(client, api_key):
    token_resp = await client.post("/v1/auth/token", headers=bearer(api_key))
    assert token_resp.status_code == 200
    jwt_token = token_resp.json()["access_token"]

    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(jwt_token))
    assert resp.status_code == 200

    # A JWT cannot be used to mint further JWTs.
    assert (await client.post("/v1/auth/token", headers=bearer(jwt_token))).status_code == 401


async def test_revoking_key_kills_key_and_its_jwts(client, api_key, admin_headers):
    jwt_token = (await client.post("/v1/auth/token", headers=bearer(api_key))).json()[
        "access_token"
    ]
    key_id = api_key.split("_")[1]
    assert (
        await client.delete(f"/v1/admin/keys/{key_id}", headers=admin_headers)
    ).status_code == 204

    for creds in (api_key, jwt_token):
        resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(creds))
        assert resp.status_code == 401


async def test_admin_endpoints_require_admin_token(client):
    resp = await client.post("/v1/admin/keys", json={"tenant_id": "x", "name": "y"})
    assert resp.status_code == 403
    resp = await client.get("/v1/admin/providers", headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 403


async def test_admin_lists_keys_without_secrets(client, api_key, admin_headers):
    resp = await client.get("/v1/admin/keys", params={"tenant_id": "acme"}, headers=admin_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert api_key not in resp.text


# ---------------------------------------------------------------- chat + routing


async def test_chat_completion_openai_shape(client, api_key):
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    assert body["nexusgate"]["deployment"] == "primary"
    assert body["nexusgate"]["cost_usd"] == pytest.approx(0.00105)
    assert resp.headers["x-nexusgate-cache"] == "miss"
    assert resp.headers["x-nexusgate-deployment"] == "primary"


async def test_fallback_when_primary_is_down(client, api_key, provider):
    provider.always_fail("primary")
    resp = await client.post(
        "/v1/chat/completions", json=chat_body(cache=False), headers=bearer(api_key)
    )
    assert resp.status_code == 200
    meta = resp.json()["nexusgate"]
    assert meta["deployment"] == "secondary"
    assert [a["outcome"] for a in meta["attempts"]] == ["error", "success"]


async def test_breaker_opens_and_is_visible_to_admin(client, api_key, provider, admin_headers):
    provider.always_fail("primary")
    for i in range(4):
        await client.post(
            "/v1/chat/completions", json=chat_body(f"q{i}", cache=False), headers=bearer(api_key)
        )
    providers = (await client.get("/v1/admin/providers", headers=admin_headers)).json()
    assert providers["breakers"]["primary"]["state"] == "open"
    assert provider.calls.count("primary") == 3  # 4th request skipped it entirely


async def test_all_providers_down_returns_503_with_attempts(client, api_key, provider):
    provider.always_fail("primary")
    provider.always_fail("secondary")
    resp = await client.post(
        "/v1/chat/completions", json=chat_body(cache=False), headers=bearer(api_key)
    )
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "5"
    error = resp.json()["error"]
    assert error["type"] == "all_providers_failed"
    assert [a["deployment"] for a in error["attempts"]] == ["primary", "secondary"]


async def test_unknown_model_is_404(client, api_key):
    resp = await client.post(
        "/v1/chat/completions", json=chat_body(model="gpt-99"), headers=bearer(api_key)
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


async def test_validation_error_envelope(client, api_key):
    resp = await client.post(
        "/v1/chat/completions", json={"model": "default", "messages": []}, headers=bearer(api_key)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request"


async def test_models_lists_route_aliases(client, api_key):
    resp = await client.get("/v1/models", headers=bearer(api_key))
    assert [m["id"] for m in resp.json()["data"]] == ["default", "single", "selfhosted"]


# ---------------------------------------------------------------- cache + metering


async def test_second_identical_request_is_a_free_cache_hit(client, api_key, provider):
    first = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    second = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))

    assert second.headers["x-nexusgate-cache"] == "hit"
    meta = second.json()["nexusgate"]
    assert meta["cached"] is True
    assert meta["cost_usd"] == 0.0
    assert meta["cache_similarity"] == pytest.approx(1.0)
    assert second.json()["choices"] == first.json()["choices"]
    assert provider.calls == ["primary"]


async def test_cache_bypass_flag(client, api_key, provider):
    for _ in range(2):
        await client.post(
            "/v1/chat/completions", json=chat_body(cache=False), headers=bearer(api_key)
        )
    assert provider.calls == ["primary", "primary"]


async def test_cache_is_tenant_isolated(client, api_key, admin_headers, provider):
    other = (
        await client.post(
            "/v1/admin/keys", json={"tenant_id": "globex", "name": "x"}, headers=admin_headers
        )
    ).json()["api_key"]
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(other))
    assert resp.headers["x-nexusgate-cache"] == "miss"
    assert provider.calls == ["primary", "primary"]


async def test_tenant_cache_purge(client, api_key, provider):
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    purge = await client.delete("/v1/cache", headers=bearer(api_key))
    assert purge.json() == {"tenant_id": "acme", "entries_deleted": 1}
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    assert resp.headers["x-nexusgate-cache"] == "miss"


async def test_admin_purge_all(client, api_key, admin_headers):
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    resp = await client.delete("/v1/admin/cache", headers=admin_headers)
    assert resp.json() == {"entries_deleted": 1}


async def test_usage_reports_spend_and_savings(client, api_key):
    for _ in range(3):  # 1 miss + 2 hits
        await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    totals = (await client.get("/v1/usage", headers=bearer(api_key))).json()["totals"]
    assert totals["requests"] == 3
    assert totals["cache_hits"] == 2
    assert totals["prompt_tokens"] == 100
    assert totals["cost_usd"] == pytest.approx(0.00105)
    assert totals["cost_saved_usd"] == pytest.approx(0.0021)


async def test_usage_splits_cost_per_1k_by_cache_hit_provider_and_self_hosted(client, api_key):
    # A blended cost per request hides which lever moved it: caching and self-hosting both push
    # it down, and they cost completely different things to arrange.
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))  # hit
    await client.post(
        "/v1/chat/completions",
        json=chat_body("Something else entirely", model="selfhosted"),
        headers=bearer(api_key),
    )

    body = (await client.get("/v1/usage", headers=bearer(api_key))).json()
    per_1k, totals = body["cost_per_1k_usd"], body["totals"]

    assert per_1k["requests"] == {"cache_hit": 1, "provider_call": 1, "self_hosted": 1}
    assert totals["self_hosted_requests"] == 1
    assert per_1k["cache_hit"] == 0.0  # a hit makes no provider call
    assert per_1k["self_hosted"] == 0.0  # the GPU is paid for by the hour, not per token
    assert per_1k["provider_call"] == pytest.approx(1.05)  # $0.00105 for the one paid call
    assert per_1k["blended"] == pytest.approx(0.35)  # ... spread over all three requests


async def test_cache_failure_degrades_to_uncached(client, api_key, services, provider):
    async def broken(*_, **__):
        raise ConnectionError("redis gone")

    services.cache.lookup = broken
    services.cache.store = broken
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=bearer(api_key))
    assert resp.status_code == 200
    assert resp.headers["x-nexusgate-cache"] == "miss"


# ---------------------------------------------------------------- rate limiting


async def test_rate_limit_returns_429_with_retry_after(client, admin_headers):
    key = (
        await client.post(
            "/v1/admin/keys",
            json={
                "tenant_id": "acme",
                "name": "tiny",
                "rate_limit_capacity": 2,
                "rate_limit_refill_per_sec": 0.5,
            },
            headers=admin_headers,
        )
    ).json()["api_key"]

    ok = [await client.get("/v1/models", headers=bearer(key)) for _ in range(2)]
    assert [r.status_code for r in ok] == [200, 200]
    assert ok[0].headers["x-ratelimit-limit"] == "2"
    assert ok[1].headers["x-ratelimit-remaining"] == "0"

    limited = await client.get("/v1/models", headers=bearer(key))
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "2"


# ---------------------------------------------------------------- lifecycle


async def test_cache_can_be_disabled(settings, redis_pair, provider, api_key):
    settings = settings.model_copy(update={"cache_enabled": False})
    text, raw = redis_pair
    services = await build_services(
        settings, redis=text, cache_redis=raw, providers={"fake": provider}, routing=make_routing()
    )
    assert services.cache is None

    app = create_app(settings, services)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.delete("/v1/cache", headers=bearer(api_key))
        assert resp.status_code == 409


async def test_shipped_config_chaos_route_falls_back_end_to_end(settings, redis_pair, api_key):
    """The real routes.yaml + real MockProvider: the offline demo a reviewer will run."""
    text, raw = redis_pair
    settings = settings.model_copy(update={"routes_file": REPO_ROOT / "config" / "routes.yaml"})
    services = await build_services(settings, redis=text, cache_redis=raw)

    app = create_app(settings, services)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post(
            "/v1/chat/completions",
            json=chat_body("ping", model="chaos", cache=False),
            headers=bearer(api_key),
        )
    assert resp.status_code == 200, resp.text
    meta = resp.json()["nexusgate"]
    assert [a["deployment"] for a in meta["attempts"]] == ["mock-broken", "mock-primary"]
    assert resp.json()["choices"][0]["message"]["content"] == "[mock-primary] You said: ping"


def test_openapi_schema_builds():
    app = create_app(Settings(env="test", log_level="WARNING"))
    paths = app.openapi()["paths"]
    assert "/v1/chat/completions" in paths
    assert "/v1/admin/keys" in paths
