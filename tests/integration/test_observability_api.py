"""Traces produced by real requests, and the Alertmanager webhook."""

from tests.conftest import ADMIN_TOKEN, chat_body
from tests.integration.test_rag_api import auth, upload_and_wait

ALERT_PAYLOAD = {
    "status": "firing",
    "receiver": "nexusgate",
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "NexusGateHighErrorRate", "severity": "critical"},
            "annotations": {"summary": "Over 5% of requests are failing"},
            "startsAt": "2026-09-20T10:00:00Z",
            "fingerprint": "abc123",
        }
    ],
}


async def test_a_completion_is_traced_with_tokens_and_cost(client, api_key, tracer):
    await client.post("/v1/chat/completions", json=chat_body(), headers=auth(api_key))

    [call] = tracer.chats
    assert call["tenant_id"] == "acme"
    assert call["messages"][-1]["content"] == "What is the capital of France?"
    assert call["model_parameters"] == {"temperature": None, "max_tokens": None}

    trace = call["trace"]
    assert trace.deployment == "primary" and trace.cached is False
    assert (trace.prompt_tokens, trace.completion_tokens) == (100, 50)
    assert trace.cost_usd > 0
    assert trace.output.endswith("What is the capital of France?")
    assert trace.attempts[0]["outcome"] == "success"


async def test_a_cache_hit_is_traced_as_cached_and_free(client, api_key, tracer):
    for _ in range(2):
        await client.post("/v1/chat/completions", json=chat_body(), headers=auth(api_key))

    first, second = (c["trace"] for c in tracer.chats)
    assert (first.cached, second.cached) == (False, True)
    assert second.cost_usd == 0.0 and second.cache_similarity is not None
    assert second.deployment is None


async def test_a_failed_route_is_traced_with_the_error(client, api_key, tracer, provider):
    provider.always_fail("primary")
    provider.always_fail("secondary")
    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(api_key))
    assert resp.status_code == 503
    assert "AllProvidersFailedError" in tracer.chats[0]["trace"].error


async def test_rag_answers_open_a_retrieval_span_around_the_completion(client, api_key, tracer):
    await upload_and_wait(client, api_key)
    await client.post(
        "/v1/rag/answer", headers=auth(api_key), json={"question": "SEV1 acknowledgement?"}
    )
    names = [name for name, _ in tracer.spans]
    assert names == ["rag.retrieve"]
    assert tracer.spans[0][1]["top_k"] == 5
    assert tracer.chats  # the completion was traced inside that span


async def test_alertmanager_webhook_records_alerts(client, admin_headers):
    resp = await client.post(
        "/v1/alerts/webhook", json=ALERT_PAYLOAD, auth=("alertmanager", ADMIN_TOKEN)
    )
    assert resp.status_code == 200 and resp.json() == {"received": 1}

    listed = (await client.get("/v1/admin/alerts", headers=admin_headers)).json()["alerts"]
    assert listed[0]["labels"]["alertname"] == "NexusGateHighErrorRate"
    assert listed[0]["annotations"]["summary"] == "Over 5% of requests are failing"

    body = (await client.get("/metrics")).text
    assert (
        'nexusgate_alerts_received_total{alertname="NexusGateHighErrorRate",status="firing"}'
        in body
    )


async def test_the_webhook_needs_the_admin_token(client):
    assert (await client.post("/v1/alerts/webhook", json=ALERT_PAYLOAD)).status_code == 401
    wrong = await client.post(
        "/v1/alerts/webhook", json=ALERT_PAYLOAD, auth=("alertmanager", "not-the-token")
    )
    assert wrong.status_code == 401


async def test_recent_alerts_need_the_admin_token(client, api_key):
    assert (await client.get("/v1/admin/alerts", headers=auth(api_key))).status_code == 403
