"""The agent over HTTP: a real run through the container, the graph endpoint, and tenant scoping."""

from app.agents.nodes import NO_CONTEXT_ANSWER
from tests.integration.test_rag_api import HANDBOOK, auth, upload_and_wait


async def research(client, key, **body):
    return await client.post(
        "/v1/agents/research",
        headers=auth(key),
        json={"question": "When is a postmortem due?", "model": "default", **body},
    )


async def test_a_run_plans_retrieves_drafts_and_reviews(client, api_key, provider):
    await upload_and_wait(client, api_key, HANDBOOK)

    resp = await research(client, api_key, max_revisions=1)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [s["node"] for s in body["steps"]] == [
        "plan",
        "retrieve",
        "draft",
        "critique",
        "revise",
        "critique",
    ]
    assert body["searches"] and body["citations"]
    assert body["revisions"] == 1
    assert body["llm_calls"] == 4  # plan, draft, critique, revise
    assert body["usage"]["total_tokens"] > 0 and body["cost_usd"] > 0
    assert len(provider.calls) == 4  # every step went through the router, not around it


async def test_the_review_can_be_switched_off(client, api_key):
    await upload_and_wait(client, api_key, HANDBOOK)

    body = (await research(client, api_key, max_revisions=0)).json()

    assert [s["node"] for s in body["steps"]] == ["plan", "retrieve", "draft", "critique"]
    assert body["llm_calls"] == 2 and body["revisions"] == 0
    assert body["steps"][-1]["note"] == "revision budget spent"


async def test_without_documents_it_answers_that_it_found_nothing_and_calls_no_model(
    client, api_key, provider
):
    body = (await research(client, api_key)).json()

    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["citations"] == [] and body["usage"] is not None  # planning still ran
    assert body["llm_calls"] == 1 and len(provider.calls) == 1


async def test_one_tenants_documents_never_reach_anothers_run(client, admin_headers, api_key):
    await upload_and_wait(client, api_key, HANDBOOK)
    other = (
        await client.post(
            "/v1/admin/keys", json={"tenant_id": "other", "name": "t2"}, headers=admin_headers
        )
    ).json()["api_key"]

    body = (await research(client, other)).json()

    assert body["answer"] == NO_CONTEXT_ANSWER and body["citations"] == []


async def test_the_graph_is_published_with_its_diagram(client, api_key):
    body = (await client.get("/v1/agents/graph", headers=auth(api_key))).json()

    assert body["entry"] == "plan"
    assert set(body["nodes"]) == {"plan", "retrieve", "draft", "critique", "revise"}
    assert body["edges"]["critique"] == ["revise", "end"]
    assert "flowchart TD" in body["mermaid"] and "revise --> critique" in body["mermaid"]


async def test_the_agent_needs_a_key(client):
    assert (await client.post("/v1/agents/research", json={"question": "hi"})).status_code == 401


async def test_an_unknown_route_fails_fast_instead_of_degrading(client, api_key, provider):
    await upload_and_wait(client, api_key, HANDBOOK)

    resp = await research(client, api_key, model="no-such-route")

    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"
    assert provider.calls == []  # it didn't retrieve, or pay for a step, before failing


async def test_the_run_is_traced_as_one_span_with_a_completion_per_step(client, api_key, tracer):
    await upload_and_wait(client, api_key, HANDBOOK)

    await research(client, api_key, max_revisions=1)

    assert ("agent.research", {"question": "When is a postmortem due?", "model": "default"}) in [
        (name, meta) for name, meta in tracer.spans
    ]
    assert len([c for c in tracer.chats if c["tenant_id"] == "acme"]) >= 4
