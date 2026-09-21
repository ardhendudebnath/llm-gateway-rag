"""Under overload the gateway degrades before failing: skip reranking, skip the cache, then shed."""

from app.core.concurrency import OverloadedError
from app.observability import metrics
from tests.conftest import chat_body
from tests.integration.test_rag_api import auth, upload_and_wait


class SaturatedReranker:
    name = "saturated"

    async def scores(self, query, passages):
        raise OverloadedError("reranker is saturated; try again shortly")


class WorkingReranker:
    name = "working"

    async def scores(self, query, passages):
        return [float(len(p)) for p in passages]


def degraded(mode: str) -> float:
    return metrics.DEGRADED.labels(mode)._value.get()


async def test_a_saturated_reranker_falls_back_to_vector_order(client, api_key, services):
    await upload_and_wait(client, api_key)
    services.rag.retriever.reranker = WorkingReranker()
    reranked = (
        await client.post("/v1/rag/search", headers=auth(api_key), json={"query": "postmortem"})
    ).json()
    assert reranked["reranked"] is True

    services.rag.retriever.reranker = SaturatedReranker()
    before = degraded("rerank_skipped")
    resp = await client.post("/v1/rag/search", headers=auth(api_key), json={"query": "postmortem"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["reranked"] is False and body["hits"]  # still answered, in vector order
    assert all(h["rerank_score"] is None for h in body["hits"])
    assert degraded("rerank_skipped") == before + 1


async def test_a_saturated_embedder_skips_the_cache_but_still_answers(
    client, api_key, services, monkeypatch
):
    async def saturated(*args, **kwargs):
        raise OverloadedError("embedder is saturated")

    monkeypatch.setattr(services.cache, "lookup", saturated)
    monkeypatch.setattr(services.cache, "store", saturated)
    before = degraded("cache_skipped")

    resp = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(api_key))

    assert resp.status_code == 200
    assert resp.json()["nexusgate"]["cached"] is False  # served by the provider instead
    assert degraded("cache_skipped") == before + 1


async def test_with_no_way_to_degrade_the_request_is_shed_with_503(
    client, api_key, services, monkeypatch
):
    async def saturated(text):
        raise OverloadedError("embedder is saturated; try again shortly")

    monkeypatch.setattr(services.rag.retriever._embedder, "embed_query", saturated)
    resp = await client.post("/v1/rag/search", headers=auth(api_key), json={"query": "anything"})

    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
    assert resp.json()["error"]["type"] == "overloaded"
