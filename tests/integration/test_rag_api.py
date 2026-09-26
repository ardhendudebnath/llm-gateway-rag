import asyncio
import time

from app.rag.service import NO_CONTEXT_ANSWER
from tests.pdfgen import make_pdf

HANDBOOK = b"""# Incident handbook

## Severity levels
A SEV1 incident means customers cannot use the product. A SEV1 must be acknowledged within
5 minutes and gets an incident commander immediately.

## Postmortems
Every SEV1 and SEV2 incident needs a written postmortem within five business days.
"""


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def upload(
    client, key, data=HANDBOOK, filename="handbook.md", content_type="text/markdown", **form
):
    return await client.post(
        "/v1/rag/documents",
        headers=auth(key),
        files={"file": (filename, data, content_type)},
        data=form,
    )


async def wait_for_job(client, key, job_id, timeout_s=5.0) -> dict:
    """Uploads are asynchronous: poll until the job finishes (tests run the inline queue)."""
    deadline = time.monotonic() + timeout_s
    while True:
        job = (await client.get(f"/v1/rag/jobs/{job_id}", headers=auth(key))).json()
        if job["status"] in {"done", "failed"}:
            return job
        assert time.monotonic() < deadline, f"job never finished: {job}"
        await asyncio.sleep(0.02)


async def upload_and_wait(client, key, *args, **kwargs) -> dict:
    resp = await upload(client, key, *args, **kwargs)
    assert resp.status_code == 202, resp.text
    return await wait_for_job(client, key, resp.json()["job_id"])


async def other_tenant_key(client, admin_headers) -> str:
    resp = await client.post(
        "/v1/admin/keys", json={"tenant_id": "globex", "name": "other"}, headers=admin_headers
    )
    return resp.json()["api_key"]


async def test_upload_list_get_and_delete(client, api_key):
    resp = await upload(client, api_key)
    assert resp.status_code == 202, resp.text
    queued = resp.json()
    assert queued["status"] in {"queued", "processing"}
    assert queued["filename"] == "handbook.md" and queued["size_bytes"] == len(HANDBOOK)

    job = await wait_for_job(client, api_key, queued["job_id"])
    assert job["status"] == "done" and job["error"] is None
    assert job["chunks"] >= 2 and job["attempts"] == 1

    listed = (await client.get("/v1/rag/documents", headers=auth(api_key))).json()
    assert [d["doc_id"] for d in listed] == [job["doc_id"]]
    doc = listed[0]
    assert doc["title"] == "Incident handbook"
    assert doc["chunks"] >= 2 and doc["chunker"].startswith("structured/")
    one = await client.get(f"/v1/rag/documents/{doc['doc_id']}", headers=auth(api_key))
    assert one.json()["filename"] == "handbook.md"

    gone = await client.delete(f"/v1/rag/documents/{doc['doc_id']}", headers=auth(api_key))
    assert gone.status_code == 204
    missing = await client.get(f"/v1/rag/documents/{doc['doc_id']}", headers=auth(api_key))
    assert missing.status_code == 404


async def test_pdf_upload_records_pages(client, api_key):
    pdf = make_pdf(["Deploys are frozen in December.", "Rollbacks need no approval."])
    job = await upload_and_wait(
        client, api_key, pdf, "policy.pdf", "application/pdf", title="Deploy policy"
    )
    assert job["status"] == "done", job
    doc = (await client.get(f"/v1/rag/documents/{job['doc_id']}", headers=auth(api_key))).json()
    assert doc["pages"] == 2

    hits = (
        await client.post(
            "/v1/rag/search",
            headers=auth(api_key),
            json={"query": "rollbacks approval", "top_k": 1},
        )
    ).json()["hits"]
    assert hits[0]["page"] == 2 and hits[0]["title"] == "Deploy policy"


async def test_search_returns_scored_passages(client, api_key):
    await upload_and_wait(client, api_key)
    resp = await client.post(
        "/v1/rag/search",
        headers=auth(api_key),
        json={"query": "written postmortem deadline", "top_k": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reranked"] is False  # no reranker configured in tests
    assert body["retrieval"] == "hybrid", "how the passages were found, not how they were ordered"
    top = body["hits"][0]
    assert "five business days" in top["text"]
    assert top["heading"] == "Incident handbook > Postmortems"
    assert top["text"].startswith("Incident handbook > Postmortems\n")  # title not repeated
    assert top["vector_score"] > body["hits"][1]["vector_score"]
    assert top["rerank_score"] is None


async def test_answer_uses_the_chat_path_with_numbered_context(client, api_key, provider):
    await upload_and_wait(client, api_key)
    resp = await client.post(
        "/v1/rag/answer",
        headers=auth(api_key),
        json={"question": "How fast must a SEV1 be acknowledged?", "top_k": 2, "cache": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert provider.calls == ["primary"]
    assert body["nexusgate"]["deployment"] == "primary"
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert any("5 minutes" in c["text"] for c in body["citations"])
    assert body["usage"]["total_tokens"] > 0


async def test_answer_without_matching_documents_skips_the_llm(client, api_key, provider):
    resp = await client.post(
        "/v1/rag/answer", headers=auth(api_key), json={"question": "What is our refund policy?"}
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": NO_CONTEXT_ANSWER,
        "citations": [],
        "usage": None,
        "nexusgate": None,
    }
    assert provider.calls == []


async def test_answer_with_unknown_route_is_404(client, api_key):
    await upload_and_wait(client, api_key)
    resp = await client.post(
        "/v1/rag/answer", headers=auth(api_key), json={"question": "SEV1?", "model": "nope"}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


async def test_other_tenants_see_nothing(client, api_key, admin_headers):
    doc = await upload_and_wait(client, api_key)
    other = await other_tenant_key(client, admin_headers)

    assert (await client.get("/v1/rag/documents", headers=auth(other))).json() == []
    search = await client.post(
        "/v1/rag/search", headers=auth(other), json={"query": "SEV1 incident postmortem"}
    )
    assert search.json()["hits"] == []
    for method in ("get", "delete"):
        resp = await client.request(
            method, f"/v1/rag/documents/{doc['doc_id']}", headers=auth(other)
        )
        assert resp.status_code == 404


async def test_unusable_uploads_are_rejected_before_they_are_queued(client, api_key, services):
    """Type and size are checked in the request; the caller doesn't wait for a job to fail."""
    pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    unsupported = await upload(client, api_key, b"PK\x03\x04...", "deck.pptx", pptx)
    assert unsupported.status_code == 415
    disguised = await upload(client, api_key, b"PK\x03\x04\x00\x00binary", "notes.md")
    assert disguised.status_code == 415
    services.rag.ingestion.max_bytes = 16
    too_big = await upload(client, api_key)
    assert too_big.status_code == 413
    assert too_big.json()["error"]["type"] == "http_error"
    assert await services.rag.job_store.list("acme") == []  # nothing was queued


async def test_a_file_with_no_text_fails_the_job_permanently(client, api_key, services):
    job = await upload_and_wait(client, api_key, b"   \n", "empty.md")
    assert job["status"] == "failed"
    assert "no extractable text" in job["error"]
    assert job["retryable"] is False and job["attempts"] == 1  # no point retrying this
    assert [j.job_id for j in await services.rag.dead_letters.list()] == [job["job_id"]]


async def test_rag_endpoints_require_auth(client):
    assert (await client.get("/v1/rag/documents")).status_code == 401
    assert (await client.post("/v1/rag/search", json={"query": "x"})).status_code == 401


async def test_readiness_includes_qdrant(client):
    body = (await client.get("/readyz")).json()
    assert body["checks"] == {"redis": "ok", "qdrant": "ok"}
