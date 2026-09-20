"""The job endpoints: polling, listing, tenant scoping, and the dead-letter admin views."""

from tests.integration.test_rag_api import HANDBOOK, auth, other_tenant_key, upload, upload_and_wait


async def test_jobs_are_listed_newest_first_for_the_caller_only(client, api_key, admin_headers):
    first = await upload_and_wait(client, api_key)
    second = await upload_and_wait(client, api_key, HANDBOOK + b"\n\n## Extra\nMore text.\n")

    jobs = (await client.get("/v1/rag/jobs", headers=auth(api_key))).json()
    assert [j["job_id"] for j in jobs] == [second["job_id"], first["job_id"]]
    assert {j["status"] for j in jobs} == {"done"}

    other = await other_tenant_key(client, admin_headers)
    assert (await client.get("/v1/rag/jobs", headers=auth(other))).json() == []
    stolen = await client.get(f"/v1/rag/jobs/{first['job_id']}", headers=auth(other))
    assert stolen.status_code == 404


async def test_unknown_job_is_404(client, api_key):
    resp = await client.get("/v1/rag/jobs/does-not-exist", headers=auth(api_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "no such job"


async def test_job_endpoints_require_auth(client):
    assert (await client.get("/v1/rag/jobs")).status_code == 401
    assert (await client.get("/v1/rag/jobs/x")).status_code == 401


async def test_dead_letters_are_visible_and_clearable_to_admins(client, api_key, admin_headers):
    failed = await upload_and_wait(client, api_key, b"\n\n", "blank.md")
    assert failed["status"] == "failed"

    listed = (await client.get("/v1/admin/dead-letters", headers=admin_headers)).json()
    assert listed["depth"] == 1
    assert listed["jobs"][0]["job_id"] == failed["job_id"]
    assert listed["jobs"][0]["retryable"] is False

    purged = await client.delete("/v1/admin/dead-letters", headers=admin_headers)
    assert purged.json() == {"entries_deleted": 1}
    assert (await client.get("/v1/admin/dead-letters", headers=admin_headers)).json()["depth"] == 0


async def test_dead_letters_need_the_admin_token(client, api_key):
    assert (await client.get("/v1/admin/dead-letters", headers=auth(api_key))).status_code == 403


async def test_metrics_expose_queue_depth_and_job_totals(client, api_key):
    await upload_and_wait(client, api_key)
    body = (await client.get("/metrics")).text
    assert 'nexusgate_rag_queue_depth{queue="ingest"} 0.0' in body
    assert 'nexusgate_rag_job_totals{outcome="done"} 1.0' in body
    assert "nexusgate_rag_dead_letter_depth 0.0" in body


async def test_upload_returns_immediately_with_a_queued_job(client, api_key):
    resp = await upload(client, api_key)
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] in {"queued", "processing"}
    assert job["doc_id"] is None and job["error"] is None
