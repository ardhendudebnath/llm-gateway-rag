"""Job records, the dead-letter queue and the payload store."""

import pytest

from app.workers.jobs import DeadLetterQueue, IngestJob, JobStatus, JobStore
from app.workers.payloads import PayloadStore


def job(tenant_id: str = "acme", job_id: str = "j1", **kw) -> IngestJob:
    return IngestJob(job_id=job_id, tenant_id=tenant_id, filename="doc.md", **kw)


@pytest.fixture
def store(redis_pair) -> JobStore:
    return JobStore(redis_pair[0], retention_seconds=3600)


async def test_save_get_and_status_transitions(store):
    saved = await store.save(job(size_bytes=42))
    assert saved.status is JobStatus.QUEUED

    saved.status = JobStatus.DONE
    saved.doc_id = "doc123"
    await store.save(saved)

    loaded = await store.get("acme", "j1")
    assert loaded.status is JobStatus.DONE
    assert (loaded.doc_id, loaded.size_bytes) == ("doc123", 42)
    assert loaded.updated_at >= loaded.created_at


async def test_records_expire_with_the_retention_window(store, redis_pair):
    await store.save(job())
    assert 0 < await redis_pair[0].ttl("ragjob:acme:j1") <= 3600


async def test_list_is_newest_first_and_tenant_scoped(store):
    for n in range(3):
        await store.save(job(job_id=f"j{n}"))
    await store.save(job(tenant_id="globex", job_id="other"))

    assert [j.job_id for j in await store.list("acme")] == ["j2", "j1", "j0"]
    assert [j.job_id for j in await store.list("acme", limit=2)] == ["j2", "j1"]
    assert [j.job_id for j in await store.list("globex")] == ["other"]
    assert await store.get("globex", "j0") is None


async def test_unknown_job_is_none(store):
    assert await store.get("acme", "nope") is None


async def test_counters_and_duration_accumulate(store):
    await store.count("submitted")
    await store.count("done")
    await store.count("done")
    await store.add_duration(1.5)
    stats = await store.stats()
    assert stats["submitted"] == 1 and stats["done"] == 2
    assert stats["failed"] == 0 and stats["retried"] == 0  # reported even when never incremented
    assert stats["duration_seconds"] == pytest.approx(1.5)


async def test_queue_depth_reads_the_broker_list(store, redis_pair):
    assert await store.queue_depth("ingest") == 0
    await redis_pair[0].rpush("ingest", "task-a", "task-b")
    assert await store.queue_depth("ingest") == 2


async def test_dead_letter_queue(redis_pair):
    dlq = DeadLetterQueue(redis_pair[0], max_entries=2)
    for n in range(3):
        await dlq.push(job(job_id=f"j{n}", status=JobStatus.FAILED, error="boom"))

    entries = await dlq.list()
    assert [j.job_id for j in entries] == ["j2", "j1"]  # newest first, capped at max_entries
    assert entries[0].error == "boom"
    assert await dlq.depth() == 2
    assert await dlq.purge() == 2
    assert await dlq.depth() == 0 and await dlq.list() == []


async def test_payloads_round_trip_and_expire(redis_pair):
    payloads = PayloadStore(redis_pair[1], ttl_seconds=60)
    payload_id = await payloads.put(b"%PDF-1.4 binary\x00bytes")

    assert await payloads.get(payload_id) == b"%PDF-1.4 binary\x00bytes"
    assert 0 < await redis_pair[1].ttl(f"ragupload:{payload_id}") <= 60
    await payloads.delete(payload_id)
    assert await payloads.get(payload_id) is None
    assert await payloads.get("never-existed") is None
