"""Submitting and running ingestion jobs: the retry policy, without a broker.

A fake queue captures what would be enqueued, so each attempt can be driven by hand and the
permanent / transient / exhausted paths are all deterministic.
"""

from datetime import UTC, datetime

import pytest

from app.rag.documents import DocumentRecord
from app.rag.ingestion import DocumentUpload
from app.rag.parsing import EmptyDocumentError
from app.workers.jobs import DeadLetterQueue, JobStatus, JobStore
from app.workers.payloads import PayloadStore
from app.workers.service import IngestionJobService, TransientJobError


class FakeQueue:
    name = "fake"

    def __init__(self):
        self.enqueued: list[tuple[str, str]] = []

    async def enqueue(self, job, payload_id):
        self.enqueued.append((job.job_id, payload_id))


class FakeIngestion:
    """Stands in for IngestionService: each call either raises a queued error or succeeds."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[DocumentUpload] = []
        self.max_bytes = 1000

    async def ingest(self, tenant_id: str, upload: DocumentUpload) -> DocumentRecord:
        self.calls.append(upload)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return DocumentRecord(
            doc_id="doc123",
            title="Doc",
            filename=upload.filename,
            content_type=upload.content_type,
            size_bytes=len(upload.data),
            pages=None,
            chunks=3,
            chunker="structured/180/40",
            created_at=datetime.now(UTC),
        )


def upload(data: bytes = b"# Doc\n\nHello.") -> DocumentUpload:
    return DocumentUpload(filename="doc.md", content_type="text/markdown", data=data, title=None)


@pytest.fixture
def make_service(redis_pair):
    text, raw = redis_pair

    def build(ingestion, max_attempts=3):
        queue = FakeQueue()
        service = IngestionJobService(
            JobStore(text, 3600),
            PayloadStore(raw, 3600),
            DeadLetterQueue(text),
            ingestion,
            queue,
            max_attempts=max_attempts,
        )
        return service, queue, JobStore(text, 3600), DeadLetterQueue(text)

    return build


async def test_submit_parks_the_payload_and_enqueues(make_service):
    service, queue, store, _ = make_service(FakeIngestion())
    job = await service.submit("acme", upload())

    assert job.status is JobStatus.QUEUED and job.size_bytes == len(upload().data)
    assert await store.get("acme", job.job_id) is not None
    assert queue.enqueued[0][0] == job.job_id
    assert (await store.stats())["submitted"] == 1


async def test_execute_ingests_and_marks_the_job_done(make_service):
    ingestion = FakeIngestion()
    service, queue, store, _ = make_service(ingestion)
    job = await service.submit("acme", upload())
    payload_id = queue.enqueued[0][1]

    done = await service.execute("acme", job.job_id, payload_id)

    assert done.status is JobStatus.DONE
    assert (done.doc_id, done.chunks, done.attempts) == ("doc123", 3, 1)
    assert ingestion.calls[0].filename == "doc.md"
    stats = await store.stats()
    assert stats["done"] == 1 and stats["duration_seconds"] >= 0
    # the parked upload is cleaned up once it is safely in Qdrant
    assert await service._payloads.get(payload_id) is None


async def test_a_bad_document_fails_at_once_without_retrying(make_service):
    ingestion = FakeIngestion(EmptyDocumentError("document contains no extractable text"))
    service, queue, store, dlq = make_service(ingestion)
    job = await service.submit("acme", upload())

    failed = await service.execute("acme", job.job_id, queue.enqueued[0][1])

    assert failed.status is JobStatus.FAILED and failed.retryable is False
    assert "no extractable text" in failed.error
    assert [j.job_id for j in await dlq.list()] == [job.job_id]
    assert (await store.stats())["retried"] == 0


async def test_a_transient_failure_asks_for_a_retry_then_succeeds(make_service):
    ingestion = FakeIngestion(ConnectionError("qdrant unreachable"))
    service, queue, store, dlq = make_service(ingestion)
    job = await service.submit("acme", upload())
    payload_id = queue.enqueued[0][1]

    with pytest.raises(TransientJobError, match="qdrant unreachable"):
        await service.execute("acme", job.job_id, payload_id)

    queued_again = await store.get("acme", job.job_id)
    assert queued_again.status is JobStatus.QUEUED  # waiting for the next attempt
    assert "qdrant unreachable" in queued_again.error
    assert await dlq.depth() == 0  # not dead yet

    done = await service.execute("acme", job.job_id, payload_id)
    assert done.status is JobStatus.DONE and done.attempts == 2
    assert (await store.stats())["retried"] == 1


async def test_the_last_attempt_fails_the_job_and_dead_letters_it(make_service):
    ingestion = FakeIngestion(*[ConnectionError("still down")] * 3)
    service, queue, _, dlq = make_service(ingestion, max_attempts=3)
    job = await service.submit("acme", upload())
    payload_id = queue.enqueued[0][1]

    for _ in (1, 2):
        with pytest.raises(TransientJobError):
            await service.execute("acme", job.job_id, payload_id)

    failed = await service.execute("acme", job.job_id, payload_id)
    assert failed.status is JobStatus.FAILED
    assert failed.retryable is True and failed.attempts == 3
    assert [j.job_id for j in await dlq.list()] == [job.job_id]


async def test_a_job_that_keeps_killing_its_worker_is_dead_lettered(make_service):
    """The poison pill: a job that never reports a failure because its worker dies.

    Redelivery after a worker is lost doesn't raise, so nothing in the retry path sees it. The
    attempt count is kept per job in Redis for exactly this case. Measured for real: one oversized
    document OOM-killed four workers in turn, and they crash-looped on it for 70 minutes.
    """
    service, queue, store, dlq = make_service(FakeIngestion(), max_attempts=3)
    job = await service.submit("acme", upload())
    payload_id = queue.enqueued[0][1]

    # Three deliveries that "died" before finishing: the record is never marked done.
    for _ in range(3):
        await store.start_attempt("acme", job.job_id)

    failed = await service.execute("acme", job.job_id, payload_id)

    assert failed.status is JobStatus.FAILED and failed.retryable is False
    assert "delivered 4 times" in failed.error and "killing its worker" in failed.error
    assert [j.job_id for j in await dlq.list()] == [job.job_id]


async def test_an_expired_payload_fails_the_job_permanently(make_service):
    service, queue, _, dlq = make_service(FakeIngestion())
    job = await service.submit("acme", upload())
    payload_id = queue.enqueued[0][1]
    await service._payloads.delete(payload_id)  # TTL elapsed before a worker picked it up

    failed = await service.execute("acme", job.job_id, payload_id)

    assert failed.status is JobStatus.FAILED and failed.retryable is False
    assert "expired" in failed.error
    assert await dlq.depth() == 1


async def test_a_missing_job_record_is_not_an_error(make_service):
    service, _, _, _ = make_service(FakeIngestion())
    assert await service.execute("acme", "gone", "payload") is None


async def test_refresh_metrics_publishes_queue_and_job_gauges(make_service, redis_pair):
    service, queue, _store, _ = make_service(FakeIngestion())
    job = await service.submit("acme", upload())
    await service.execute("acme", job.job_id, queue.enqueued[0][1])
    await redis_pair[0].rpush("ingest", "queued-task")

    await service.refresh_metrics("ingest")

    from app.observability import metrics

    assert metrics.RAG_QUEUE_DEPTH.labels("ingest")._value.get() == 1
    assert metrics.RAG_JOB_TOTALS.labels("done")._value.get() == 1
    assert metrics.RAG_DLQ_DEPTH._value.get() == 0
