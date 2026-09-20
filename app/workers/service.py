"""Submitting and running ingestion jobs.

The API submits (park the bytes, create the job record, enqueue) and returns 202 straight away.
A worker later runs ``execute``. All the retry *policy* lives here so it can be tested without a
broker; the queue only decides *when* a retry runs.

Failures are split in two, because they need opposite handling:

* **Permanent** - the upload itself is wrong (not a supported type, no extractable text, too big,
  payload expired). Retrying would fail identically, so the job fails at once and goes to the
  dead-letter queue.
* **Transient** - Qdrant is down, Redis blipped, the model failed to load. Worth retrying with
  backoff; after the last attempt the job fails and goes to the dead-letter queue too.
"""

import logging
import time
import uuid

from app.core.logging import request_id_var
from app.observability import metrics
from app.rag.ingestion import DocumentTooLargeError, DocumentUpload, IngestionService
from app.rag.parsing import EmptyDocumentError, UnsupportedDocumentError
from app.workers.jobs import DeadLetterQueue, IngestJob, JobStatus, JobStore
from app.workers.payloads import PayloadStore

log = logging.getLogger(__name__)

# Retrying these would fail in exactly the same way.
PERMANENT_ERRORS = (UnsupportedDocumentError, EmptyDocumentError, DocumentTooLargeError)


class TransientJobError(Exception):
    """The job failed for a reason worth retrying; the queue schedules the next attempt."""


class IngestionJobService:
    def __init__(
        self,
        jobs: JobStore,
        payloads: PayloadStore,
        dead_letters: DeadLetterQueue,
        ingestion: IngestionService,
        queue,  # JobQueue; set late because the inline queue needs this service
        *,
        max_attempts: int,
    ):
        self._jobs = jobs
        self._payloads = payloads
        self._dead_letters = dead_letters
        self._ingestion = ingestion
        self.queue = queue
        self.max_attempts = max_attempts

    async def submit(self, tenant_id: str, upload: DocumentUpload) -> IngestJob:
        payload_id = await self._payloads.put(upload.data)
        job = await self._jobs.save(
            IngestJob(
                job_id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                filename=upload.filename,
                content_type=upload.content_type,
                title=upload.title,
                size_bytes=len(upload.data),
                request_id=request_id_var.get(),
            )
        )
        await self._jobs.count("submitted")
        await self.queue.enqueue(job, payload_id)
        log.info(
            "ingestion job queued",
            extra={"job_id": job.job_id, "tenant_id": tenant_id, "bytes": job.size_bytes},
        )
        return job

    async def execute(
        self, tenant_id: str, job_id: str, payload_id: str, *, attempt: int
    ) -> IngestJob | None:
        """Run one attempt. Raises TransientJobError when the caller should retry later."""
        job = await self._jobs.get(tenant_id, job_id)
        if job is None:  # the record outlived its retention, or never existed
            log.warning("ingestion job record is gone", extra={"job_id": job_id})
            return None

        # Log this attempt under the request id of the upload that created the job, so one
        # request can be followed from the API's access log into the worker's.
        if job.request_id:
            request_id_var.set(job.request_id)
        job.status = JobStatus.PROCESSING
        job.attempts = attempt
        await self._jobs.save(job)

        data = await self._payloads.get(payload_id)
        if data is None:
            return await self._fail(
                job, "uploaded file expired before a worker picked it up", False
            )

        started = time.perf_counter()
        try:
            record = await self._ingestion.ingest(
                tenant_id,
                DocumentUpload(
                    filename=job.filename,
                    content_type=job.content_type,
                    data=data,
                    title=job.title,
                ),
            )
        except PERMANENT_ERRORS as e:
            return await self._fail(job, f"{type(e).__name__}: {e}", False)
        except Exception as e:
            if attempt >= self.max_attempts:
                return await self._fail(job, f"{type(e).__name__}: {e}", True)
            job.status = JobStatus.QUEUED
            job.error = f"{type(e).__name__}: {e}"
            await self._jobs.save(job)
            await self._jobs.count("retried")
            log.warning(
                "ingestion attempt failed, retrying",
                extra={"job_id": job_id, "attempt": attempt, "error": job.error},
            )
            raise TransientJobError(job.error) from e

        await self._payloads.delete(payload_id)
        job.status = JobStatus.DONE
        job.doc_id = record.doc_id
        job.chunks = record.chunks
        job.error = None
        job.retryable = None
        await self._jobs.save(job)
        await self._jobs.count("done")
        await self._jobs.add_duration(time.perf_counter() - started)
        log.info(
            "ingestion job done",
            extra={"job_id": job_id, "doc_id": record.doc_id, "chunks": record.chunks},
        )
        return job

    async def _fail(self, job: IngestJob, error: str, retryable: bool) -> IngestJob:
        job.status = JobStatus.FAILED
        job.error = error
        job.retryable = retryable
        await self._jobs.save(job)
        await self._jobs.count("failed")
        await self._dead_letters.push(job)
        log.error(
            "ingestion job failed",
            extra={"job_id": job.job_id, "error": error, "retryable": retryable},
        )
        return job

    async def refresh_metrics(self, queue_name: str) -> None:
        """Called when /metrics is scraped: queue depth and totals live in Redis, not a process."""
        metrics.RAG_QUEUE_DEPTH.labels(queue_name).set(await self._jobs.queue_depth(queue_name))
        metrics.RAG_DLQ_DEPTH.set(await self._dead_letters.depth())
        stats = await self._jobs.stats()
        metrics.RAG_JOB_SECONDS.set(stats.pop("duration_seconds", 0.0))
        for outcome, total in stats.items():
            metrics.RAG_JOB_TOTALS.labels(outcome).set(total)
