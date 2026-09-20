"""Where a submitted job actually runs.

Two implementations behind one interface, so the API code path is identical either way:

* ``CeleryJobQueue`` - production. Publishes a small message to Redis; a worker pod runs it.
* ``InlineJobQueue`` - dev and tests. Runs the job in this process as a background task, so the
  endpoint still returns 202 immediately and clients still poll for status, with no broker and no
  worker to start.
"""

import asyncio
import logging
from typing import Protocol

from app.workers.jobs import IngestJob
from app.workers.service import IngestionJobService, TransientJobError

log = logging.getLogger(__name__)

TASK_INGEST = "nexusgate.ingest_document"


class JobQueue(Protocol):
    name: str

    async def enqueue(self, job: IngestJob, payload_id: str) -> None: ...


class CeleryJobQueue:
    def __init__(self, celery_app, queue_name: str):
        self.name = "celery"
        self._app = celery_app
        self._queue = queue_name

    async def enqueue(self, job: IngestJob, payload_id: str) -> None:
        # Kombu's publish is blocking; keep it off the event loop.
        await asyncio.to_thread(
            self._app.send_task,
            TASK_INGEST,
            kwargs={
                "job_id": job.job_id,
                "tenant_id": job.tenant_id,
                "payload_id": payload_id,
            },
            queue=self._queue,
        )


class InlineJobQueue:
    """Runs jobs in-process, retrying with backoff like the Celery task does."""

    def __init__(self, backoff_seconds: float = 0.1):
        self.name = "inline"
        self._service: IngestionJobService | None = None
        self._backoff = backoff_seconds
        self._tasks: set[asyncio.Task] = set()

    def bind(self, service: IngestionJobService) -> None:
        self._service = service

    async def enqueue(self, job: IngestJob, payload_id: str) -> None:
        task = asyncio.create_task(self._run(job, payload_id))
        self._tasks.add(task)  # keep a reference: bare tasks can be garbage collected mid-flight
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job: IngestJob, payload_id: str) -> None:
        for attempt in range(1, self._service.max_attempts + 1):
            try:
                await self._service.execute(job.tenant_id, job.job_id, payload_id, attempt=attempt)
                return
            except TransientJobError:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
            except Exception:
                log.exception("inline ingestion job crashed", extra={"job_id": job.job_id})
                return
