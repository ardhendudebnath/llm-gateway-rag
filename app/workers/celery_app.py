"""Celery worker entry point.

    celery -A app.workers.celery_app:celery_app worker --queues ingest --concurrency 2

Each worker process keeps one event loop and one set of services, so the embedding and reranking
models load once per process rather than once per job.

Delivery settings worth knowing: ``task_acks_late`` plus ``task_reject_on_worker_lost`` mean a job
whose worker is killed mid-flight is redelivered instead of vanishing, and re-running it is safe
because ingestion is idempotent (content-addressed document ids, deterministic point ids).
``worker_prefetch_multiplier=1`` stops a worker from reserving jobs it cannot start yet.
"""

import asyncio
import logging

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from app.core.config import Settings, get_settings
from app.core.container import Services, build_services
from app.core.logging import configure_logging
from app.workers.queue import TASK_INGEST
from app.workers.service import TransientJobError

log = logging.getLogger(__name__)

_settings: Settings = get_settings()
_loop: asyncio.AbstractEventLoop | None = None
_services: Services | None = None

celery_app = Celery("nexusgate", broker=_settings.broker_url)
celery_app.conf.update(
    task_default_queue=_settings.ingest_queue,
    task_serializer="json",
    accept_content=["json"],
    result_backend=None,  # job state lives in our own Redis records, not in Celery's backend
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_hijack_root_logger=False,  # keep our JSON formatter
    timezone="UTC",
    enable_utc=True,
    broker_transport_options={"visibility_timeout": _settings.job_visibility_timeout_seconds},
)


def _run(coro):
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


def _get_services() -> Services:
    global _services
    if _services is None:
        _services = _run(build_services(_settings))
    return _services


@worker_process_init.connect
def _init_worker(**_kwargs) -> None:
    configure_logging(_settings.log_level)
    _get_services()  # load models and open connections before the first job, not during it
    log.info("worker process ready", extra={"queue": _settings.ingest_queue})


@worker_process_shutdown.connect
def _shutdown_worker(**_kwargs) -> None:
    if _services is not None:
        _run(_services.aclose())


# The service stops retrying at `job_max_attempts` and fails the job itself, so Celery's own
# ceiling should never be reached; it is here so a bug can't turn into an endless retry loop.
@celery_app.task(bind=True, name=TASK_INGEST, max_retries=_settings.job_max_attempts - 1)
def ingest_document(self, job_id: str, tenant_id: str, payload_id: str) -> dict:
    """Run one ingestion attempt. Retry policy and failure handling live in IngestionJobService."""
    services = _get_services()
    try:
        job = _run(services.rag.jobs.execute(tenant_id, job_id, payload_id))
    except TransientJobError as e:
        # Backoff only; how many attempts a job gets is the service's decision, counted per job
        # in Redis so that a redelivery after a worker died still counts.
        countdown = _settings.job_retry_backoff_seconds * 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown) from e
    return {"job_id": job_id, "status": job.status.value if job else "gone"}
