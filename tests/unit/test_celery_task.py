"""The Celery task is a thin wrapper: it numbers attempts and turns a transient failure into a
retry. The policy it delegates to is tested in test_job_service.py."""

from types import SimpleNamespace

import pytest

from app.workers import celery_app as mod
from app.workers.jobs import IngestJob, JobStatus
from app.workers.service import TransientJobError


class FakeJobs:
    def __init__(self, outcome=None):
        self.calls: list[dict] = []
        self._outcome = outcome

    async def execute(self, tenant_id, job_id, payload_id, *, attempt):
        self.calls.append(
            {"tenant_id": tenant_id, "job_id": job_id, "payload_id": payload_id, "attempt": attempt}
        )
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return IngestJob(
            job_id=job_id, tenant_id=tenant_id, filename="doc.md", status=JobStatus.DONE
        )


@pytest.fixture
def jobs(monkeypatch):
    fake = FakeJobs()

    def use(outcome=None):
        fake._outcome = outcome
        monkeypatch.setattr(
            mod, "_get_services", lambda: SimpleNamespace(rag=SimpleNamespace(jobs=fake))
        )
        return fake

    return use


def test_the_task_runs_the_first_attempt_and_reports_the_status(jobs):
    fake = jobs()
    result = mod.ingest_document(job_id="j1", tenant_id="acme", payload_id="p1")
    assert result == {"job_id": "j1", "status": "done"}
    assert fake.calls == [{"tenant_id": "acme", "job_id": "j1", "payload_id": "p1", "attempt": 1}]


def test_a_transient_failure_retries_with_a_rising_attempt_number(jobs):
    fake = jobs(TransientJobError("qdrant unreachable"))
    # Run through .apply(): called directly, Celery re-raises instead of scheduling a retry.
    # Eager mode runs each retry inline, so this also proves retries are bounded.
    result = mod.ingest_document.apply(
        kwargs={"job_id": "j1", "tenant_id": "acme", "payload_id": "p1"}, throw=False
    )
    assert [c["attempt"] for c in fake.calls] == [1, 2, 3]
    assert result.state == "FAILURE"  # Celery's ceiling; the service normally stops first


def test_a_vanished_job_is_reported_not_crashed(jobs):
    fake = jobs()
    fake.execute = _returns_none
    assert mod.ingest_document(job_id="gone", tenant_id="acme", payload_id="p1") == {
        "job_id": "gone",
        "status": "gone",
    }


async def _returns_none(*args, **kwargs):
    return None


def test_the_worker_is_configured_for_safe_redelivery():
    conf = mod.celery_app.conf
    assert conf.task_acks_late is True  # a killed worker's job is redelivered
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1  # don't hoard jobs a slow one would block
    assert conf.task_serializer == "json"
