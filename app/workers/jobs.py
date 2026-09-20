"""Ingestion job records and the dead-letter queue, both in Redis.

    ragjob:{tenant}:{job_id}   STRING  IngestJob JSON, expires after the retention window
    ragjobs:{tenant}           ZSET    job_id -> created_at (newest-first listing)
    ragjobs:dlq                LIST    jobs that exhausted their retries, or failed permanently
    ragjobs:stats              HASH    outcome counters + total processing time

Counters live in Redis rather than in a process, because the worker that runs a job has no HTTP
endpoint for Prometheus to scrape. The API reads them when /metrics is scraped, so the numbers are
the same whichever process did the work.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field
from redis.asyncio import Redis

DLQ_KEY = "ragjobs:dlq"
STATS_KEY = "ragjobs:stats"
OUTCOMES = ("submitted", "done", "failed", "retried")


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class IngestJob(BaseModel):
    job_id: str
    tenant_id: str
    filename: str
    content_type: str | None = None
    title: str | None = None
    size_bytes: int = 0
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    doc_id: str | None = None
    chunks: int | None = None
    error: str | None = None
    retryable: bool | None = Field(default=None, description="Set when a job fails.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _job_key(tenant_id: str, job_id: str) -> str:
    return f"ragjob:{tenant_id}:{job_id}"


def _index_key(tenant_id: str) -> str:
    return f"ragjobs:{tenant_id}"


class JobStore:
    def __init__(self, redis: Redis, retention_seconds: int):
        self._redis = redis
        self._retention = retention_seconds

    async def save(self, job: IngestJob) -> IngestJob:
        job.updated_at = datetime.now(UTC)
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(_job_key(job.tenant_id, job.job_id), job.model_dump_json(), ex=self._retention)
        pipe.zadd(_index_key(job.tenant_id), {job.job_id: job.created_at.timestamp()})
        pipe.expire(_index_key(job.tenant_id), self._retention)
        await pipe.execute()
        return job

    async def get(self, tenant_id: str, job_id: str) -> IngestJob | None:
        raw = await self._redis.get(_job_key(tenant_id, job_id))
        return IngestJob.model_validate_json(raw) if raw else None

    async def list(self, tenant_id: str, limit: int = 20) -> list[IngestJob]:
        job_ids = await self._redis.zrevrange(_index_key(tenant_id), 0, limit - 1)
        if not job_ids:
            return []
        raws = await self._redis.mget([_job_key(tenant_id, j) for j in job_ids])
        return [IngestJob.model_validate_json(r) for r in raws if r]

    async def count(self, outcome: str) -> None:
        await self._redis.hincrby(STATS_KEY, outcome, 1)

    async def add_duration(self, seconds: float) -> None:
        await self._redis.hincrbyfloat(STATS_KEY, "duration_seconds", seconds)

    async def stats(self) -> dict[str, float]:
        raw = await self._redis.hgetall(STATS_KEY)
        stats = {outcome: 0.0 for outcome in OUTCOMES}
        stats["duration_seconds"] = 0.0
        stats.update({k: float(v) for k, v in raw.items()})
        return stats

    async def queue_depth(self, queue_name: str) -> int:
        """How many jobs are waiting in the broker. Celery on Redis keeps one list per queue."""
        return int(await self._redis.llen(queue_name))


class DeadLetterQueue:
    """Jobs that will not be retried again, kept for inspection through the admin API."""

    def __init__(self, redis: Redis, max_entries: int = 1000):
        self._redis = redis
        self._max_entries = max_entries

    async def push(self, job: IngestJob) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.lpush(DLQ_KEY, job.model_dump_json())
        pipe.ltrim(DLQ_KEY, 0, self._max_entries - 1)
        await pipe.execute()

    async def list(self, limit: int = 50) -> list[IngestJob]:
        return [
            IngestJob.model_validate_json(r)
            for r in await self._redis.lrange(DLQ_KEY, 0, limit - 1)
        ]

    async def depth(self) -> int:
        return int(await self._redis.llen(DLQ_KEY))

    async def purge(self) -> int:
        depth = await self.depth()
        await self._redis.delete(DLQ_KEY)
        return depth
