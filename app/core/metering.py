"""Per-API-key usage metering: requests, tokens, spend and cache savings, bucketed by UTC day.

This is the same data a billing system needs; ``GET /v1/usage`` exposes it per key.
"""

from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel
from redis.asyncio import Redis


class DailyUsage(BaseModel):
    date: date
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0


def _key(key_id: str, day: date) -> str:
    return f"usage:{key_id}:{day.isoformat()}"


class UsageMeter:
    def __init__(self, redis: Redis, retention_days: int):
        self._redis = redis
        self._retention = timedelta(days=retention_days)

    async def record(
        self,
        key_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cached: bool,
        cost_saved_usd: float = 0.0,
        now: datetime | None = None,
    ) -> None:
        key = _key(key_id, (now or datetime.now(UTC)).date())
        pipe = self._redis.pipeline(transaction=False)
        pipe.hincrby(key, "requests", 1)
        if cached:
            pipe.hincrby(key, "cache_hits", 1)
            pipe.hincrbyfloat(key, "cost_saved_usd", cost_saved_usd)
        else:
            pipe.hincrby(key, "prompt_tokens", prompt_tokens)
            pipe.hincrby(key, "completion_tokens", completion_tokens)
            pipe.hincrbyfloat(key, "cost_usd", cost_usd)
        pipe.expire(key, int(self._retention.total_seconds()))
        await pipe.execute()

    async def daily(self, key_id: str, days: int, today: date | None = None) -> list[DailyUsage]:
        today = today or datetime.now(UTC).date()
        dates = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
        pipe = self._redis.pipeline(transaction=False)
        for d in dates:
            pipe.hgetall(_key(key_id, d))
        rows = await pipe.execute()
        return [DailyUsage(date=d, **row) for d, row in zip(dates, rows, strict=True)]
