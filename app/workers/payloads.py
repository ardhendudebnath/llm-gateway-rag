"""Uploaded bytes, parked in Redis while a job waits for a worker.

The file never travels through the broker: the queued message carries only this key. Celery
messages are meant to be small, and a 10 MB PDF in a Redis list would be copied on every
redelivery. The payload expires on its own, so an upload whose job is never run cannot leak.
"""

import uuid

from redis.asyncio import Redis


def _key(payload_id: str) -> str:
    return f"ragupload:{payload_id}"


class PayloadStore:
    def __init__(self, redis: Redis, ttl_seconds: int):
        self._redis = redis  # the bytes-decoding client: payloads are binary
        self._ttl = ttl_seconds

    async def put(self, data: bytes) -> str:
        payload_id = uuid.uuid4().hex
        await self._redis.set(_key(payload_id), data, ex=self._ttl)
        return payload_id

    async def get(self, payload_id: str) -> bytes | None:
        return await self._redis.get(_key(payload_id))

    async def delete(self, payload_id: str) -> None:
        await self._redis.delete(_key(payload_id))
