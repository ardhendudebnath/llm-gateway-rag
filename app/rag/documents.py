"""Per-tenant registry of ingested documents (metadata only; chunks and vectors live in Qdrant).

ragdoc:{tenant}:{doc_id}   STRING  DocumentRecord JSON
ragdocs:{tenant}           ZSET    doc_id -> created_at (newest-first listing)
"""

from datetime import datetime

from pydantic import BaseModel
from redis.asyncio import Redis


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    filename: str
    content_type: str | None
    size_bytes: int
    pages: int | None
    chunks: int
    chunker: str
    created_at: datetime


def _record_key(tenant_id: str, doc_id: str) -> str:
    return f"ragdoc:{tenant_id}:{doc_id}"


def _index_key(tenant_id: str) -> str:
    return f"ragdocs:{tenant_id}"


class DocumentRegistry:
    def __init__(self, redis: Redis):
        self._redis = redis

    async def put(self, tenant_id: str, record: DocumentRecord) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(_record_key(tenant_id, record.doc_id), record.model_dump_json())
        pipe.zadd(_index_key(tenant_id), {record.doc_id: record.created_at.timestamp()})
        await pipe.execute()

    async def get(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        raw = await self._redis.get(_record_key(tenant_id, doc_id))
        return DocumentRecord.model_validate_json(raw) if raw else None

    async def list(self, tenant_id: str) -> list[DocumentRecord]:
        doc_ids = await self._redis.zrevrange(_index_key(tenant_id), 0, -1)
        if not doc_ids:
            return []
        raws = await self._redis.mget([_record_key(tenant_id, d) for d in doc_ids])
        return [DocumentRecord.model_validate_json(r) for r in raws if r]

    async def delete(self, tenant_id: str, doc_id: str) -> bool:
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(_record_key(tenant_id, doc_id))
        pipe.zrem(_index_key(tenant_id), doc_id)
        deleted, _ = await pipe.execute()
        return bool(deleted)
