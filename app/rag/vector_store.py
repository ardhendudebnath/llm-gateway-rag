"""Qdrant-backed chunk store, used directly through ``qdrant-client`` (no framework in between).

Multi-tenancy is one collection with a ``tenant_id`` payload field, the layout Qdrant recommends for
many small tenants. ``tenant_id`` gets a keyword index with ``is_tenant=True``, so Qdrant co-locates
each tenant's vectors, and **every** read and delete carries a tenant filter: there is no code path
that searches across tenants.

Point IDs are deterministic (UUIDv5 of tenant, document and chunk index), so re-ingesting a
document, or a retried ingestion job in week 5, overwrites points instead of duplicating them.
"""

import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.chunking import Chunk

_POINT_NAMESPACE = uuid.UUID("5f0c3f9e-2d4b-4e0a-9a51-6b8f2c1d7e93")


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    chunk_index: int
    text: str
    title: str
    page: int | None
    heading: str | None
    score: float


def point_id(tenant_id: str, doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{tenant_id}/{doc_id}/{chunk_index}"))


def _tenant_filter(tenant_id: str, doc_id: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    if doc_id is not None:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    return models.Filter(must=must)


class QdrantChunkStore:
    def __init__(self, client: AsyncQdrantClient, collection: str):
        self._client = client
        self.collection = collection

    async def setup(self, dim: int) -> None:
        if not await self._client.collection_exists(self.collection):
            try:
                await self._client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
            except (UnexpectedResponse, ValueError) as e:
                # Replicas starting together race to create it; losing that race is fine.
                if "already exists" not in str(e).lower():
                    raise
        # Idempotent on the server. The in-memory client (tests, the public demo) has no payload
        # indexes and says so on every start; that is expected, so keep it out of the logs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            await self._client.create_payload_index(
                self.collection,
                "tenant_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )
            await self._client.create_payload_index(
                self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
            )

    async def replace_document(
        self,
        tenant_id: str,
        doc_id: str,
        title: str,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Store a document's chunks, dropping any left over from a previous, longer version."""
        await self.delete_document(tenant_id, doc_id)
        points = [
            models.PointStruct(
                id=point_id(tenant_id, doc_id, c.index),
                vector=vec.astype(np.float32).tolist(),
                payload={
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "chunk_index": c.index,
                    "text": c.text,
                    "title": title,
                    "page": c.page,
                    "heading": c.heading,
                },
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            await self._client.upsert(
                self.collection, points=points[start : start + 256], wait=True
            )

    async def delete_document(self, tenant_id: str, doc_id: str) -> None:
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_tenant_filter(tenant_id, doc_id)),
            wait=True,
        )

    async def search(self, tenant_id: str, vector: np.ndarray, limit: int) -> list[StoredChunk]:
        result = await self._client.query_points(
            self.collection,
            query=vector.astype(np.float32).tolist(),
            query_filter=_tenant_filter(tenant_id),
            limit=limit,
            with_payload=True,
        )
        return [
            StoredChunk(
                doc_id=p.payload["doc_id"],
                chunk_index=p.payload["chunk_index"],
                text=p.payload["text"],
                title=p.payload["title"],
                page=p.payload.get("page"),
                heading=p.payload.get("heading"),
                score=float(p.score),
            )
            for p in result.points
        ]

    async def ping(self) -> None:
        await self._client.collection_exists(self.collection)

    async def aclose(self) -> None:
        await self._client.close()
