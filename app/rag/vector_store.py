"""Qdrant-backed chunk store, used directly through ``qdrant-client`` (no framework in between).

Multi-tenancy is one collection with a ``tenant_id`` payload field, the layout Qdrant recommends for
many small tenants. ``tenant_id`` gets a keyword index with ``is_tenant=True``, so Qdrant co-locates
each tenant's vectors, and **every** read and delete carries a tenant filter: there is no code path
that searches across tenants.

Point IDs are deterministic (UUIDv5 of tenant, document and chunk index), so re-ingesting a
document, or a retried ingestion job in week 5, overwrites points instead of duplicating them.
"""

import logging
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.chunking import Chunk
from app.rag.lexical import document_terms, query_terms

log = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("5f0c3f9e-2d4b-4e0a-9a51-6b8f2c1d7e93")
LEXICAL = "lexical"  # the sparse vector's name inside the collection
RRF = "rrf"  # reciprocal rank fusion: ranks only
DBSF = "dbsf"  # distribution-based score fusion: normalised scores
PREFETCH_FLOOR = 20  # rows each half offers the fusion, however few are finally returned


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


def _stored(point) -> StoredChunk:
    return StoredChunk(
        doc_id=point.payload["doc_id"],
        chunk_index=point.payload["chunk_index"],
        text=point.payload["text"],
        title=point.payload["title"],
        page=point.payload.get("page"),
        heading=point.payload.get("heading"),
        score=float(point.score),
    )


def _tenant_filter(tenant_id: str, doc_id: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    if doc_id is not None:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    return models.Filter(must=must)


class QdrantChunkStore:
    def __init__(self, client: AsyncQdrantClient, collection: str):
        self._client = client
        self.collection = collection
        self._lexical = False

    @property
    def supports_lexical(self) -> bool:
        """Whether this collection carries the sparse vector hybrid retrieval needs."""
        return self._lexical

    async def _has_lexical_vector(self) -> bool:
        try:
            info = await self._client.get_collection(self.collection)
            return LEXICAL in (info.config.params.sparse_vectors or {})
        except Exception:
            log.warning("could not inspect the collection's vectors", exc_info=True)
            return False

    async def setup(self, dim: int, lexical: bool = False) -> None:
        """`lexical` also declares the sparse vector that hybrid retrieval searches.

        The dense vector stays unnamed, so nothing about the existing layout changes. A sparse
        vector cannot be added to a collection after the fact, so a collection created before
        hybrid retrieval existed keeps working and `supports_lexical` reports False: the gateway
        then searches dense-only rather than failing, and says so once at start-up.
        """
        if not await self._client.collection_exists(self.collection):
            try:
                await self._client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                    sparse_vectors_config=(
                        {LEXICAL: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                        if lexical
                        else None
                    ),
                )
            except (UnexpectedResponse, ValueError) as e:
                # Replicas starting together race to create it; losing that race is fine.
                if "already exists" not in str(e).lower():
                    raise
        self._lexical = lexical and await self._has_lexical_vector()
        if lexical and not self._lexical:
            log.warning(
                "this collection has no lexical vector, so retrieval stays dense-only; to enable "
                "hybrid search, re-create the collection and re-ingest",
                extra={"collection": self.collection},
            )
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
                vector=self._vectors(c.text, vec),
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

    def _vectors(self, text: str, dense: np.ndarray):
        """The dense vector alone, or both vectors when the collection has a lexical one.

        When a collection carries named sparse vectors, the unnamed dense vector is addressed as
        the empty name, which is why the shape of this differs between the two cases.
        """
        embedding = dense.astype(np.float32).tolist()
        if not self._lexical:
            return embedding
        sparse = document_terms(text)
        return {
            "": embedding,
            LEXICAL: models.SparseVector(indices=sparse.indices, values=sparse.values),
        }

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
            using="" if self._lexical else None,
            query_filter=_tenant_filter(tenant_id),
            limit=limit,
            with_payload=True,
        )
        return [_stored(p) for p in result.points]

    async def hybrid_search(
        self,
        tenant_id: str,
        vector: np.ndarray,
        query: str,
        limit: int,
        fusion: str = RRF,
    ) -> list[StoredChunk]:
        """Dense and lexical search, fused by Qdrant.

        One request, not two: each half runs as a prefetch and the server fuses them, so nothing is
        paged back to the gateway to be merged here. The tenant filter is applied to both halves,
        exactly as the dense-only path does.

        Two fusion methods, because they behave differently and the eval measures both:

        * ``rrf`` ranks by reciprocal rank. It needs no weights and cannot be skewed by one half's
          scale — but it discards magnitude, so a passage the lexical half matched exactly (BM25
          4.6 against 0.14 for its neighbours) counts no more than the dense half's best guess, and
          can lose a tie to it.
        * ``dbsf`` normalises each half's scores and adds them, so an overwhelming lexical match
          wins. The price is sensitivity to score distributions, which differ per query.

        Each half prefetches at least ``PREFETCH_FLOOR`` rows regardless of ``limit``: fusion can
        only rank what the halves handed it, and a half that returns exactly ``limit`` rows cannot
        rescue a passage the other half ranked first.
        """
        if not self._lexical:
            return await self.search(tenant_id, vector, limit)
        sparse = query_terms(query)
        tenant = _tenant_filter(tenant_id)
        depth = max(limit, PREFETCH_FLOOR)
        prefetch = [
            models.Prefetch(
                query=vector.astype(np.float32).tolist(),
                using="",
                filter=tenant,
                limit=depth,
            )
        ]
        if sparse:  # a query of only punctuation has no terms to look up
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=LEXICAL,
                    filter=tenant,
                    limit=depth,
                )
            )
        result = await self._client.query_points(
            self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.DBSF if fusion == DBSF else models.Fusion.RRF
            ),
            query_filter=tenant,
            limit=limit,
            with_payload=True,
        )
        return [_stored(p) for p in result.points]

    async def ping(self) -> None:
        await self._client.collection_exists(self.collection)

    async def aclose(self) -> None:
        await self._client.close()
