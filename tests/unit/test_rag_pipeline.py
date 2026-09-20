"""Ingestion, the Qdrant chunk store and retrieval, against an in-process Qdrant."""

from types import SimpleNamespace

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.embeddings import HashingEmbedder
from app.rag.chunking import Chunk, build_chunker
from app.rag.documents import DocumentRegistry
from app.rag.ingestion import DocumentTooLargeError, DocumentUpload, IngestionService
from app.rag.retrieval import Retriever
from app.rag.vector_store import QdrantChunkStore, point_id

COLLECTION = "test_chunks"

ONCALL = b"""# On-call policy

## Escalation
If the primary on-call engineer does not acknowledge a page within 5 minutes, the secondary is
paged. After 15 minutes without acknowledgement the engineering manager is paged.

## Compensation
Each weekend on-call shift is compensated with one extra day of leave.
"""

RETENTION = b"""# Data retention

Application logs are kept for 30 days. Audit logs are kept for seven years. Customer data is
deleted within 30 days of an account being closed.
"""


class KeywordReranker:
    """Deterministic stand-in for a cross-encoder: counts query words present in the passage."""

    name = "keyword"

    async def scores(self, query, passages):
        terms = set(query.lower().split())
        return [float(sum(t in p.lower() for t in terms)) for p in passages]


@pytest.fixture
async def rag(redis_pair):
    client = AsyncQdrantClient(location=":memory:")
    embedder = HashingEmbedder()
    store = QdrantChunkStore(client, COLLECTION)
    await store.setup(embedder.dim)
    registry = DocumentRegistry(redis_pair[0])
    ingestion = IngestionService(
        store,
        registry,
        embedder,
        build_chunker("structured", 60, 10),
        chunker_name="structured/60/10",
        max_bytes=100_000,
    )
    yield SimpleNamespace(client=client, store=store, registry=registry, ingestion=ingestion)
    await client.close()


def upload(data: bytes, filename: str = "doc.md", title: str | None = None) -> DocumentUpload:
    return DocumentUpload(filename=filename, content_type="text/markdown", data=data, title=title)


async def count_points(client, tenant_id: str) -> int:
    flt = models.Filter(
        must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    )
    return (await client.count(COLLECTION, count_filter=flt, exact=True)).count


async def test_ingest_then_search_finds_the_relevant_chunk(rag):
    record = await rag.ingestion.ingest("acme", upload(ONCALL, "oncall.md"))
    await rag.ingestion.ingest("acme", upload(RETENTION, "retention.md"))
    assert record.title == "On-call policy"
    assert record.chunks >= 2

    retriever = Retriever(rag.store, HashingEmbedder(), None, candidates=10)
    [top] = await retriever.search("acme", "how long are audit logs kept", top_k=1)
    assert top.chunk.title == "Data retention"
    assert "seven years" in top.chunk.text


async def test_tenants_cannot_retrieve_or_delete_each_others_documents(rag):
    record = await rag.ingestion.ingest("tenant-a", upload(ONCALL))
    retriever = Retriever(rag.store, HashingEmbedder(), None, candidates=10)

    assert await retriever.search("tenant-b", "on-call escalation after 5 minutes", 5) == []
    assert await rag.registry.list("tenant-b") == []
    assert await rag.ingestion.delete("tenant-b", record.doc_id) is False
    assert await count_points(rag.client, "tenant-a") == record.chunks  # untouched


async def test_same_file_from_two_tenants_is_stored_separately(rag):
    a = await rag.ingestion.ingest("tenant-a", upload(ONCALL))
    b = await rag.ingestion.ingest("tenant-b", upload(ONCALL))
    assert a.doc_id == b.doc_id  # content-addressed...
    assert point_id("tenant-a", a.doc_id, 0) != point_id("tenant-b", b.doc_id, 0)  # ...not shared
    await rag.ingestion.delete("tenant-a", a.doc_id)
    assert await count_points(rag.client, "tenant-b") == b.chunks


async def test_reingesting_the_same_file_is_idempotent(rag):
    first = await rag.ingestion.ingest("acme", upload(ONCALL))
    second = await rag.ingestion.ingest("acme", upload(ONCALL))
    assert first.doc_id == second.doc_id
    assert await count_points(rag.client, "acme") == first.chunks
    assert len(await rag.registry.list("acme")) == 1


async def test_replacing_a_document_drops_its_stale_chunks(rag):
    embedder = HashingEmbedder()
    long_version = [Chunk(i, f"chunk number {i}") for i in range(5)]
    await rag.store.replace_document(
        "acme", "doc1", "Doc", long_version, await embedder.embed_documents(["x"] * 5)
    )
    await rag.store.replace_document(
        "acme", "doc1", "Doc", long_version[:2], await embedder.embed_documents(["x"] * 2)
    )
    assert await count_points(rag.client, "acme") == 2


async def test_delete_removes_chunks_and_record(rag):
    record = await rag.ingestion.ingest("acme", upload(ONCALL))
    assert await rag.ingestion.delete("acme", record.doc_id) is True
    assert await count_points(rag.client, "acme") == 0
    assert await rag.registry.get("acme", record.doc_id) is None
    assert await rag.ingestion.delete("acme", record.doc_id) is False


async def test_title_is_explicit_then_h1_then_filename(rag):
    assert (await rag.ingestion.ingest("t", upload(ONCALL, title=" Custom "))).title == "Custom"
    assert (await rag.ingestion.ingest("t", upload(RETENTION))).title == "Data retention"
    plain = await rag.ingestion.ingest("t", upload(b"No heading here.", "runbook-v2.txt"))
    assert plain.title == "runbook-v2"


async def test_oversized_upload_is_rejected(rag):
    rag.ingestion.max_bytes = 10
    with pytest.raises(DocumentTooLargeError):
        await rag.ingestion.ingest("acme", upload(ONCALL))


async def test_reranker_reorders_candidates_and_can_be_bypassed(rag):
    await rag.ingestion.ingest("acme", upload(ONCALL))
    await rag.ingestion.ingest("acme", upload(RETENTION))
    retriever = Retriever(rag.store, HashingEmbedder(), KeywordReranker(), candidates=10)

    reranked = await retriever.search("acme", "weekend compensation leave", top_k=3)
    assert "extra day of leave" in reranked[0].chunk.text
    assert [h.rerank_score for h in reranked] == sorted(
        (h.rerank_score for h in reranked), reverse=True
    )

    plain = await retriever.search("acme", "weekend compensation leave", top_k=3, rerank=False)
    assert all(h.rerank_score is None for h in plain)
    assert len(plain) == 3


async def test_reranker_sees_more_candidates_than_it_returns(rag):
    for i in range(6):
        await rag.ingestion.ingest("acme", upload(f"# Doc {i}\nfiller text {i}".encode()))
    seen: list[int] = []

    class CountingReranker(KeywordReranker):
        async def scores(self, query, passages):
            seen.append(len(passages))
            return await super().scores(query, passages)

    retriever = Retriever(rag.store, HashingEmbedder(), CountingReranker(), candidates=5)
    assert len(await retriever.search("acme", "filler", top_k=2)) == 2
    assert seen == [5]


class RacingClient:
    """Another replica creates the collection between our existence check and our create call."""

    def __init__(self, error: Exception):
        self._error = error
        self.index_calls = 0

    async def collection_exists(self, name):
        return False

    async def create_collection(self, *args, **kwargs):
        raise self._error

    async def create_payload_index(self, *args, **kwargs):
        self.index_calls += 1


async def test_store_setup_tolerates_losing_the_create_race():
    client = RacingClient(ValueError(f"Collection {COLLECTION} already exists!"))
    await QdrantChunkStore(client, COLLECTION).setup(dim=8)
    assert client.index_calls == 2


async def test_store_setup_raises_other_errors():
    error = UnexpectedResponse(500, "Internal Server Error", b"boom", None)
    with pytest.raises(UnexpectedResponse):
        await QdrantChunkStore(RacingClient(error), COLLECTION).setup(dim=8)
