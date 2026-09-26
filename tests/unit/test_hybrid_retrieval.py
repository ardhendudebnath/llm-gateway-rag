"""Hybrid retrieval: fusion finds what dense search alone ranks below the cut."""

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

from app.core.embeddings import HashingEmbedder
from app.rag.chunking import Chunk
from app.rag.retrieval import Retriever
from app.rag.vector_store import DBSF, LEXICAL, RRF, QdrantChunkStore

TENANT = "acme"

# Rows that differ by one token, which is the case dense embeddings blur together.
ROWS = [
    "NG-1015 | The API key is valid but belongs to another tenant",
    "NG-1017 | The tenant's monthly token budget is exhausted",
    "NG-1021 | The request exceeds the per-key rate limit",
    "NG-1023 | The prompt is longer than the model's context window",
    "NG-1026 | The uploaded document is larger than 10 MB",
]


class RankedEmbedder:
    """Dense similarity that decreases down the rows, whatever the query says.

    A deliberate stand-in for the real failure: asked about the last row, dense search returns the
    first ones, because to an embedding these rows are nearly the same text. The eval measures the
    real version of this on the large corpus — recall@5 0.722 for identifier-only queries. Here it
    is made deterministic so the test proves what fusion adds rather than winning a coin toss.
    """

    dim = 8
    LAST = len(ROWS) - 1  # the row dense search ranks worst

    async def embed_documents(self, texts):
        vectors = []
        for i, _ in enumerate(texts):
            vector = np.zeros(self.dim, dtype=np.float32)
            vector[0], vector[1] = 1.0 - 0.2 * i, 0.2 * i
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors)

    async def embed_query(self, text):
        return np.eye(1, self.dim, 0).astype(np.float32)[0]


@pytest.fixture
async def store_of():
    clients = []

    async def build(lexical: bool, embedder=None) -> QdrantChunkStore:
        embedder = embedder or RankedEmbedder()
        client = AsyncQdrantClient(location=":memory:")
        clients.append(client)
        store = QdrantChunkStore(client, "chunks")
        await store.setup(embedder.dim, lexical=lexical)
        chunks = [Chunk(index=i, text=row, page=None, heading=None) for i, row in enumerate(ROWS)]
        vectors = await embedder.embed_documents([c.text for c in chunks])
        await store.replace_document(TENANT, "codes", "error-codes.md", chunks, vectors)
        return store, embedder

    yield build
    for client in clients:
        await client.close()


async def test_a_collection_asked_for_lexical_vectors_reports_that_it_has_them(store_of):
    store, _ = await store_of(lexical=True)
    assert store.supports_lexical is True

    dense_only, _ = await store_of(lexical=False)
    assert dense_only.supports_lexical is False


WORST = ROWS[RankedEmbedder.LAST].split(" |")[0]  # "NG-1026": last by dense similarity


async def test_asking_for_hybrid_on_an_existing_dense_collection_degrades_and_says_so(caplog):
    """The migration path: a sparse vector cannot be added to a collection after the fact, so a
    deployment that predates hybrid keeps working dense-only until it is re-created and re-ingested.
    """
    client = AsyncQdrantClient(location=":memory:")
    try:
        before = QdrantChunkStore(client, "chunks")
        await before.setup(RankedEmbedder.dim, lexical=False)  # as an older release created it

        after = QdrantChunkStore(client, "chunks")
        with caplog.at_level("WARNING"):
            await after.setup(RankedEmbedder.dim, lexical=True)

        assert after.supports_lexical is False
        assert "no lexical vector" in caplog.text
        assert "re-ingest" in caplog.text, "the operator is told what to do about it"
    finally:
        await client.close()


async def test_a_collection_that_cannot_be_inspected_is_treated_as_dense_only():
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantChunkStore(client, "chunks")

        async def broken(*_args, **_kwargs):
            raise ConnectionError("qdrant gone")

        client.get_collection = broken
        await store.setup(RankedEmbedder.dim, lexical=True)

        assert store.supports_lexical is False, "unknown means no, not a crash"
    finally:
        await client.close()


async def test_rank_fusion_brings_the_named_row_into_the_candidate_set(store_of):
    store, embedder = await store_of(lexical=True)
    vector = await embedder.embed_query(WORST)

    fused = await store.hybrid_search(TENANT, vector, WORST, limit=3)

    assert any(h.text.startswith(WORST) for h in fused), "the lexical half brought it into reach"


async def test_neither_fusion_promises_the_exact_match_the_top_slot(store_of):
    """Fusion gets a passage into reach; it does not decide it wins.

    This row scores 4.6 in the lexical half against 0.14 for its neighbours and still does not come
    first: RRF sees only "rank 1", and DBSF min-max normalises each half, which sends the *worst*
    dense hit to zero there. Which of the two orders better is a statistical question the eval
    answers over 102 questions (hit@1 0.794 for RRF, 0.873 for DBSF), not something to assert from
    one hand-made case — and it is why the reranker still earns its place on top of either.
    """
    store, embedder = await store_of(lexical=True)
    vector = await embedder.embed_query(WORST)

    for fusion in (RRF, DBSF):
        fused = await store.hybrid_search(TENANT, vector, WORST, limit=3, fusion=fusion)
        assert any(h.text.startswith(WORST) for h in fused), f"{fusion} lost it entirely"
        assert not fused[0].text.startswith(WORST), f"{fusion} is not expected to promote it here"


async def test_the_choice_of_fusion_reaches_the_server(store_of):
    """Rank fusion produces coarse, tied scores; score fusion produces distinct ones.

    The two agree on the order for five rows, so the scores are what shows the parameter arrived.
    """
    store, embedder = await store_of(lexical=True)
    vector = await embedder.embed_query(WORST)

    by_rank = await store.hybrid_search(TENANT, vector, WORST, limit=5, fusion=RRF)
    by_score = await store.hybrid_search(TENANT, vector, WORST, limit=5, fusion=DBSF)

    assert {h.text for h in by_rank} == {h.text for h in by_score}, "the same candidates"
    assert len({h.score for h in by_rank}) < len({h.score for h in by_score})


async def test_fusion_lifts_the_exact_match_above_the_dense_cut(store_of):
    """The measurable gain: dense search ranks this row last, fusion puts it third."""
    store, embedder = await store_of(lexical=True)
    vector = await embedder.embed_query(WORST)

    fused = [h.text[:7] for h in await store.hybrid_search(TENANT, vector, WORST, limit=5)]
    dense = [h.text[:7] for h in await store.search(TENANT, vector, limit=5)]

    assert dense.index(WORST) == len(ROWS) - 1, "dense retrieval ranks it worst of all"
    assert fused.index(WORST) < dense.index(WORST)


async def test_dense_search_alone_cannot(store_of):
    # The control for the test above: same data, same query, no lexical vector.
    store, embedder = await store_of(lexical=False)
    vector = await embedder.embed_query(WORST)

    hits = await store.search(TENANT, vector, limit=2)

    assert not any(h.text.startswith(WORST) for h in hits), "dense ranks it below the cut"


async def test_hybrid_search_on_a_dense_only_collection_still_answers(store_of):
    """A collection created before hybrid existed must keep working, not raise."""
    store, embedder = await store_of(lexical=False)

    hits = await store.hybrid_search(TENANT, await embedder.embed_query("NG-1023"), "NG-1023", 3)

    assert len(hits) == 3


async def test_a_query_with_no_terms_falls_back_to_the_dense_half(store_of):
    store, embedder = await store_of(lexical=True)

    hits = await store.hybrid_search(TENANT, await embedder.embed_query("???"), "???", limit=2)

    assert len(hits) == 2, "punctuation has no terms to look up, but the request still answers"


async def test_the_tenant_filter_applies_to_both_halves(store_of):
    store, embedder = await store_of(lexical=True)

    hits = await store.hybrid_search(
        "someone-else", await embedder.embed_query("NG-1023"), "NG-1023", limit=5
    )

    assert hits == [], "no path searches across tenants, fused or not"


async def test_lexical_vectors_are_written_for_every_chunk(store_of):
    store, _ = await store_of(lexical=True)

    points, _ = await store._client.scroll("chunks", limit=10, with_vectors=True)

    assert len(points) == len(ROWS)
    for point in points:
        assert point.vector[LEXICAL].indices, f"chunk {point.payload['chunk_index']} has no terms"


async def test_the_retriever_only_claims_hybrid_when_the_collection_can_do_it(store_of):
    lexical, embedder = await store_of(lexical=True)
    dense_only, _ = await store_of(lexical=False)

    assert Retriever(lexical, embedder, None, candidates=5, hybrid=True).hybrid is True
    assert Retriever(dense_only, embedder, None, candidates=5, hybrid=True).hybrid is False
    assert Retriever(lexical, embedder, None, candidates=5, hybrid=False).hybrid is False


async def test_the_retriever_uses_fusion_when_it_is_on(store_of):
    store, embedder = await store_of(lexical=True)

    hybrid = await Retriever(store, embedder, None, candidates=5, hybrid=True).search(
        TENANT, WORST, top_k=3
    )
    dense = await Retriever(store, embedder, None, candidates=5, hybrid=False).search(
        TENANT, WORST, top_k=3
    )

    assert any(h.chunk.text.startswith(WORST) for h in hybrid)
    assert not any(h.chunk.text.startswith(WORST) for h in dense)


async def test_hybrid_also_answers_a_paraphrase_with_a_real_embedder(store_of):
    """Fusion must not cost anything on the queries dense retrieval was already good at."""
    store, embedder = await store_of(lexical=True, embedder=HashingEmbedder())

    hits = await Retriever(store, embedder, None, candidates=5, hybrid=True).search(
        TENANT, "the prompt is too long for the model", top_k=2
    )

    assert any(h.chunk.text.startswith("NG-1023") for h in hits)
