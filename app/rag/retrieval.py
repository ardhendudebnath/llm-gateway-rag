"""Two-stage retrieval: vector search over the tenant's chunks, then (optionally) a cross-encoder
rerank of the top candidates."""

import time
from dataclasses import dataclass

from app.core.concurrency import OverloadedError
from app.core.embeddings import Embedder
from app.observability import metrics
from app.rag.reranking import Reranker
from app.rag.vector_store import RRF, QdrantChunkStore, StoredChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: StoredChunk
    rerank_score: float | None = None

    @property
    def vector_score(self) -> float:
        return self.chunk.score


class Retriever:
    def __init__(
        self,
        store: QdrantChunkStore,
        embedder: Embedder,
        reranker: Reranker | None,
        *,
        candidates: int,
        hybrid: bool = False,
        fusion: str = RRF,
    ):
        self._store = store
        self._embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.fusion = fusion
        # Hybrid needs a collection that carries the lexical vector; without one, dense-only is the
        # honest answer rather than an error, and the store has already said so at start-up.
        self.hybrid = hybrid and store.supports_lexical

    async def search(
        self, tenant_id: str, query: str, top_k: int, *, rerank: bool = True
    ) -> list[RetrievedChunk]:
        use_reranker = rerank and self.reranker is not None

        start = time.perf_counter()
        vector = await self._embedder.embed_query(query)
        metrics.RAG_STAGE_LATENCY.labels("embed_query").observe(time.perf_counter() - start)

        start = time.perf_counter()
        limit = max(self.candidates, top_k) if use_reranker else top_k
        if self.hybrid:
            hits = await self._store.hybrid_search(tenant_id, vector, query, limit, self.fusion)
        else:
            hits = await self._store.search(tenant_id, vector, limit)
        metrics.RAG_STAGE_LATENCY.labels(
            "hybrid_search" if self.hybrid else "vector_search"
        ).observe(time.perf_counter() - start)

        if not use_reranker or not hits:
            return [RetrievedChunk(h) for h in hits[:top_k]]

        start = time.perf_counter()
        try:
            scores = await self.reranker.scores(query, [h.text for h in hits])
        except OverloadedError:
            # Degrade rather than fail. With hybrid retrieval the unreranked order finds just as
            # much (recall@5 1.000 either way in the eval); what is lost is ordering, hit@1 0.922
            # -> 0.794. Callers see rerank_score=None.
            metrics.DEGRADED.labels("rerank_skipped").inc()
            return [RetrievedChunk(h) for h in hits[:top_k]]
        metrics.RAG_STAGE_LATENCY.labels("rerank").observe(time.perf_counter() - start)
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [RetrievedChunk(h, s) for h, s in ranked[:top_k]]
