"""Second-stage reranking.

Vector search compares a query embedding with passage embeddings computed independently of each
other. A cross-encoder reads the query and a passage *together*, which is far more accurate but
too slow to run over a whole corpus, so it only re-scores the vector search's top candidates.
"""

from collections.abc import Sequence
from typing import Protocol

from app.core.concurrency import InferenceGate


class Reranker(Protocol):
    name: str

    async def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage; higher is more relevant. Scale is model-specific.

        May raise ``OverloadedError`` when saturated; the retriever then falls back to vector order.
        """
        ...


class CrossEncoderReranker:
    """ONNX cross-encoder via ``fastembed`` (default: ms-marco MiniLM-L-6, ~80 MB, CPU-friendly)."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        gate: InferenceGate | None = None,
        max_wait_seconds: float | None = None,
    ):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir, threads=threads)
        # Reranking is the expensive step and it is optional (vector order alone scored recall@5
        # 0.979 in the eval). So under load it gets a wait budget rather than a long queue: if it
        # can't start in time, the retriever skips it instead of making the caller wait.
        self._gate = gate or InferenceGate("reranker", max_concurrency=2, max_queue=8)
        self._max_wait = max_wait_seconds

    async def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw = await self._gate.run(
            lambda: list(self._model.rerank(query, list(passages))), max_wait=self._max_wait
        )
        return [float(s) for s in raw]
