"""Second-stage reranking.

Vector search compares a query embedding with passage embeddings computed independently of each
other. A cross-encoder reads the query and a passage *together*, which is far more accurate but
too slow to run over a whole corpus, so it only re-scores the vector search's top candidates.
"""

import asyncio
from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    name: str

    async def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage; higher is more relevant. Scale is model-specific."""
        ...


class CrossEncoderReranker:
    """ONNX cross-encoder via ``fastembed`` (default: ms-marco MiniLM-L-6, ~80 MB, CPU-friendly)."""

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir, threads=threads)

    async def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw = await asyncio.to_thread(lambda: list(self._model.rerank(query, list(passages))))
        return [float(s) for s in raw]
