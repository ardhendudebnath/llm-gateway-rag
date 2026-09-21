"""Text embedders shared by the semantic cache and the RAG pipeline.

All vectors are L2-normalised float32, so cosine similarity is a plain dot product.

Three entry points, because retrieval models are asymmetric:

* ``embed``           query-vs-query similarity (the semantic cache compares prompts to prompts)
* ``embed_query``     a search query, to be matched against passages
* ``embed_documents`` passages to be indexed, in one batch

BGE-family models expect an instruction prefix on retrieval queries but not on passages;
fastembed's ``query_embed``/``passage_embed`` apply the right one per model.
"""

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

from app.core.concurrency import InferenceGate

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    async def embed(self, text: str) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` matrix."""
        ...


class HashingEmbedder:
    """Dependency-free feature-hashing embedder (unigrams + bigrams).

    It captures *lexical* overlap ("what's the capital of France?" vs "what is the capital of
    france"), not paraphrases. Used for tests, zero-setup dev, and as the lexical baseline in the
    retrieval eval; production should use ``FastEmbedEmbedder``.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in pairwise(tokens)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in features:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(vec)


class FastEmbedEmbedder:
    """Open-weight ONNX embedding model (default BAAI/bge-small-en-v1.5) via ``fastembed``.

    Install with ``pip install -e .[embeddings]``. Inference runs in a worker thread so the event
    loop is never blocked. ``cache_dir`` points at pre-downloaded weights (the container image
    bakes them in, so pods never download models at startup).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        gate: InferenceGate | None = None,
    ):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
        # Bounded concurrency: see app/core/concurrency.py for the OOM this prevents.
        self._gate = gate or InferenceGate("embedder", max_concurrency=2, max_queue=64)

    async def embed(self, text: str) -> np.ndarray:
        vec = await self._gate.run(lambda: next(iter(self._model.embed([text]))))
        return _normalise(np.asarray(vec, dtype=np.float32))

    async def embed_query(self, text: str) -> np.ndarray:
        vec = await self._gate.run(lambda: next(iter(self._model.query_embed(text))))
        return _normalise(np.asarray(vec, dtype=np.float32))

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = await self._gate.run(lambda: list(self._model.passage_embed(list(texts))))
        return np.stack([_normalise(np.asarray(r, dtype=np.float32)) for r in rows])


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
