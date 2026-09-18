"""Query embedders for the semantic cache. All return L2-normalised float32 vectors, so cosine
similarity is a plain dot product."""

import asyncio
import hashlib
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    async def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free feature-hashing embedder (unigrams + bigrams).

    It captures *lexical* near-duplicates ("what's the capital of France?" vs "what is the capital
    of france"), not paraphrases. Used for tests and zero-setup dev; production should use
    ``FastEmbedEmbedder`` so paraphrases hit the cache too.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, text: str) -> np.ndarray:
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
    loop is never blocked.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    async def embed(self, text: str) -> np.ndarray:
        vec = await asyncio.to_thread(lambda: next(iter(self._model.embed([text]))))
        return _normalise(np.asarray(vec, dtype=np.float32))


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
