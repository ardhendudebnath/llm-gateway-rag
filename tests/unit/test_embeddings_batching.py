"""Document embedding is batched, so one big document cannot exhaust a worker's memory."""

import numpy as np

from app.core.embeddings import FastEmbedEmbedder


class FakeModel:
    """Stands in for fastembed's TextEmbedding, recording how much it was asked to do at once."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.batches: list[int] = []

    def embed(self, texts, **_kwargs):
        return iter([np.ones(self.dim, dtype=np.float32) for _ in texts])

    def passage_embed(self, texts, **_kwargs):
        texts = list(texts)
        self.batches.append(len(texts))
        return [np.full(self.dim, i + 1, dtype=np.float32) for i in range(len(texts))]


def embedder(batch_size: int) -> tuple[FastEmbedEmbedder, FakeModel]:
    model = FakeModel()
    return FastEmbedEmbedder("fake-model", batch_size=batch_size, model=model), model


async def test_a_large_document_is_embedded_in_bounded_batches():
    # The bug this covers: every chunk went into ONNX in one call, so a 160 KB upload
    # (~660 chunks) pushed a worker past its 2 GiB limit and it was OOM-killed.
    emb, model = embedder(batch_size=3)

    vectors = await emb.embed_documents([f"chunk {i}" for i in range(7)])

    assert model.batches == [3, 3, 1]  # never the whole document at once
    assert vectors.shape == (7, emb.dim)


async def test_every_vector_is_normalised_whichever_batch_it_came_from():
    emb, _ = embedder(batch_size=2)

    vectors = await emb.embed_documents([f"chunk {i}" for i in range(5)])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


async def test_no_chunks_means_no_inference():
    emb, model = embedder(batch_size=4)

    vectors = await emb.embed_documents([])

    assert vectors.shape == (0, emb.dim) and model.batches == []
