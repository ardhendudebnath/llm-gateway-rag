"""Retrieval evaluation: precision@k, recall@k, MRR and hit@1 on a labelled question set, for each
combination of chunking strategy, embedding model and reranker worth comparing.

    python -m eval.retrieval_eval               # every config (needs the `embeddings` extra)
    python -m eval.retrieval_eval --only hash   # lexical baseline only: no model download
    python -m eval.retrieval_eval --write       # also rewrite eval/RESULTS.md and results.json

It drives the production code path (``IngestionService`` and ``Retriever``) against an in-process
Qdrant, so what is measured is what the API serves.

Relevance is judged by *evidence spans*, not chunk ids: a retrieved passage is relevant if it
contains one of the question's evidence phrases (ignoring case, whitespace and table pipes). One set
of labels therefore scores every chunking strategy, and a chunker that cuts a fact in half is
penalised for it, which is part of what we want to measure.

Metrics, averaged over questions:

* recall@k     share of a question's evidence spans found in the top k passages
* precision@k  share of the top k passages that contain an evidence span. Most questions have one
               span that lives in one chunk, so the ceiling is about 1/k; compare configs against
               each other, not against 1.0
* MRR          1 / rank of the first relevant passage (0 if none is in the top k)
* hit@1        share of questions whose top passage is relevant
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.core.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from app.rag.chunking import build_chunker
from app.rag.ingestion import DocumentUpload, IngestionService
from app.rag.reranking import CrossEncoderReranker, Reranker
from app.rag.retrieval import Retriever
from app.rag.vector_store import QdrantChunkStore

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
QA_FILE = EVAL_DIR / "labeled_qa.jsonl"
TENANT = "eval"

EMBEDDING_MODELS = {"bge-small": "BAAI/bge-small-en-v1.5"}
RERANKER_MODELS = {"minilm": "Xenova/ms-marco-MiniLM-L-6-v2"}


@dataclass(frozen=True)
class Question:
    id: str
    doc: str
    question: str
    evidence: list[str]
    answer: str


@dataclass(frozen=True)
class EvalConfig:
    chunker: str  # fixed | sentence | structured
    max_words: int
    overlap_words: int
    embedder: str  # hash | bge-small
    reranker: str | None = None  # None | minilm
    candidates: int = 20  # vector hits handed to the reranker

    @property
    def name(self) -> str:
        rerank = f" + {self.reranker} rerank of top {self.candidates}" if self.reranker else ""
        chunking = f"{self.chunker}-{self.max_words}/{self.overlap_words}"
        return f"{self.embedder} + {chunking}{rerank}"


# Each group changes one variable against the baseline structured-180/40 with bge-small.
CONFIGS = [
    EvalConfig("structured", 180, 40, "hash"),  # lexical baseline: no semantic model at all
    EvalConfig("fixed", 180, 0, "bge-small"),  # chunking strategy
    EvalConfig("fixed", 180, 40, "bge-small"),
    EvalConfig("sentence", 180, 40, "bge-small"),
    EvalConfig("structured", 180, 40, "bge-small"),
    EvalConfig("structured", 100, 25, "bge-small"),  # chunk size
    EvalConfig("structured", 300, 60, "bge-small"),
    EvalConfig("fixed", 180, 40, "bge-small", "minilm"),  # reranking
    EvalConfig("structured", 180, 40, "bge-small", "minilm"),
    EvalConfig("structured", 180, 40, "bge-small", "minilm", candidates=10),
]


def normalise(text: str) -> str:
    return re.sub(r"[\s|]+", " ", text.lower()).strip()


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(corpus_dir.glob("*.md"))}


def load_questions(qa_file: Path = QA_FILE) -> list[Question]:
    with qa_file.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


class _NullRegistry:
    """The eval needs chunks in Qdrant, not document records in Redis."""

    async def put(self, tenant_id, record) -> None:
        return None


class ModelCache:
    """Loads each embedding/reranking model once across all configs."""

    def __init__(self, cache_dir: str | None = None, threads: int | None = None):
        self._cache_dir = cache_dir
        self._threads = threads
        self._embedders: dict[str, Embedder] = {"hash": HashingEmbedder()}
        self._rerankers: dict[str, Reranker] = {}

    def embedder(self, name: str) -> Embedder:
        if name not in self._embedders:
            self._embedders[name] = FastEmbedEmbedder(
                EMBEDDING_MODELS[name], self._cache_dir, self._threads
            )
        return self._embedders[name]

    def reranker(self, name: str | None) -> Reranker | None:
        if name is None:
            return None
        if name not in self._rerankers:
            self._rerankers[name] = CrossEncoderReranker(
                RERANKER_MODELS[name], self._cache_dir, self._threads
            )
        return self._rerankers[name]


def score(passages: list[str], evidence: list[str], k: int) -> dict[str, float]:
    spans = [normalise(e) for e in evidence]
    texts = [normalise(p) for p in passages[:k]]
    relevant = [any(s in t for s in spans) for t in texts]
    first = next((rank for rank, rel in enumerate(relevant, start=1) if rel), None)
    return {
        "precision": sum(relevant) / k,
        "recall": sum(any(s in t for t in texts) for s in spans) / len(spans),
        "mrr": 1 / first if first else 0.0,
        "hit1": float(bool(relevant and relevant[0])),
    }


async def run_config(
    config: EvalConfig,
    corpus: dict[str, bytes],
    questions: list[Question],
    models: ModelCache,
    k: int = 5,
) -> dict:
    embedder = models.embedder(config.embedder)
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantChunkStore(client, "eval")
        await store.setup(embedder.dim)
        ingestion = IngestionService(
            store,
            _NullRegistry(),
            embedder,
            build_chunker(config.chunker, config.max_words, config.overlap_words),
            chunker_name=config.name,
            max_bytes=10 * 1024 * 1024,
        )
        chunks = 0
        for filename, data in corpus.items():
            record = await ingestion.ingest(TENANT, DocumentUpload(filename, "text/markdown", data))
            chunks += record.chunks

        retriever = Retriever(
            store, embedder, models.reranker(config.reranker), candidates=config.candidates
        )
        per_question, latencies = [], []
        for q in questions:
            start = time.perf_counter()
            hits = await retriever.search(TENANT, q.question, top_k=k)
            latencies.append((time.perf_counter() - start) * 1000)
            per_question.append({"id": q.id, **score([h.chunk.text for h in hits], q.evidence, k)})
    finally:
        await client.close()

    def mean(key: str) -> float:
        return round(statistics.fmean(r[key] for r in per_question), 3)

    return {
        "config": config.name,
        "settings": asdict(config),
        "chunks": chunks,
        f"precision@{k}": mean("precision"),
        f"recall@{k}": mean("recall"),
        "mrr": mean("mrr"),
        "hit@1": mean("hit1"),
        "ms_per_query_p50": round(statistics.median(latencies), 1),
        "misses": [r["id"] for r in per_question if r["recall"] < 1],
    }


def markdown_table(results: list[dict], k: int) -> str:
    lines = [
        f"| Config | Chunks | P@{k} | R@{k} | MRR | Hit@1 | ms/query (p50) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['chunks']} | {r[f'precision@{k}']:.3f} | {r[f'recall@{k}']:.3f}"
            f" | {r['mrr']:.3f} | {r['hit@1']:.3f} | {r['ms_per_query_p50']} |"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--only", help="run configs whose name contains this text")
    parser.add_argument("--model-cache-dir", help="where fastembed stores model weights")
    parser.add_argument(
        "--threads", type=int, default=4, help="ONNX threads per model (the API's default)"
    )
    parser.add_argument("--write", action="store_true", help="write RESULTS.md and results.json")
    args = parser.parse_args()

    corpus, questions = load_corpus(), load_questions()
    configs = [c for c in CONFIGS if not args.only or args.only in c.name]
    words = sum(len(d.decode("utf-8").split()) for d in corpus.values())
    print(f"{len(corpus)} documents, {words} words, {len(questions)} questions, k={args.k}\n")

    models = ModelCache(args.model_cache_dir, args.threads)
    results = []
    for config in configs:
        print(f"-> {config.name}", file=sys.stderr, flush=True)
        results.append(await run_config(config, corpus, questions, models, args.k))

    table = markdown_table(results, args.k)
    print(table)
    # Quality first; among equally good configs, the faster one wins.
    best = max(results, key=lambda r: (r[f"recall@{args.k}"], r["mrr"], -r["ms_per_query_p50"]))
    print(f"\nBest: {best['config']}. Questions it misses: {', '.join(best['misses']) or 'none'}")

    if args.write:
        summary = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "documents": len(corpus),
            "words": words,
            "questions": len(questions),
            "k": args.k,
            "results": results,
        }
        (EVAL_DIR / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (EVAL_DIR / "RESULTS.md").write_text(
            f"# Retrieval eval results\n\n"
            f"Generated by `python -m eval.retrieval_eval --write` on {summary['generated_at']}.\n"
            f"{len(corpus)} documents ({words} words), {len(questions)} labelled questions, "
            f"k={args.k}. Timings: CPU only, {args.threads} ONNX threads per model, in-process "
            f"Qdrant.\n\n{table}\n\n"
            f"Best by recall@{args.k}, then MRR, then latency: **{best['config']}**. "
            f"Questions it misses: {', '.join(best['misses']) or 'none'}.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
