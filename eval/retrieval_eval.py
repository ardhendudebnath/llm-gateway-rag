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
    # "semantic": a paraphrase, which is what dense retrieval is good at.
    # "lexical":  names an exact identifier (NG-1017, EMBED_MAX_QUEUE, invoice-mailer) that sits
    #             among near-identical neighbours, which is where dense retrieval breaks down.
    kind: str = "semantic"


@dataclass(frozen=True)
class EvalConfig:
    chunker: str  # fixed | sentence | structured
    max_words: int
    overlap_words: int
    embedder: str  # hash | bge-small
    reranker: str | None = None  # None | minilm
    candidates: int = 20  # vector hits handed to the reranker
    hybrid: bool = False  # BM25 sparse vectors fused with the dense ones (app/rag/lexical.py)
    fusion: str = "rrf"  # rrf (ranks) | dbsf (normalised scores)

    @property
    def name(self) -> str:
        rerank = f" + {self.reranker} rerank of top {self.candidates}" if self.reranker else ""
        chunking = f"{self.chunker}-{self.max_words}/{self.overlap_words}"
        hybrid = f" + bm25 {self.fusion}" if self.hybrid else ""
        return f"{self.embedder} + {chunking}{hybrid}{rerank}"


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
    # Hybrid: the same pipeline with BM25 sparse vectors fused in by reciprocal rank. The pair
    # with and without the reranker shows whether fusion earns its place on its own.
    EvalConfig("structured", 180, 40, "bge-small", hybrid=True),
    EvalConfig("structured", 180, 40, "bge-small", "minilm", hybrid=True),
    # Score fusion instead of rank fusion: does an exact lexical match deserve more than "rank 1"?
    EvalConfig("structured", 180, 40, "bge-small", hybrid=True, fusion="dbsf"),
    EvalConfig("structured", 180, 40, "bge-small", "minilm", hybrid=True, fusion="dbsf"),
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
        await store.setup(embedder.dim, lexical=config.hybrid)
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
            store,
            embedder,
            models.reranker(config.reranker),
            candidates=config.candidates,
            hybrid=config.hybrid,
            fusion=config.fusion,
        )
        per_question, latencies = [], []
        for q in questions:
            start = time.perf_counter()
            hits = await retriever.search(TENANT, q.question, top_k=k)
            latencies.append((time.perf_counter() - start) * 1000)
            per_question.append(
                {
                    "id": q.id,
                    "kind": q.kind,
                    **score([h.chunk.text for h in hits], q.evidence, k),
                }
            )
    finally:
        await client.close()

    def mean(key: str, rows: list[dict] | None = None) -> float:
        return round(statistics.fmean(r[key] for r in (rows or per_question)), 3)

    # Reported per kind as well as overall: a config that lifts the average by wrecking one half of
    # the set is not an improvement, and the two kinds fail for different reasons.
    by_kind = {}
    for kind in sorted({r["kind"] for r in per_question}):
        rows = [r for r in per_question if r["kind"] == kind]
        by_kind[kind] = {
            "questions": len(rows),
            f"recall@{k}": mean("recall", rows),
            "mrr": mean("mrr", rows),
            "hit@1": mean("hit1", rows),
            "misses": [r["id"] for r in rows if r["recall"] < 1],
        }

    return {
        "config": config.name,
        "settings": asdict(config),
        "chunks": chunks,
        f"precision@{k}": mean("precision"),
        f"recall@{k}": mean("recall"),
        "mrr": mean("mrr"),
        "hit@1": mean("hit1"),
        "ms_per_query_p50": round(statistics.median(latencies), 1),
        "by_kind": by_kind,
        "misses": [r["id"] for r in per_question if r["recall"] < 1],
    }


def markdown_table(results: list[dict], k: int) -> str:
    kinds = sorted({kind for r in results for kind in r.get("by_kind", {})})
    header = ["Config", "Chunks", f"P@{k}", f"R@{k}"]
    header += [f"R@{k} {kind}" for kind in kinds]
    header += ["MRR", "Hit@1", "ms/query (p50)"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---" + "|---:" * (len(header) - 1) + "|",
    ]
    for r in results:
        by_kind = r.get("by_kind", {})
        cells = [
            r["config"],
            str(r["chunks"]),
            f"{r[f'precision@{k}']:.3f}",
            f"{r[f'recall@{k}']:.3f}",
        ]
        cells += [
            f"{by_kind[kind][f'recall@{k}']:.3f}" if kind in by_kind else "—" for kind in kinds
        ]
        cells += [f"{r['mrr']:.3f}", f"{r['hit@1']:.3f}", str(r["ms_per_query_p50"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--corpus",
        choices=["small", "large", "both"],
        default="small",
        help="'small' is the 13 hand-written documents; 'large' adds generated distractors, which "
        "is where retrieval starts to be a hard problem (see eval/generate_corpus.py); 'both' runs "
        "each in turn, which is what RESULTS.md reports",
    )
    parser.add_argument("--only", help="run configs whose name contains this text")
    parser.add_argument("--model-cache-dir", help="where fastembed stores model weights")
    parser.add_argument(
        "--threads", type=int, default=4, help="ONNX threads per model (the API's default)"
    )
    parser.add_argument("--write", action="store_true", help="write RESULTS.md and results.json")
    args = parser.parse_args()

    questions = load_questions()
    configs = [c for c in CONFIGS if not args.only or args.only in c.name]
    models = ModelCache(args.model_cache_dir, args.threads)
    sizes = ["small", "large"] if args.corpus == "both" else [args.corpus]

    runs = []
    for size in sizes:
        corpus = load_corpus(large_corpus() if size == "large" else CORPUS_DIR)
        words = sum(len(d.decode("utf-8").split()) for d in corpus.values())
        print(
            f"\n{size} corpus: {len(corpus)} documents, {words} words, "
            f"{len(questions)} questions, k={args.k}\n"
        )
        results = []
        for config in configs:
            print(f"-> [{size}] {config.name}", file=sys.stderr, flush=True)
            results.append(await run_config(config, corpus, questions, models, args.k))
        # Quality first; among equally good configs, the faster one wins.
        best = max(results, key=lambda r: (r[f"recall@{args.k}"], r["mrr"], -r["ms_per_query_p50"]))
        table = markdown_table(results, args.k)
        print(table)
        print(
            f"\nBest: {best['config']}. Questions it misses: {', '.join(best['misses']) or 'none'}"
        )
        runs.append(
            {
                "corpus": size,
                "documents": len(corpus),
                "words": words,
                "results": results,
                "best": best["config"],
                "table": table,
            }
        )

    if args.write:
        write_results(runs, questions, args)
    return 0


def large_corpus() -> Path:
    """Generate the distractor corpus into a temp directory: one seed, nothing committed."""
    from tempfile import mkdtemp

    from eval.generate_corpus import generate

    out = Path(mkdtemp(prefix="nexusgate-eval-corpus-"))
    generate(out)
    return out


def write_results(runs: list[dict], questions: list[Question], args) -> None:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    (EVAL_DIR / "results.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "questions": len(questions),
                "kinds": {
                    kind: sum(1 for q in questions if q.kind == kind)
                    for kind in sorted({q.kind for q in questions})
                },
                "k": args.k,
                "runs": [{k: v for k, v in run.items() if k != "table"} for run in runs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    kinds = ", ".join(
        f"{sum(1 for q in questions if q.kind == kind)} {kind}"
        for kind in sorted({q.kind for q in questions})
    )
    sections = []
    for run in runs:
        sections.append(
            f"## {run['corpus'].capitalize()} corpus: {run['documents']} documents "
            f"({run['words']} words)\n\n{run['table']}\n\nBest by recall@{args.k}, then MRR, then "
            f"latency: **{run['best']}**.\n"
        )
    (EVAL_DIR / "RESULTS.md").write_text(
        f"# Retrieval eval results\n\n"
        f"Generated by `python -m eval.retrieval_eval --corpus both --write` on {generated_at}. "
        f"{len(questions)} labelled questions ({kinds}), k={args.k}. Timings: CPU only, "
        f"{args.threads} ONNX threads per model, in-process Qdrant.\n\n"
        f"The small corpus is the 13 hand-written documents. The large one adds generated "
        f"distractors ([`generate_corpus.py`](generate_corpus.py)) so that a question about one "
        f"identifier competes with thousands of rows that look like it; at 70 chunks every config "
        f"scores recall@5 1.000, which measures nothing.\n\n" + "\n".join(sections),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
