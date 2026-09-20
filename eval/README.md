# Retrieval evaluation

How well does the RAG pipeline find the passage that answers a question, and which chunking,
embedding and reranking choices earn their cost? [`retrieval_eval.py`](retrieval_eval.py) answers
that with numbers. Latest run: [`RESULTS.md`](RESULTS.md) (raw data in `results.json`).

```bash
python -m eval.retrieval_eval               # all configs; downloads ~220 MB of models once
python -m eval.retrieval_eval --only hash   # lexical baseline only, no downloads
python -m eval.retrieval_eval --write       # regenerate RESULTS.md and results.json
```

The harness drives the production code (`IngestionService` → Qdrant → `Retriever`), so the
numbers describe what the API serves, not a notebook re-implementation.

## Method

**Corpus.** [`corpus/`](corpus) is a small engineering handbook for a *fictional* company, "Acme
Cloud": 9 Markdown documents, 2,530 words. It is fictional on purpose: an LLM can't answer the
questions from memory, so the eval measures retrieval alone. It is also deliberately repetitive:
"30 days", "24 hours" and "within 5 minutes" each appear in several documents with different
meanings, so matching numbers is not enough.

**Labels.** [`labeled_qa.jsonl`](labeled_qa.jsonl) holds 48 questions, mostly paraphrased away
from the source wording. 6 of them need two facts, and one of those draws on two documents. Each
question is labelled with *evidence spans*: short exact phrases from the corpus that contain the
answer. A retrieved passage counts as relevant if it contains a span, ignoring case, whitespace and
table pipes. Labelling spans instead of chunk ids means one label set scores every chunking
strategy, and a chunker that cuts a fact in half is penalised. A unit test checks every span
still exists verbatim in the corpus, so the labels can't silently rot.

**Metrics** (k = 5, averaged over questions):

| Metric | Meaning |
|---|---|
| Recall@5 | Share of a question's evidence spans found in the top 5 passages |
| Precision@5 | Share of the top 5 passages that contain a span. Most answers live in one chunk, so the ceiling is about 0.2; compare configs with each other, not with 1.0 |
| MRR | 1 / rank of the first relevant passage |
| Hit@1 | Share of questions whose first passage is relevant: what a model reading only the top passage would see |

## Findings

From the latest [`RESULTS.md`](RESULTS.md):

1. **Semantic embeddings matter most.** On the same chunks, the lexical baseline (feature
   hashing) reaches recall@5 0.719; `bge-small-en-v1.5` reaches 0.979. Paraphrased questions share
   few words with their answers.
2. **Overlap fixes facts cut at chunk boundaries.** Fixed 180-word windows go from recall@5 0.927
   with no overlap to 0.958 with 40 words of overlap. The misses without overlap are facts split
   across two windows.
3. **Structure beats sentence boundaries.** Packing whole sentences scores the same as fixed
   windows with overlap (0.958 recall@5, 0.854 MRR). Starting a chunk at every heading and
   prefixing it with "title > heading path" lifts MRR to 0.924 and hit@1 from 0.771 to 0.875. A
   chunk that says "within 30 days" now also says it is about *customer data deletion*.
4. **Chunk size barely matters here.** With structure-aware chunking, most sections are shorter
   than 100 words, so 100-, 180- and 300-word limits produce 52–54 chunks and near-identical
   scores. Larger documents would make this setting matter; the default stays at 180/40.
5. **Reranking closes the remaining gap, at a price.** A MiniLM cross-encoder over the top 20
   vector hits reaches recall@5 1.000 and hit@1 0.979, but costs ~230 ms per query on CPU against
   ~6 ms for vector search alone. Reranking only the top 10 halves the cost (~107 ms) and drops
   recall@5 back to 0.979. Structure-aware chunks also make reranking cheaper than fixed windows
   (231 vs 423 ms), because the cross-encoder reads shorter passages.
6. **A 4× latency bug, found by measuring.** The first run measured reranking at ~1,000 ms. ONNX
   Runtime gives every model session a thread pool the size of the machine (24 threads here), and
   the embedder's and reranker's spinning pools fought over the CPU. Capping each model at 4
   threads (`NEXUSGATE_MODEL_THREADS`) brought the same work down to ~230 ms, with identical scores.

## Defaults chosen from these results

| Setting | Value | Why |
|---|---|---|
| Chunking | `structured`, 180 words, 40 overlap | Best recall/MRR without a reranker |
| Embeddings | `BAAI/bge-small-en-v1.5` in Kubernetes | +26 points of recall@5 over lexical; 384-d, fast on CPU |
| Reranker | MiniLM cross-encoder over the top 20 in Kubernetes | Perfect recall@5 on this set for ~230 ms, small next to an LLM call |
| Dev and tests | Hashing embedder, no reranker | No model downloads; deterministic tests |

## Limitations

- **Small and self-written.** One author wrote both the corpus and the questions, which flatters
  every config: real questions are messier. Treat the numbers as a regression baseline and for
  comparing configs, not as a claim about production accuracy.
- **Mostly single-hop.** Only 6 of 48 questions need two facts, and only one crosses documents.
- **Exact-span matching** can mark a relevant passage as irrelevant if the fact is phrased twice
  in different words. Spans were chosen to be unique, and the tests check that each one exists.
- **Retrieval only.** Whether the generated answer is faithful to the passages (groundedness,
  citation accuracy) needs an LLM-judged eval; that's a planned next step.
