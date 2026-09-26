# Retrieval evaluation

How well does the RAG pipeline find the passage that answers a question, and which chunking,
embedding, fusion and reranking choices earn their cost? [`retrieval_eval.py`](retrieval_eval.py)
answers that with numbers. Latest run: [`RESULTS.md`](RESULTS.md) (raw data in `results.json`).

```bash
python -m eval.retrieval_eval                        # small corpus, all configs (~220 MB of models once)
python -m eval.retrieval_eval --only hash            # lexical baseline only, no downloads
python -m eval.retrieval_eval --corpus large         # with generated distractors
python -m eval.retrieval_eval --corpus both --write  # regenerate RESULTS.md and results.json
```

The harness drives the production code (`IngestionService` → Qdrant → `Retriever`), so the numbers
describe what the API serves, not a notebook re-implementation.

## Method

**Corpus.** [`corpus/`](corpus) is an engineering handbook for a *fictional* company: 13 Markdown
documents, 4,067 words. Fictional on purpose, so an LLM cannot answer from memory and the eval
measures retrieval alone. It is deliberately repetitive — "30 days", "24 hours" and "within 5
minutes" recur with different meanings — and four of the documents are reference tables (error
codes, a service catalogue, alert thresholds, configuration defaults) whose rows differ by a single
token.

**A second, larger corpus, because the first one stopped measuring anything.** At 70 chunks every
config with a reranker scored recall@5 1.000: with five slots and seventy candidates, retrieval is
not a hard problem. [`generate_corpus.py`](generate_corpus.py) adds several hundred generated
documents in the same shapes — 400 services, 600 error codes, 300 settings, 200 alerts — so a
question about `NG-1017` competes with thousands of rows that look like it. The filler is
deterministic from one seed, generated rather than committed, and every identifier a label depends
on is reserved so no question gains a second right answer. The hand-written documents are copied in
unchanged, so the labels still point at real text.

**Labels.** [`labeled_qa.jsonl`](labeled_qa.jsonl) holds 102 questions, each labelled with *evidence
spans*: short exact phrases from the corpus that contain the answer. A retrieved passage counts as
relevant if it contains a span, ignoring case, whitespace and table pipes. Labelling spans rather
than chunk ids means one label set scores every chunking strategy, and a chunker that cuts a fact in
half is penalised for it. Unit tests check every span still exists verbatim, so the labels cannot
silently rot.

Results are broken down by how the question is asked, because the three kinds fail for different
reasons:

| Kind | Count | Example | Why it is here |
|---|---:|---|---|
| semantic | 55 | "Who gets paged if the primary hasn't acknowledged after a quarter of an hour?" | A paraphrase sharing few words with the answer: what dense retrieval is for |
| lexical | 29 | "What does error NG-1017 mean?" | Names an identifier, in a sentence, among near-identical neighbours |
| terse | 18 | "NG-1017" | An identifier pasted in with no context at all, as into a search box |

**Metrics** (k = 5, averaged over questions):

| Metric | Meaning |
|---|---|
| Recall@5 | Share of a question's evidence spans found in the top 5 passages |
| Precision@5 | Share of the top 5 passages containing a span. Most answers live in one chunk, so the ceiling is about 0.2; compare configs with each other, not with 1.0 |
| MRR | 1 / rank of the first relevant passage |
| Hit@1 | Share of questions whose first passage is relevant: what a model reading only the top passage would see |

## Findings

From the latest [`RESULTS.md`](RESULTS.md). The large-corpus numbers are the interesting ones.

1. **Distractors are what make retrieval hard, not question phrasing.** The same pipeline
   (`bge-small` + structured chunks) scores recall@5 0.971 on 70 chunks and 0.873 on 227. Adding
   documents that nobody asked about is what moves the number.
2. **Dense retrieval is weakest exactly where a search box is used.** On the large corpus it reaches
   0.982 on paraphrases and 0.722 on bare identifiers. The nearest neighbours of `NG-1017` are the
   codes either side of it. The crude lexical baseline, worse everywhere else (0.578 overall), beats
   it on those queries: 0.833.
3. **Fusing the two fixes it completely, and cheaply.** BM25 sparse vectors fused with the dense
   ones by reciprocal rank ([`app/rag/lexical.py`](../app/rag/lexical.py)) reach recall@5 **1.000 on
   all three kinds at 14.7 ms**. Reranked dense retrieval reaches 0.922 at 515 ms — fusion is both
   better at finding the passage and ~35× cheaper than reranking to fix the same problem.
4. **Rank fusion and score fusion buy different things.** RRF ignores score magnitude, so a passage
   the lexical half matched exactly (BM25 4.6 against 0.14 for its neighbours) counts only as
   "rank 1" and can lose a tie to the dense half's best guess. DBSF normalises each half and adds,
   which puts it first far more often — hit@1 0.873 against 0.794, MRR 0.924 against 0.884 — at the
   cost of a point of recall (0.990, terse 0.944) and an assumption that the two score
   distributions are comparable. RRF is the default because recall decides whether an answer is
   possible at all.
5. **The reranker still earns its place, for ordering.** Hybrid + rerank gives the best MRR (0.948)
   and hit@1 (0.922) against hybrid alone with RRF (0.884 / 0.794) — it puts the right passage
   *first* far more often. It costs recall: 0.980, because the cross-encoder pushes a correct
   passage out of the top 5 for two questions. Fusion maximises what is findable; reranking
   maximises what is read first. Both are on in production, and `rerank=false` is now a defensible
   500 ms saving when only recall matters.
6. **Structure-aware chunking matters more as the corpus grows.** On the small corpus, structured
   chunks beat fixed windows by 0.025 recall@5 (0.971 vs 0.946); on the large corpus, by 0.123
   (0.873 vs 0.750). Prefixing each chunk with its "title > heading path" is what distinguishes one
   near-identical table row from the next.
7. **Overlap fixes facts cut at a boundary.** Fixed 180-word windows: recall@5 0.647 with no
   overlap, 0.750 with 40 words of it, on the large corpus.
8. **Chunk size barely matters, still.** 100/25 produces 350 chunks against 227 for 180/40 and
   scores slightly worse (0.863 vs 0.873): more, smaller chunks mean more near-duplicates competing.
9. **A 4× latency bug, found by measuring.** The first run measured reranking at ~1,000 ms. ONNX
   Runtime gives every model session a thread pool the size of the machine (24 threads here), and
   the embedder's and reranker's spinning pools fought over the CPU. Capping each model at 4 threads
   (`NEXUSGATE_MODEL_THREADS`) brought the same work down to ~230 ms, with identical scores.

## Defaults chosen from these results

| Setting | Value | Why |
|---|---|---|
| Chunking | `structured`, 180 words, 40 overlap | Best recall and MRR at both corpus sizes, and the gap widens with distractors |
| Embeddings | `BAAI/bge-small-en-v1.5` in Kubernetes | +30 points of recall@5 over lexical hashing on the large corpus; 384-d, fast on CPU |
| Hybrid retrieval | On (`NEXUSGATE_RETRIEVAL_HYBRID`) | Recall@5 0.873 → 1.000 on the large corpus for ~7 ms, and it is the only thing that fixes identifier lookups |
| Reranker | MiniLM cross-encoder over the top 20 | Hit@1 0.765 → 0.922 on fused candidates; skippable when latency matters more than ordering |
| Dev and tests | Hashing embedder, no reranker, hybrid on | No model downloads; deterministic tests |

## Limitations

- **Self-written labels.** One author wrote the corpus and the questions, which flatters every
  config: real questions are messier. Treat the numbers as a regression baseline and a way to
  compare configs, not as a claim about production accuracy.
- **The distractors are generated**, not real documents. They are realistic in shape and volume,
  which is what retrieval competes against, but they are repetitive in ways real corpora are not.
- **Mostly single-hop.** Only 6 questions need two facts, and only one crosses documents.
- **Exact-span matching** can mark a relevant passage irrelevant if the fact is phrased twice in
  different words. Spans were chosen to be unique, and tests check each one exists.
- **Retrieval only.** Whether the generated answer is faithful to the passages (groundedness,
  citation accuracy) needs an LLM-judged eval; that needs a real provider, so the offline suite
  cannot answer it.
