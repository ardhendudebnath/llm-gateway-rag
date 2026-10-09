# Answer evaluation

Retrieval is measured in [README.md](README.md). This measures what comes after it: given the
passages, is the answer correct, does it stay within them, does it cite them, and does it admit
when they don't hold the answer? It runs against the self-hosted model, Qwen2.5-1.5B-Instruct
(4-bit, CPU), the one model this project can run without an API key.

```bash
python -m eval.answer_eval --admin-token "$ADMIN_TOKEN"                 # the `local` route
python -m eval.answer_eval --model mock --label control                 # the negative control
python -m eval.answer_eval --resummarise eval/answers/rag-local.json    # after a label change
```

Raw results: [`answers/rag-local.json`](answers/rag-local.json) (every question, answer and score),
[`answers/control.json`](answers/control.json), and the hand review,
[`answers/rag-local.review.jsonl`](answers/rag-local.review.jsonl).

## Method

**End to end, through the API.** [`answer_eval.py`](answer_eval.py) gives a fresh tenant the 13
corpus documents through the upload endpoint and the ingestion workers, then sends every question
to `/v1/rag/answer` with `temperature: 0` and the cache off: hybrid retrieval, reranking, the top
5 passages, the production prompt. What is scored is what a client receives.

**The questions.** The 102 labelled questions of the retrieval eval, which carry a gold answer and
the evidence spans that hold it, plus 14 [unanswerable ones](unanswerable.jsonl): in-domain
questions (parental leave, the approved programming languages, the budget for a team offsite) that
the corpus does not answer, each listing terms whose appearance in the corpus would make it
answerable. A test fails if one appears.

**Deterministic scoring, no LLM judge.** The only model to hand is the 1.5B one being judged.
Per question:

| | |
|---|---|
| retrieved | a passage given to the model holds an evidence span; if none does, the question tested retrieval |
| correct | every number and identifier of the gold answer, and at least half of the content words it adds to the question |
| cited / cited right | the answer cites a passage inline (`[n]`) / cites one that holds the evidence |
| unsupported | numbers or identifiers in the answer that appear in neither the question nor any passage |
| abstained | it says it doesn't know, as the prompt asks when the passages fall short |

**Checked that it can fail, three ways.**

- *A negative control.* The `mock` route answers by echoing the question. It scores 0.000 on every
  metric. Before the scorer discounted words the question already holds, it gave three echoes
  credit ("Which team owns billing-reconciler?" earned half of "The Revenue team" for "team").
- *Self-consistency.* Every gold answer scores correct against itself; a unit test holds both.
- *A hand review of every answer* ([`rag-local.review.jsonl`](answers/rag-local.review.jsonl)),
  each judged correct, terse (answers the question, omits a detail the gold gives), incomplete
  (misses a fact the question needs) or wrong. The scorer is strict by design — correct means
  *all* the gold facts — and agrees with the strict hand verdict on **99 of 102**. The three
  misses: a synonym it can't see ("exceeds 2%" for "above 2%"), and two answers it credited that
  say the opposite of the gold, once with a "not" ("the deploy pipeline will not fail") and once
  with a "you can" that its own clause contradicts. Lexical matching cannot read negation; that is
  what an LLM judge would add. On abstention it agrees on all 14.

Reading the answers also caught two of my labels being wrong. "Which paging tool sends the on-call
alerts?" was meant to be unanswerable, and the model answered it correctly: the corpus names
*Beacon*, which my absence terms (PagerDuty, Opsgenie) never looked for. And a per-diem question
turned out to be answered by a meal allowance that applies to every trip. Both were dropped, and
the saved run re-summarised with `--resummarise`; the counts below are after that.

## Results

`local` route, 102 answerable and 14 unanswerable questions, 0 errors. Scorer figures, with the
hand review beside them:

| Answerable (102) | Scorer | Hand review |
|---|---|---|
| Evidence among the passages given | **1.000** | |
| Correct (all gold facts) | 0.676 | 68 (0.667) |
| Answers the question, nothing false | | 79 (0.775): 68 correct + 11 terse |
| Incomplete | | 7 |
| Wrong | | 16 (0.157) |
| Cites a passage inline | **0.039** | |
| Contains a number or identifier found in no passage | 0.000 | |

| By kind, hand review | n | Correct + terse | Wrong |
|---|---|---|---|
| semantic (a paraphrase) | 55 | 49 (0.89) | 5 |
| lexical (names an identifier) | 29 | 21 (0.72) | 5 |
| terse (a bare identifier, e.g. `NG-1017`) | 18 | 9 (0.50) | 6 |

| Unanswerable (14) | |
|---|---|
| Abstained, correctly | **8** (0.571) |
| Fabricated an answer from nearby facts | 3 |
| Misapplied a nearby fact | 1 |
| Answered something else | 2 |

Latency per answer, two at a time on 4 CPU threads: p50 23.5 s, p95 48 s.

## Findings

**1. Every error is the model's, not retrieval's.** The evidence was among the five passages for
all 102 questions. Hybrid retrieval and reranking delivered; what the model did with the passages
is the whole story.

**2. It reads prose well and tables badly.** 89% of paraphrased questions get a usable answer, 72%
of questions naming an identifier, and half of the bare identifiers, a third of which are answered
*wrongly*. The wrong ones read another row of the same table: NG-1012's meaning for NG-1017,
NG-1029's for both NG-1026 and NG-1031, NG-2017 named for NG-2021, the alert with runbook RB-114
named for RB-121. Once it found the right row of NG-2008 and read out its remedy column instead of
its meaning. The right row was in context every time, among near-identical ones: the problem
hybrid retrieval solved for search, reappearing in generation.

**3. It almost never cites.** 4 answers in 102 carry a `[n]`, though the prompt asks for one on
every claim. Anything that verifies an answer through its citations cannot rely on this model.

**4. It doesn't invent numbers, but it borrows them.** No answer contained a number or identifier
absent from its passages. Yet it answered 6 of 14 unanswerable questions, 3 by assembling a claim
from nearby facts: "the probation period is six rotations" (new hires shadow *two* rotations),
"the maximum is 3 feature flags", "the Platform team owns the mobile app". The unsupported-specifics
check can't see these, because the borrowed number is in *some* passage. Abstention is the metric
that catches them, and it is right only 57% of the time.

**5. Slow, and the prompt is why.** 23.5 s per answer is mostly the ~1,000-token prompt of five
passages being read at ~130 tokens/s ([SELF_HOSTED.md](../loadtest/SELF_HOSTED.md)).

## What these numbers are for

They are a baseline the gateway can now be measured against, not a verdict on a 1.5B model. The
same run answers whether the research agent's critique-and-revise loop fixes the neighbouring-row
errors, whether a few-shot example makes the model cite, whether fewer passages trade accuracy for
latency, and how far a hosted model is ahead — each a re-run with one flag changed.

## Limitations

- **Lexical scoring can't read negation or synonyms.** It agreed with the hand review on 99/102,
  and the three misses were all of this kind. A stronger LLM judge would close the gap; it needs a
  provider key.
- **One model, one run, temperature 0.** Deterministic enough to compare configurations, not a
  measure of how the model varies with sampling.
- **The corpus is mine.** 13 short documents written for these evals, with the tables that make
  identifiers hard. A real document set would score differently.
