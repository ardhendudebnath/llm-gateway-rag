"""Keep the retrieval eval honest: labels must match the corpus, and the harness must run."""

from eval.retrieval_eval import (
    EvalConfig,
    ModelCache,
    load_corpus,
    load_questions,
    normalise,
    run_config,
    score,
)


def test_question_set_is_well_formed():
    questions = load_questions()
    corpus = load_corpus()
    assert 30 <= len(questions) <= 50  # the spec asks for 30-50 labelled questions
    assert len({q.id for q in questions}) == len(questions)
    assert {q.doc for q in questions} <= set(corpus)


def test_every_evidence_span_exists_verbatim_in_the_corpus():
    corpus_text = [normalise(d.decode("utf-8")) for d in load_corpus().values()]
    missing = [
        (q.id, span)
        for q in load_questions()
        for span in q.evidence
        if not any(normalise(span) in doc for doc in corpus_text)
    ]
    assert missing == []


def test_scoring():
    passages = ["noise", "The DEADLINE is\nfive days", "more noise"]
    assert score(passages, ["deadline is five days"], k=3) == {
        "precision": 1 / 3,
        "recall": 1.0,
        "mrr": 0.5,
        "hit1": 0.0,
    }
    assert score(passages, ["deadline is five days", "absent"], k=3)["recall"] == 0.5
    assert score(passages, ["absent"], k=3)["mrr"] == 0.0


async def test_harness_runs_end_to_end_on_the_lexical_baseline():
    questions = load_questions()[:5]
    result = await run_config(
        EvalConfig("structured", 180, 40, "hash"), load_corpus(), questions, ModelCache(), k=5
    )
    assert result["chunks"] > 0
    for metric in ("precision@5", "recall@5", "mrr", "hit@1"):
        assert 0.0 <= result[metric] <= 1.0
