"""The answer eval's scoring, and the unanswerable set it relies on. A scorer that credits an
echo of the question, or misses a made-up number, would make every answer number meaningless, so
these pin both directions. Runs in milliseconds, needs no models and no stack."""

import pytest

from app.rag.service import NO_CONTEXT_ANSWER
from eval.answer_eval import (
    Result,
    abstained,
    is_correct,
    load_unanswerable,
    score,
    specifics,
    unsupported,
)
from eval.retrieval_eval import load_corpus, load_questions

QUESTIONS = load_questions()
UNANSWERABLE = load_unanswerable()


def test_every_gold_answer_scores_correct_against_itself():
    wrong = [q.id for q in QUESTIONS if not is_correct(q.answer, q.answer, q.question)]
    assert wrong == []


def test_echoing_the_question_never_scores_correct():
    """What the mock route does. Before the scorer discounted words the question already holds,
    three echoes scored correct ("Which team owns billing-reconciler?" earned half of "The
    Revenue team" for the word "team")."""
    echoed = [
        q.id for q in QUESTIONS if is_correct(f"You said: {q.question}", q.answer, q.question)
    ]
    assert echoed == []


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("The engineering manager is paged [2].", "The engineering manager.", True),
        ("Mondays at 10:00 CET [1].", "Every Monday at 10:00 CET.", True),
        ("A stipend of 300 euros for each week [1].", "300 euros per week on call.", True),
        ("It is 250 euros a week.", "300 euros per week on call.", False),  # the wrong number
        ("Wait fifteen minutes, then page the manager.", "After 15 minutes.", True),
        ("Up to 10000 requests.", "10,000 requests.", True),
        ("The incident commander.", "The engineering manager.", False),
    ],
)
def test_correctness_needs_the_gold_facts(answer, gold, expected):
    assert is_correct(answer, gold) is expected


def test_specifics_ignore_citation_markers_and_list_numbering():
    assert specifics("Drain the queue [1].\n1. Then wait [2].\n2. Then restart.") == set()
    assert specifics("Error NG-1017 within a 15-minute window") == {
        "ng-1017",
        "1017",
        "15-minute",
        "15",
    }


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "The passages do not mention a parental leave policy.",
        "There is no information about per-diem rates.",
        "The context doesn't say which tool sends pages.",
        "This is not specified in the documents.",
        "I don't have enough information to determine the minimum password length.",
        NO_CONTEXT_ANSWER,
    ],
)
def test_abstentions_are_recognised(answer):
    assert abstained(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Don't deploy on Fridays [1].",
        "The engineering manager is paged [2].",
        "Not before 10:00 CET on Mondays [1].",
        "No, the incident commander coordinates and does not debug [3].",
    ],
)
def test_answers_are_not_mistaken_for_abstentions(answer):
    assert not abstained(answer)


def test_made_up_specifics_are_caught_and_cited_ones_are_not():
    passages = ["Drain the queue, then wait 90 seconds before restarting the worker."]
    question = "How long should I wait before restarting the billing worker?"
    assert unsupported("Wait 90 seconds [1].", question, passages) == []
    assert unsupported("Wait 120 seconds, per runbook RB-77 [1].", question, passages) == [
        "120",
        "77",
        "rb-77",
    ]


def test_score_separates_retrieval_from_citation():
    passages = [
        "Unrelated text.",
        "After 15 minutes without acknowledgement, the engineering manager is paged.",
    ]
    evidence = ["After 15 minutes without acknowledgement, the engineering manager is paged"]

    cited_wrong = Result("q", "semantic", "Who is paged next?", "The engineering manager.")
    cited_wrong.answer = "The engineering manager [1][7]."
    score(cited_wrong, passages, {1}, evidence)
    assert cited_wrong.retrieved and cited_wrong.correct and cited_wrong.cited
    assert not cited_wrong.cited_right  # [1] is the unrelated passage
    assert cited_wrong.invalid_citations == [7]  # there is no passage 7

    not_retrieved = Result("q", "semantic", "Who is paged next?", "The engineering manager.")
    not_retrieved.answer = "I don't know."
    score(not_retrieved, ["Unrelated text."], set(), evidence)
    assert not not_retrieved.retrieved and not not_retrieved.correct and not_retrieved.abstained


def test_the_unanswerable_set_is_unanswerable_from_the_corpus():
    """Each question names terms that would have to appear for the corpus to answer it. If a new
    document adds one, the question no longer belongs here and this says which."""
    corpus = {name: data.decode("utf-8").lower() for name, data in load_corpus().items()}
    leaks = {
        u.id: [(term, name) for term in u.absent for name, text in corpus.items() if term in text]
        for u in UNANSWERABLE
    }
    assert {qid: found for qid, found in leaks.items() if found} == {}


def test_the_unanswerable_set_is_distinct_and_does_not_repeat_a_labelled_question():
    ids = [u.id for u in UNANSWERABLE]
    assert len(ids) == len(set(ids)) >= 12
    labelled = {q.question for q in QUESTIONS}
    assert not labelled & {u.question for u in UNANSWERABLE}
