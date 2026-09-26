"""The labelled retrieval set has to stay usable: a span that is in no document scores zero
forever, and nothing else would notice. Runs in milliseconds, needs no models."""

import pytest

from eval.retrieval_eval import CORPUS_DIR, load_corpus, load_questions, normalise

QUESTIONS = load_questions()


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    return {name: normalise(data.decode("utf-8")) for name, data in load_corpus().items()}


def test_the_set_is_not_empty_and_ids_are_unique():
    assert len(QUESTIONS) > 50
    ids = [q.id for q in QUESTIONS]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.id)
def test_every_evidence_span_appears_in_the_corpus(question, corpus):
    for span in question.evidence:
        holders = [name for name, text in corpus.items() if normalise(span) in text]
        assert holders, f"{question.id}: no document contains {span!r}"


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.id)
def test_every_question_names_a_document_that_exists(question):
    assert (CORPUS_DIR / question.doc).is_file()


def test_every_question_kind_is_represented():
    # Three ways a user asks, which fail for different reasons: a paraphrase (dense retrieval's
    # strength), a sentence naming an identifier, and an identifier pasted in on its own.
    assert {q.kind for q in QUESTIONS} == {"semantic", "lexical", "terse"}
