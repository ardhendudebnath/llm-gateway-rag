"""BM25 term weights: what gets indexed, and how the weights behave."""

import pytest

from app.rag.lexical import document_terms, query_terms, term_id, terms


def test_identifiers_are_indexed_whole_and_in_pieces():
    """So that both "EMBED_MAX_QUEUE" and "embed max queue" find the same row."""
    assert terms("EMBED_MAX_QUEUE") == ["embed_max_queue", "embed", "max", "queue"]
    assert terms("NG-1017") == ["ng-1017", "ng", "1017"]
    assert terms("invoice-mailer is tier 3") == [
        "invoice-mailer",
        "invoice",
        "mailer",
        "is",
        "tier",
        "3",
    ]


def test_punctuation_and_case_are_dropped():
    assert terms("Retention: 7 years (audit logs).") == [
        "retention",
        "7",
        "years",
        "audit",
        "logs",
    ]
    assert terms("!!! ???") == []


def test_term_ids_are_stable_and_fit_the_sparse_index():
    # A hash, not a vocabulary: nothing to build or migrate. It must be the same in every process,
    # which rules out Python's salted hash().
    assert term_id("ng-1017") == term_id("ng-1017")
    assert term_id("ng-1017") != term_id("ng-1018")
    assert 0 <= term_id("anything") <= 0x7FFFFFFF


def test_a_repeated_term_is_weighted_more_but_not_proportionally():
    once = dict(zip(*_pairs(document_terms("alpha beta gamma")), strict=True))
    thrice = dict(zip(*_pairs(document_terms("alpha alpha alpha beta gamma")), strict=True))
    alpha = term_id("alpha")

    assert thrice[alpha] > once[alpha], "more mentions, more weight"
    assert thrice[alpha] < 3 * once[alpha], "saturating, as BM25 does"


def test_a_term_in_a_short_chunk_outweighs_the_same_term_in_a_long_one():
    short = dict(zip(*_pairs(document_terms("quota exceeded")), strict=True))
    long = dict(zip(*_pairs(document_terms("quota exceeded " + "filler " * 200)), strict=True))

    assert short[term_id("quota")] > long[term_id("quota")]


def test_a_query_weights_every_distinct_term_once():
    sparse = query_terms("NG-1017 NG-1017 budget")

    assert set(sparse.values) == {1.0}, "the IDF is Qdrant's job, from its own statistics"
    assert len(sparse.indices) == len(set(sparse.indices))
    assert term_id("ng-1017") in sparse.indices and term_id("budget") in sparse.indices


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_text_with_no_terms_produces_an_empty_vector(text):
    assert not document_terms(text)
    assert not query_terms(text)


def _pairs(sparse):
    return sparse.indices, sparse.values
