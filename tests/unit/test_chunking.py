import pytest

from app.rag.chunking import FixedWindowChunker, SentenceChunker, build_chunker
from app.rag.parsing import Section

DOC = """# On-call

## Rotation
Engineers rotate weekly. The handover happens every Monday at 10:00. Each shift has a primary.

## Escalation
- If the primary does not acknowledge within 5 minutes, the secondary is paged.
- After 15 minutes the engineering manager is paged.
"""


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_fixed_windows_cover_every_word_with_the_requested_overlap():
    chunks = FixedWindowChunker(max_words=10, overlap_words=3).chunk([Section(words(25))])
    texts = [c.text.split() for c in chunks]
    assert [len(t) for t in texts] == [10, 10, 10, 4]
    assert texts[1][:3] == texts[0][-3:]  # overlap
    assert {w for t in texts for w in t} == set(words(25).split())
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_fixed_window_without_overlap_has_disjoint_chunks():
    chunks = FixedWindowChunker(max_words=10, overlap_words=0).chunk([Section(words(20))])
    assert [c.text for c in chunks] == [words(10), words(10, start=10)]


@pytest.mark.parametrize(("max_words", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_are_rejected(max_words, overlap):
    with pytest.raises(ValueError):
        FixedWindowChunker(max_words, overlap)
    with pytest.raises(ValueError):
        SentenceChunker(max_words, overlap)


def test_sentence_chunker_never_splits_a_sentence():
    text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
    chunks = SentenceChunker(max_words=9, overlap_words=0).chunk([Section(text)])
    assert [c.text for c in chunks] == [
        "Alpha beta gamma delta.\nEpsilon zeta eta theta.",
        "Iota kappa lambda mu.",
    ]


def test_sentence_chunker_overlaps_by_whole_sentences():
    text = "One two three. Four five six. Seven eight nine."
    chunks = SentenceChunker(max_words=6, overlap_words=3).chunk([Section(text)])
    assert [c.text for c in chunks] == [
        "One two three.\nFour five six.",
        "Four five six.\nSeven eight nine.",
    ]


def test_trailing_overlap_alone_is_not_emitted_as_a_chunk():
    chunks = SentenceChunker(max_words=6, overlap_words=3).chunk([Section("One two three.")])
    assert [c.text for c in chunks] == ["One two three."]


def test_an_over_long_sentence_is_hard_split():
    chunks = SentenceChunker(max_words=4, overlap_words=0).chunk([Section(words(10) + ".")])
    assert [len(c.text.split()) for c in chunks] == [4, 4, 2]


def test_sentence_chunker_keeps_headings_as_text_and_tracks_the_path():
    chunks = SentenceChunker(max_words=40, overlap_words=0).chunk([Section(DOC)])
    assert chunks[0].text.startswith("On-call\nRotation\nEngineers rotate weekly.")
    assert "Escalation" in chunks[0].text  # no section breaks: packed together
    assert chunks[0].heading == "On-call"


def test_structured_chunker_breaks_at_headings_and_prefixes_context():
    chunks = build_chunker("structured", 40, 0).chunk([Section(DOC)], title="Handbook")
    assert [c.heading for c in chunks] == ["On-call > Rotation", "On-call > Escalation"]
    assert chunks[1].text == (
        "Handbook > On-call > Escalation\n"
        "- If the primary does not acknowledge within 5 minutes, the secondary is paged.\n"
        "- After 15 minutes the engineering manager is paged."
    )


def test_pages_and_indices_carry_across_sections():
    sections = [Section("Page one text.", page=1), Section("Page two text.", page=2)]
    chunks = build_chunker("sentence", 50, 10).chunk(sections)
    assert [(c.index, c.page) for c in chunks] == [(0, 1), (1, 2)]


def test_empty_input_gives_no_chunks():
    assert build_chunker("fixed", 10, 2).chunk([Section("")]) == []
    assert build_chunker("structured", 10, 2).chunk([Section("   ")]) == []
