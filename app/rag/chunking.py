"""Chunking strategies. ``eval/retrieval_eval.py`` compares them on the labelled QA set.

Sizes are counted in whitespace-separated words, not model tokens: deterministic, dependency-free,
and close enough (English prose runs ~1.3 BPE tokens per word, so a 180-word chunk is ~240
tokens, well inside the 512-token window of the embedding and reranking models).

* ``fixed``      Sliding windows of N words with W words of overlap. The textbook baseline: cheap,
                 but it cuts sentences (and the facts inside them) in half at every boundary.
* ``sentence``   Packs whole sentences, list items and table rows up to N words, then starts the
                 next chunk with the trailing units that fit in W words. No fact is ever split
                 unless a single sentence is longer than N.
* ``structured`` ``sentence`` plus document structure: a Markdown heading always starts a new
                 chunk, and every chunk is prefixed with "<title> > <heading path>" so a chunk that
                 says "the threshold is 5 minutes" still carries *which* threshold it is about.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.rag.parsing import Section

ChunkStrategy = Literal["fixed", "sentence", "structured"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_STANDALONE_LINE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\|)")  # list items and table rows


@dataclass(frozen=True)
class Chunk:
    index: int  # position within the document, 0-based
    text: str  # what gets embedded, reranked and shown to the LLM
    page: int | None = None
    heading: str | None = None


class Chunker(Protocol):
    def chunk(self, sections: Sequence[Section], title: str | None = None) -> list[Chunk]: ...


def _words(text: str) -> int:
    return len(text.split())


class FixedWindowChunker:
    def __init__(self, max_words: int, overlap_words: int):
        if max_words <= 0 or not 0 <= overlap_words < max_words:
            raise ValueError("need max_words > 0 and 0 <= overlap_words < max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, sections: Sequence[Section], title: str | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.max_words - self.overlap_words
        for section in sections:
            words = section.text.split()
            for start in range(0, len(words), step):
                window = words[start : start + self.max_words]
                chunks.append(Chunk(len(chunks), " ".join(window), section.page))
                if start + self.max_words >= len(words):
                    break
        return chunks


@dataclass(frozen=True)
class _Unit:
    text: str
    heading: str | None
    starts_section: bool = False
    is_heading: bool = False


def _sentences(lines: list[str]) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(" ".join(lines)) if s.strip()]


def _split_units(text: str) -> list[_Unit]:
    """Split text into sentences / list items / table rows, tracking the Markdown heading path."""
    units: list[_Unit] = []
    stack: list[tuple[int, str]] = []  # (heading level, heading text)
    pending_section = False
    paragraph: list[str] = []

    def add(pieces: list[str], *, is_heading: bool = False) -> None:
        nonlocal pending_section
        heading_path = " > ".join(t for _, t in stack) or None
        for piece in pieces:
            units.append(_Unit(piece, heading_path, pending_section, is_heading))
            pending_section = False

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        heading = _HEADING.match(line)
        standalone = bool(line) and _STANDALONE_LINE.match(line) is not None
        if not line or heading or standalone:  # each of these ends the current paragraph
            add(_sentences(paragraph))
            paragraph.clear()
        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
            pending_section = True
            add([heading.group(2)], is_heading=True)
        elif standalone:
            add([line])
        elif line:
            paragraph.append(line)
    add(_sentences(paragraph))
    return units


class SentenceChunker:
    def __init__(self, max_words: int, overlap_words: int, *, structured: bool = False):
        if max_words <= 0 or not 0 <= overlap_words < max_words:
            raise ValueError("need max_words > 0 and 0 <= overlap_words < max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.structured = structured

    def chunk(self, sections: Sequence[Section], title: str | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            for heading, body in self._pack(_split_units(section.text)):
                prefix = _context_prefix(title, heading) if self.structured else ""
                text = f"{prefix}\n{body}" if prefix else body
                chunks.append(Chunk(len(chunks), text, section.page, heading))
        return chunks

    def _pack(self, units: list[_Unit]) -> list[tuple[str | None, str]]:
        out: list[tuple[str | None, str]] = []
        current: list[_Unit] = []
        fresh = 0  # units in `current` not already emitted as another chunk's overlap

        def emit() -> None:
            if fresh:
                heading = current[len(current) - fresh].heading
                out.append((heading, "\n".join(u.text for u in current)))

        for unit in units:
            size = _words(unit.text)
            if self.structured and unit.starts_section and current:
                emit()
                current, fresh = [], 0
            if self.structured and unit.is_heading:
                continue  # already in every chunk's "title > heading path" prefix
            if size > self.max_words:  # a single over-long sentence: hard-split it
                emit()
                words = unit.text.split()
                for start in range(0, len(words), self.max_words):
                    piece = " ".join(words[start : start + self.max_words])
                    out.append((unit.heading, piece))
                current, fresh = [], 0
                continue
            if current and sum(_words(u.text) for u in current) + size > self.max_words:
                emit()
                current = self._overlap(current, size)
                fresh = 0
            current.append(unit)
            fresh += 1
        emit()
        return out

    def _overlap(self, previous: list[_Unit], incoming: int) -> list[_Unit]:
        """Trailing units of the previous chunk that fit the overlap budget and the next chunk."""
        carried: list[_Unit] = []
        budget = min(self.overlap_words, self.max_words - incoming)
        for unit in reversed(previous):
            size = _words(unit.text)
            if size > budget:
                break
            carried.insert(0, unit)
            budget -= size
        return carried


def _context_prefix(title: str | None, heading: str | None) -> str:
    """ "<title> > <heading path>", without repeating a title that is also the top heading (H1)."""
    if title and heading and heading.split(" > ")[0] == title:
        title = None
    return " > ".join(p for p in (title, heading) if p)


def build_chunker(strategy: ChunkStrategy, max_words: int, overlap_words: int) -> Chunker:
    if strategy == "fixed":
        return FixedWindowChunker(max_words, overlap_words)
    return SentenceChunker(max_words, overlap_words, structured=strategy == "structured")
