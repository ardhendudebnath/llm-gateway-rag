"""BM25 term weights for Qdrant's sparse vectors — the lexical half of hybrid retrieval.

Dense embeddings are good at paraphrases and bad at identifiers. Asked for `NG-1017` among six
hundred codes that all read alike, the nearest neighbours are the *neighbouring* codes: the eval
measures recall@5 dropping to 0.72 on queries that are nothing but an identifier, while a crude
lexical method beats it there. BM25 is exact where embeddings are approximate, so the two are fused
rather than chosen between (see ``QdrantChunkStore.hybrid_search``).

This is a deliberate 60 lines rather than another model download:

* **Qdrant applies the IDF**, server-side, from its own statistics (``Modifier.IDF``). Only term
  frequencies are sent, so nothing here has to track what the rest of the corpus contains.
* **Term ids are hashes**, so there is no vocabulary to build, persist or migrate. Collisions cost a
  little precision and are bounded by a 32-bit space; a vocabulary would cost a schema.
* **Identifiers are indexed whole and in pieces.** `EMBED_MAX_QUEUE` yields the whole token *and*
  `embed`, `max`, `queue`, so both "EMBED_MAX_QUEUE" and "embed max queue" find it. This is the one
  piece of tokenisation this domain genuinely needs.
* **The document length normaliser uses a fixed average**, not the true corpus average, which would
  make ingestion stateful and every chunk's weights depend on ingestion order. Chunks are capped at
  a known size, so the approximation is close and, more importantly, stable.

Stop words are left in: IDF gives them almost no weight anyway, and a hand-written list is one more
thing to be wrong about English.
"""

import re
from dataclasses import dataclass

# BM25's usual constants: k1 saturates term frequency, b controls length normalisation.
K1 = 1.2
B = 0.75
AVERAGE_LENGTH = 180.0  # chunk_max_words; see the module docstring
_TERM = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_PARTS = re.compile(r"[._-]")


def terms(text: str) -> list[str]:
    """Tokens of `text`: identifiers whole, plus their parts when they have any."""
    tokens: list[str] = []
    for match in _TERM.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        parts = [p for p in _PARTS.split(token) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def term_id(term: str) -> int:
    """A stable 32-bit id for a term. Stable across processes, unlike hash()."""
    from zlib import crc32

    return crc32(term.encode("utf-8")) & 0x7FFFFFFF


@dataclass(frozen=True)
class SparseTerms:
    indices: list[int]
    values: list[float]

    def __bool__(self) -> bool:
        return bool(self.indices)


def document_terms(text: str, average_length: float = AVERAGE_LENGTH) -> SparseTerms:
    """BM25 term frequencies for a stored chunk."""
    tokens = terms(text)
    if not tokens:
        return SparseTerms([], [])
    counts: dict[int, int] = {}
    for token in tokens:
        key = term_id(token)
        counts[key] = counts.get(key, 0) + 1
    norm = K1 * (1 - B + B * len(tokens) / average_length)
    return SparseTerms(
        indices=list(counts),
        values=[tf * (K1 + 1) / (tf + norm) for tf in counts.values()],
    )


def query_terms(text: str) -> SparseTerms:
    """One weight per distinct query term; Qdrant multiplies in the IDF."""
    unique = dict.fromkeys(term_id(token) for token in terms(text))
    return SparseTerms(indices=list(unique), values=[1.0] * len(unique))
