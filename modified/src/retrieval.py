"""Lexical evidence retrieval with BM25 (no embeddings).

A BM25 index over the paper's own **sentences** (a sentence is the unit of
evidence -- no chunker needed). Retrieval is used only to *select* candidate
sentences from the whole paper; the actual meaning/definition strings are then
extracted from those sentences by the grammar/structure code in
:mod:`.nlp_methods`. Nothing here generates text.

BM25 is a strong fit for this task because the evidence almost always shares an
exact term with the query -- the symbol name (``eta``), the equation number, or
the concept noun -- which is precisely what lexical matching rewards. It is also
deterministic, fast, and needs no GPU or downloaded model, which keeps the
pipeline easy to run and to justify.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _bm25_tokens(text: str) -> List[str]:
    """Math-aware tokeniser for BM25.

    Lower-cased alphanumeric word tokens, plus character bigrams of short
    notation-like tokens (``h0``, ``etad``) so that subscripted symbols and
    notation fragments still match even when the prose spells them slightly
    differently. Never returns an empty list (``BM25Okapi`` dislikes empty docs).
    """

    words = _TOKEN_RE.findall(text.lower())
    tokens: List[str] = list(words)
    for word in words:
        if len(word) <= 4 and not word.isalpha():
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens or ["__empty__"]


class BM25Index:
    """Thin wrapper over ``rank_bm25.BM25Okapi`` for a fixed sentence corpus."""

    def __init__(self, docs: Sequence[str]) -> None:
        self.docs = list(docs)
        self._bm25 = None
        if self.docs:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_bm25_tokens(doc) for doc in self.docs])

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of ``query`` against every document."""

        if self._bm25 is None:
            return np.zeros(0, dtype=float)
        return np.asarray(self._bm25.get_scores(_bm25_tokens(query)), dtype=float)


class BM25Retriever:
    """Rank a paper's sentences against a text query with BM25.

    The corpus is the paper's sentences. ``retrieve`` returns the highest-scoring
    sentence indices (optionally restricted to a subset, e.g. only sentences that
    mention a particular symbol), so a definition placed far from the equation is
    still found.
    """

    def __init__(self, sentences: Sequence[str]) -> None:
        self.sentences = list(sentences)
        self._bm25 = BM25Index(self.sentences)

    def retrieve(
        self, query: str, k: int = 8, allowed: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Return up to ``k`` ``(sentence_index, bm25_score)`` pairs."""

        if not self.sentences or not query.strip():
            return []
        scores = self._bm25.scores(query)
        if allowed is None:
            indices: List[int] = list(range(len(self.sentences)))
        else:
            indices = [i for i in allowed if 0 <= i < len(self.sentences)]
        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:k]]

    def sentences_matching(self, predicate: Callable[[str], bool]) -> List[int]:
        """Indices of corpus sentences for which ``predicate(sentence)`` is True."""

        return [i for i, sentence in enumerate(self.sentences) if predicate(sentence)]
