"""
src/meaning.py

Extractive equation meaning from surrounding paragraph text.

DESIGN NOTE — why IDF must be corpus-wide
------------------------------------------
TF-IDF distinguishes informative sentences from boilerplate by comparing
term frequencies against how rare each word is *across the whole corpus*.
If IDF is computed over only the local context window (2-3 sentences), every
word appears roughly once, all IDF values are equal, and the ranking reduces
to plain word count.  Corpus-wide IDF tells us that "Hamiltonian" appears in
most physics papers (low IDF, down-weighted) while "superradiant" appears in
few (high IDF, up-weighted), so a sentence containing the latter is more
likely to be the specific description of *this* equation.

DESIGN NOTE — extraction only (no generation)
----------------------------------------------
Every string written to the meaning field is a verbatim sentence taken from
the paper's own text.  The audit trail records exactly which sentence was
chosen and which strategy selected it.  If no suitable sentence is found the
field is left empty — never filled with model-generated text.

Pipeline usage:
  1. extractor = MeaningExtractor()
  2. extractor.fit_corpus(all_context_texts)   # once, before any extract call
  3. meaning = extractor.extract(eq_num, latex, ctx_before, ctx_after, audit)
"""

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── spaCy optional import ─────────────────────────────────────────────────────
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.info("spaCy not available — gazetteer will use plain string search")

# ── Named-equation gazetteer ──────────────────────────────────────────────────
# Phrases that identify well-known equations by name.  When one of these
# appears in the context window the matching sentence is used directly.
_GAZETTEER: List[str] = [
    "Schrödinger equation", "Schrodinger equation",
    "Hamiltonian", "Lindblad equation", "Lindblad master equation",
    "master equation", "von Neumann equation",
    "Fokker-Planck equation", "Bloch equation",
    "Maxwell equation", "Maxwell's equation",
    "Dirac equation", "Heisenberg equation",
    "Liouville equation", "Liouville-von Neumann equation",
    "density matrix", "density operator",
    "partition function", "Green's function", "Green function",
    "Wigner function", "Husimi function",
    "Jaynes-Cummings", "Rabi model",
    "tight-binding", "Bogoliubov",
]
_GAZETTEER_LOWER = [p.lower() for p in _GAZETTEER]

# ── TF-IDF stop words ─────────────────────────────────────────────────────────
_STOP: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "we", "our", "us", "as", "if", "when", "where",
    "which", "who", "whom", "what", "how", "not", "no", "also", "both",
    "each", "all", "any", "such", "given", "here", "thus", "hence",
    "equation", "eq", "fig", "figure", "table", "section", "note",
    "above", "below", "following", "shown", "see",
})

# Sentence splitter: split on ./?/! followed by whitespace + capital letter.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\(])')

# Pattern to detect a citation of equation number N in plain text.
# Matches "(3)", "Eq. (3)", "equation (3)", "(A.2)", etc.
_EQ_CITE_RE_TMPL = r'(?:eq(?:uation)?s?\.?\s*)?\(\s*{n}\s*\)'


class MeaningExtractor:
    """Extract a one-sentence meaning for each equation from surrounding text.

    Must be fitted on the whole corpus before use (see :meth:`fit_corpus`).

    Parameters
    ----------
    None

    Examples
    --------
    >>> me = MeaningExtractor()
    >>> me.fit_corpus(all_contexts)
    >>> meaning = me.extract("1", r"E=mc^2", ctx_before, ctx_after, audit)
    """

    def __init__(self) -> None:
        self._idf: Dict[str, float] = {}
        self._fitted: bool = False
        self._nlp = None
        self._matcher = None

    # ── corpus fitting ────────────────────────────────────────────────────────

    def fit_corpus(self, texts: List[str]) -> None:
        """Build corpus-wide IDF from all collected context windows.

        Must be called once with ALL context strings from ALL equations
        before any :meth:`extract` call.  Calling it again rebuilds IDF
        from scratch.

        Parameters
        ----------
        texts : list of str
            Every ``context_before`` and ``context_after`` string collected
            from the full equation set.  Duplicates are fine — they add
            document-frequency weight as expected.
        """
        n_docs = len(texts)
        if n_docs == 0:
            logger.warning("fit_corpus called with empty text list — IDF will be empty")
            return

        df: Counter = Counter()
        for text in texts:
            words = set(self._tokenize(text))
            df.update(words)

        # smoothed IDF: log((N+1)/(df+1)) + 1
        self._idf = {
            word: math.log((n_docs + 1) / (count + 1)) + 1.0
            for word, count in df.items()
        }
        self._fitted = True
        logger.info(
            "MeaningExtractor: fitted IDF on %d context texts, vocab size=%d",
            n_docs, len(self._idf),
        )

        # build spaCy PhraseMatcher if available
        if _SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
                patterns = [self._nlp.make_doc(p) for p in _GAZETTEER]
                self._matcher.add("NAMED_EQ", patterns)
                logger.info("MeaningExtractor: spaCy PhraseMatcher ready")
            except OSError:
                logger.warning(
                    "spaCy model en_core_web_sm not found — "
                    "falling back to plain-string gazetteer"
                )
                self._nlp = None
                self._matcher = None

    # ── public extraction ─────────────────────────────────────────────────────

    def extract(
        self,
        eq_num: str,
        latex: str,
        context_before: str,
        context_after: str,
        audit,
    ) -> str:
        """Return the best one-sentence meaning for an equation.

        Strategy (tried in order):
          1. Sentence in the context window that cites this equation's number.
          2. Sentence containing a named-equation phrase from the gazetteer.
          3. Highest TF-IDF-scored sentence in the context window.

        Parameters
        ----------
        eq_num : str
            The equation number label (e.g. ``"3"`` for equation (3)).
        latex : str
            The LaTeX source of the equation (used to build TF-IDF query).
        context_before : str
            Text immediately before the equation.
        context_after : str
            Text immediately after the equation.
        audit : AuditTrail
            Receives one entry describing which strategy matched.

        Returns
        -------
        str
            A verbatim sentence from the paper, or ``""`` if nothing found.
        """
        if not self._fitted:
            audit.log("meaning", "MeaningExtractor not fitted — call fit_corpus first")
            return ""

        sentences = self._collect_sentences(context_before, context_after)
        if not sentences:
            audit.log("meaning", f"eq ({eq_num}): no context sentences available")
            return ""

        full_context = (context_before + " " + context_after).strip()

        # ── strategy 1: sentence that cites this equation number ──────────────
        cite_result = self._find_citation_sentence(eq_num, sentences)
        if cite_result:
            audit.log(
                "meaning",
                f"eq ({eq_num}): strategy=citation_match, "
                f"sentence={cite_result[:80]!r}",
            )
            return cite_result

        # ── strategy 2: gazetteer match ───────────────────────────────────────
        gaz_result = self._find_gazetteer_sentence(sentences)
        if gaz_result:
            audit.log(
                "meaning",
                f"eq ({eq_num}): strategy=gazetteer_match, "
                f"sentence={gaz_result[:80]!r}",
            )
            return gaz_result

        # ── strategy 3: TF-IDF ranking ────────────────────────────────────────
        ranked = self._rank_by_tfidf(sentences)
        if ranked:
            best_score, best_sent = ranked[0]
            audit.log(
                "meaning",
                f"eq ({eq_num}): strategy=tfidf_rank, score={best_score:.3f}, "
                f"sentence={best_sent[:80]!r}",
            )
            return best_sent

        audit.log("meaning", f"eq ({eq_num}): no meaning sentence found")
        return ""

    # ── internal helpers ──────────────────────────────────────────────────────

    def _collect_sentences(
        self, context_before: str, context_after: str
    ) -> List[str]:
        """Split both context strings into sentences and merge."""
        sentences: List[str] = []
        for text in (context_before, context_after):
            if text:
                sentences.extend(self._split_sentences(text))
        # deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on ./?/! + whitespace + capital."""
        parts = _SENT_SPLIT_RE.split(text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 10]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase alphabetic tokens, stop-words removed."""
        return [
            w for w in re.findall(r'[a-z]+', text.lower())
            if w not in _STOP and len(w) > 1
        ]

    def _find_citation_sentence(
        self, eq_num: str, sentences: List[str]
    ) -> Optional[str]:
        """Return first sentence that contains a citation of eq_num."""
        pattern = re.compile(
            _EQ_CITE_RE_TMPL.format(n=re.escape(eq_num)),
            re.IGNORECASE,
        )
        for sent in sentences:
            if pattern.search(sent):
                return sent
        return None

    def _find_gazetteer_sentence(self, sentences: List[str]) -> Optional[str]:
        """Return first sentence matching a named-equation phrase."""
        if self._matcher is not None and self._nlp is not None:
            # spaCy PhraseMatcher path
            for sent in sentences:
                doc = self._nlp(sent)
                matches = self._matcher(doc)
                if matches:
                    return sent
        else:
            # plain string fallback
            for sent in sentences:
                sent_lower = sent.lower()
                for phrase in _GAZETTEER_LOWER:
                    if phrase in sent_lower:
                        return sent
        return None

    def _rank_by_tfidf(
        self, sentences: List[str]
    ) -> List[Tuple[float, str]]:
        """Score sentences by mean IDF of their non-stop words."""
        scored: List[Tuple[float, str]] = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            if not tokens:
                continue
            score = sum(self._idf.get(t, 0.0) for t in tokens) / len(tokens)
            scored.append((score, sent))
        return sorted(scored, key=lambda x: x[0], reverse=True)
