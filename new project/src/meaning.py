"""
src/meaning.py

Extractive equation meaning from surrounding text and BM25-retrieved evidence.

Pipeline (Stage 3 — per equation):
  1. Retrieve top-k chunks from the paper-level BM25 index using a deterministic
     query built from the equation number and LaTeX symbol names.
  2. Split every retrieved chunk and both context-window strings into sentences.
  3. Apply _is_prose() to discard math-heavy fragments.
  4. Score each surviving sentence with the weighted signal table.
  5. Return the highest-scoring sentence verbatim if its normalised score ≥
     SCORE_THRESHOLD; otherwise return "".

Weighted signals (sum; divide by MAX_SIGNAL_SCORE to normalise):
  +5  Sentence explicitly cites this equation number (Eq. (N), equation (N), …)
  +3  Contains a definitional cue word (defines/describes/represents/gives)
  +3  Matches a named-equation gazetteer entry (Hamiltonian, Schrödinger, …)
  +2  Chunk is in the paragraph immediately before or after the equation
  +1  Chunk is in the same section as the equation
  +1  Chunk is the top-1 BM25-scored result for this query
  -1  Sentence is too short (< MIN_MEANING_LEN) or too long (> MAX_MEANING_LEN)

Context-window sentences receive the +2 adjacent-paragraph and +1 same-section
bonuses by default, because they are drawn from the equation's immediate prose
neighbourhood.

No text generation.  Every string written to the meaning field is a verbatim
sentence from the source paper.  The audit trail records the equation number,
BM25 score, chunk location, character span, active signals, and the chosen
sentence.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
_GAZETTEER_LOWER: List[str] = [p.lower() for p in _GAZETTEER]

# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\(])')

# ── Context cleaning patterns ─────────────────────────────────────────────────
# Layer 1: LaTeXML HTML artifact tokens
_ARTIFACT_RE = re.compile(
    r'\bitalic[-_]\S+'        # italic_x and italic-x variants
    r'|\bstart_[A-Z]\w*'
    r'|\bend_[A-Z]\w*'
    r'|\bover[a-z]*_ARG\b'
    r'|\broad_[A-Za-z_]+'
    r'|\bcaligraphic_\S+'     # caligraphic_W, caligraphic_A, etc.
    r'|\bbold_[a-z]\S*'       # bold_italic_P etc.
)
# Layer 2: Unicode mathematical alphanumeric chars (U+1D400–U+1D7FF)
_MATH_UNICODE_RE = re.compile(r'[\U0001D400-\U0001D7FF⁡-⁤]')
# Layer 3: LaTeX subscript/superscript brace notation that appears literally
#   in some LaTeXML HTML text nodes, e.g. "n_{0}" or "H^{+}"
_LATEX_SCRIPT_RE = re.compile(r'[_^]\{[^{}]*\}')
# Layer 4: Bare LaTeX commands (with or without non-nested brace arg)
#   e.g. \rho_{t} → \rho already gone from layer 3 fix; bare \mathcal{A}
_RAW_LATEX_RE = re.compile(r'\\[a-zA-Z]+(?:\{[^{}]*\})*')
# Layer 5: Bare subscript/superscript words emitted by some LaTeXML versions
_SCRIPT_WORD_RE = re.compile(
    r'\b(?:subscript|superscript|postsubscript|postsuperscript)\b',
    re.IGNORECASE,
)
# Layer 6: Orphan braces left after partial LaTeX stripping (e.g. from \ket{})
_ORPHAN_BRACE_RE = re.compile(r'[{}]')

# ── Equation citation pattern ─────────────────────────────────────────────────
_EQ_CITE_RE_TMPL = r'(?:eq(?:uation)?s?\.?\s*)?\(\s*{n}\s*\)'

# ── Definitional cue pattern ──────────────────────────────────────────────────
_DEF_CUE_RE = re.compile(
    r'\b(?:define[sd]?|describes?|represents?|gives?|'
    r'denotes?|is\s+defined\s+(?:as|by))\b',
    re.IGNORECASE,
)

# ── Stop words (for tokeniser used in query construction only) ────────────────
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
    "italic", "start", "end", "postsubscript", "postsuperscript",
    "arg", "over", "displaystyle", "rm", "bf", "operatorname",
})

# ── Weighted scoring constants ────────────────────────────────────────────────
_W_EQ_NUM       = 5    # sentence explicitly cites this equation number
_W_DEFINITIONAL = 3    # definitional cue word present
_W_GAZETTEER    = 3    # named-equation gazetteer match
_W_PARA_ADJ     = 2    # chunk in immediately adjacent paragraph
_W_SECTION      = 1    # chunk in same section as equation
_W_BM25_TOP1    = 1    # chunk is the top-1 BM25 result
_W_BADLEN       = -1   # sentence is too short or too long

MAX_SIGNAL_SCORE: float = float(
    _W_EQ_NUM + _W_DEFINITIONAL + _W_GAZETTEER + _W_PARA_ADJ + _W_SECTION + _W_BM25_TOP1
)   # 15.0

SCORE_THRESHOLD: float = 0.15   # normalised; ≥ 2 raw points required

MIN_MEANING_LEN: int = 30    # chars — shorter sentences are likely fragments
MAX_MEANING_LEN: int = 400   # chars — longer sentences are likely multi-eq blocks
RETRIEVAL_TOP_K: int = 12    # chunks retrieved per equation query

# ── Paragraph adjacency helper ────────────────────────────────────────────────
_PARA_NUM_RE = re.compile(r'^(.+\.)p(\d+)$')


def _is_adjacent_para(eq_para_id: str, cand_para_id: str) -> bool:
    """Return True if *cand_para_id* is the paragraph immediately before or after
    *eq_para_id* (same section prefix, numeric suffix differs by exactly 1)."""
    if not eq_para_id or not cand_para_id or eq_para_id == cand_para_id:
        return False
    m_eq   = _PARA_NUM_RE.match(eq_para_id)
    m_cand = _PARA_NUM_RE.match(cand_para_id)
    if m_eq and m_cand and m_eq.group(1) == m_cand.group(1):
        return abs(int(m_cand.group(2)) - int(m_eq.group(2))) == 1
    return False


def _is_prose(text: str) -> bool:
    """Return True if *text* contains enough alphabetic content to be prose.

    Rejects strings that are mostly single-character math tokens or symbols
    (e.g. "H = E α ψ + β σ" would fail; "the total Hamiltonian H is …" passes).
    Requires at least 3 whitespace-separated tokens and at least 40 % of those
    tokens to carry two or more consecutive alphabetic characters.
    """
    tokens = re.findall(r'\S+', text)
    if len(tokens) < 3:
        return False
    prose = sum(1 for t in tokens if len(re.sub(r'[^a-zA-Z]', '', t)) >= 2)
    return prose / len(tokens) >= 0.40


class MeaningExtractor:
    """Extract a one-sentence meaning for each equation from surrounding text.

    Must be fitted once before use (see :meth:`fit_corpus`).

    Approach: retrieve → score → threshold.  All candidate sentences are scored
    with a fixed weighted table; the highest-scoring sentence is returned
    verbatim if its normalised score meets SCORE_THRESHOLD.  No LLM or
    generative model is involved at any stage.

    Examples
    --------
    >>> me = MeaningExtractor()
    >>> me.fit_corpus(all_context_texts)
    >>> meaning = me.extract("3", r"H = E \\psi", ctx_b, ctx_a, audit,
    ...                      retriever=bm25, eq_section_id="S2", eq_para_id="S2.p3")
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._nlp   = None
        self._matcher = None

    # ── corpus fitting ─────────────────────────────────────────────────────────

    def fit_corpus(self, texts: List[str]) -> None:
        """Initialise the spaCy PhraseMatcher for the gazetteer signal.

        Must be called once with ALL context strings from ALL equations before
        any :meth:`extract` call.  The *texts* argument is accepted for
        backward compatibility but corpus-wide IDF is no longer computed;
        only the gazetteer PhraseMatcher is built here.

        Parameters
        ----------
        texts : list of str
            Context texts from the full equation set (may be empty).
        """
        self._fitted = True

        if _SPACY_AVAILABLE:
            try:
                self._nlp     = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
                patterns      = [self._nlp.make_doc(p) for p in _GAZETTEER]
                self._matcher.add("NAMED_EQ", patterns)
                logger.info("MeaningExtractor: spaCy PhraseMatcher ready")
            except OSError:
                logger.warning(
                    "spaCy model en_core_web_sm not found — "
                    "falling back to plain-string gazetteer"
                )
                self._nlp     = None
                self._matcher = None

        logger.info(
            "MeaningExtractor.fit_corpus: ready (gazetteer=%d entries, "
            "texts_received=%d)",
            len(_GAZETTEER), len(texts),
        )

    # ── public extraction ──────────────────────────────────────────────────────

    def extract(
        self,
        eq_num: str,
        latex: str,
        context_before: str,
        context_after: str,
        audit: Any,
        retriever: Any = None,
        eq_section_id: str = "",
        eq_para_id: str = "",
        used_sentences: Optional[Set[str]] = None,
    ) -> str:
        """Return the best one-sentence meaning for an equation.

        Retrieves up to RETRIEVAL_TOP_K chunks from the paper BM25 index,
        then scores every sentence (from retrieved chunks and from the local
        context windows) with the weighted signal table.  The highest-scoring
        prose sentence is returned verbatim if its normalised score meets
        SCORE_THRESHOLD.

        Parameters
        ----------
        eq_num : str
            Equation number label (e.g. ``"3"`` for equation (3)).
        latex : str
            LaTeX source — used to build the BM25 query.
        context_before : str
            Text immediately before the equation.
        context_after : str
            Text immediately after the equation.
        audit : AuditTrail
            Receives entries describing the chosen sentence and signals.
        retriever : BM25Retriever or None, optional
            Paper-level BM25 index.  When ``None``, only context windows
            are searched.
        eq_section_id : str, optional
            Section ID of the equation's paragraph (for same-section signal).
        eq_para_id : str, optional
            Paragraph ID of the equation (for adjacent-paragraph signal).

        Returns
        -------
        str
            A verbatim sentence from the paper, or ``""`` if nothing passes
            the threshold.
        """
        if not self._fitted:
            audit.log("meaning", f"eq=({eq_num}): extractor not fitted")
            return ""

        # ── step 1: retrieve candidates via BM25 ─────────────────────────────
        retrieved_hits: List[Dict] = []
        if retriever is not None:
            latex_terms = " ".join(self._tokenize(latex))
            query       = f"equation {eq_num} eq {eq_num} {latex_terms}".strip()
            retrieved_hits = retriever.search(query, top_k=RETRIEVAL_TOP_K)
            audit.log(
                "retrieval",
                f"eq=({eq_num}): BM25 query={query[:60]!r} hits={len(retrieved_hits)}",
            )

        top1_chunk_id: str = retrieved_hits[0]["chunk_id"] if retrieved_hits else ""

        # ── step 2: build candidate pool ──────────────────────────────────────
        # Each candidate is a dict with text + metadata.
        candidates: List[Dict] = []

        # From retrieved chunks — split each chunk into sentences.
        for hit in retrieved_hits:
            chunk_id    = hit["chunk_id"]
            bm25_score  = hit["score"]
            chunk_sec   = hit.get("section_id",  "")
            chunk_para  = hit.get("paragraph_id", "")
            base_offset = hit.get("char_offset",  0)
            chunk_text  = self._clean_context(hit["text"])

            for sent, local_off in self._split_sentences_with_offset(chunk_text):
                if _is_prose(sent):
                    candidates.append({
                        "text":         sent,
                        "section_id":   chunk_sec,
                        "paragraph_id": chunk_para,
                        "chunk_id":     chunk_id,
                        "char_offset":  base_offset + local_off,
                        "bm25_score":   bm25_score,
                        "from_top1":    chunk_id == top1_chunk_id,
                        "from_context": False,
                    })

        # From context windows — treated as adjacent-paragraph prose.
        clean_before = self._clean_context(context_before)
        clean_after  = self._clean_context(context_after)
        for ctext in (clean_after, clean_before):
            for sent, local_off in self._split_sentences_with_offset(ctext):
                if _is_prose(sent):
                    candidates.append({
                        "text":         sent,
                        "section_id":   eq_section_id,   # same section assumed
                        "paragraph_id": "",
                        "chunk_id":     "context_window",
                        "char_offset":  local_off,
                        "bm25_score":   0.0,
                        "from_top1":    False,
                        "from_context": True,
                    })

        # Deduplicate by text (preserve order: retrieved first).
        seen: set = set()
        unique: List[Dict] = []
        for c in candidates:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique.append(c)
        candidates = unique

        if not candidates:
            audit.log("meaning", f"eq=({eq_num}): no prose candidates found")
            return ""

        # ── step 3: score each candidate ──────────────────────────────────────
        eq_cite_re = re.compile(
            _EQ_CITE_RE_TMPL.format(n=re.escape(eq_num)),
            re.IGNORECASE,
        )

        scored: List[Tuple[int, float, Dict, List[str]]] = []
        for c in candidates:
            text    = c["text"]
            score   = 0
            signals: List[str] = []

            # +5: sentence explicitly cites this equation number
            if (eq_cite_re.search(text)
                    and len(text) > 30
                    and not text.startswith('(')):
                score += _W_EQ_NUM
                signals.append("eq_num")

            # +3: definitional cue word
            if _DEF_CUE_RE.search(text):
                score += _W_DEFINITIONAL
                signals.append("definitional")

            # +3: named-equation gazetteer match
            if self._matches_gazetteer(text):
                score += _W_GAZETTEER
                signals.append("gazetteer")

            # +2: adjacent paragraph (structural or context-window)
            if c["from_context"]:
                score += _W_PARA_ADJ
                signals.append("context_window")
            elif _is_adjacent_para(eq_para_id, c["paragraph_id"]):
                score += _W_PARA_ADJ
                signals.append("para_adj")

            # +1: same section
            if eq_section_id and c["section_id"] == eq_section_id:
                score += _W_SECTION
                signals.append("same_section")

            # +1: top-1 BM25 result
            if c["from_top1"]:
                score += _W_BM25_TOP1
                signals.append("bm25_top1")

            # -1: too short or too long
            if len(text) < MIN_MEANING_LEN or len(text) > MAX_MEANING_LEN:
                score += _W_BADLEN
                signals.append("badlen")

            scored.append((score, c["bm25_score"], c, signals))

        # Sort: primary = raw score (desc); secondary = BM25 score (desc).
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # ── step 4: threshold, dedup, and return ──────────────────────────────
        # Walk candidates best-first; skip already-used sentences (per-paper
        # dedup) and below-threshold entries.
        for best_score, best_bm25, best_cand, best_signals in scored:
            normalised = best_score / MAX_SIGNAL_SCORE
            if normalised < SCORE_THRESHOLD:
                audit.log(
                    "meaning",
                    f"eq=({eq_num}): strategy=weighted "
                    f"score={best_score}/{int(MAX_SIGNAL_SCORE)} "
                    f"({normalised:.2f}) below threshold={SCORE_THRESHOLD:.2f}",
                )
                return ""  # remaining candidates are only worse
            if used_sentences is not None and best_cand["text"] in used_sentences:
                continue   # skip duplicate; try next-best candidate
            break
        else:
            return ""

        if used_sentences is not None:
            used_sentences.add(best_cand["text"])

        char_start = best_cand["char_offset"]
        char_end   = char_start + len(best_cand["text"])
        audit.log(
            "meaning",
            f"eq=({eq_num}): strategy=weighted "
            f"bm25={best_bm25:.1f} "
            f"score={best_score}/{int(MAX_SIGNAL_SCORE)} "
            f"chunk={best_cand['chunk_id']} "
            f"char_span=[{char_start},{char_end}] "
            f"signals={best_signals} "
            f"sentence={best_cand['text'][:80]!r}",
        )
        return best_cand["text"]

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_context(text: str) -> str:
        """Strip LaTeXML artifacts, Unicode math chars, and raw LaTeX fragments.

        Six layers applied in order so each pass works on progressively
        cleaner text without cascading double-spaces.
        """
        text = text.replace('\xa0', ' ')          # non-breaking space
        text = _ARTIFACT_RE.sub('', text)         # italic_x, start_ROW, …
        text = _MATH_UNICODE_RE.sub('', text)     # U+1D400–U+1D7FF
        text = _LATEX_SCRIPT_RE.sub('', text)     # _{t}, ^{2}, etc.
        text = _RAW_LATEX_RE.sub('', text)        # \rho, \mathcal{A}, etc.
        text = _SCRIPT_WORD_RE.sub('', text)      # bare "subscript" words
        text = _ORPHAN_BRACE_RE.sub('', text)     # stray { } after stripping
        text = re.sub(r'  +', ' ', text).strip()
        return text

    @staticmethod
    def _split_sentences_with_offset(text: str) -> List[Tuple[str, int]]:
        """Split *text* into (sentence, char_offset) pairs.

        Uses the same sentence-boundary regex as the old strategy code.
        Sentences shorter than 10 chars are discarded.
        """
        if not text:
            return []
        parts: List[Tuple[str, int]] = []
        pos = 0
        for m in _SENT_SPLIT_RE.finditer(text):
            fragment = text[pos:m.start()].strip()
            if len(fragment) > 10:
                parts.append((fragment, pos))
            pos = m.end()
        tail = text[pos:].strip()
        if len(tail) > 10:
            parts.append((tail, pos))
        return parts

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase alphabetic tokens with stop words removed (for BM25 query)."""
        return [
            w for w in re.findall(r'[a-z]+', text.lower())
            if w not in _STOP and len(w) > 1
        ]

    def _matches_gazetteer(self, sent: str) -> bool:
        """Return True if *sent* contains a named-equation gazetteer phrase."""
        if self._matcher is not None and self._nlp is not None:
            return bool(self._matcher(self._nlp(sent)))
        sent_lower = sent.lower()
        return any(phrase in sent_lower for phrase in _GAZETTEER_LOWER)
