"""
src/relations.py

Stage 3: extract relations between equations in the same paper.

For each equation A, emit one entry per other equation B in the paper:

    {B: {"grade": "strong|potential|none", "description": "<text>"}}

Four signals drive the grade:

Signal 1 — Cross-reference (strong)
    PRIMARY: structural DOM cross-reference edge from the Task A XRef graph
    (passed as cross_refs parameter).  An XRef(to_eq_num=B) from a paragraph
    in the same section as equation A is treated as authoritative evidence.
    FALLBACK: regex search in A's BM25-expanded context for the citation pattern
    "Eq. (B)" / "equation (B)".
    Description: verbatim sentence from the XRef (≤200 chars) or verbatim
    sentence from text search.  Fixed label: "explicit citation".

Signal 2 — Jaccard symbol overlap (potential)
    |A.symbols ∩ B.symbols| / |A.symbols ∪ B.symbols| >= JACCARD_MIN.
    Uses only symbols that passed the confidence gate in symbols.py (the output
    dict already contains only high/medium confidence symbols).
    Description: "shares symbols: sym1, sym2, …" template.

Signal 3 — Derivation cue words (strong)
    A's context contains a derivation-cue phrase AND a reference to B's number
    in the same sentence.  Cue words: substituting, derived from, follows from,
    combining, plugging, yields, into eq, using eq.
    Description: verbatim cue sentence (≤200 chars).

Signal 4 — Semantic similarity (potential)
    cosine(SBERT(A.text), SBERT(B.text)) >= SBERT_MIN.  Falls back to TF-IDF.
    Description: "semantically similar (cosine=X.XX)".

Grading:
    strong    — signal 1 or 3 (textual / structural evidence)
    potential — signal 2 or 4 (structural overlap / semantic similarity)
    none      — no signal fired; description ""

Description vocabulary (fixed labels):
    "explicit citation"                 — signal 1 (no verbatim sentence)
    verbatim citing sentence (≤200 ch)  — signal 1 (when sentence is available)
    verbatim derivation sentence        — signal 3
    "derived from"                      — signal 3 (when no verbatim sentence)
    "shares symbols: X, Y, …"          — signal 2
    "semantically similar (cosine=…)"   — signal 4
    ""                                  — grade = none

Usage
-----
    extractor = RelationExtractor()
    all_relations = extractor.compute_all_relations(
        paper_build, eq_audits,
        retriever=bm25,
        cross_refs=parsed_paper.cross_refs,
        eq_section_ids={"1": "S1", "2": "S2"},
    )
    # all_relations: {eq_num: {other_eq_num: {"grade": ..., "description": ...}}}
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audit import AuditTrail

logger = logging.getLogger(__name__)

# ── Description artifact cleaner (mirrors meaning.py _clean_context layers) ──
_DESC_ARTIFACT_RE = re.compile(
    r'\bitalic[-_]\S+'
    r'|\bstart_[A-Z]\w*'
    r'|\bend_[A-Z]\w*'
    r'|\broad_[A-Za-z_]+'
    r'|\bcaligraphic_\S+'
    r'|\bbold_[a-z]\S*'
    r'|[\U0001D400-\U0001D7FF]'       # Unicode math block
    r'|[_^]\{[^{}]*\}'                # _{t}, ^{2}
    r'|\\[a-zA-Z]+(?:\{[^{}]*\})*'   # \rho, \mathcal{A}
    r'|\b(?:subscript|superscript|postsubscript|postsuperscript)\b'
)
_DESC_ORPHAN_RE = re.compile(r'[{}]')


def _clean_description(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = _DESC_ARTIFACT_RE.sub('', text)
    text = _DESC_ORPHAN_RE.sub('', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text

# ── scikit-learn optional import ──────────────────────────────────────────────
_SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    _SKLEARN_AVAILABLE = True
except Exception:
    logger.info(
        "scikit-learn not available — "
        "signal 4 will use built-in TF-IDF cosine fallback"
    )

# ── SBERT optional import ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.info(
        "sentence-transformers not available — "
        "signal 4 will use TF-IDF cosine fallback"
    )

# ── Paragraph-ID → section-ID extractor ──────────────────────────────────────
# Matches IDs like "S2.SS3.p4" → group(1) = "S2.SS3", or "S1.p1" → "S1".
_PARA_SECT_RE = re.compile(r'^(.+?)\.p\d+$')

# ── Paragraph numeric-suffix extractor ────────────────────────────────────────
_PARA_NUM_RE = re.compile(r'^(.+\.)p(\d+)$')


def _is_adjacent_para(para_a: str, para_b: str) -> bool:
    """Return True if *para_b* is the paragraph immediately before or after *para_a*.

    Requires the same section prefix and a trailing numeric suffix that
    differs by exactly 1 (e.g. ``S2.p3`` and ``S2.p4``).
    """
    if not para_a or not para_b or para_a == para_b:
        return False
    m_a = _PARA_NUM_RE.match(para_a)
    m_b = _PARA_NUM_RE.match(para_b)
    if m_a and m_b and m_a.group(1) == m_b.group(1):
        return abs(int(m_b.group(2)) - int(m_a.group(2))) == 1
    return False

# ── Citation regex template (matches meaning.py _EQ_CITE_RE_TMPL exactly) ────
# {n} is replaced with re.escape(eq_num_b) at match time.
_EQ_CITE_RE_TMPL: str = r'(?:eq(?:uation)?s?\.?\s*)?\(\s*{n}\s*\)'

# ── Sentence boundary splitter ────────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\(])')

# ── Derivation cue phrases (lower-cased for case-insensitive substring search) ─
_DERIVE_CUES: frozenset = frozenset({
    "substituting",
    "inserting",
    "derived from",
    "follows from",
    "combining",
    "plugging",
    "yields",
    "into eq",     # catches "into Eq.", "into equation"
    "using eq",    # catches "using Eq.", "using equation"
})

# ── Thresholds (named constants — no magic numbers inline) ────────────────────
JACCARD_MIN: float = 0.30   # minimum Jaccard for symbol-overlap signal
SBERT_MIN:   float = 0.50   # minimum SBERT cosine for semantic signal
TFIDF_MIN:   float = 0.60   # TF-IDF cosine threshold (fallback for signal 4)
SBERT_MODEL: str   = "all-MiniLM-L6-v2"
DESCRIPTION_MAX_LEN: int = 200   # truncation limit for verbatim excerpts

# Number of chunks retrieved to expand equation A's context when searching
# for cross-references to equation B.  Kept at 3 because we query by A's
# equation number, so high-precision results are expected.
RETRIEVAL_TOP_K_RELATIONS: int = 3


# ── Built-in TF-IDF cosine (no sklearn dependency) ───────────────────────────

def _tfidf_cosine_builtin(
    texts: List[str],
    eq_nums: List[str],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise TF-IDF cosine similarity without sklearn.

    Uses term-frequency vectors (raw counts) with smoothed IDF weights
    computed over the provided texts.  Returns a dict keyed by
    ``(eq_a, eq_b)`` pairs.

    Parameters
    ----------
    texts : list of str
        One text per equation, same order as *eq_nums*.
    eq_nums : list of str
        Equation number labels corresponding to *texts*.

    Returns
    -------
    dict
        ``{(eq_a, eq_b): cosine_score}`` for all ordered pairs ``i != j``.
    """
    _STOP = frozenset({
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "be", "been",
        "this", "that", "it", "we", "as", "if", "not", "no",
    })

    def _tokenize(text: str) -> List[str]:
        return [w for w in re.findall(r'[a-z]+', text.lower())
                if w not in _STOP and len(w) > 1]

    # build vocabulary and term-frequency vectors
    tokenized = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for toks in tokenized:
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    n_docs = len(texts)
    n_vocab = len(vocab)
    if n_vocab == 0:
        return {}

    # document frequency
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))

    # IDF (smoothed)
    idf = {
        word: math.log((n_docs + 1) / (df[word] + 1)) + 1.0
        for word in vocab
    }

    # TF-IDF vectors as numpy arrays
    vecs = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = np.zeros(n_vocab, dtype=float)
        for tok, cnt in tf.items():
            if tok in vocab:
                vec[vocab[tok]] = cnt * idf[tok]
        norm = np.linalg.norm(vec)
        vecs.append(vec / norm if norm > 0 else vec)

    # pairwise cosine
    result: Dict[Tuple[str, str], float] = {}
    for i, eq_a in enumerate(eq_nums):
        for j, eq_b in enumerate(eq_nums):
            if i != j:
                result[(eq_a, eq_b)] = float(np.dot(vecs[i], vecs[j]))
    return result


# ── Sort helper ───────────────────────────────────────────────────────────────

def _sort_eq_nums(eq_nums: List[str]) -> List[str]:
    """Sort equation numbers numerically where possible, else alphabetically.

    Parameters
    ----------
    eq_nums : list of str
        Equation number labels (e.g. ``["3", "1", "A.1", "2"]``).

    Returns
    -------
    list of str
        Sorted labels: pure integers first (ascending), then strings.
    """
    def _key(s: str) -> Tuple:
        try:
            return (0, int(s), "")
        except ValueError:
            return (1, 0, s)

    return sorted(eq_nums, key=_key)


class RelationExtractor:
    """Extract spec-format relations between equations in the same paper.

    Loads SBERT once at construction time.  If unavailable, falls back to
    TF-IDF cosine for signal 4 — logged per paper in the audit trail.

    Examples
    --------
    >>> ext = RelationExtractor()
    >>> rels = ext.compute_all_relations(paper_build, eq_audits)
    >>> rels["1"]["2"]
    {"grade": "strong", "description": "The first term of Eq. (2) is..."}
    """

    def __init__(self) -> None:
        self._sbert = None
        if _SBERT_AVAILABLE:
            try:
                self._sbert = SentenceTransformer(SBERT_MODEL)
                logger.info("RelationExtractor: SBERT model '%s' loaded", SBERT_MODEL)
            except Exception as exc:
                logger.warning(
                    "RelationExtractor: could not load SBERT (%s) — "
                    "falling back to TF-IDF cosine", exc
                )

    # ── public API ────────────────────────────────────────────────────────────

    def compute_all_relations(
        self,
        paper_equations: Dict[str, Dict],
        eq_audits: Dict[str, AuditTrail],
        retriever: Any = None,
        cross_refs: Optional[List] = None,
        eq_section_ids: Optional[Dict[str, str]] = None,
        eq_para_ids: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """Compute spec-format relations for every pair in a paper.

        Must be called *after* meaning and symbols have been extracted for
        all equations (signal 2 uses symbol keys; signal 4 encodes meaning).

        Signal 1 uses the structural DOM cross-reference graph (*cross_refs*)
        as the primary evidence source, falling back to BM25-expanded text
        search when the graph is unavailable or when no matching XRef is found.

        Parameters
        ----------
        paper_equations : dict
            ``{eq_num: {"equation": str, "meaning": str, "symbols": dict,
            "_ctx_before": str, "_ctx_after": str}}``.
        eq_audits : dict
            ``{eq_num: AuditTrail}`` — one audit entry appended per pair.
        retriever : BM25Retriever or None, optional
        cross_refs : list of XRef or None, optional
            Structural cross-reference graph from ParsedPaper.  Each XRef has
            ``to_eq_num``, ``from_para_id``, and ``from_sentence`` attributes.
        eq_section_ids : dict or None, optional
            ``{eq_num: section_id}`` — used to filter XRefs by section proximity.

        Returns
        -------
        dict
            ``{eq_num: {other_eq_num: {"grade": str, "description": str}}}``
            Single-equation papers return ``{eq_num: {}}``.
        """
        eq_nums = _sort_eq_nums(list(paper_equations.keys()))

        if len(eq_nums) <= 1:
            return {e: {} for e in eq_nums}

        # Pre-compute pairwise semantic similarity for signal 4 (one batch call).
        sim_matrix, sim_method = self._compute_similarity_matrix(
            paper_equations, eq_nums
        )

        # Build structural XRef index: to_eq_num → [(from_para, from_section, from_sentence)]
        xref_index: Dict[str, List[Tuple[str, str, str]]] = {}
        if cross_refs:
            for xref in cross_refs:
                to_eq   = getattr(xref, "to_eq_num",    "") or ""
                from_p  = getattr(xref, "from_para_id", "") or ""
                from_s  = getattr(xref, "from_sentence","") or ""
                m = _PARA_SECT_RE.match(from_p)
                from_sect = m.group(1) if m else ""
                if to_eq:
                    xref_index.setdefault(to_eq, []).append((from_p, from_sect, from_s))

        result: Dict[str, Dict] = {}

        for eq_a in eq_nums:
            data_a      = paper_equations[eq_a]
            ctx_a_local = (
                data_a.get("_ctx_before", "") + " " +
                data_a.get("_ctx_after",  "")
            ).strip()
            eq_a_sect   = (eq_section_ids or {}).get(eq_a, "")
            eq_a_para   = (eq_para_ids    or {}).get(eq_a, "")

            # Expand A's context with BM25-retrieved chunks for text-based signals.
            if retriever is not None:
                query_a    = f"equation {eq_a} eq {eq_a} ({eq_a})"
                retrieved_a = retriever.retrieve_text(
                    query_a, top_k=RETRIEVAL_TOP_K_RELATIONS
                )
                ctx_a = (ctx_a_local + " " + retrieved_a).strip()
            else:
                ctx_a = ctx_a_local

            syms_a  = set(data_a.get("symbols", {}).keys())
            audit_a = eq_audits.get(eq_a) or AuditTrail()

            eq_a_rels: Dict = {}

            for eq_b in eq_nums:
                if eq_b == eq_a:
                    continue

                syms_b = set(paper_equations[eq_b].get("symbols", {}).keys())

                fired: List[str] = []
                grade       = "none"
                description = ""
                jaccard     = 0.0
                sim_score   = 0.0

                # Signal 1 — cross-reference (strong) ────────────────────────
                # Primary: structural DOM XRef in same/adjacent paragraph as equation A.
                xref_graph_sent: Optional[str] = None
                for from_para, from_sect, from_sent in xref_index.get(eq_b, []):
                    if eq_a_para:
                        match = (from_para == eq_a_para or
                                 _is_adjacent_para(eq_a_para, from_para))
                    else:
                        match = (not eq_a_sect or from_sect == eq_a_sect)
                    if match:
                        xref_graph_sent = from_sent or None
                        break

                # Fallback: regex search in A's LOCAL context only (not BM25-
                # expanded).  The BM25-expanded context aggregates text from
                # multiple sections and can fan a single citation sentence out
                # as "strong" evidence for every equation in the paper.
                # The structural graph already provides the correct primary
                # signal; the text fallback is only needed for PDF papers or
                # when the graph has no entry for this pair.
                cite_sent: Optional[str] = (
                    xref_graph_sent
                    or self._find_citation_sentence(eq_b, ctx_a_local)
                )
                if cite_sent:
                    fired.append("crossref+graph" if xref_graph_sent else "crossref")

                # Signal 3 — derivation cue (strong) ─────────────────────────
                cue_sent: Optional[str] = self._find_derivation_cue_sentence(
                    eq_b, ctx_a
                )
                if cue_sent:
                    fired.append("deriv_cue")

                # Assign grade and description for strong signals ─────────────
                if cite_sent or cue_sent:
                    grade = "strong"
                    evidence    = cite_sent or cue_sent
                    description = _clean_description(evidence)[:DESCRIPTION_MAX_LEN]
                else:
                    # Signal 2 — Jaccard symbol overlap (potential) ───────────
                    union       = syms_a | syms_b
                    shared_syms = syms_a & syms_b
                    jaccard     = len(shared_syms) / len(union) if union else 0.0
                    if jaccard >= JACCARD_MIN:
                        fired.append(f"jaccard={jaccard:.2f}")
                        grade       = "potential"
                        description = f"shares symbols: {', '.join(sorted(shared_syms))}"

                    # Signal 4 — semantic similarity (potential) ───────────────
                    sim_score = sim_matrix.get((eq_a, eq_b), 0.0)
                    threshold = SBERT_MIN if self._sbert is not None else TFIDF_MIN
                    if sim_score >= threshold:
                        fired.append(f"{sim_method}={sim_score:.2f}")
                        grade = "potential"
                        if not description:
                            description = (
                                f"semantically similar (cosine={sim_score:.2f})"
                            )

                # Audit: signals + evidence + key scores ──────────────────────
                audit_a.log(
                    "relations",
                    f"eq=({eq_a})~({eq_b}): grade={grade} "
                    f"signals={fired} "
                    f"jaccard={jaccard:.2f} "
                    f"sim={sim_score:.2f} "
                    f"evidence={description[:80]!r}",
                )

                eq_a_rels[eq_b] = {"grade": grade, "description": description}

            result[eq_a] = eq_a_rels

        return result

    # ── signal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _sentence_around(context: str, match: "re.Match") -> str:
        """Extract the sentence that contains *match* from *context*.

        Expands left to the previous sentence-ending punctuation and right
        to the next one.  This avoids the known splitter bug where
        ``_SENT_SPLIT_RE`` breaks "Eq. (2)" into two fragments.

        Parameters
        ----------
        context : str
            Full context string.
        match : re.Match
            A regex match object whose span lies within *context*.

        Returns
        -------
        str
            The extracted sentence, stripped.
        """
        start, end = match.start(), match.end()
        # walk left to the nearest sentence-ending punctuation (. ! ?)
        # or the start of the string
        left = start
        while left > 0 and context[left - 1] not in ".!?":
            left -= 1
        # walk right to the nearest sentence-ending punctuation
        right = end
        while right < len(context) and context[right] not in ".!?":
            right += 1
        if right < len(context):
            right += 1   # include the punctuation
        return context[left:right].strip()

    def _find_citation_sentence(
        self, eq_num_b: str, context: str
    ) -> Optional[str]:
        """Return verbatim sentence in *context* that cites *eq_num_b*.

        Searches the full context string for the citation pattern, then
        extracts the sentence around the match.  This avoids the sentence-
        splitter fragmentation that occurs with "Eq. (2)" patterns.

        Parameters
        ----------
        eq_num_b : str
            The equation number we are looking for references to.
        context : str
            Combined context_before + context_after of equation A.

        Returns
        -------
        str or None
            First matching sentence, or ``None`` if none found.
        """
        pattern = re.compile(
            _EQ_CITE_RE_TMPL.format(n=re.escape(eq_num_b)),
            re.IGNORECASE,
        )
        m = pattern.search(context)
        if not m:
            return None
        sent = self._sentence_around(context, m)
        # Reject fragments: too short or starts with bare parenthesis
        if len(sent) > 20 and not sent.startswith("("):
            return sent
        return None

    def _find_derivation_cue_sentence(
        self, eq_num_b: str, context: str
    ) -> Optional[str]:
        """Return a sentence containing both a derivation-cue word and a citation of eq_num_b.

        Parameters
        ----------
        eq_num_b : str
            Equation number that must be referenced in the sentence.
        context : str
            Combined context of equation A.

        Returns
        -------
        str or None
        """
        cite_re = re.compile(
            _EQ_CITE_RE_TMPL.format(n=re.escape(eq_num_b)),
            re.IGNORECASE,
        )
        for m in cite_re.finditer(context):
            sent = self._sentence_around(context, m)
            if any(cue in sent.lower() for cue in _DERIVE_CUES):
                return sent
        return None

    # ── similarity matrix ─────────────────────────────────────────────────────

    def _get_encode_texts(
        self,
        paper_equations: Dict[str, Dict],
        eq_nums: List[str],
    ) -> List[str]:
        """Select the best text to encode per equation (meaning > context).

        Parameters
        ----------
        paper_equations : dict
            Full paper dict including ``"meaning"`` and context fields.
        eq_nums : list of str
            Ordered equation numbers to encode.

        Returns
        -------
        list of str
            One text string per equation, same order as *eq_nums*.
        """
        texts: List[str] = []
        for eq_num in eq_nums:
            data = paper_equations[eq_num]
            text = data.get("meaning", "").strip()
            if not text:
                ctx_b = data.get("_ctx_before", "")
                ctx_a = data.get("_ctx_after",  "")
                text  = (ctx_b + " " + ctx_a).strip()[:500]
            texts.append(text if text else " ")
        return texts

    def _compute_similarity_matrix(
        self,
        paper_equations: Dict[str, Dict],
        eq_nums: List[str],
    ) -> Tuple[Dict[Tuple[str, str], float], str]:
        """Compute pairwise cosine similarity for all equations in the paper.

        Uses SBERT if available, otherwise falls back to TF-IDF cosine.
        Encodes all equation texts in a single batched call for efficiency.

        Parameters
        ----------
        paper_equations : dict
        eq_nums : list of str

        Returns
        -------
        sim_matrix : dict
            ``{(eq_a, eq_b): cosine_score}`` for all ordered pairs.
        method : str
            ``"sbert"`` or ``"tfidf"`` — logged to the audit trail.
        """
        sim: Dict[Tuple[str, str], float] = {}
        texts = self._get_encode_texts(paper_equations, eq_nums)

        if self._sbert is not None:
            try:
                embeddings = self._sbert.encode(texts, convert_to_numpy=True)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normed = embeddings / norms
                cos_matrix = normed @ normed.T
                for i, eq_a in enumerate(eq_nums):
                    for j, eq_b in enumerate(eq_nums):
                        if i != j:
                            sim[(eq_a, eq_b)] = float(cos_matrix[i, j])
                return sim, "sbert"
            except Exception as exc:
                logger.warning(
                    "SBERT encoding failed (%s) — falling back to TF-IDF", exc
                )

        # TF-IDF cosine fallback (built-in, no sklearn required)
        non_empty = [t for t in texts if t.strip()]
        if len(non_empty) < 2:
            return sim, "tfidf"
        try:
            if _SKLEARN_AVAILABLE:
                vectorizer   = TfidfVectorizer(
                    min_df=1,
                    stop_words="english",
                    token_pattern=r"(?u)\b\w+\b",
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                cos_matrix   = _sk_cosine(tfidf_matrix)
                for i, eq_a in enumerate(eq_nums):
                    for j, eq_b in enumerate(eq_nums):
                        if i != j:
                            sim[(eq_a, eq_b)] = float(cos_matrix[i, j])
            else:
                # Pure-Python TF-IDF cosine (no external dependencies)
                sim.update(_tfidf_cosine_builtin(texts, eq_nums))
        except Exception as exc:
            logger.warning("TF-IDF similarity computation failed: %s", exc)

        return sim, "tfidf"
