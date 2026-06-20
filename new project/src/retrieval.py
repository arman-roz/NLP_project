"""
src/retrieval.py

BM25 (primary) and TF-IDF (ablation) retrievers over the structured chunk
views of a single arXiv paper, with an optional MiniLM encoder reranker.

One retriever instance is built per paper by calling ``fit()`` once with that
paper's :class:`~src.chunks.Chunk` objects.  The retriever is then queried
independently for each equation.  Results are fully deterministic: BM25 and
TF-IDF scores are arithmetic functions of fixed token counts; MiniLM
deterministic given its fixed pre-trained weights.

Compliance block
----------------
ALLOWED:  encode text → dense or sparse vector; cosine or BM25 score →
          rank existing paper sentences; SELECT and copy a sentence verbatim.
FORBIDDEN: use any model (BM25, TF-IDF, MiniLM, or any LLM / generative
           system) to PRODUCE, GENERATE, COMPLETE, or PARAPHRASE any string
           written to output JSON fields (``meaning``, ``symbols.*``,
           ``relations.*.description``).  Every output string must be a
           verbatim fragment from the source paper or a deterministic template
           filled with extracted values.

Usage
-----
    from src.chunks import DocumentParser
    from src.retrieval import BM25Retriever, meaning_query, symbol_query

    parser = DocumentParser()
    paper  = parser.parse(html_bytes, arxiv_id, endpoint, equations)

    ret = BM25Retriever()
    ret.fit(paper.eq_neighborhood_chunks + paper.sentence_chunks)

    hits = ret.search(meaning_query("3", "H E"), top_k=8)
    # hits[0] == {
    #   "chunk_id": "2401.13506:S2:sentence:4",
    #   "score": 5.41,
    #   "chunk_type": "sentence",
    #   "text": "...",
    #   "matched_terms": ["equation", "3", "hamiltonian"],
    # }
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── BM25+ hyper-parameters ──────────────────────────────────────────────────────
BM25_K1: float    = 1.5   # term-frequency saturation rate
BM25_B: float     = 0.75  # length normalisation strength (0 = off, 1 = full)
BM25_DELTA: float = 1.0   # BM25+ floor — matched terms always contribute ≥ δ

# ── Default retrieval depth ─────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 5

# ── SBERT model name (pinned; no network call at query time once cached) ────────
SBERT_MODEL: str = "all-MiniLM-L6-v2"

# ── English stop words ──────────────────────────────────────────────────────────
# Physics domain stop words (e.g. "not", "no") kept so that negations in
# definitional prose ("H is not the total energy") are not silently dropped.
_STOP: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "we", "our", "us", "as", "if", "when", "where",
    "which", "who", "what", "how",
})

# LaTeXML artifact token prefixes — excluded even if they survive chunk cleaning.
_ARTIFACT_PREFIXES: frozenset = frozenset({
    "italic", "start", "end", "over", "road",
    "superscript", "subscript", "postsubscript", "postsuperscript",
})

# ── Greek symbol surface-form lookup ───────────────────────────────────────────
# Keys: LaTeX command strings.  Values: lowercase surface forms used in prose
# and Unicode glyphs (both appear in arXiv HTML after LaTeXML rendering).
# Used by :func:`symbol_query` to widen retrieval for Greek symbols.
_GREEK_SURFACE: Dict[str, List[str]] = {
    r"\alpha":      ["alpha", "α"],
    r"\beta":       ["beta", "β"],
    r"\gamma":      ["gamma", "γ"],
    r"\Gamma":      ["gamma", "Γ"],
    r"\delta":      ["delta", "δ"],
    r"\Delta":      ["delta", "Δ"],
    r"\epsilon":    ["epsilon", "ε"],
    r"\varepsilon": ["epsilon", "varepsilon", "ε"],
    r"\zeta":       ["zeta", "ζ"],
    r"\eta":        ["eta", "η"],
    r"\theta":      ["theta", "θ"],
    r"\Theta":      ["theta", "Θ"],
    r"\iota":       ["iota", "ι"],
    r"\kappa":      ["kappa", "κ"],
    r"\lambda":     ["lambda", "λ"],
    r"\Lambda":     ["lambda", "Λ"],
    r"\mu":         ["mu", "μ"],
    r"\nu":         ["nu", "ν"],
    r"\xi":         ["xi", "ξ"],
    r"\pi":         ["pi", "π"],
    r"\Pi":         ["pi", "Π"],
    r"\rho":        ["rho", "ρ"],
    r"\sigma":      ["sigma", "σ"],
    r"\Sigma":      ["sigma", "Σ"],
    r"\tau":        ["tau", "τ"],
    r"\upsilon":    ["upsilon", "υ"],
    r"\phi":        ["phi", "φ"],
    r"\Phi":        ["phi", "Φ"],
    r"\chi":        ["chi", "χ"],
    r"\psi":        ["psi", "ψ"],
    r"\Psi":        ["psi", "Ψ"],
    r"\omega":      ["omega", "ω"],
    r"\Omega":      ["omega", "Ω"],
}

# ── Module-level model cache for MiniLM ────────────────────────────────────────
# Loaded lazily on first call to rerank_with_minilm(); never loaded otherwise.
_sbert_cache: Dict[str, Any] = {}

# ── Tokenizer ───────────────────────────────────────────────────────────────────
_TOK_RE = re.compile(r'[a-z0-9]+')


def _tokenize(text: str) -> List[str]:
    """Tokenize *text* for BM25/TF-IDF indexing.

    Lowercase; keeps alphabetic words AND digit strings so that equation
    numbers (``"3"``, ``"1a"``) are preserved as BM25 vocabulary terms.
    Removes English stop words and LaTeXML artifact-prefix tokens.

    Unlike the old regex ``[a-z]+``, this keeps ``[a-z0-9]+`` so that
    ``"(3)"`` tokenises to ``["3"]``, ``"Eq. 1a"`` to ``["eq", "1a"]``, and
    the query ``"Eq 3 Equation 3"`` can match ``"equation (3)"`` in chunk text.

    Parameters
    ----------
    text : str
        Raw text from a Chunk or a query template string.

    Returns
    -------
    list of str
        Filtered, lowercased, alphanumeric tokens.
    """
    tokens = []
    for w in _TOK_RE.findall(text.lower()):
        if w in _STOP or w in _ARTIFACT_PREFIXES:
            continue
        # keep numeric tokens (equation numbers, even single digit) and words ≥ 2 chars
        if w.isdigit() or len(w) >= 2:
            tokens.append(w)
    return tokens


# ── Deterministic query template helpers ───────────────────────────────────────

def meaning_query(eq_num: str, top_symbols: str = "") -> str:
    """Build a deterministic BM25 query for retrieving meaning sentences.

    Targets sentences that describe or define the equation, using common
    definitional cue words alongside the equation number.

    Parameters
    ----------
    eq_num : str
        Equation number string, e.g. ``"3"`` or ``"1a"``.
    top_symbols : str, optional
        Space-separated symbol surface forms from the equation, e.g.
        ``"H E alpha"``.  Appended to widen symbol-based recall.

    Returns
    -------
    str
        Deterministic query string, ready for :meth:`BM25Retriever.search`.

    Examples
    --------
    >>> meaning_query("3", "H E")
    'Eq 3 Equation 3 describes represents gives defines called known as H E'
    """
    base = (
        f"Eq {eq_num} Equation {eq_num} "
        "describes represents gives defines called known as"
    )
    cue = top_symbols.strip()
    return f"{base} {cue}".rstrip() if cue else base


def symbol_query(symbol: str, greek_forms: Optional[List[str]] = None) -> str:
    """Build a deterministic BM25 query for retrieving symbol definition sentences.

    Expands Greek LaTeX commands to text / Unicode surface forms so that
    definitional prose (``"where α is the fine-structure constant"``) is
    retrieved even when the chunk text does not contain the LaTeX command.

    Parameters
    ----------
    symbol : str
        Symbol in LaTeX or ASCII form, e.g. ``r"\\alpha"``, ``"H"``,
        ``"E_k"``.
    greek_forms : list of str, optional
        Surface-form overrides.  When provided, replaces the built-in Greek
        lookup.  Pass multiple forms to widen recall for unusual symbols.

    Returns
    -------
    str
        Deterministic query string.

    Examples
    --------
    >>> symbol_query(r"\\alpha")
    'alpha α where denotes represents is defined as called corresponds to'
    >>> symbol_query("H")
    'H where denotes represents is defined as called corresponds to'
    """
    if greek_forms is not None:
        forms = greek_forms
    else:
        forms = _GREEK_SURFACE.get(symbol, [symbol])
    forms_str = " ".join(forms[:4])  # cap to keep the query focused
    return f"{forms_str} where denotes represents is defined as called corresponds to"


def relation_query(eq_a: str, eq_b: str) -> str:
    """Build a deterministic BM25 query for retrieving relation evidence sentences.

    Targets sentences that discuss both equations together using derivation,
    substitution, or equivalence cue words.

    Parameters
    ----------
    eq_a : str
        Source equation number, e.g. ``"3"``.
    eq_b : str
        Target equation number, e.g. ``"7"``.

    Returns
    -------
    str
        Deterministic query string.

    Examples
    --------
    >>> relation_query("3", "7")
    'Eq 3 Eq 7 using substituting inserting follows from derived from yields equivalent special case'
    """
    return (
        f"Eq {eq_a} Eq {eq_b} "
        "using substituting inserting follows from derived from "
        "yields equivalent special case"
    )


# ── BM25Retriever ───────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25+ retriever over the structured chunk views of a single paper.

    Call :meth:`fit` once to build the inverted index, then call
    :meth:`search` for each query.  Results are fully deterministic for a
    given chunk list and query string: BM25+ is arithmetic; ``sorted()`` is
    stable (ties resolved by chunk insertion order).

    BM25+ variant (Lü & Callan 2011): the δ floor (``BM25_DELTA``) ensures
    every matched term contributes a strictly positive score, preventing
    high-document-frequency terms like ``"equation"`` or ``"energy"`` from
    receiving a zero contribution.

    Parameters
    ----------
    None — call :meth:`fit` to supply Chunk objects.

    Examples
    --------
    >>> ret = BM25Retriever()
    >>> ret.fit(paper.eq_neighborhood_chunks + paper.sentence_chunks)
    >>> hits = ret.search(meaning_query("3", "H E"), top_k=8)
    >>> hits[0]["chunk_id"]
    '2401.13506:S2.SS3:sentence:4'
    >>> hits[0]["matched_terms"]
    ['3', 'defines', 'eq', 'equation', 'gives']
    """

    def __init__(self) -> None:
        self._chunks:     List[Any]           = []
        self._tok_chunks: List[List[str]]     = []
        self._idf:        Dict[str, float]    = {}
        self._avgdl:      float               = 1.0
        self._n:          int                 = 0

    def fit(self, chunks: List[Any]) -> None:
        """Build the BM25+ inverted index from a list of Chunk objects.

        Must be called before :meth:`search`.  A second call replaces the
        previous index entirely.

        Parameters
        ----------
        chunks : list of Chunk
            Chunk objects from :class:`~src.chunks.DocumentParser`.  Typically
            the concatenation of ``eq_neighborhood_chunks`` (primary pool) and
            ``sentence_chunks`` (fallback pool) from one
            :class:`~src.chunks.ParsedPaper`.

        Returns
        -------
        None
        """
        self._chunks     = list(chunks)
        self._n          = len(self._chunks)
        self._tok_chunks = [_tokenize(c.text) for c in self._chunks]

        if self._n == 0:
            self._idf   = {}
            self._avgdl = 1.0
            logger.debug("BM25Retriever: built empty index")
            return

        # Document-frequency count for IDF
        df: Counter = Counter()
        for toks in self._tok_chunks:
            df.update(set(toks))

        # BM25+ IDF: log((N − df + 0.5) / (df + 0.5) + 1) — always positive
        self._idf = {
            w: math.log((self._n - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for w, cnt in df.items()
        }
        self._avgdl = sum(len(t) for t in self._tok_chunks) / self._n

        logger.debug(
            "BM25Retriever: fit %d chunks  vocab=%d  avgdl=%.1f",
            self._n, len(self._idf), self._avgdl,
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Return top-*k* chunks ranked by BM25+ score.

        Applies optional metadata filters before ranking so that the filter
        does not waste top-*k* budget on irrelevant chunk types or sections.

        Parameters
        ----------
        query : str
            Free-text query.  Use :func:`meaning_query`, :func:`symbol_query`,
            or :func:`relation_query` for deterministic, pre-tested templates.
        top_k : int
            Maximum number of results to return (after filtering).
        filters : dict, optional
            Restrict the candidate set before BM25 scoring.  Supported keys:

            ``"chunk_type"`` : str or list of str
                Keep only chunks whose ``chunk_type`` is this value (or in
                this list), e.g. ``"sentence"`` or
                ``["sentence", "equation_neighborhood"]``.
            ``"section_id"`` : str
                Keep only chunks from this section, e.g. ``"S2.SS3"``.
            ``"eq_nums_nearby"`` : list of str
                Keep only chunks that have at least one of these equation
                numbers in their ``eq_nums_nearby`` list.

        Returns
        -------
        list of dict
            Each element contains:

            ``"chunk_id"``      : str   — unique chunk identifier
            ``"score"``         : float — BM25+ score (higher = more relevant)
            ``"chunk_type"``    : str   — ``"sentence"``, ``"paragraph"``,
                                         or ``"equation_neighborhood"``
            ``"text"``          : str   — cleaned prose text
            ``"matched_terms"`` : list of str — query tokens that hit this chunk

            Sorted descending by ``"score"``; ties in insertion order (stable).
        """
        if self._n == 0:
            return []
        query_toks = _tokenize(query)
        if not query_toks:
            return []

        candidate_ids = list(range(self._n))
        if filters:
            candidate_ids = self._apply_filters(candidate_ids, filters)
        if not candidate_ids:
            return []

        scores, matched = self._bm25_scores_filtered(query_toks, candidate_ids)

        ranked: List[Tuple[float, int]] = sorted(
            (
                (scores[k], k)
                for k in range(len(candidate_ids))
                if scores[k] > 0.0
            ),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, k in ranked:
            i = candidate_ids[k]
            c = self._chunks[i]
            results.append({
                "chunk_id":      c.chunk_id,
                "score":         score,
                "chunk_type":    c.chunk_type,
                "text":          c.text,
                "matched_terms": matched[k],
                "section_id":    getattr(c, "section_id",   ""),
                "paragraph_id":  getattr(c, "paragraph_id", ""),
                "char_offset":   getattr(c, "char_offset",  0),
            })

        logger.debug(
            "BM25Retriever.search: query=%r filters=%s → %d/%d hits",
            query[:60], filters, len(results), len(candidate_ids),
        )
        return results

    def retrieve_text(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Return joined text of top-*k* chunks (convenience wrapper).

        Called by :class:`~src.meaning.MeaningExtractor`,
        :class:`~src.symbols.SymbolExtractor`, and
        :class:`~src.relations.RelationExtractor` when a retriever is wired in.

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        str
            Space-joined chunk texts in descending score order, or ``""`` if
            nothing was retrieved.
        """
        return " ".join(h["text"] for h in self.search(query, top_k=top_k))

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Alias for :meth:`search` without filters (backward compatibility).

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        list of dict
        """
        return self.search(query, top_k=top_k)

    # ── private helpers ─────────────────────────────────────────────────────────

    def _apply_filters(
        self,
        indices: List[int],
        filters: Dict,
    ) -> List[int]:
        """Return the subset of *indices* whose chunks satisfy all *filters*.

        Parameters
        ----------
        indices : list of int
            Candidate chunk indices to filter.
        filters : dict
            See :meth:`search` for supported keys.

        Returns
        -------
        list of int
        """
        ct_filter  = filters.get("chunk_type")
        sec_filter = filters.get("section_id")
        eq_filter  = filters.get("eq_nums_nearby")
        eq_set     = set(eq_filter) if eq_filter else None

        result = []
        for i in indices:
            c = self._chunks[i]
            if ct_filter is not None:
                allowed = (
                    ct_filter
                    if isinstance(ct_filter, (list, tuple, set))
                    else [ct_filter]
                )
                if c.chunk_type not in allowed:
                    continue
            if sec_filter is not None and c.section_id != sec_filter:
                continue
            if eq_set is not None and not (set(c.eq_nums_nearby) & eq_set):
                continue
            result.append(i)
        return result

    def _bm25_scores_filtered(
        self,
        query_toks: List[str],
        candidate_ids: List[int],
    ) -> Tuple[List[float], List[List[str]]]:
        """Score *candidate_ids* against *query_toks* with BM25+.

        BM25+ formula per term *t*, document *d*::

            IDF(t) × [ tf(t,d)×(K1+1) / (tf(t,d) + K1×norm(d)) + δ ]

        where ``norm(d) = 1 − B + B × |d| / avgdl``.

        Parameters
        ----------
        query_toks : list of str
            Pre-tokenised query terms (stop words already removed).
        candidate_ids : list of int
            Indices into ``self._chunks`` / ``self._tok_chunks`` to score.
            Scores are returned parallel to this list, not to the full chunk list.

        Returns
        -------
        scores : list of float
            BM25+ score for each candidate (parallel to *candidate_ids*).
        matched : list of list of str
            Sorted matched query terms for each candidate (parallel to
            *candidate_ids*).
        """
        n_cands  = len(candidate_ids)
        scores:  List[float] = [0.0] * n_cands
        matched: List[set]   = [set()  for _ in range(n_cands)]

        for qt in query_toks:
            idf = self._idf.get(qt, 0.0)
            if idf == 0.0:
                continue
            for k, i in enumerate(candidate_ids):
                toks = self._tok_chunks[i]
                f    = toks.count(qt)
                if f == 0:
                    continue
                dl        = len(toks)
                norm      = 1.0 - BM25_B + BM25_B * dl / max(self._avgdl, 1.0)
                tf_score  = (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * norm)
                scores[k] += idf * (tf_score + BM25_DELTA)
                matched[k].add(qt)

        return scores, [sorted(m) for m in matched]


# ── TfidfRetriever ──────────────────────────────────────────────────────────────

class TfidfRetriever:
    """TF-IDF retriever — ablation baseline, not used in the default pipeline.

    Drop-in replacement for :class:`BM25Retriever` for ablation studies
    comparing BM25 vs TF-IDF cosine.  Requires ``scikit-learn``.  All public
    methods have identical signatures so the two classes are interchangeable
    for any caller.

    Parameters
    ----------
    None — call :meth:`fit` to supply Chunk objects.
    """

    def __init__(self) -> None:
        self._chunks:     List[Any] = []
        self._vectorizer: Any       = None
        self._matrix:     Any       = None
        self._n:          int       = 0

    def fit(self, chunks: List[Any]) -> None:
        """Build TF-IDF index from a list of Chunk objects.

        Parameters
        ----------
        chunks : list of Chunk

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If ``scikit-learn`` is not installed.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "TfidfRetriever requires scikit-learn: pip install scikit-learn"
            ) from exc

        self._chunks = list(chunks)
        self._n      = len(self._chunks)

        if self._n == 0:
            logger.debug("TfidfRetriever: built empty index")
            return

        # analyzer=_tokenize delegates tokenisation to our shared function,
        # ensuring the same vocabulary as BM25Retriever for fair comparison.
        self._vectorizer = TfidfVectorizer(
            analyzer=_tokenize,
            sublinear_tf=True,  # apply 1 + log(tf) to reduce burstiness
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(
            [c.text for c in self._chunks]
        )
        logger.debug(
            "TfidfRetriever: fit %d chunks  vocab=%d",
            self._n, len(self._vectorizer.vocabulary_),
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Return top-*k* chunks ranked by TF-IDF cosine similarity.

        Parameters
        ----------
        query : str
        top_k : int
        filters : dict, optional
            Same keys as :meth:`BM25Retriever.search`.

        Returns
        -------
        list of dict
            Same format as :meth:`BM25Retriever.search` (``"score"`` here is
            cosine similarity, 0–1).

        Raises
        ------
        ImportError
            If ``scikit-learn`` is not installed.
        """
        if self._n == 0 or self._vectorizer is None:
            return []
        query_toks = set(_tokenize(query))
        if not query_toks:
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise ImportError("TfidfRetriever requires scikit-learn") from exc

        candidate_ids = list(range(self._n))
        if filters:
            candidate_ids = self._apply_filters(candidate_ids, filters)
        if not candidate_ids:
            return []

        query_vec   = self._vectorizer.transform([query])
        cand_matrix = self._matrix[candidate_ids]
        sims        = cosine_similarity(query_vec, cand_matrix)[0]  # (n_cands,)

        ranked: List[Tuple[float, int]] = sorted(
            (
                (float(sims[k]), k)
                for k in range(len(candidate_ids))
                if sims[k] > 0.0
            ),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, k in ranked:
            i = candidate_ids[k]
            c = self._chunks[i]
            chunk_toks = set(_tokenize(c.text))
            results.append({
                "chunk_id":      c.chunk_id,
                "score":         score,
                "chunk_type":    c.chunk_type,
                "text":          c.text,
                "matched_terms": sorted(query_toks & chunk_toks),
                "section_id":    getattr(c, "section_id",   ""),
                "paragraph_id":  getattr(c, "paragraph_id", ""),
                "char_offset":   getattr(c, "char_offset",  0),
            })
        return results

    def retrieve_text(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Return joined text of top-*k* chunks (convenience wrapper).

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        str
        """
        return " ".join(h["text"] for h in self.search(query, top_k=top_k))

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Alias for :meth:`search` without filters (backward compatibility).

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        list of dict
        """
        return self.search(query, top_k=top_k)

    def _apply_filters(
        self,
        indices: List[int],
        filters: Dict,
    ) -> List[int]:
        """Return the subset of *indices* whose chunks satisfy all *filters*.

        Parameters
        ----------
        indices : list of int
        filters : dict

        Returns
        -------
        list of int
        """
        ct_filter  = filters.get("chunk_type")
        sec_filter = filters.get("section_id")
        eq_filter  = filters.get("eq_nums_nearby")
        eq_set     = set(eq_filter) if eq_filter else None

        result = []
        for i in indices:
            c = self._chunks[i]
            if ct_filter is not None:
                allowed = (
                    ct_filter
                    if isinstance(ct_filter, (list, tuple, set))
                    else [ct_filter]
                )
                if c.chunk_type not in allowed:
                    continue
            if sec_filter is not None and c.section_id != sec_filter:
                continue
            if eq_set is not None and not (set(c.eq_nums_nearby) & eq_set):
                continue
            result.append(i)
        return result


# ── Optional MiniLM encoder reranker ────────────────────────────────────────────

def rerank_with_minilm(
    query: str,
    candidates: List[Dict],
    model_name: str = SBERT_MODEL,
) -> List[Dict]:
    """Rerank BM25/TF-IDF candidates by MiniLM cosine similarity (opt-in only).

    Encodes *query* and candidate texts with ``all-MiniLM-L6-v2`` (encoder
    only, no text generation), computes cosine similarity, and reorders the
    list.  The candidate dicts are never modified — only reordered and
    augmented with ``"minilm_score"``.

    This function is **never called automatically** by the pipeline; it must
    be invoked explicitly by the caller.  If the model cannot be loaded
    (offline lab, ``sentence-transformers`` not installed), *candidates* are
    returned unchanged and a WARNING is logged — the pipeline does not crash.

    Compliance: MiniLM is an encoder used for similarity SCORING only.  It
    never produces or modifies any string that is written to the output JSON.

    Parameters
    ----------
    query : str
        The same query passed to :meth:`BM25Retriever.search`.
    candidates : list of dict
        Output of :meth:`BM25Retriever.search` or :meth:`TfidfRetriever.search`.
    model_name : str, optional
        SBERT model identifier.  Defaults to :data:`SBERT_MODEL`.
        Pin ``SBERT_MODEL = "all-MiniLM-L6-v2"`` at module level for
        reproducibility.

    Returns
    -------
    list of dict
        Same dicts as *candidates*, reordered by ``"minilm_score"`` descending.
        Each dict gains a ``"minilm_score"`` key (float, cosine similarity 0–1).
        The original BM25 ``"score"`` is preserved unchanged.
    """
    if not candidates:
        return candidates

    # Lazy load — model is cached after first call; no network at query time.
    if model_name not in _sbert_cache:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("rerank_with_minilm: loading model %r", model_name)
            _sbert_cache[model_name] = SentenceTransformer(model_name)
            logger.info("rerank_with_minilm: model %r loaded", model_name)
        except Exception as exc:
            logger.warning(
                "rerank_with_minilm: cannot load %r (%s) — returning BM25 order",
                model_name, exc,
            )
            return candidates

    try:
        import numpy as np
    except ImportError as exc:
        logger.warning("rerank_with_minilm: numpy unavailable (%s) — skipping rerank", exc)
        return candidates

    model  = _sbert_cache[model_name]
    texts  = [c["text"] for c in candidates]
    q_vec  = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    d_vecs = model.encode(texts,   convert_to_numpy=True, show_progress_bar=False)

    # Cosine similarity: L2-normalise then dot product
    q_norm = q_vec  / (np.linalg.norm(q_vec,  axis=1, keepdims=True) + 1e-10)
    d_norm = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-10)
    sims   = (d_norm @ q_norm.T).flatten()  # shape (n_candidates,)

    logger.debug(
        "rerank_with_minilm: model=%r  query=%r  scores=%s",
        model_name,
        query[:60],
        " ".join(f"{s:.3f}" for s in sims),
    )

    reranked = [
        {**cand, "minilm_score": float(sim)}
        for cand, sim in zip(candidates, sims)
    ]
    return sorted(reranked, key=lambda x: x["minilm_score"], reverse=True)


# ── Backward-compatibility shim ────────────────────────────────────────────────

class _DictChunkAdapter:
    """Wrap a retriever dict (``{id, text, _chunk}``) to look like a Chunk.

    Used internally by :class:`Retriever` when old dict-format chunks are
    passed.  Prefers the embedded ``_chunk`` object for metadata; falls back
    to dict keys when ``_chunk`` is absent.
    """

    __slots__ = (
        "chunk_id", "chunk_type", "text",
        "section_id", "paragraph_id", "char_offset", "eq_nums_nearby",
    )

    def __init__(self, d: Dict) -> None:
        c: Any               = d.get("_chunk")
        self.text: str       = d.get("text", "")
        if c is not None:
            self.chunk_id:       str       = c.chunk_id
            self.chunk_type:     str       = c.chunk_type
            self.section_id:     str       = c.section_id
            self.paragraph_id:   str       = getattr(c, "paragraph_id", "")
            self.char_offset:    int       = getattr(c, "char_offset", 0)
            self.eq_nums_nearby: List[str] = c.eq_nums_nearby
        else:
            self.chunk_id       = d.get("id", "")
            self.chunk_type     = "unknown"
            self.section_id     = ""
            self.paragraph_id   = ""
            self.char_offset    = 0
            self.eq_nums_nearby = []


class Retriever(BM25Retriever):
    """Backward-compatible dict-based wrapper around :class:`BM25Retriever`.

    Accepts the old ``List[Dict]`` format produced by
    :func:`~src.chunks.chunks_for_retriever` *in addition to* Chunk objects.
    When dicts are passed, they are wrapped by :class:`_DictChunkAdapter` so
    that :class:`BM25Retriever` can index and filter them.

    Prefer :class:`BM25Retriever` for new code.

    Parameters
    ----------
    chunks : list of dict or list of Chunk
        Either the old ``{"id", "text", "_chunk"}`` dict format or raw Chunk
        objects — both are accepted.
    """

    def __init__(self, chunks: List[Any]) -> None:
        super().__init__()
        if not chunks:
            return
        if isinstance(chunks[0], dict):
            self.fit([_DictChunkAdapter(c) for c in chunks])
        else:
            self.fit(chunks)
