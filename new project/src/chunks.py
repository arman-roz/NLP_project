"""
src/chunks.py

Document model, chunk views, and cross-reference graph for arXiv HTML papers.

Replaces the old sliding-window ``Chunker`` with a structured, DOM-aware
document model built directly from arXiv's LaTeXML HTML output.

Three chunk views
-----------------
sentence_chunks         — one ``Chunk`` per sentence; the unit from which
                          meaning and symbol evidence is selected.
paragraph_chunks        — one ``Chunk`` per ``div.ltx_para``; BM25 baseline
                          with broader context.
eq_neighborhood_chunks  — one ``Chunk`` per equation: section title +
                          preceding prose + LaTeX + following prose, all
                          within the same ``div.ltx_para``.  Default chunk
                          for meaning and relations retrieval.

Cross-reference graph
---------------------
``build_cross_reference_index`` emits one ``XRef`` per (paragraph, equation)
edge.  Two evidence sources, tried in order:

1. Structural (preferred) — ``<a class="ltx_ref">`` links whose href
   fragment resolves to a known equation table id.  When the fragment is
   not in the table-id map, the link's visible text (always the bare
   equation number) is used instead; this is labelled
   ``"structural_text"`` in the ``source`` field.

2. Lexical (fallback) — regex ``_EQ_CITE_CAPTURE_RE`` over sentence text
   for sentences not already covered by a structural edge.  Covers
   PDF-sourced papers and HTML papers where DOM links were absent.

Usage
-----
    parser = DocumentParser()
    paper  = parser.parse(html_bytes, arxiv_id, source_endpoint, equations)
    hits   = parser.symbol_chunk_view(paper, ["H", r"\\Hamiltonian", "ℋ"])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Sentence splitter (same pattern as meaning.py) ────────────────────────────
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\(])')

# ── Cleaning regexes — mirror and extend meaning.py two-layer strip ───────────
# Layer 1: LaTeXML HTML artifact tokens
_ARTIFACT_RE = re.compile(
    r'\bitalic_\w+'
    r'|\bstart_[A-Z]\w*'
    r'|\bend_[A-Z]\w*'
    r'|\bover[a-z]*_ARG\b'
    r'|\broad_[A-Za-z_]+'
    # Additional: bare "superscript"/"subscript" words left after layer-1 strip
    r'|\b(?:superscript|subscript|POSTSUBSCRIPT|POSTSUPERSCRIPT)\b'
)
# Layer 2: Unicode mathematical alphanumeric characters (U+1D400–U+1D7FF)
# and invisible operators (U+2061–U+2064)
_MATH_UNICODE_RE  = re.compile(r'[\U0001D400-\U0001D7FF⁡-⁤]')
# Layer 3: zero-width and non-printable Unicode characters
_ZERO_WIDTH_RE    = re.compile(r'[​‌‍﻿­]')
# Layer 4: LaTeX subscript/superscript brace notation that appears literally
#   in some LaTeXML HTML text nodes, e.g. "n_{0}" or "H^{+}"
_LATEX_SCRIPT_RE  = re.compile(r'[_^]\{[^{}]*\}')
# Layer 5: bare LaTeX commands (with or without brace argument) that survive
#   as text in some arXiv HTML renderings, e.g. "\rho_{t}", "\mathcal{A}"
_RAW_LATEX_RE     = re.compile(r'\\[a-zA-Z]+(?:\{[^{}]*\})*')
# Collapse internal whitespace after all stripping
_WS_RE            = re.compile(r'\s+')

# ── Cross-reference patterns ───────────────────────────────────────────────────
# Fragment IDs from ltx_ref hrefs that identify equation tables.
# Examples: S2.E1, A4.EGx1, S3.E1.E2, A1.E12
# Heuristic: fragment contains ".E" followed by digits (possibly "Gx" before).
_EQ_FRAG_RE = re.compile(r'\.[Ee](?:Gx)?\d+')

# Lexical capture: equation number from prose patterns like
# "(3)", "Eq. (3)", "equation (3)", "Eqs. (3) and (4)"
_EQ_CITE_CAPTURE_RE = re.compile(
    r'(?:eq(?:uation)?s?\.?\s*)?'
    r'\(\s*(\d+[a-z]?(?:\.\d+)?)\s*\)',
    re.IGNORECASE,
)

# ── Thresholds / window constants ─────────────────────────────────────────────
MIN_SENT_LEN: int = 10   # discard sentences shorter than this
MAX_EVIDENCE_LEN: int = 300  # truncate XRef.from_sentence to this length


# ── Public data classes ────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A retrievable text unit from an arXiv paper.

    Attributes
    ----------
    chunk_id : str
        Unique identifier in the form
        ``"{arxiv_id}:{section_id}:{chunk_type}:{n}"``.
    arxiv_id : str
        arXiv paper identifier (e.g. ``"2401.13506"``).
    chunk_type : str
        One of ``"sentence"``, ``"paragraph"``, or
        ``"equation_neighborhood"``.
    text : str
        Cleaned plain text of this chunk.
    section_id : str
        ``id`` attribute of the innermost ``<section>`` containing this
        chunk (e.g. ``"S2.SS3"``).  Empty string for content outside
        any section.
    section_title : str
        Text of the section's ``ltx_title`` element, stripped of
        leading/trailing whitespace.
    paragraph_id : str
        ``id`` attribute of the containing ``div.ltx_para``
        (e.g. ``"S2.SS3.p1"``).
    eq_nums_nearby : list of str
        Equation numbers of ``ltx_eqn_table`` elements within the same
        ``div.ltx_para``.
    char_offset : int
        Character offset of this chunk's text within the cleaned
        full text of its source paragraph.  For paragraph and
        equation-neighborhood chunks this is always 0.
    source_endpoint : str
        The arXiv endpoint URL used to fetch the paper
        (e.g. ``"https://arxiv.org/html/2401.13506"``).
    """

    chunk_id:        str
    arxiv_id:        str
    chunk_type:      str
    text:            str
    section_id:      str
    section_title:   str
    paragraph_id:    str
    eq_nums_nearby:  List[str]
    char_offset:     int
    source_endpoint: str


@dataclass
class XRef:
    """One directed cross-reference edge from a paragraph to an equation.

    Attributes
    ----------
    from_para_id : str
        ``id`` of the ``div.ltx_para`` that contains the citation.
    from_sentence : str
        The citing sentence (verbatim, truncated to
        :data:`MAX_EVIDENCE_LEN` characters).
    to_eq_num : str
        Target equation number string (no parentheses).
    source : str
        Evidence source:
        ``"structural"``      — href fragment matched a table id,
        ``"structural_text"`` — href fragment unresolved; link text used,
        ``"lexical"``         — regex match on prose; no DOM link.
    """

    from_para_id:  str
    from_sentence: str
    to_eq_num:     str
    source:        str


@dataclass
class ParsedPaper:
    """All structural data and pre-built chunk views for one paper.

    Attributes
    ----------
    arxiv_id : str
    source_endpoint : str
    sentence_chunks : list of Chunk
        One chunk per sentence across all paragraphs.
    paragraph_chunks : list of Chunk
        One chunk per ``div.ltx_para``.
    eq_neighborhood_chunks : list of Chunk
        One chunk per equation: section title + preceding prose + LaTeX +
        following prose.
    cross_refs : list of XRef
        All detected equation cross-references, structural and lexical.
    eq_id_to_num : dict
        ``{table_id: eq_num}`` mapping built from ``ltx_eqn_table`` ids,
        exposed for downstream audit use.
    """

    arxiv_id:               str
    source_endpoint:        str
    sentence_chunks:        List[Chunk]
    paragraph_chunks:       List[Chunk]
    eq_neighborhood_chunks: List[Chunk]
    cross_refs:             List[XRef]
    eq_id_to_num:           Dict[str, str]


# ── Internal intermediate (not exported) ──────────────────────────────────────

@dataclass
class _ParaRecord:
    """Internal: cleaned paragraph data extracted from the DOM."""

    para_id:       str
    section_id:    str
    section_title: str
    full_text:     str        # cleaned prose, equations excluded
    eq_nums:       List[str]  # equation numbers whose tables live in this para


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip LaTeXML artifacts and Unicode noise from plain text.

    Applies six cleaning layers in order:

    1. LaTeXML artifact tokens (``italic_X``, ``start_POSTSUBSCRIPT``,
       bare ``superscript``/``subscript`` words left by earlier strip).
    2. Unicode mathematical alphanumeric block (U+1D400–U+1D7FF) and
       invisible operator characters (U+2061–U+2064).
    3. Zero-width characters (U+200B, U+200C, U+200D, U+FEFF, U+00AD).
    4. LaTeX subscript/superscript notation: ``_{t}``, ``^{2}`` etc. that
       appear as literal characters in some LaTeXML HTML text nodes.
    5. Bare LaTeX commands: ``\\rho``, ``\\mathcal{A}`` etc. that survive
       as text in some arXiv HTML renderings.
    6. Whitespace collapse (including non-breaking spaces).

    Parameters
    ----------
    text : str
        Raw text possibly containing LaTeXML rendering artifacts.

    Returns
    -------
    str
        Cleaned text with collapsed whitespace.
    """
    text = text.replace('\xa0', ' ')          # non-breaking space → space
    text = _ARTIFACT_RE.sub(' ', text)        # italic_x, start_ROW, …
    text = _MATH_UNICODE_RE.sub(' ', text)    # U+1D400–U+1D7FF
    text = _ZERO_WIDTH_RE.sub('', text)       # zero-width chars
    text = _LATEX_SCRIPT_RE.sub('', text)     # _{t}, ^{2}, etc.
    text = _RAW_LATEX_RE.sub('', text)        # \rho, \mathcal{A}, etc.
    text = _WS_RE.sub(' ', text)
    return text.strip()


# ── Sentence splitting with offset tracking ───────────────────────────────────

def _split_sentences(text: str) -> List[Tuple[str, int]]:
    """Split *text* into (sentence, char_offset) pairs.

    Uses the same ``_SENT_SPLIT_RE`` pattern as ``meaning.py`` so that
    sentence boundaries are consistent across the pipeline.  Offsets are
    character positions within *text* (before stripping), enabling
    provenance tracing back to the paragraph.

    Parameters
    ----------
    text : str
        Cleaned paragraph prose.

    Returns
    -------
    list of (str, int)
        Each tuple is ``(sentence_text, start_offset_in_text)``.
        Sentences shorter than :data:`MIN_SENT_LEN` are discarded.
    """
    # Collect split points
    split_pos: List[int] = [0]
    for m in _SENT_SPLIT_RE.finditer(text):
        split_pos.append(m.end())
    split_pos.append(len(text))

    result: List[Tuple[str, int]] = []
    for i in range(len(split_pos) - 1):
        start = split_pos[i]
        end   = split_pos[i + 1]
        sent  = text[start:end].strip()
        if len(sent) >= MIN_SENT_LEN:
            result.append((sent, start))
    return result


# ── DocumentParser ─────────────────────────────────────────────────────────────

class DocumentParser:
    """Parse arXiv HTML into structured chunk views and cross-references.

    Build one instance for the entire run (stateless, reusable).  Call
    :meth:`parse` once per paper; the returned :class:`ParsedPaper`
    holds all three chunk views and the cross-reference graph.

    Examples
    --------
    >>> parser = DocumentParser()
    >>> paper  = parser.parse(html_bytes, "2401.13506",
    ...                       "https://arxiv.org/html/2401.13506",
    ...                       equations)
    >>> len(paper.sentence_chunks)
    843
    >>> paper.cross_refs[0].source
    'structural'
    """

    # ── public entry point ─────────────────────────────────────────────────────

    def parse(
        self,
        html_bytes:      bytes,
        arxiv_id:        str,
        source_endpoint: str,
        equations:       Dict[str, Dict],
    ) -> ParsedPaper:
        """Parse one arXiv HTML paper into all chunk views.

        Parameters
        ----------
        html_bytes : bytes
            Raw HTML content of the paper (UTF-8 or Latin-1 encoded).
        arxiv_id : str
            arXiv identifier, e.g. ``"2401.13506"``.
        source_endpoint : str
            URL used to fetch the paper, e.g.
            ``"https://arxiv.org/html/2401.13506"``.
        equations : dict
            ``{eq_num: {"equation": latex_str, ...}}`` as produced by
            ``EquationExtractor``.  Used to embed LaTeX in
            equation-neighborhood chunks.

        Returns
        -------
        ParsedPaper
            Contains sentence, paragraph, and equation-neighborhood chunk
            views plus the cross-reference graph.  All lists are in
            document order (deterministic).
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "BeautifulSoup4 is required for DocumentParser: "
                "pip install beautifulsoup4 lxml"
            ) from exc

        soup = BeautifulSoup(html_bytes, "lxml")

        # Step 1 — equation table id → number map
        eq_id_to_num: Dict[str, str] = self._build_eq_id_map(soup)

        # Step 2 — paragraph records (section + cleaned prose + nearby eqs)
        para_records: List[_ParaRecord] = self._collect_paragraphs(
            soup, eq_id_to_num
        )

        # Step 3 — three chunk views
        chunk_counter = [0]  # mutable int in a list so closures can update it
        sentence_chunks  = self._make_sentence_chunks(
            para_records, arxiv_id, source_endpoint, chunk_counter
        )
        paragraph_chunks = self._make_paragraph_chunks(
            para_records, arxiv_id, source_endpoint, chunk_counter
        )
        eq_nbhd_chunks   = self._make_eq_neighborhood_chunks(
            soup, equations, eq_id_to_num, arxiv_id, source_endpoint, chunk_counter
        )

        # Step 4 — cross-reference graph
        cross_refs = self._make_cross_refs(soup, eq_id_to_num, para_records)

        logger.info(
            "DocumentParser: %s  paras=%d  sent=%d  para_chunks=%d  "
            "eq_nbhd=%d  xrefs=%d",
            arxiv_id, len(para_records), len(sentence_chunks),
            len(paragraph_chunks), len(eq_nbhd_chunks), len(cross_refs),
        )
        return ParsedPaper(
            arxiv_id               = arxiv_id,
            source_endpoint        = source_endpoint,
            sentence_chunks        = sentence_chunks,
            paragraph_chunks       = paragraph_chunks,
            eq_neighborhood_chunks = eq_nbhd_chunks,
            cross_refs             = cross_refs,
            eq_id_to_num           = eq_id_to_num,
        )

    # ── named chunk-view methods (operate on pre-built ParsedPaper) ───────────

    def build_sentence_chunks(self, paper: ParsedPaper) -> List[Chunk]:
        """Return the pre-built sentence-level chunks for *paper*.

        Parameters
        ----------
        paper : ParsedPaper

        Returns
        -------
        list of Chunk
        """
        return paper.sentence_chunks

    def build_paragraph_chunks(self, paper: ParsedPaper) -> List[Chunk]:
        """Return the pre-built paragraph-level chunks for *paper*.

        Parameters
        ----------
        paper : ParsedPaper

        Returns
        -------
        list of Chunk
        """
        return paper.paragraph_chunks

    def build_equation_neighborhood_chunks(
        self, paper: ParsedPaper
    ) -> List[Chunk]:
        """Return the pre-built equation-neighborhood chunks for *paper*.

        Each chunk spans: section title + preceding prose + equation LaTeX +
        following prose, all within the same ``div.ltx_para``.

        Parameters
        ----------
        paper : ParsedPaper

        Returns
        -------
        list of Chunk
        """
        return paper.eq_neighborhood_chunks

    def build_cross_reference_index(self, paper: ParsedPaper) -> List[XRef]:
        """Return the pre-built cross-reference graph for *paper*.

        Parameters
        ----------
        paper : ParsedPaper

        Returns
        -------
        list of XRef
        """
        return paper.cross_refs

    def symbol_chunk_view(
        self,
        paper:        ParsedPaper,
        symbol_forms: List[str],
    ) -> List[Chunk]:
        """Return chunks whose text contains any surface form of a symbol.

        Searches all three chunk views and deduplicates by ``chunk_id``.
        Case-sensitive for single-letter symbols (``H`` vs ``h``); the
        caller should pass all relevant forms.

        Parameters
        ----------
        paper : ParsedPaper
            Pre-built paper views to search.
        symbol_forms : list of str
            Surface forms to match: ASCII name (``"H"``), LaTeX command
            (``"\\psi"``), Unicode (``"ψ"``), or subscripted form
            (``"omega_c"``).  Each form is matched as a substring.

        Returns
        -------
        list of Chunk
            Matching chunks in document order (sentence_chunks first,
            then paragraph_chunks, then eq_neighborhood_chunks).
            Deduplicated by ``chunk_id``.
        """
        if not symbol_forms:
            return []

        seen_ids: set  = set()
        result:   List[Chunk] = []
        all_chunks = (
            paper.sentence_chunks
            + paper.paragraph_chunks
            + paper.eq_neighborhood_chunks
        )
        for chunk in all_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            t = chunk.text
            # Case-sensitive: "H" ≠ "h" for single-letter physics symbols
            if any(f in t for f in symbol_forms):
                result.append(chunk)
                seen_ids.add(chunk.chunk_id)
        return result

    # ── internal: DOM traversal helpers ───────────────────────────────────────

    @staticmethod
    def _build_eq_id_map(soup: object) -> Dict[str, str]:
        """Map equation table ``id`` attribute → equation number string.

        Strips surrounding parentheses from the number so the map values
        match the keys used in the ``equations`` dict from
        ``EquationExtractor``.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.

        Returns
        -------
        dict
            ``{"S2.E1": "1", "A4.EGx1": "2", ...}``
        """
        result: Dict[str, str] = {}
        for table in soup.select("table.ltx_eqn_table"):
            tid  = table.get("id")
            span = table.find(class_="ltx_tag_equation")
            if tid and span:
                num = span.get_text().strip().strip("()")
                result[tid] = num
        return result

    @staticmethod
    def _section_info(para) -> Tuple[str, str]:
        """Return (section_id, section_title) for the innermost parent section.

        Uses ``find_parent('section')`` so the innermost (most specific)
        section is returned.  Title is the text of the first direct child
        element with class ``ltx_title``.

        Parameters
        ----------
        para : Tag
            A ``div.ltx_para`` BeautifulSoup tag.

        Returns
        -------
        tuple of (str, str)
            ``("", "")`` when no parent section exists.
        """
        sec = para.find_parent("section")
        if sec is None:
            return ("", "")
        sec_id = sec.get("id", "")
        # Title is a direct child element — don't recurse into sub-sections
        title_tag = sec.find(class_="ltx_title", recursive=False)
        title = title_tag.get_text().strip() if title_tag else ""
        return (sec_id, title)

    @staticmethod
    def _para_eq_nums(para, eq_id_to_num: Dict[str, str]) -> List[str]:
        """Return equation numbers for tables that are direct children of *para*.

        Only tables that are direct children (``recursive=False``) are
        included, so nested structures don't bleed into the wrong paragraph.

        Parameters
        ----------
        para : Tag
        eq_id_to_num : dict

        Returns
        -------
        list of str
            Equation number strings in document order, deduplicated.
        """
        nums: List[str] = []
        seen: set       = set()
        for child in para.children:
            if not hasattr(child, "get"):
                continue  # NavigableString
            classes = child.get("class") or []
            if "ltx_eqn_table" in classes:
                tid = child.get("id", "")
                num = eq_id_to_num.get(tid)
                if num and num not in seen:
                    nums.append(num)
                    seen.add(num)
        return nums

    @staticmethod
    def _para_prose_text(para) -> str:
        """Extract cleaned prose from a ``div.ltx_para`` element.

        Collects text only from ``p.ltx_p`` elements that are direct
        children of *para* (``recursive=False``), skipping equation tables
        and other non-prose content.  The texts are joined with a single
        space and cleaned with :func:`_clean_text`.

        Parameters
        ----------
        para : Tag
            A ``div.ltx_para`` BeautifulSoup tag.

        Returns
        -------
        str
            Cleaned prose text.  Empty string if no ``p.ltx_p`` children.
        """
        p_texts: List[str] = []
        for child in para.children:
            if not hasattr(child, "get"):
                continue
            classes = child.get("class") or []
            if child.name == "p" and "ltx_p" in classes:
                p_texts.append(child.get_text())
        return _clean_text(" ".join(p_texts))

    # ── internal: paragraph collection ────────────────────────────────────────

    def _collect_paragraphs(
        self,
        soup:         object,
        eq_id_to_num: Dict[str, str],
    ) -> List[_ParaRecord]:
        """Walk all ``div.ltx_para`` elements and build para records.

        Parameters
        ----------
        soup : BeautifulSoup
        eq_id_to_num : dict

        Returns
        -------
        list of _ParaRecord
            In document order.
        """
        records: List[_ParaRecord] = []
        for para in soup.select("div.ltx_para"):
            para_id = para.get("id", "")
            if not para_id:
                continue  # skip paras without an id (uncommon; defensive)

            sec_id, sec_title = self._section_info(para)
            eq_nums           = self._para_eq_nums(para, eq_id_to_num)
            full_text         = self._para_prose_text(para)

            if not full_text:
                continue  # equation-only paras have no prose

            records.append(_ParaRecord(
                para_id       = para_id,
                section_id    = sec_id,
                section_title = sec_title,
                full_text     = full_text,
                eq_nums       = eq_nums,
            ))
        return records

    # ── internal: chunk view builders ─────────────────────────────────────────

    def _make_sentence_chunks(
        self,
        para_records:    List[_ParaRecord],
        arxiv_id:        str,
        source_endpoint: str,
        counter:         List[int],
    ) -> List[Chunk]:
        """Build one sentence chunk per sentence across all paragraphs.

        Parameters
        ----------
        para_records : list of _ParaRecord
        arxiv_id : str
        source_endpoint : str
        counter : list of int
            Single-element mutable counter shared with other builders so
            all chunk_ids in the paper are globally unique.

        Returns
        -------
        list of Chunk
        """
        chunks: List[Chunk] = []
        for rec in para_records:
            for sent_text, offset in _split_sentences(rec.full_text):
                n          = counter[0]; counter[0] += 1
                chunk_id   = f"{arxiv_id}:{rec.section_id}:sentence:{n}"
                chunks.append(Chunk(
                    chunk_id        = chunk_id,
                    arxiv_id        = arxiv_id,
                    chunk_type      = "sentence",
                    text            = sent_text,
                    section_id      = rec.section_id,
                    section_title   = rec.section_title,
                    paragraph_id    = rec.para_id,
                    eq_nums_nearby  = list(rec.eq_nums),
                    char_offset     = offset,
                    source_endpoint = source_endpoint,
                ))
        return chunks

    def _make_paragraph_chunks(
        self,
        para_records:    List[_ParaRecord],
        arxiv_id:        str,
        source_endpoint: str,
        counter:         List[int],
    ) -> List[Chunk]:
        """Build one paragraph chunk per ``div.ltx_para``.

        Parameters
        ----------
        para_records : list of _ParaRecord
        arxiv_id : str
        source_endpoint : str
        counter : list of int

        Returns
        -------
        list of Chunk
        """
        chunks: List[Chunk] = []
        for rec in para_records:
            n        = counter[0]; counter[0] += 1
            chunk_id = f"{arxiv_id}:{rec.section_id}:paragraph:{n}"
            chunks.append(Chunk(
                chunk_id        = chunk_id,
                arxiv_id        = arxiv_id,
                chunk_type      = "paragraph",
                text            = rec.full_text,
                section_id      = rec.section_id,
                section_title   = rec.section_title,
                paragraph_id    = rec.para_id,
                eq_nums_nearby  = list(rec.eq_nums),
                char_offset     = 0,
                source_endpoint = source_endpoint,
            ))
        return chunks

    def _make_eq_neighborhood_chunks(
        self,
        soup:            object,
        equations:       Dict[str, Dict],
        eq_id_to_num:    Dict[str, str],
        arxiv_id:        str,
        source_endpoint: str,
        counter:         List[int],
    ) -> List[Chunk]:
        """Build one neighborhood chunk per equation.

        Each chunk contains: section_title + preceding prose in the same
        ``div.ltx_para`` + equation LaTeX + following prose in that para.
        This is the default retrieval unit for meaning and relations.

        Only equations present in *equations* (from ``EquationExtractor``)
        are included.  Equations in the document but not in *equations*
        (e.g. beyond the 7-per-paper cap) are skipped.

        Parameters
        ----------
        soup : BeautifulSoup
        equations : dict
            ``{eq_num: {"equation": str, ...}}`` from EquationExtractor.
        eq_id_to_num : dict
        arxiv_id : str
        source_endpoint : str
        counter : list of int

        Returns
        -------
        list of Chunk
        """
        # Build reverse map: eq_num → table element (first occurrence)
        num_to_table: Dict[str, object] = {}
        for table in soup.select("table.ltx_eqn_table"):
            tid = table.get("id", "")
            num = eq_id_to_num.get(tid)
            if num and num not in num_to_table:
                num_to_table[num] = table

        chunks: List[Chunk] = []
        # Iterate in sorted equation-number order for determinism
        for eq_num in sorted(equations.keys(), key=lambda x: (
            (0, int(x)) if x.isdigit() else (1, 0)
        )):
            table = num_to_table.get(eq_num)
            if table is None:
                continue  # this equation wasn't in the HTML

            para = table.find_parent("div", class_="ltx_para")
            if para is None:
                continue

            sec_id, sec_title = self._section_info(para)
            eq_nums_in_para   = self._para_eq_nums(para, eq_id_to_num)
            latex             = equations[eq_num].get("equation", "")

            # Split prose in the containing para around the equation table
            before_texts: List[str] = []
            after_texts:  List[str] = []
            found_table            = False

            for child in para.children:
                if not hasattr(child, "get"):
                    continue  # NavigableString
                classes = child.get("class") or []
                if child is table:
                    found_table = True
                elif child.name == "p" and "ltx_p" in classes:
                    text_raw = _clean_text(child.get_text())
                    if found_table:
                        after_texts.append(text_raw)
                    else:
                        before_texts.append(text_raw)

            # Compose neighborhood text: title, before, latex, after
            # Each part separated by a newline for readability.
            parts: List[str] = []
            if sec_title:
                parts.append(sec_title)
            if before_texts:
                parts.append(" ".join(before_texts))
            if latex:
                parts.append(latex)
            if after_texts:
                parts.append(" ".join(after_texts))
            nbhd_text = " ".join(p for p in parts if p)

            if not nbhd_text:
                continue

            n        = counter[0]; counter[0] += 1
            chunk_id = f"{arxiv_id}:{sec_id}:equation_neighborhood:{n}"
            chunks.append(Chunk(
                chunk_id        = chunk_id,
                arxiv_id        = arxiv_id,
                chunk_type      = "equation_neighborhood",
                text            = nbhd_text,
                section_id      = sec_id,
                section_title   = sec_title,
                paragraph_id    = para.get("id", ""),
                eq_nums_nearby  = eq_nums_in_para,
                char_offset     = 0,
                source_endpoint = source_endpoint,
            ))
        return chunks

    # ── internal: cross-reference graph ───────────────────────────────────────

    def _make_cross_refs(
        self,
        soup:         object,
        eq_id_to_num: Dict[str, str],
        para_records: List[_ParaRecord],
    ) -> List[XRef]:
        """Build the cross-reference graph from structural DOM links and lexical patterns.

        Two passes:

        **Pass 1 — structural** (preferred):
        Scans all ``<a class="ltx_ref">`` elements whose ``href`` fragment
        looks like an equation reference.  Each such link is resolved to an
        equation number via *eq_id_to_num*; when the fragment is absent from
        the map the link's visible text is used instead
        (``source="structural_text"``).

        **Pass 2 — lexical** (fallback):
        Applies :data:`_EQ_CITE_CAPTURE_RE` to every sentence in every
        paragraph that was not already covered by a structural edge in the
        same direction.

        Per-edge logging records which source produced each ``XRef``.

        Parameters
        ----------
        soup : BeautifulSoup
        eq_id_to_num : dict
        para_records : list of _ParaRecord

        Returns
        -------
        list of XRef
            In document order; structural edges before lexical edges.
        """
        xrefs: List[XRef] = []
        # Track (from_para_id, to_eq_num) pairs already covered structurally
        structural_covered: set = set()

        # ── pass 1: structural ────────────────────────────────────────────────
        for a in soup.select("a.ltx_ref"):
            href = a.get("href", "")
            if "#" not in href:
                continue

            frag = href.rsplit("#", 1)[-1]

            # Skip non-equation refs: sections, figures, tables, etc.
            # Equation fragments contain ".E" followed by digits
            if not _EQ_FRAG_RE.search(frag):
                continue

            # Try id-map resolution first
            resolved = eq_id_to_num.get(frag)
            if resolved is not None:
                source = "structural"
            else:
                # Fall back to visible link text when fragment not in map.
                # arXiv LaTeXML equation refs always render the bare number.
                link_text = a.get_text().strip()
                if re.fullmatch(r'\d+[a-z]?(?:\.\d+)?', link_text):
                    resolved = link_text
                    source   = "structural_text"
                else:
                    continue  # unresolvable — skip

            # Find containing paragraph and sentence
            para     = a.find_parent("div", class_="ltx_para")
            para_id  = para.get("id", "") if para else ""
            p_elem   = a.find_parent("p", class_="ltx_p")
            evidence = (
                _clean_text(p_elem.get_text())[:MAX_EVIDENCE_LEN]
                if p_elem else ""
            )

            xref = XRef(
                from_para_id  = para_id,
                from_sentence = evidence,
                to_eq_num     = resolved,
                source        = source,
            )
            xrefs.append(xref)
            structural_covered.add((para_id, resolved))
            logger.debug(
                "xref structural: para=%r → eq=%r  src=%s  frag=%r",
                para_id, resolved, source, frag,
            )

        # ── pass 2: lexical fallback ───────────────────────────────────────────
        for rec in para_records:
            for sent_text, _ in _split_sentences(rec.full_text):
                for m in _EQ_CITE_CAPTURE_RE.finditer(sent_text):
                    eq_num = m.group(1)
                    edge   = (rec.para_id, eq_num)
                    if edge in structural_covered:
                        continue  # already have a structural edge for this pair
                    xrefs.append(XRef(
                        from_para_id  = rec.para_id,
                        from_sentence = sent_text[:MAX_EVIDENCE_LEN],
                        to_eq_num     = eq_num,
                        source        = "lexical",
                    ))
                    structural_covered.add(edge)  # one lexical edge per pair
                    logger.debug(
                        "xref lexical: para=%r → eq=%r  sent=%r",
                        rec.para_id, eq_num, sent_text[:60],
                    )

        return xrefs


# ── Backward-compatibility shim for Retriever ─────────────────────────────────

def chunk_to_retriever_dict(chunk: Chunk) -> Dict:
    """Convert a :class:`Chunk` to the dict format expected by :class:`~src.retrieval.Retriever`.

    Parameters
    ----------
    chunk : Chunk
        A chunk from any of the three chunk views.

    Returns
    -------
    dict
        ``{"id": chunk_id, "text": text, "_chunk": chunk}``.
        The ``"_chunk"`` key preserves the original object for provenance.
    """
    return {
        "id":     chunk.chunk_id,
        "text":   chunk.text,
        "_chunk": chunk,
    }


def chunks_for_retriever(chunks: List[Chunk]) -> List[Dict]:
    """Convert a list of :class:`Chunk` objects to retriever-compatible dicts.

    Parameters
    ----------
    chunks : list of Chunk

    Returns
    -------
    list of dict
    """
    return [chunk_to_retriever_dict(c) for c in chunks]
