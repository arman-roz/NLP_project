"""
src/extraction.py

Equation extraction from arXiv HTML (primary) and PDF (fallback).

HTML path  — parse LaTeXML-generated HTML with BeautifulSoup.  For each
             numbered equation (identified by a <span class="ltx_tag_equation">),
             retrieve LaTeX in priority order:
               1. <annotation encoding="application/x-tex"> — original LaTeX
               2. alttext attribute on <math>               — equivalent copy
               3. Recursive MathML-to-LaTeX conversion      — lossy fallback
             surrounding paragraph text is also captured for use by
             meaning.py and symbols.py.

PDF path   — use PyMuPDF (fitz) page.get_text().  Detect numbered equations
             by the right-margin pattern "(N)" and collect the text on the
             same line.  LaTeX quality is low (rendered Unicode, no structure),
             but equation numbers are reliably found.

Limit: first MAX_EQUATIONS_PER_PAPER numbered equations per paper.
Every paper returns a dict (empty if no numbered equations found).

No LLM, no external API, no text generation — only parsing and regex.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from bs4 import BeautifulSoup, NavigableString, Tag

from .audit import AuditTrail

logger = logging.getLogger(__name__)

# Maximum numbered equations to extract per paper (spec requirement)
MAX_EQUATIONS_PER_PAPER: int = 7

# ── MathML identifier → LaTeX command (Greek letters + special symbols) ───────
# Used when converting <mi> elements during MathML-to-LaTeX fallback.
_MI_MAP: Dict[str, str] = {
    # lowercase Greek
    "α": r"\alpha",    "β": r"\beta",    "γ": r"\gamma",   "δ": r"\delta",
    "ε": r"\epsilon",  "ζ": r"\zeta",    "η": r"\eta",     "θ": r"\theta",
    "ι": r"\iota",     "κ": r"\kappa",   "λ": r"\lambda",  "μ": r"\mu",
    "ν": r"\nu",       "ξ": r"\xi",      "π": r"\pi",      "ρ": r"\rho",
    "σ": r"\sigma",    "τ": r"\tau",     "υ": r"\upsilon",  "φ": r"\phi",
    "χ": r"\chi",      "ψ": r"\psi",     "ω": r"\omega",
    # uppercase Greek
    "Γ": r"\Gamma",    "Δ": r"\Delta",   "Θ": r"\Theta",   "Λ": r"\Lambda",
    "Ξ": r"\Xi",       "Π": r"\Pi",      "Σ": r"\Sigma",   "Υ": r"\Upsilon",
    "Φ": r"\Phi",      "Ψ": r"\Psi",     "Ω": r"\Omega",
    # special identifiers
    "ℏ": r"\hbar",     "ħ": r"\hbar",    "∞": r"\infty",
}

# ── MathML operator → LaTeX command ──────────────────────────────────────────
# Used when converting <mo> elements.  Invisible operators are dropped.
_MO_MAP: Dict[str, str] = {
    "∂": r"\partial",   "∇": r"\nabla",    "∞": r"\infty",
    "∫": r"\int",       "∮": r"\oint",     "∑": r"\sum",     "∏": r"\prod",
    "≤": r"\leq",       "≥": r"\geq",      "≠": r"\neq",     "≈": r"\approx",
    "∈": r"\in",        "∉": r"\notin",    "⊂": r"\subset",  "⊃": r"\supset",
    "∪": r"\cup",       "∩": r"\cap",      "∧": r"\wedge",   "∨": r"\vee",
    "→": r"\rightarrow","←": r"\leftarrow","↔": r"\leftrightarrow",
    "⟨": r"\langle",   "⟩": r"\rangle",
    "†": r"\dagger",    "‡": r"\ddagger",  "×": r"\times",   "÷": r"\div",
    "±": r"\pm",        "∓": r"\mp",       "·": r"\cdot",    "⋅": r"\cdot",
    "∝": r"\propto",    "∼": r"\sim",      "≃": r"\simeq",   "≡": r"\equiv",
    "⊗": r"\otimes",    "⊕": r"\oplus",   "‖": r"\|",
    # invisible operators (U+2062 invisible times, U+2063 invisible sep) → drop
    "⁢": "",        "⁣": "",      "⁡": "",
}

# ── mover accent characters → LaTeX accent command ────────────────────────────
_ACCENT_MAP: Dict[str, str] = {
    "^": r"\hat",   "ˆ": r"\hat",
    "¯": r"\bar",   "‾": r"\bar",
    "˙": r"\dot",   "˜": r"\tilde",  "~": r"\tilde",
    "⃗": r"\vec",   "→": r"\vec",
    "̈": r"\ddot",
}

# Regex to identify a printed equation number like "(1)", "(A.2)", "(1a)"
_EQ_NUMBER_RE = re.compile(
    r"^\s*\(\s*([\dA-Za-z]+(?:[.\-][\dA-Za-z]+)*)\s*\)\s*$"
)

# Regex for finding equation numbers in raw PDF text lines
_PDF_EQ_LINE_RE = re.compile(
    r"^(.*?)\s{2,}\(\s*([\dA-Za-z]+(?:[.\-][\dA-Za-z]+)*)\s*\)\s*$"
)


# ── EquationExtractor ─────────────────────────────────────────────────────────

class EquationExtractor:
    """Extract numbered equations from arXiv HTML or PDF content.

    Tries the HTML path first (rich MathML → LaTeX conversion),
    falls back to PDF text extraction when HTML is unavailable.

    Parameters
    ----------
    max_equations : int, optional
        Maximum numbered equations to keep per paper.  Default: 7.

    Examples
    --------
    >>> extractor = EquationExtractor()
    >>> audit = AuditTrail()
    >>> equations = extractor.extract(fetch_result, audit)
    >>> equations["1"]["equation"]
    'E = mc^{2}'
    """

    def __init__(self, max_equations: int = MAX_EQUATIONS_PER_PAPER) -> None:
        self.max_equations = max_equations

    # ── public interface ───────────────────────────────────────────────────────

    def extract(self, fetch_result: dict, audit: AuditTrail) -> dict:
        """Extract numbered equations from a fetched paper.

        Dispatches to the HTML or PDF path based on the source recorded
        by the Fetcher.  Always returns a dict; empty if no numbered
        equations are found or if the paper could not be fetched.

        Parameters
        ----------
        fetch_result : dict
            Dict returned by ``Fetcher.fetch_paper``.  Must contain keys
            ``'content'``, ``'source'``, and ``'arxiv_id'``.
        audit : AuditTrail
            Paper-level audit log; receives one summary entry.

        Returns
        -------
        dict
            ``{eq_number: {equation, source, latex_method,
            context_before, context_after, _audit}}``
            The ``_audit`` value is an :class:`AuditTrail` instance that
            downstream modules (meaning, symbols, relations) append to.
            ``context_before`` and ``context_after`` are stripped before
            the final JSON is written.
        """
        arxiv_id = fetch_result.get("arxiv_id", "unknown")
        source = fetch_result.get("source", "none")
        content = fetch_result.get("content")

        # paper fetched as 'none' → return empty dict (paper still gets a JSON key)
        if source == "none" or content is None:
            audit.log("extract", f"{arxiv_id}: source=none, returning empty equations")
            return {}

        if source == "html":
            equations = self._extract_from_html(content, arxiv_id, audit)
        else:
            equations = self._extract_from_pdf(content, arxiv_id, audit)

        audit.log(
            "extract",
            f"{arxiv_id}: source={source}, found {len(equations)} numbered equation(s)",
        )
        return equations

    # ── HTML extraction ────────────────────────────────────────────────────────

    def _extract_from_html(
        self, html_bytes: bytes, arxiv_id: str, audit: AuditTrail
    ) -> dict:
        """Extract equations from arXiv LaTeXML HTML content.

        Strategy (top-down from equation containers):
          1. Parse with BeautifulSoup (lxml parser).
          2. Find every ``<table class="ltx_eqn_table">`` — the universal
             container for all numbered equations in LaTeXML HTML.
          3. Three sub-cases based on table classes and number-tag type:

             a. ``ltx_equation`` table — standalone equation.
                The number tag (``ltx_tag_equation``) and the ``<math>``
                element may be in different ``<tr>`` rows due to rowspan
                rendering; searching at table level always finds the math.

             b. ``ltx_equationgroup`` table, one row per equation —
                eqnarray / align / gather with individually numbered rows.
                Each ``<tr>`` is scoped independently; if a row's number td
                has ``rowspan="0"`` (number placed below the content row),
                the fallback searches sibling rows within the same ``<tbody>``.

             c. ``ltx_equationgroup`` table, shared group number —
                a single ``ltx_tag_equationgroup`` span labels all rows.
                All ``<math>`` elements are collected and their LaTeX joined
                with ``\\\\`` so the group is stored as one equation entry.

          4. LaTeX is cleaned (``\\displaystyle`` prefix and ``%\\n``
             line-continuation artefacts stripped) before storage.
          5. Surrounding paragraph text is captured for downstream modules.

        Parameters
        ----------
        html_bytes : bytes
            Raw HTML bytes from cache.
        arxiv_id : str
            Used only for logging.
        audit : AuditTrail
            Receives one entry per equation found.

        Returns
        -------
        dict
            Equation dicts keyed by equation number string.
        """
        try:
            soup = BeautifulSoup(html_bytes, "lxml")
        except Exception as exc:
            audit.log("extract_html", f"{arxiv_id}: parse error — {exc}")
            logger.error("HTML parse error for %s: %s", arxiv_id, exc)
            return {}

        eq_tables = soup.find_all("table", class_="ltx_eqn_table")
        audit.log("extract_html", f"{arxiv_id}: found {len(eq_tables)} equation tables")

        equations: dict = {}

        for table in eq_tables:
            if len(equations) >= self.max_equations:
                break

            table_classes = table.get("class", [])

            if "ltx_equationgroup" in table_classes:
                grp_tag = table.find("span", class_="ltx_tag_equationgroup")
                if grp_tag:
                    # case (c): whole group carries one shared number
                    self._record_group_equation(table, grp_tag, equations, audit)
                else:
                    # case (b): individually numbered rows within the group
                    for tr in table.find_all("tr"):
                        if len(equations) >= self.max_equations:
                            break
                        self._record_row_equation(tr, table, equations, audit)
            else:
                # case (a): ltx_equation — standalone, one math per table
                self._record_standalone_equation(table, equations, audit)

        return equations

    def _record_standalone_equation(
        self, table: Tag, equations: dict, audit: AuditTrail
    ) -> None:
        """Handle a ``table.ltx_equation`` (standalone numbered equation).

        Searches for the number tag at table level, then the ``<math>``
        element at table level — so rowspan placement of the number in a
        separate row from the content is handled transparently.
        """
        tag = table.find("span", class_="ltx_tag_equation")
        if tag is None:
            return
        eq_num = self._parse_eq_number(tag.get_text())
        if eq_num is None or eq_num in equations:
            return
        math_elem = table.find("math")
        if math_elem is None:
            audit.log("extract_html", f"eq ({eq_num}): standalone table has no <math>, skipped")
            return
        self._store_equation(eq_num, math_elem, table, equations, audit)

    def _record_row_equation(
        self, tr: Tag, table: Tag, equations: dict, audit: AuditTrail
    ) -> None:
        """Handle one numbered row inside a ``table.ltx_equationgroup``.

        LaTeXML sometimes places the number ``<td rowspan="0">`` in a
        separate ``<tbody>`` from the content rows (e.g. when the equation
        spans multiple display lines).  When the number row has no
        ``<math>``, this method searches backwards through all rows in the
        parent table to find the nearest preceding row that contains math —
        that row holds the content belonging to this equation number.
        """
        tag = tr.find("span", class_="ltx_tag_equation")
        if tag is None:
            return
        eq_num = self._parse_eq_number(tag.get_text())
        if eq_num is None or eq_num in equations:
            return

        math_elem = tr.find("math")
        if math_elem is None:
            # rowspan issue: the number td is in a different tbody from the
            # content.  Walk backwards through all rows in the table to find
            # the nearest preceding row that has a <math> element.
            all_rows = table.find_all("tr")
            tr_idx = next((i for i, r in enumerate(all_rows) if r is tr), -1)
            for i in range(tr_idx - 1, -1, -1):
                m = all_rows[i].find("math")
                if m:
                    math_elem = m
                    break

        if math_elem is None:
            audit.log("extract_html", f"eq ({eq_num}): no <math> in row or preceding rows, skipped")
            return
        self._store_equation(eq_num, math_elem, table, equations, audit)

    def _record_group_equation(
        self, table: Tag, grp_tag: Tag, equations: dict, audit: AuditTrail
    ) -> None:
        """Handle a ``table.ltx_equationgroup`` with a single shared number.

        Collects LaTeX from every row in the group and joins them with
        ``\\\\`` so the complete multi-line system is stored as one entry.
        """
        eq_num = self._parse_eq_number(grp_tag.get_text())
        if eq_num is None or eq_num in equations:
            return
        math_elems = table.find_all("math")
        if not math_elems:
            audit.log("extract_html", f"eq ({eq_num}): equationgroup has no <math>, skipped")
            return

        parts: List[str] = []
        best_method = "mathml"
        for m in math_elems:
            latex, method = self._get_latex_from_math(m)
            latex = self._clean_latex(latex)
            if latex:
                parts.append(latex)
            if method == "annotation":
                best_method = "annotation"
            elif method == "alttext" and best_method != "annotation":
                best_method = "alttext"

        latex = " \\\\\n".join(parts)
        context_before, context_after = self._get_surrounding_text_html(table)
        eq_audit = AuditTrail()
        eq_audit.log(
            "extract_html",
            (
                f"found eq ({eq_num}), source=html, latex_method={best_method} "
                f"(group {len(parts)} rows), latex_preview={latex[:60]!r}"
            ),
        )
        equations[eq_num] = {
            "equation": latex,
            "source": "html",
            "latex_method": best_method,
            "context_before": context_before,
            "context_after": context_after,
            "_audit": eq_audit,
        }
        logger.debug("HTML eq (%s): method=%s (group), latex=%r", eq_num, best_method, latex[:50])

    def _store_equation(
        self,
        eq_num: str,
        math_elem: Tag,
        table: Tag,
        equations: dict,
        audit: AuditTrail,
    ) -> None:
        """Extract LaTeX from a math element and store the equation dict."""
        latex, latex_method = self._get_latex_from_math(math_elem)
        latex = self._clean_latex(latex)
        context_before, context_after = self._get_surrounding_text_html(table)
        eq_audit = AuditTrail()
        eq_audit.log(
            "extract_html",
            (
                f"found eq ({eq_num}), source=html, "
                f"latex_method={latex_method}, "
                f"latex_preview={latex[:60]!r}"
            ),
        )
        equations[eq_num] = {
            "equation": latex,
            "source": "html",
            "latex_method": latex_method,
            "context_before": context_before,
            "context_after": context_after,
            "_audit": eq_audit,
        }
        logger.debug("HTML eq (%s): method=%s, latex=%r", eq_num, latex_method, latex[:50])

    # ── PDF extraction ─────────────────────────────────────────────────────────

    def _extract_from_pdf(
        self, pdf_bytes: bytes, arxiv_id: str, audit: AuditTrail
    ) -> dict:
        """Extract equations from PDF content using PyMuPDF.

        Strategy:
          1. Open PDF from bytes with ``fitz.open``.
          2. Extract text from each page with ``page.get_text()``.
          3. Find lines that end with a right-aligned equation number
             pattern ``(N)``.
          4. Two content layouts are handled:

             a. Inline layout — the equation is on the same line as the
                number (e.g. ``E = mc²   (1)``).  The text to the left of
                the number is used directly.

             b. Multi-line layout — the equation spans several lines and
                only the terminating line carries the number (the content
                before the number is just ``,`` or ``.``).  In this case
                the method walks backwards from the number line to collect
                equation-body lines, stopping when it reaches regular prose
                text (a line with ≥ 3 space-separated words each ≥ 4 chars).

          5. Context is taken from the two prose lines immediately before
             and after the equation block.

        Quality note:
            PDF extraction yields rendered Unicode (e.g. ``ψ``, ``∇``) rather
            than LaTeX commands.  This is an inherent limitation of reading
            PDF without the LaTeX source.  Recorded in the audit trail.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF bytes from cache.
        arxiv_id : str
            Used only for logging.
        audit : AuditTrail
            Receives one entry per equation found.

        Returns
        -------
        dict
            Equation dicts keyed by equation number string.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            audit.log("extract_pdf", f"{arxiv_id}: open error — {exc}")
            logger.error("PDF open error for %s: %s", arxiv_id, exc)
            return {}

        equations: dict = {}
        all_lines: List[Tuple[str, int]] = []  # (line_text, page_num)

        for page_num, page in enumerate(doc):
            for line in page.get_text().splitlines():
                all_lines.append((line, page_num + 1))

        doc.close()
        audit.log("extract_pdf", f"{arxiv_id}: {len(all_lines)} lines extracted from pdf")

        for line_idx, (line, page_num) in enumerate(all_lines):
            if len(equations) >= self.max_equations:
                break

            match = _PDF_EQ_LINE_RE.match(line)
            if not match:
                continue

            eq_content = match.group(1).strip()
            eq_num = match.group(2).strip()

            if eq_num in equations:
                continue

            # No content-length filter here.  Some PDFs render equation bodies
            # as positioned glyphs that are invisible to text extraction — the
            # number-terminator line then carries only ',' or '.'.  We still
            # record the equation: the number itself is the valuable datum, and
            # the audit trail documents the limitation.

            # context: nearest prose line (≥4 words) before and after
            before_lines: List[str] = []
            for i in range(line_idx - 1, max(0, line_idx - 10) - 1, -1):
                t = all_lines[i][0].strip()
                if t and len(t.split()) >= 4:
                    before_lines.append(t)
                    break

            after_lines: List[str] = []
            for i in range(line_idx + 1, min(len(all_lines), line_idx + 10)):
                t = all_lines[i][0].strip()
                if t and len(t.split()) >= 4:
                    after_lines.append(t)
                    break

            eq_audit = AuditTrail()
            eq_audit.log(
                "extract_pdf",
                (
                    f"found eq ({eq_num}), source=pdf, page={page_num}, "
                    f"content={eq_content[:60]!r} [unicode, not latex]"
                ),
            )

            equations[eq_num] = {
                "equation": eq_content,
                "source": "pdf",
                "latex_method": "pdf_text",
                "context_before": " ".join(before_lines),
                "context_after": " ".join(after_lines),
                "_audit": eq_audit,
            }

            logger.debug("PDF eq (%s): %r (page %d)", eq_num, eq_content[:50], page_num)

        return equations

    # ── LaTeX extraction from a <math> element ─────────────────────────────────

    def _get_latex_from_math(self, math_elem: Tag) -> Tuple[str, str]:
        """Extract LaTeX from a MathML ``<math>`` element.

        Tries three methods in priority order:

        1. ``<annotation encoding="application/x-tex">`` — the original LaTeX
           that LaTeXML preserved verbatim.  Most reliable.
        2. ``alttext`` attribute on ``<math>`` — equivalent copy placed by
           LaTeXML; present whenever the annotation tag is.
        3. Recursive MathML-to-LaTeX conversion — lossy algorithmic fallback
           used only when both (1) and (2) are absent.

        Parameters
        ----------
        math_elem : Tag
            BeautifulSoup ``<math>`` tag.

        Returns
        -------
        latex : str
            LaTeX string (may be approximate for method 3).
        method : str
            One of ``'annotation'``, ``'alttext'``, or ``'mathml'``.
        """
        # method 1: annotation tag (verbatim original LaTeX)
        annotation = math_elem.find(
            "annotation", attrs={"encoding": "application/x-tex"}
        )
        if annotation:
            latex = annotation.get_text().strip()
            if latex:
                return latex, "annotation"

        # method 2: alttext attribute (equivalent to annotation in LaTeXML)
        alttext = math_elem.get("alttext", "").strip()
        if alttext:
            return alttext, "alttext"

        # method 3: recursive MathML → LaTeX conversion (lossy)
        latex = self._mathml_to_latex(math_elem).strip()
        return latex, "mathml"

    # ── LaTeX post-processing ─────────────────────────────────────────────────

    @staticmethod
    def _clean_latex(latex: str) -> str:
        """Strip rendering artefacts introduced by LaTeXML.

        Two artefacts appear in annotation tags:

        ``\\displaystyle`` — LaTeXML prefixes every row of an align/eqnarray
        cell with this command to force display-math size.  It is a rendering
        detail, not part of the original equation.

        ``%\\n`` — LaTeX source line continuations (comment + newline) that
        authors use to break long equations across source lines.  LaTeXML
        copies them verbatim into the annotation; they are invisible in the
        rendered output and carry no mathematical meaning.
        """
        latex = re.sub(r"^\\displaystyle\s*", "", latex.strip())
        latex = latex.replace("%\n", "")
        return latex.strip()

    # ── MathML → LaTeX recursive converter ────────────────────────────────────

    def _mathml_to_latex(self, elem) -> str:
        """Recursively convert a MathML element to a LaTeX-like string.

        Handles the most common MathML elements produced by LaTeXML.
        Unknown elements fall back to concatenating their children's text.
        This method is only used when neither the annotation tag nor the
        alttext attribute is available.

        Parameters
        ----------
        elem : Tag or NavigableString
            A BeautifulSoup node.

        Returns
        -------
        str
            Approximate LaTeX representation.
        """
        # plain text node
        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            return _MO_MAP.get(text, text)  # map operators if known

        # get local tag name (strip namespace prefix if present)
        tag = getattr(elem, "name", None)
        if tag is None:
            return ""
        tag = tag.split(":")[-1] if ":" in tag else tag

        # recursively convert all child nodes
        children = list(elem.children)
        child_latex = [self._mathml_to_latex(c) for c in children]
        # filter empty strings but keep meaningful ones
        child_latex = [c for c in child_latex if c]

        # ── element-specific handling ──────────────────────────────────────────

        if tag == "math":
            # top-level element: just join children
            return " ".join(child_latex)

        if tag == "semantics":
            # semantics has (MathML-content, annotation).  Use first child only
            # (skip the annotation child — we already handled it above).
            non_annotation = [
                c for c in children
                if not (hasattr(c, "name") and c.name == "annotation")
            ]
            return " ".join(self._mathml_to_latex(c) for c in non_annotation if c)

        if tag == "annotation":
            # skip: we already extracted this at the parent level
            return ""

        if tag == "mi":
            # math identifier: map Greek/special chars, else return as-is
            text = elem.get_text()
            return _MI_MAP.get(text, text)

        if tag == "mn":
            # math number: return digits as-is
            return elem.get_text()

        if tag == "mo":
            # math operator: map known Unicode operators
            text = elem.get_text()
            return _MO_MAP.get(text, text)

        if tag == "mrow":
            # generic row grouping: join children
            return " ".join(child_latex)

        if tag == "msup":
            # superscript: base^{exp}
            if len(child_latex) >= 2:
                return f"{{{child_latex[0]}}}^{{{child_latex[1]}}}"
            return "".join(child_latex)

        if tag == "msub":
            # subscript: base_{sub}
            if len(child_latex) >= 2:
                return f"{{{child_latex[0]}}}_{{{child_latex[1]}}}"
            return "".join(child_latex)

        if tag == "msubsup":
            # combined subscript and superscript: base_{sub}^{sup}
            if len(child_latex) >= 3:
                return f"{{{child_latex[0]}}}_{{{child_latex[1]}}}^{{{child_latex[2]}}}"
            return "".join(child_latex)

        if tag == "mfrac":
            # fraction: \frac{numerator}{denominator}
            if len(child_latex) >= 2:
                return rf"\frac{{{child_latex[0]}}}{{{child_latex[1]}}}"
            return "".join(child_latex)

        if tag == "msqrt":
            # square root: \sqrt{content}
            inner = " ".join(child_latex)
            return rf"\sqrt{{{inner}}}"

        if tag == "mroot":
            # nth root: \sqrt[index]{radicand}
            if len(child_latex) >= 2:
                return rf"\sqrt[{child_latex[1]}]{{{child_latex[0]}}}"
            return " ".join(child_latex)

        if tag == "mtext":
            # text: \text{content}
            return rf"\text{{{elem.get_text()}}}"

        if tag == "mspace":
            # horizontal space: use thin space
            return r"\,"

        if tag == "mover":
            # accent or overline: \hat{base}, \bar{base}, etc.
            if len(child_latex) >= 2:
                base = child_latex[0]
                # identify the accent from the second child's text
                accent_char = elem.find_all(recursive=False)[1].get_text() if len(list(elem.find_all(recursive=False))) >= 2 else ""
                accent_cmd = _ACCENT_MAP.get(accent_char, r"\hat")
                return rf"{accent_cmd}{{{base}}}"
            return " ".join(child_latex)

        if tag == "munder":
            # underset: \underset{below}{base}
            if len(child_latex) >= 2:
                return rf"\underset{{{child_latex[1]}}}{{{child_latex[0]}}}"
            return " ".join(child_latex)

        if tag == "munderover":
            # common for \sum_{lower}^{upper} or \int_{lower}^{upper}
            if len(child_latex) >= 3:
                return f"{{{child_latex[0]}}}_{{{child_latex[1]}}}^{{{child_latex[2]}}}"
            return " ".join(child_latex)

        if tag in ("mtable", "mtr", "mtd"):
            # matrix / table: join with appropriate separators
            if tag == "mtd":
                return " ".join(child_latex)
            if tag == "mtr":
                return " & ".join(child_latex)
            if tag == "mtable":
                rows = " \\\\ ".join(child_latex)
                return rf"\begin{{matrix}} {rows} \end{{matrix}}"

        if tag == "mpadded":
            return " ".join(child_latex)

        if tag in ("mstyle", "merror", "mphantom", "menclose"):
            return " ".join(child_latex)

        # unknown element: concatenate children's text as best-effort
        return " ".join(child_latex)

    # ── helper: scope for HTML equation ───────────────────────────────────────

    def _get_html_scope(self, tag_span: Tag) -> Optional[Tag]:
        """Find the DOM scope containing one numbered equation.

        For equation arrays (align environments), each ``<tr>`` row holds
        one numbered equation.  For standalone display equations, the scope
        is the nearest ``ltx_equation`` container.

        Parameters
        ----------
        tag_span : Tag
            The ``<span class="ltx_tag_equation">`` element.

        Returns
        -------
        Tag or None
        """
        # equation arrays: each row is a separate equation
        tr = tag_span.find_parent("tr")
        if tr is not None:
            return tr

        # standalone equation: nearest ltx_equation table or div
        return tag_span.find_parent(
            lambda t: t.name in ("table", "div", "span")
            and any("ltx_equation" in cls for cls in t.get("class", []))
        )

    # ── helper: parse printed equation number ─────────────────────────────────

    @staticmethod
    def _parse_eq_number(raw_text: str) -> Optional[str]:
        """Extract the equation number from a tag's text.

        Parameters
        ----------
        raw_text : str
            Text content of the ``ltx_tag_equation`` span, e.g. ``"(1)"``
            or ``"(A.2)"``.

        Returns
        -------
        str or None
            The number string without parentheses (``"1"``, ``"A.2"``),
            or ``None`` if the text does not match the expected pattern.
        """
        match = _EQ_NUMBER_RE.match(raw_text.strip())
        return match.group(1) if match else None

    # ── helper: surrounding text from HTML ────────────────────────────────────

    @staticmethod
    def _get_surrounding_text_html(scope: Tag) -> Tuple[str, str]:
        """Extract paragraph text immediately before and after an equation.

        Walks up to the nearest paragraph-like container (``ltx_para`` div
        or ``<p>`` element) and collects the text of adjacent sibling
        elements.  This text is used by meaning.py and symbols.py for
        pattern matching — it is NOT written to the final JSON.

        Parameters
        ----------
        scope : Tag
            The equation's scope element (``<tr>`` or ``ltx_equation`` tag).

        Returns
        -------
        context_before : str
        context_after : str
        """
        # navigate up to a paragraph container
        para = scope.find_parent(
            lambda t: t.name == "div" and "ltx_para" in t.get("class", [])
        ) or scope.find_parent("p") or scope.parent

        if para is None:
            return "", ""

        before_parts: List[str] = []
        after_parts: List[str] = []
        found_scope = False

        for child in para.children:
            if isinstance(child, NavigableString):
                continue
            # detect when we have passed the equation scope
            if child is scope or (hasattr(child, "find") and child.find(lambda t: t is scope)):
                found_scope = True
                continue
            text = child.get_text(" ", strip=True)
            if not text:
                continue
            if not found_scope:
                before_parts.append(text)
            else:
                after_parts.append(text)

        # take the two closest siblings on each side
        context_before = " ".join(before_parts[-2:])
        context_after = " ".join(after_parts[:2])
        return context_before, context_after
