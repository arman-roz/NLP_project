"""
src/fragmenter.py

Stage 1 of the three-stage pipeline: raw HTML fragment extraction.

For every arXiv HTML paper, finds all <table class="ltx_eqn_table">
elements and stores each one as a self-contained dict with its raw HTML
and key metadata.  PDF papers produce no HTML fragments (recorded as an
empty list with source='pdf').

Why this stage exists
---------------------
A complete arXiv HTML file is typically 1–5 MB.  Inspecting why a single
equation was missed or why LaTeX is malformed requires searching through
thousands of lines.  By extracting and storing only the equation tables
upfront, two things become possible:

  1. Targeted debugging — open eq_fragments.json, search for the paper
     ID, and immediately see every equation container the parser found.
     No need to open the original HTML.

  2. Pattern analysis — feed the fragment list for a subset of papers to
     an LLM or a grep to detect structural patterns (e.g. how many tables
     lack an annotation tag, how many use ltx_equationgroup vs ltx_equation).

No equation cap is applied here.  All tables in the paper are saved.
The 7-equation limit is enforced downstream in extraction.py (Stage 2).

Output shape per paper
----------------------
[
  {
    "table_index":        0,
    "classes":            ["ltx_equation", "ltx_eqn_table"],
    "eq_num":             "(1)",
    "has_math":           true,
    "has_annotation":     true,
    "annotation_preview": "I_{\\rm OFF}(x,y)=...",
    "alttext_preview":    null,
    "html":               "<table ...>...</table>"
  },
  ...
]
"""

import logging
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class FragmentExtractor:
    """Extract raw HTML fragments of equation tables from arXiv HTML.

    Each fragment is one ``<table class="ltx_eqn_table">`` element,
    serialised back to a string and stored with metadata that lets a
    reader quickly assess what the extractor will find without re-parsing
    the full HTML.

    Examples
    --------
    >>> fe = FragmentExtractor()
    >>> fragments = fe.extract("2401.13506", html_bytes)
    >>> len(fragments)
    19
    >>> fragments[0]["eq_num"]
    '(1)'
    >>> fragments[0]["has_annotation"]
    True
    """

    def extract(self, arxiv_id: str, html_bytes: bytes) -> List[Dict]:
        """Parse HTML and return one dict per equation table found.

        Parameters
        ----------
        arxiv_id : str
            Used only for logging.
        html_bytes : bytes
            Raw HTML bytes (from cache or Fetcher).

        Returns
        -------
        list of dict
            One entry per ``<table class="ltx_eqn_table">``.
            Empty list if the HTML fails to parse or contains no tables.
        """
        try:
            soup = BeautifulSoup(html_bytes, "lxml")
        except Exception as exc:
            logger.error("FragmentExtractor: parse error for %s: %s", arxiv_id, exc)
            return []

        tables = soup.find_all("table", class_="ltx_eqn_table")
        logger.debug("FragmentExtractor: %s — %d equation tables found", arxiv_id, len(tables))

        return [self._to_fragment(i, table) for i, table in enumerate(tables)]

    # ── private ────────────────────────────────────────────────────────────────

    def _to_fragment(self, index: int, table: Tag) -> Dict:
        """Convert one equation table Tag to a fragment dict."""
        classes = table.get("class", [])

        # equation number: prefer individual tag, fall back to group tag
        num_span = (
            table.find("span", class_="ltx_tag_equation")
            or table.find("span", class_="ltx_tag_equationgroup")
        )
        eq_num = num_span.get_text().strip() if num_span else "?"

        # annotation tag — verbatim LaTeX preserved by LaTeXML
        annotation = table.find("annotation", attrs={"encoding": "application/x-tex"})
        has_annotation = annotation is not None
        annotation_preview: Optional[str] = None
        if annotation:
            annotation_preview = annotation.get_text()[:120].strip()

        # math element and alttext fallback
        math_elem = table.find("math")
        alttext_preview: Optional[str] = None
        if math_elem and not has_annotation:
            raw = math_elem.get("alttext", "").strip()
            if raw:
                alttext_preview = raw[:120]

        return {
            "table_index":        index,
            "classes":            classes,
            "eq_num":             eq_num,
            "has_math":           bool(math_elem),
            "has_annotation":     has_annotation,
            "annotation_preview": annotation_preview,
            "alttext_preview":    alttext_preview,
            "html":               str(table),
        }
