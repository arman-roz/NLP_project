"""
experiments/compare_extraction.py

Side-by-side comparison of the three LaTeX extraction methods for the
first 10 numbered equations found across the first papers in paper_list_44.txt.

Three columns compared:
  (a) annotation  — <annotation encoding="application/x-tex"> tag (original LaTeX)
  (b) alttext     — alttext attribute on <math> (equivalent to annotation)
  (c) mathml      — recursive MathML-to-LaTeX conversion (lossy algorithmic fallback)
  (d) pdf_text    — PyMuPDF text extraction (unicode, not LaTeX)

Run from the project root:
    python experiments/compare_extraction.py

Output is printed to stdout and also saved to experiments/comparison_results.txt.
"""

import sys
import os
import textwrap
from pathlib import Path

# allow importing from src/ when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import AuditTrail
from src.acquisition import read_paper_list, Fetcher
from src.extraction import EquationExtractor
from bs4 import BeautifulSoup

# ── config ─────────────────────────────────────────────────────────────────────
PAPER_LIST_PATH = "paper_list_44.txt"
CACHE_DIR = "data/cache"
TARGET_EQUATIONS = 10   # how many equations to compare
MAX_PAPERS_TO_SCAN = 5  # scan at most this many papers to find TARGET_EQUATIONS
COL_WIDTH = 55          # character width for each displayed column


def _get_all_three(html_bytes: bytes) -> list:
    """Extract equations using all three HTML methods plus PDF-style text.

    For each numbered equation in the HTML, collect:
      - annotation tag LaTeX  (method a)
      - alttext attribute      (method b)
      - MathML conversion      (method c)

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML bytes from cache.

    Returns
    -------
    list of dict
        Each dict has keys: eq_num, annotation, alttext, mathml.
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    extractor = EquationExtractor()
    results = []

    tag_spans = soup.find_all("span", class_="ltx_tag_equation")

    for tag_span in tag_spans:
        eq_num = extractor._parse_eq_number(tag_span.get_text())
        if eq_num is None:
            continue

        scope = extractor._get_html_scope(tag_span)
        if scope is None:
            continue

        math_elem = scope.find("math")
        if math_elem is None:
            continue

        # (a) annotation tag
        annotation_tag = math_elem.find(
            "annotation", attrs={"encoding": "application/x-tex"}
        )
        annotation_latex = annotation_tag.get_text().strip() if annotation_tag else "N/A"

        # (b) alttext attribute
        alttext_latex = math_elem.get("alttext", "N/A").strip() or "N/A"

        # (c) MathML-to-LaTeX conversion
        mathml_latex = extractor._mathml_to_latex(math_elem).strip() or "N/A"

        results.append({
            "eq_num": eq_num,
            "annotation": annotation_latex,
            "alttext": alttext_latex,
            "mathml": mathml_latex,
        })

    return results


def _wrap(text: str, width: int) -> list:
    """Wrap text to a fixed column width, returning a list of lines.

    Parameters
    ----------
    text : str
        Text to wrap.
    width : int
        Column character width.

    Returns
    -------
    list of str
    """
    if not text or text == "N/A":
        return ["N/A"]
    return textwrap.wrap(text, width=width) or ["(empty)"]


def print_comparison(results: list, output_file=None) -> None:
    """Pretty-print the side-by-side comparison table.

    Parameters
    ----------
    results : list of dict
        Output of :func:`_get_all_three`.
    output_file : file-like object or None
        If provided, also write to this file.
    """
    def out(line=""):
        print(line)
        if output_file:
            output_file.write(line + "\n")

    sep = "=" * (COL_WIDTH * 3 + 10)
    row_sep = "-" * (COL_WIDTH * 3 + 10)

    out(sep)
    out(f"{'Eq':^6}  {'(a) annotation / alttext':^{COL_WIDTH}}  {'(b) mathml conversion':^{COL_WIDTH}}  {'notes':^20}")
    out(sep)

    for r in results:
        eq_num = r["eq_num"]
        ann = r["annotation"]
        alt = r["alttext"]
        mml = r["mathml"]

        # annotation and alttext are usually identical — show annotation,
        # flag if they differ
        primary = ann if ann != "N/A" else alt
        match_note = ""
        if ann != "N/A" and alt != "N/A" and ann != alt:
            match_note = "ann≠alt"

        primary_lines = _wrap(primary, COL_WIDTH)
        mml_lines = _wrap(mml, COL_WIDTH)
        n_lines = max(len(primary_lines), len(mml_lines))

        out(row_sep)
        out(f"Eq ({eq_num})")
        for i in range(n_lines):
            pl = primary_lines[i] if i < len(primary_lines) else ""
            ml = mml_lines[i] if i < len(mml_lines) else ""
            note = match_note if i == 0 else ""
            out(f"  {pl:<{COL_WIDTH}}  {ml:<{COL_WIDTH}}  {note}")

        # show if methods match
        if ann == mml and ann != "N/A":
            out(f"  → methods (a) and (b) MATCH")
        elif primary != "N/A" and mml != "N/A":
            out(f"  → methods DIFFER (annotation is authoritative)")

    out(sep)


def main() -> None:
    """Run the comparison over the first TARGET_EQUATIONS numbered equations.

    Reads paper_list_44.txt, fetches papers (using cache if available),
    extracts equations by all three methods, and prints the comparison.
    """
    paper_ids = read_paper_list(PAPER_LIST_PATH)
    fetcher = Fetcher(cache_dir=CACHE_DIR)
    extractor = EquationExtractor()

    all_results = []
    papers_scanned = 0

    print(f"Scanning up to {MAX_PAPERS_TO_SCAN} papers for {TARGET_EQUATIONS} equations...\n")

    for arxiv_id in paper_ids[:MAX_PAPERS_TO_SCAN]:
        if len(all_results) >= TARGET_EQUATIONS:
            break

        audit = AuditTrail()
        fetch_result = fetcher.fetch_paper(arxiv_id, audit)
        papers_scanned += 1

        if fetch_result["source"] != "html" or fetch_result["content"] is None:
            print(f"  {arxiv_id}: no HTML, skipping for this comparison")
            continue

        paper_results = _get_all_three(fetch_result["content"])
        print(f"  {arxiv_id}: found {len(paper_results)} numbered equations")

        for r in paper_results:
            r["arxiv_id"] = arxiv_id
            all_results.append(r)
            if len(all_results) >= TARGET_EQUATIONS:
                break

    print(f"\nScanned {papers_scanned} paper(s), collected {len(all_results)} equations.\n")

    # write comparison to file and stdout
    out_path = Path("experiments/comparison_results.txt")
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        print_comparison(all_results, output_file=f)

    print(f"\nResults also saved to {out_path}")


if __name__ == "__main__":
    main()
