"""
experiments/test_pipeline.py

Quick end-to-end test of acquisition + extraction on the first N papers.

Run from the project root:
    python experiments/test_pipeline.py

What it does:
  1. Reads paper_list_44.txt
  2. Fetches the first TEST_PAPERS papers (HTML-first, PDF-fallback)
  3. Extracts numbered equations from each
  4. Prints a readable summary to the console
  5. Saves the raw extracted data to data/output/test_output.json

No meaning / symbols / relations yet — this is purely to verify that
acquisition and extraction are working correctly on real arXiv papers.
"""

import json
import logging
import sys
from pathlib import Path

# allow imports from src/ when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition import Fetcher, read_paper_list
from src.audit import AuditTrail
from src.extraction import EquationExtractor

# ── paths (resolved relative to this file so the script works from any cwd) ───
_HERE        = Path(__file__).resolve().parent          # experiments/
_PROJECT     = _HERE.parent                             # project/
_REPO_ROOT   = _PROJECT.parent                          # NLP project/

PAPER_LIST_PATH = str(_REPO_ROOT / "paper_list_44.txt")
CACHE_DIR       = str(_PROJECT / "data" / "cache")
OUTPUT_PATH     = str(_PROJECT / "data" / "output" / "test_output.json")
TEST_PAPERS     = 100      # how many papers to test with
LOG_LEVEL       = logging.INFO


def setup_logging() -> None:
    """Configure console logging for the test run."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def print_separator(char: str = "─", width: int = 70) -> None:
    """Print a visual separator line."""
    print(char * width)


def print_paper_result(arxiv_id: str, fetch_result: dict, equations: dict) -> None:
    """Print a human-readable summary for one paper.

    Parameters
    ----------
    arxiv_id : str
        The arXiv ID.
    fetch_result : dict
        Result from Fetcher.fetch_paper.
    equations : dict
        Result from EquationExtractor.extract.
    """
    print_separator("═")
    source = fetch_result.get("source", "none")
    html_s = fetch_result.get("html_status")
    pdf_s  = fetch_result.get("pdf_status")

    print(f"  Paper   : {arxiv_id}")
    print(f"  Source  : {source}  (html_status={html_s}, pdf_status={pdf_s})")
    print(f"  Equations found: {len(equations)}")
    print_separator()

    if not equations:
        print("  (no numbered equations extracted)")
        return

    for eq_num, eq_data in equations.items():
        latex        = eq_data.get("equation", "")
        method       = eq_data.get("latex_method", "?")
        ctx_before   = eq_data.get("context_before", "")[:120]
        ctx_after    = eq_data.get("context_after",  "")[:120]

        print(f"\n  Equation ({eq_num})")
        print(f"    LaTeX method : {method}")
        print(f"    LaTeX        : {latex[:120]}")
        if ctx_before:
            print(f"    Context ←    : {ctx_before!r}")
        if ctx_after:
            print(f"    Context →    : {ctx_after!r}")


def build_output(arxiv_id: str, equations: dict) -> dict:
    """Build a JSON-serialisable dict for one paper.

    Strips internal-only keys (_audit, context_before, context_after)
    and converts the AuditTrail objects to dicts.

    Parameters
    ----------
    arxiv_id : str
        The arXiv ID.
    equations : dict
        Raw equation dicts from EquationExtractor.

    Returns
    -------
    dict
        Clean dict ready for json.dump.
    """
    paper_out = {}
    for eq_num, eq_data in equations.items():
        audit_obj = eq_data.get("_audit")
        paper_out[eq_num] = {
            "equation"    : eq_data.get("equation", ""),
            "source"      : eq_data.get("source", ""),
            "latex_method": eq_data.get("latex_method", ""),
            # placeholders — filled by meaning/symbols/relations later
            "meaning"     : "",
            "symbols"     : {},
            "relations"   : {},
            "audit-trail" : audit_obj.to_dict() if audit_obj else {},
        }
    return paper_out


def main() -> None:
    """Run the acquisition + extraction test."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # ── load paper list ────────────────────────────────────────────────────────
    paper_ids = read_paper_list(PAPER_LIST_PATH)
    test_ids  = paper_ids[:TEST_PAPERS]
    print(f"\nTesting on first {TEST_PAPERS} papers: {test_ids}\n")

    fetcher   = Fetcher(cache_dir=CACHE_DIR)
    extractor = EquationExtractor()
    output    = {}   # {arxiv_id: {eq_num: {...}}}

    total_equations = 0

    for arxiv_id in test_ids:
        print(f"\nProcessing {arxiv_id} ...")

        # ── acquire ────────────────────────────────────────────────────────────
        paper_audit = AuditTrail()
        fetch_result = fetcher.fetch_paper(arxiv_id, paper_audit)

        # ── extract ────────────────────────────────────────────────────────────
        equations = extractor.extract(fetch_result, paper_audit)

        # ── display ────────────────────────────────────────────────────────────
        print_paper_result(arxiv_id, fetch_result, equations)

        # ── accumulate ─────────────────────────────────────────────────────────
        output[arxiv_id] = build_output(arxiv_id, equations)
        total_equations += len(equations)

    # ── summary ────────────────────────────────────────────────────────────────
    print_separator("═")
    print(f"\nSUMMARY")
    print(f"  Papers processed : {len(test_ids)}")
    print(f"  Total equations  : {total_equations}")
    print_separator()

    sources = {"html": 0, "pdf": 0, "none": 0}
    methods = {"annotation": 0, "alttext": 0, "mathml": 0, "pdf_text": 0}
    for paper_data in output.values():
        for eq in paper_data.values():
            s = eq.get("source", "none")
            m = eq.get("latex_method", "")
            sources[s] = sources.get(s, 0) + 1
            methods[m] = methods.get(m, 0) + 1

    print("  Source breakdown:")
    for k, v in sources.items():
        print(f"    {k:10s} : {v}")
    print("  LaTeX method breakdown:")
    for k, v in methods.items():
        if v:
            print(f"    {k:12s} : {v}")

    # ── save to JSON ───────────────────────────────────────────────────────────
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
