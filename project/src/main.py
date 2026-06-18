"""
src/main.py

Pipeline orchestrator — Equations Knowledge Graph
Exam ID 44, OTH Amberg-Weiden, Summer 2026.

Three-stage pipeline
--------------------
Stage 1  data/output/eq_fragments.json
         Raw HTML of every <table class="ltx_eqn_table"> from each
         HTML paper.  One list of fragment dicts per paper.  PDF and
         unavailable papers are recorded with an empty fragment list.
         Use this file to debug equation detection or analyse HTML
         patterns across the corpus.

Stage 2  data/output/eq_extracted.json
         Structured extraction: equation number → LaTeX + context.
         Capped at 7 equations per paper, deduped to first occurrence.
         source and latex_method are stored here (intermediate data)
         so the extraction quality can be verified before running the
         more expensive Stage 3 semantic annotation.

Stage 3  data/output/eq_final.json  (spec-compliant output)
         Final knowledge-graph entries: equation → meaning + symbols
         + relations + audit-trail.  source and latex_method are inside
         the audit-trail only, not at the top level (spec requirement).

Spec constraints enforced
--------------------------
* First 7 numbered equations per paper (MAX_EQUATIONS_PER_PAPER = 7).
* Stop after 350 total equations; always finish the current paper first.
* 15-second crawl delay after every network request (in acquisition.py).
  Cache hits make zero network calls — no sleep incurred.
* Deduplicate papers to first occurrence (in acquisition.py).
* No text generation: every string in meaning/symbols is a verbatim
  phrase extracted from the paper's own text.
* Only /abs, /html, /pdf arXiv endpoints — never /src or /e-print.

Cache-first fetch
-----------------
For papers already in cache the pipeline reads directly from disk and
skips the Fetcher entirely.  This avoids the 404 + 15-second sleep that
occurs when a PDF-only paper is checked for HTML availability on every
re-run.  For papers not yet in cache the Fetcher is called, which makes
the network request and caches the result for future runs.

Usage
-----
Run from the project root:
    python project/src/main.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# make project/src importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition import Fetcher, read_paper_list
from src.audit import AuditTrail
from src.extraction import EquationExtractor
from src.fragmenter import FragmentExtractor
from src.meaning import MeaningExtractor
from src.relations import RelationExtractor
from src.symbols import SymbolExtractor

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_PROJECT    = _HERE.parent
_REPO_ROOT  = _PROJECT.parent

PAPER_LIST_PATH = str(_REPO_ROOT / "paper_list_44.txt")
CACHE_DIR       = _PROJECT / "data" / "cache"
OUTPUT_DIR      = _PROJECT / "data" / "output"

STAGE1_PATH = OUTPUT_DIR / "eq_fragments.json"
STAGE2_PATH = OUTPUT_DIR / "eq_extracted.json"
STAGE3_PATH = OUTPUT_DIR / "eq_final.json"

MAX_TOTAL_EQUATIONS: int = 350

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── cache-first fetch ──────────────────────────────────────────────────────────

def _from_cache(arxiv_id: str) -> Optional[Dict]:
    """Return a fetch-result dict from disk cache, or None if not cached.

    Checks HTML cache first, then PDF.  If neither exists the caller
    should fall back to the Fetcher (which will make a network request).

    Returns
    -------
    dict or None
        Compatible shape with ``Fetcher.fetch_paper()`` output:
        ``{arxiv_id, content, source}``.
    """
    html_path = CACHE_DIR / f"{arxiv_id}_html.html"
    pdf_path  = CACHE_DIR / f"{arxiv_id}_pdf.pdf"

    if html_path.exists() and html_path.stat().st_size > 0:
        return {
            "arxiv_id": arxiv_id,
            "content":  html_path.read_bytes(),
            "source":   "html",
        }
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return {
            "arxiv_id": arxiv_id,
            "content":  pdf_path.read_bytes(),
            "source":   "pdf",
        }
    return None


# ── main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_ids = read_paper_list(PAPER_LIST_PATH)
    logger.info("Paper list: %d unique IDs", len(paper_ids))

    fetcher      = Fetcher(cache_dir=str(CACHE_DIR))
    frag_ext     = FragmentExtractor()
    eq_ext       = EquationExtractor()
    meaning_ext  = MeaningExtractor()
    symbol_ext   = SymbolExtractor(mode="regex+dep")
    relation_ext = RelationExtractor()

    # ── stage 1 + 2: acquisition and extraction ────────────────────────────────
    stage1: Dict      = {}   # arxiv_id → list of fragment dicts
    stage2_clean: Dict = {}   # arxiv_id → {eq_num: {equation, source, latex_method, context_*}}
    all_equations: Dict = {}  # arxiv_id → full equations dict (includes _audit + context)
    total_equations: int = 0

    logger.info("=== Stage 1 + 2: acquisition and extraction ===")

    for arxiv_id in paper_ids:
        # ── fetch (cache-first, network as fallback) ───────────────────────────
        paper_audit  = AuditTrail()
        fetch_result = _from_cache(arxiv_id)

        if fetch_result is None:
            logger.info("%s: not in cache — fetching from network", arxiv_id)
            fetch_result = fetcher.fetch_paper(arxiv_id, paper_audit)
        else:
            logger.info("%s: loaded from cache (source=%s)", arxiv_id, fetch_result["source"])

        source  = fetch_result.get("source", "none")
        content = fetch_result.get("content")

        # ── stage 1: HTML fragments ────────────────────────────────────────────
        if source == "html" and content is not None:
            fragments = frag_ext.extract(arxiv_id, content)
            stage1[arxiv_id] = {"source": "html", "fragments": fragments}
            logger.info("%s  source=html  tables_found=%d", arxiv_id, len(fragments))
        else:
            stage1[arxiv_id] = {"source": source, "fragments": []}
            logger.info("%s  source=%s  tables_found=0", arxiv_id, source)

        # ── stage 2: structured extraction ────────────────────────────────────
        equations = eq_ext.extract(fetch_result, paper_audit)
        all_equations[arxiv_id] = equations

        # clean version for stage 2 JSON (no private _audit key, no raw context)
        stage2_clean[arxiv_id] = {
            eq_num: {
                "equation":       eq_data.get("equation", ""),
                "source":         eq_data.get("source", source),
                "latex_method":   eq_data.get("latex_method", ""),
                "context_before": eq_data.get("context_before", ""),
                "context_after":  eq_data.get("context_after", ""),
            }
            for eq_num, eq_data in equations.items()
        }

        total_equations += len(equations)
        logger.info(
            "%s  equations=%d  total=%d",
            arxiv_id, len(equations), total_equations,
        )

        # stop acquiring new papers once the 350 limit is reached
        # (current paper is always finished before stopping)
        if total_equations >= MAX_TOTAL_EQUATIONS:
            logger.info(
                "Reached %d equations after paper %s — stopping acquisition",
                total_equations, arxiv_id,
            )
            break

    # save stage 1 and stage 2 to disk
    _save_json(stage1, STAGE1_PATH)
    logger.info("Stage 1 saved → %s  (%d papers)", STAGE1_PATH.name, len(stage1))

    _save_json(stage2_clean, STAGE2_PATH)
    logger.info("Stage 2 saved → %s  (%d equations total)", STAGE2_PATH.name, total_equations)

    # ── fit corpus-wide IDF ────────────────────────────────────────────────────
    logger.info("=== Fitting corpus-wide IDF ===")
    all_contexts = []
    for equations in all_equations.values():
        for eq_data in equations.values():
            cb = eq_data.get("context_before", "")
            ca = eq_data.get("context_after",  "")
            if cb:
                all_contexts.append(cb)
            if ca:
                all_contexts.append(ca)

    meaning_ext.fit_corpus(all_contexts)
    logger.info("IDF fitted on %d context texts from %d equations",
                len(all_contexts), total_equations)

    # ── stage 3: meaning + symbols + relations ─────────────────────────────────
    logger.info("=== Stage 3: meaning, symbols, relations ===")
    stage3: Dict = {}

    for arxiv_id, equations in all_equations.items():
        if not equations:
            stage3[arxiv_id] = {}
            continue

        paper_eq_nums = list(equations.keys())
        paper_out: Dict = {}

        for eq_num, eq_data in equations.items():
            latex   = eq_data.get("equation", "")
            ctx_b   = eq_data.get("context_before", "")
            ctx_a   = eq_data.get("context_after",  "")
            eq_audit: AuditTrail = eq_data.get("_audit") or AuditTrail()

            # move source + latex_method into audit-trail (spec compliance:
            # these must not appear as top-level keys in the final output)
            eq_audit.log("source",       eq_data.get("source", "unknown"))
            eq_audit.log("latex_method", eq_data.get("latex_method", "unknown"))

            # meaning — single best sentence from the paper's own text
            meaning = meaning_ext.extract(eq_num, latex, ctx_b, ctx_a, eq_audit)

            # symbols — verbatim definitional phrases from context
            symbols = symbol_ext.extract(latex, ctx_b, ctx_a, eq_audit)

            # relations — citation links from context; shared symbols added below
            relations = relation_ext.extract(eq_num, ctx_b, ctx_a, paper_eq_nums, eq_audit)

            paper_out[eq_num] = {
                "equation":    latex,
                "meaning":     meaning,
                "symbols":     symbols,
                "relations":   relations,
                "audit-trail": eq_audit.to_dict(),
            }

        # second pass: compute shared-symbol relations across all equations
        # in this paper (requires every equation's symbols to be known first)
        shared = relation_ext.compute_shared_symbols(paper_out)
        for eq_num in paper_out:
            paper_out[eq_num]["relations"]["shares_symbol_with"] = shared.get(eq_num, [])

        stage3[arxiv_id] = paper_out

    _save_json(stage3, STAGE3_PATH)
    logger.info("Stage 3 saved → %s", STAGE3_PATH.name)

    # ── summary ────────────────────────────────────────────────────────────────
    total_final   = sum(len(v) for v in stage3.values())
    papers_empty  = sum(1 for v in stage3.values() if not v)
    papers_full   = sum(1 for v in stage3.values() if len(v) == 7)
    papers_html   = sum(1 for v in stage1.values() if v["source"] == "html")
    papers_pdf    = sum(1 for v in stage1.values() if v["source"] == "pdf")
    papers_none   = sum(1 for v in stage1.values() if v["source"] == "none")

    logger.info("=== Pipeline complete ===")
    logger.info("Papers processed    : %d", len(stage3))
    logger.info("  source=html       : %d", papers_html)
    logger.info("  source=pdf        : %d", papers_pdf)
    logger.info("  source=none       : %d", papers_none)
    logger.info("Equations total     : %d", total_final)
    logger.info("  papers with 7 eqs : %d", papers_full)
    logger.info("  papers with 0 eqs : %d", papers_empty)
    logger.info("Output files:")
    logger.info("  %s", STAGE1_PATH)
    logger.info("  %s", STAGE2_PATH)
    logger.info("  %s", STAGE3_PATH)


# ── helpers ────────────────────────────────────────────────────────────────────

def _save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
