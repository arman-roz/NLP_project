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
from src.chunks import DocumentParser
from src.extraction import EquationExtractor
from src.fragmenter import FragmentExtractor
from src.meaning import MeaningExtractor
from src.relations import RelationExtractor
from src.retrieval import BM25Retriever
from src.symbols import SymbolExtractor
from src.validate import ValidationError, Validator

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
    doc_parser   = DocumentParser()
    validator    = Validator()

    # ── stage 1 + 2: acquisition and extraction ────────────────────────────────
    stage1: Dict      = {}   # arxiv_id → list of fragment dicts
    stage2_clean: Dict = {}   # arxiv_id → {eq_num: {equation, source, latex_method, context_*}}
    all_equations: Dict = {}  # arxiv_id → full equations dict (includes _audit + context)
    all_content: Dict  = {}  # arxiv_id → fetch_result kept for Stage 3 chunking
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
        # keep fetch result so Stage 3 can build a per-paper BM25 index
        all_content[arxiv_id] = fetch_result

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

        # ── build per-paper document model + BM25 retriever ──────────────────
        # Parse the cached HTML into structured chunk views.  Equation-
        # neighborhood chunks are used as the primary retrieval pool because
        # they always contain the relevant section title, surrounding prose,
        # and equation LaTeX in one unit.  Sentence chunks are concatenated
        # as a broader fallback pool.
        fetch_result_paper = all_content.get(arxiv_id, {})
        paper_source       = fetch_result_paper.get("source", "none")
        paper_content      = fetch_result_paper.get("content")

        retriever    = None
        parsed_paper = None
        if paper_source == "html" and paper_content:
            # Build URL from arxiv_id; version suffix is omitted since the
            # cached HTML may be any version (content doesn't change for
            # retrieval purposes).
            source_endpoint = f"https://arxiv.org/html/{arxiv_id}"
            parsed_paper    = doc_parser.parse(
                paper_content, arxiv_id, source_endpoint, equations
            )
            # Primary pool: equation neighborhoods; fallback: sentences.
            # BM25Retriever.fit() takes Chunk objects directly.
            primary_chunks  = parsed_paper.eq_neighborhood_chunks
            fallback_chunks = parsed_paper.sentence_chunks
            combined        = primary_chunks + fallback_chunks
            if combined:
                retriever = BM25Retriever()
                retriever.fit(combined)
            logger.info(
                "%s  source=html  eq_nbhd=%d  sentences=%d  xrefs=%d  retriever=%s",
                arxiv_id,
                len(primary_chunks),
                len(fallback_chunks),
                len(parsed_paper.cross_refs),
                "BM25" if retriever is not None else "none",
            )
        else:
            logger.info(
                "%s  source=%s  retriever=none (no HTML available)",
                arxiv_id, paper_source,
            )

        # ── pass A: meaning + symbols for every equation ──────────────────────
        # Build per-equation location metadata (section/paragraph IDs) from
        # the equation neighborhood chunks so that the weighted meaning scorer
        # and the relations extractor can use section/paragraph proximity
        # signals.  Available only for HTML papers (parsed_paper is None for PDF).
        eq_section_ids: Dict[str, str] = {}
        eq_para_ids:    Dict[str, str] = {}
        if parsed_paper is not None:
            for chunk in parsed_paper.eq_neighborhood_chunks:
                for eq_n in chunk.eq_nums_nearby:
                    if eq_n not in eq_section_ids:
                        eq_section_ids[eq_n] = chunk.section_id
                        eq_para_ids[eq_n]    = chunk.paragraph_id

        # Context is kept in paper_build for relations (pass B) then stripped.
        paper_build: Dict = {}
        eq_audits:   Dict = {}
        used_meaning_sentences: set = set()  # per-paper dedup for meaning

        for eq_num, eq_data in equations.items():
            latex    = eq_data.get("equation", "")
            ctx_b    = eq_data.get("context_before", "")
            ctx_a    = eq_data.get("context_after",  "")
            eq_audit: AuditTrail = eq_data.get("_audit") or AuditTrail()

            # move source + latex_method into audit-trail (spec compliance:
            # these must not appear as top-level keys in the final output)
            eq_audit.log("source",       eq_data.get("source", "unknown"))
            eq_audit.log("latex_method", eq_data.get("latex_method", "unknown"))

            # meaning — single best sentence, weighted-signal scoring
            meaning = meaning_ext.extract(
                eq_num, latex, ctx_b, ctx_a, eq_audit,
                retriever=retriever,
                eq_section_id=eq_section_ids.get(eq_num, ""),
                eq_para_id=eq_para_ids.get(eq_num, ""),
                used_sentences=used_meaning_sentences,
            )

            # symbols — MathML <mi> identification + BM25 retrieval + confidence gate
            symbols = symbol_ext.extract(
                eq_num, latex, ctx_b, ctx_a, eq_audit,
                retriever=retriever,
                html_content=paper_content if paper_source == "html" else None,
                parsed_paper=parsed_paper,
            )

            paper_build[eq_num] = {
                "equation":    latex,
                "meaning":     meaning,
                "symbols":     symbols,
                # kept for pass B only — not written to stage3
                "_ctx_before": ctx_b,
                "_ctx_after":  ctx_a,
            }
            eq_audits[eq_num] = eq_audit

        # ── pass B: relations (needs all meanings + symbols from pass A) ──────
        # cross_refs: structural DOM cross-reference graph from ParsedPaper.
        # eq_section_ids: maps eq_num → section_id for XRef proximity filtering.
        all_relations = relation_ext.compute_all_relations(
            paper_build, eq_audits,
            retriever=retriever,
            cross_refs=parsed_paper.cross_refs if parsed_paper is not None else None,
            eq_section_ids=eq_section_ids if eq_section_ids else None,
            eq_para_ids=eq_para_ids if eq_para_ids else None,
        )

        # ── assemble final output (exactly the five spec keys) ────────────────
        paper_out: Dict = {}
        for eq_num in paper_build:
            paper_out[eq_num] = {
                "equation":    paper_build[eq_num]["equation"],
                "meaning":     paper_build[eq_num]["meaning"],
                "symbols":     paper_build[eq_num]["symbols"],
                "relations":   all_relations.get(eq_num, {}),
                "audit-trail": eq_audits[eq_num].to_dict(),
            }

        stage3[arxiv_id] = paper_out

    _save_json(stage3, STAGE3_PATH)
    logger.info("Stage 3 saved → %s", STAGE3_PATH.name)

    # ── schema validation ──────────────────────────────────────────────────────
    val_errors = validator.validate(stage3, fail_loudly=False)
    if val_errors:
        logger.warning("Schema validation found %d issue(s):", len(val_errors))
        for err in val_errors:
            logger.warning("  %s", err)
    else:
        logger.info("Schema validation passed — output conforms to spec")

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
