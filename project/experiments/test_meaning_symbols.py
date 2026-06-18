"""
experiments/test_meaning_symbols.py

Tests meaning.py and symbols.py on the already-cached 100 papers.

What it does:
  1. Reads the paper list and uses the first TEST_PAPERS papers.
  2. For each paper: loads from cache (zero network calls), runs extraction
     to get equations with context_before / context_after.
  3. Fits MeaningExtractor on the full corpus of contexts (corpus-wide IDF).
  4. For every equation:
       - Extracts meaning (single best sentence from the paper).
       - Extracts symbols in BOTH modes so you can compare:
           mode "regex"      — definitional regex patterns only
           mode "regex+dep"  — regex first, spaCy dependency fallback
  5. Prints a per-paper and summary report to the console.
  6. Saves output to data/output/test_meaning_symbols_output.json
     following the spec shape: {equation, meaning, symbols, relations, audit-trail}.

Run from the project root:
    python project/experiments/test_meaning_symbols.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition import read_paper_list
from src.audit import AuditTrail
from src.extraction import EquationExtractor
from src.meaning import MeaningExtractor
from src.symbols import SymbolExtractor

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_PROJECT     = _HERE.parent
_REPO_ROOT   = _PROJECT.parent

PAPER_LIST_PATH = str(_REPO_ROOT / "paper_list_44.txt")
CACHE_DIR       = _PROJECT / "data" / "cache"
OUTPUT_PATH     = str(_PROJECT / "data" / "output" / "test_meaning_symbols_output.json")

TEST_PAPERS = 100
LOG_LEVEL   = logging.WARNING


def setup_logging() -> None:
    # force UTF-8 output so Greek/math characters in extracted text don't crash
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def sep(char="=", width=70):
    print(char * width)


def _load_from_cache(arxiv_id: str) -> dict:
    """Read a paper directly from the cache — zero network calls.

    Returns a fetch_result dict compatible with EquationExtractor.extract().
    Checks HTML cache first, then PDF. Returns source='none' if neither exists.
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
    return {"arxiv_id": arxiv_id, "content": None, "source": "none"}


def main() -> None:
    setup_logging()

    paper_ids = read_paper_list(PAPER_LIST_PATH)
    test_ids  = paper_ids[:TEST_PAPERS]
    print(f"\nPapers to process : {len(test_ids)}")

    extractor       = EquationExtractor()
    meaning_ext     = MeaningExtractor()
    symbol_ext_re   = SymbolExtractor(mode="regex")
    symbol_ext_dep  = SymbolExtractor(mode="regex+dep")

    # ── pass 1: extract all equations and collect context texts ────────────────
    print("Pass 1 - extracting equations from cache (no network) ...")
    all_papers: dict = {}   # arxiv_id -> {eq_num: eq_data_with_context}

    for arxiv_id in test_ids:
        paper_audit  = AuditTrail()
        fetch_result = _load_from_cache(arxiv_id)
        equations    = extractor.extract(fetch_result, paper_audit)
        all_papers[arxiv_id] = equations

    total_equations = sum(len(eqs) for eqs in all_papers.values())
    print(f"  Equations extracted : {total_equations}")

    # ── fit corpus-wide IDF ────────────────────────────────────────────────────
    print("Fitting corpus-wide IDF ...")
    all_contexts = []
    for equations in all_papers.values():
        for eq_data in equations.values():
            cb = eq_data.get("context_before", "")
            ca = eq_data.get("context_after",  "")
            if cb:
                all_contexts.append(cb)
            if ca:
                all_contexts.append(ca)

    meaning_ext.fit_corpus(all_contexts)
    print(f"  Corpus texts used   : {len(all_contexts)}")

    # ── pass 2: meaning + symbols ──────────────────────────────────────────────
    print("Pass 2 - extracting meaning and symbols ...\n")

    output: dict        = {}
    strategy_counts     = Counter()   # citation_match / gazetteer_match / tfidf_rank / none
    meanings_found      = 0
    total_syms_regex    = 0
    total_syms_dep      = 0
    extra_by_dep        = 0           # symbols found by dep but not regex

    for arxiv_id, equations in all_papers.items():
        sep("=")
        source = next(
            (v.get("source", "?") for v in equations.values()), "none"
        ) if equations else "none"
        print(f"  {arxiv_id}   source={source}   equations={len(equations)}")
        sep("-")

        paper_out: dict = {}

        for eq_num, eq_data in equations.items():
            latex   = eq_data.get("equation", "")
            ctx_b   = eq_data.get("context_before", "")
            ctx_a   = eq_data.get("context_after",  "")
            eq_audit: AuditTrail = eq_data.get("_audit") or AuditTrail()

            # ── meaning ───────────────────────────────────────────────────────
            meaning = meaning_ext.extract(eq_num, latex, ctx_b, ctx_a, eq_audit)
            if meaning:
                meanings_found += 1
                # find which strategy was used from the audit trail
                for method, detail in eq_audit.entries():
                    if method == "meaning":
                        for strat in ("citation_match", "gazetteer_match", "tfidf_rank"):
                            if strat in detail:
                                strategy_counts[strat] += 1
                                break
                        else:
                            if "no meaning" not in detail:
                                strategy_counts["other"] += 1
                        break
            else:
                strategy_counts["none"] += 1

            # ── symbols — both modes ──────────────────────────────────────────
            sym_audit_re  = AuditTrail()
            sym_audit_dep = AuditTrail()

            syms_regex  = symbol_ext_re.extract(latex, ctx_b, ctx_a, sym_audit_re)
            syms_dep    = symbol_ext_dep.extract(latex, ctx_b, ctx_a, sym_audit_dep)

            extra   = {k: v for k, v in syms_dep.items() if k not in syms_regex}
            total_syms_regex += len(syms_regex)
            total_syms_dep   += len(syms_dep)
            extra_by_dep     += len(extra)

            # merge dep results into eq_audit for the output
            for m, d in sym_audit_dep.entries():
                eq_audit.log(m, d)

            # ── console print for this equation ───────────────────────────────
            print(f"\n  Equation ({eq_num})")
            print(f"    LaTeX   : {latex[:90]}")
            print(f"    Meaning : {meaning[:100] if meaning else '(none found)'}")
            if syms_regex:
                for sym, defn in syms_regex.items():
                    extra_flag = " [+dep]" if sym in extra else ""
                    print(f"    Symbol  : {sym} = {defn[:60]}{extra_flag}")
            if extra:
                for sym, defn in extra.items():
                    print(f"    Symbol* : {sym} = {defn[:60]}  <- dep-only")
            if not syms_dep:
                print("    Symbols : (none found)")

            # ── build output entry (spec shape) ──────────────────────────────
            paper_out[eq_num] = {
                "equation"   : latex,
                "meaning"    : meaning,
                "symbols"    : syms_dep,   # full set (regex + dep)
                "relations"  : {},          # filled by relations.py
                "audit-trail": eq_audit.to_dict(),
            }

        output[arxiv_id] = paper_out

    # ── summary ────────────────────────────────────────────────────────────────
    sep("=")
    print("\nSUMMARY")
    sep("-")
    print(f"  Papers processed        : {len(test_ids)}")
    print(f"  Total equations         : {total_equations}")
    print()
    print(f"  Meanings found          : {meanings_found} / {total_equations}")
    print(f"    via citation match    : {strategy_counts['citation_match']}")
    print(f"    via gazetteer         : {strategy_counts['gazetteer_match']}")
    print(f"    via TF-IDF rank       : {strategy_counts['tfidf_rank']}")
    print(f"    none found            : {strategy_counts['none']}")
    print()
    print(f"  Symbol definitions")
    print(f"    regex-only mode       : {total_syms_regex}")
    print(f"    regex+dep mode        : {total_syms_dep}")
    print(f"    extra found by dep    : {extra_by_dep}")
    sep()

    # ── save output ────────────────────────────────────────────────────────────
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    sep("-")
    print(f"\n  Saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
