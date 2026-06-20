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
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill

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
EXPORT_DIR      = Path(r"D:\OTH Sem 4\NLP project\project\testing exports")

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
    strategy_counts     = Counter()   # citation_match / gazetteer_match / definitional_match / tfidf_rank / none
    meanings_found      = 0
    total_syms_regex    = 0
    total_syms_dep      = 0
    extra_by_dep        = 0           # symbols found by dep but not regex

    # ── audit row collection (for export) ─────────────────────────────────────
    audit_rows: list    = []          # one dict per equation
    paper_summary: dict = {}          # arxiv_id -> summary dict

    _STRAT_RE = re.compile(r"strategy=(\w+)")

    for arxiv_id, equations in all_papers.items():
        sep("=")
        source = next(
            (v.get("source", "?") for v in equations.values()), "none"
        ) if equations else "none"
        print(f"  {arxiv_id}   source={source}   equations={len(equations)}")
        sep("-")

        paper_out: dict = {}
        paper_meanings  = 0
        paper_nones     = 0

        for eq_num, eq_data in equations.items():
            latex   = eq_data.get("equation", "")
            ctx_b   = eq_data.get("context_before", "")
            ctx_a   = eq_data.get("context_after",  "")
            eq_audit: AuditTrail = eq_data.get("_audit") or AuditTrail()

            # ── meaning ───────────────────────────────────────────────────────
            meaning = meaning_ext.extract(eq_num, latex, ctx_b, ctx_a, eq_audit)
            strategy = "none"
            if meaning:
                meanings_found += 1
                paper_meanings += 1
                for method, detail in eq_audit.entries():
                    if method == "meaning":
                        m = _STRAT_RE.search(detail)
                        if m:
                            strategy = m.group(1)
                        strategy_counts[strategy] += 1
                        break
            else:
                paper_nones += 1
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
            for m_key, m_detail in sym_audit_dep.entries():
                eq_audit.log(m_key, m_detail)

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

            # ── collect audit row for export ──────────────────────────────────
            syms_dep_preview = "; ".join(
                f"{k}={v[:40]}" for k, v in list(syms_dep.items())[:5]
            )
            audit_rows.append({
                "arxiv_id":          arxiv_id,
                "source":            source,
                "eq_num":            eq_num,
                "latex":             latex[:120],
                "meaning":           meaning[:220] if meaning else "",
                "strategy":          strategy,
                "symbols_regex_cnt": len(syms_regex),
                "symbols_dep_cnt":   len(syms_dep),
                "symbols_dep_keys":  ", ".join(syms_dep.keys()),
                "symbols_dep_preview": syms_dep_preview,
            })

        output[arxiv_id] = paper_out

        paper_summary[arxiv_id] = {
            "source":        source,
            "total_eqs":     len(equations),
            "meanings_found": paper_meanings,
            "meanings_none": paper_nones,
        }

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
    print(f"    via definitional      : {strategy_counts['definitional_match']}")
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
    print(f"\n  Saved to {OUTPUT_PATH}")

    # ── export Excel + JSON audit ──────────────────────────────────────────────
    _export_audit(
        audit_rows       = audit_rows,
        paper_summary    = paper_summary,
        strategy_counts  = strategy_counts,
        total_equations  = total_equations,
        meanings_found   = meanings_found,
    )
    print()


def _export_audit(
    audit_rows: list,
    paper_summary: dict,
    strategy_counts: Counter,
    total_equations: int,
    meanings_found: int,
) -> None:
    """Write Excel and JSON audit exports to EXPORT_DIR."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── fill colours ──────────────────────────────────────────────────────────
    _RED    = PatternFill("solid", fgColor="FFCCCC")
    _GREEN  = PatternFill("solid", fgColor="CCFFCC")
    _BLUE   = PatternFill("solid", fgColor="CCE5FF")
    _ORANGE = PatternFill("solid", fgColor="FFE5CC")
    _YELLOW = PatternFill("solid", fgColor="FFFFCC")
    _HEADER = PatternFill("solid", fgColor="D9D9D9")
    _BOLD   = Font(bold=True)

    _STRAT_COLOUR = {
        "citation_match":    _GREEN,
        "gazetteer_match":   _BLUE,
        "definitional_match": _ORANGE,
        "tfidf_rank":        _YELLOW,
        "none":              _RED,
    }

    wb = openpyxl.Workbook()

    # ── Sheet 1: Summary (one row per paper) ──────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "Summary"

    sum_headers = [
        "arxiv_id", "source", "total_eqs",
        "meanings_found", "meanings_none",
        "citation_match", "gazetteer_match", "definitional_match",
        "tfidf_rank", "none_found",
    ]
    ws_sum.append(sum_headers)
    for cell in ws_sum[1]:
        cell.fill = _HEADER
        cell.font = _BOLD

    # per-paper strategy breakdown: count from audit_rows
    paper_strat: dict = {}
    for row in audit_rows:
        pid  = row["arxiv_id"]
        strat = row["strategy"]
        if pid not in paper_strat:
            paper_strat[pid] = Counter()
        paper_strat[pid][strat] += 1

    for arxiv_id, info in paper_summary.items():
        sc = paper_strat.get(arxiv_id, Counter())
        ws_sum.append([
            arxiv_id,
            info["source"],
            info["total_eqs"],
            info["meanings_found"],
            info["meanings_none"],
            sc["citation_match"],
            sc["gazetteer_match"],
            sc["definitional_match"],
            sc["tfidf_rank"],
            sc["none"],
        ])
        # highlight rows with at least one "none found"
        if info["meanings_none"] > 0:
            for cell in ws_sum[ws_sum.max_row]:
                cell.fill = _RED

    # totals row
    ws_sum.append([
        "TOTAL", "", total_equations, meanings_found,
        total_equations - meanings_found,
        strategy_counts["citation_match"],
        strategy_counts["gazetteer_match"],
        strategy_counts["definitional_match"],
        strategy_counts["tfidf_rank"],
        strategy_counts["none"],
    ])
    for cell in ws_sum[ws_sum.max_row]:
        cell.font = _BOLD

    # column widths
    for col, width in zip("ABCDEFGHIJ", [14, 6, 10, 14, 13, 15, 16, 20, 12, 11]):
        ws_sum.column_dimensions[col].width = width

    # ── Sheet 2: Equations (one row per equation) ─────────────────────────────
    ws_eq = wb.create_sheet("Equations")

    eq_headers = [
        "arxiv_id", "source", "eq_num",
        "latex", "meaning", "strategy",
        "symbols_regex_cnt", "symbols_dep_cnt",
        "symbols_dep_keys", "symbols_dep_preview",
    ]
    ws_eq.append(eq_headers)
    for cell in ws_eq[1]:
        cell.fill = _HEADER
        cell.font = _BOLD

    for row in audit_rows:
        ws_eq.append([
            row["arxiv_id"],
            row["source"],
            row["eq_num"],
            row["latex"],
            row["meaning"],
            row["strategy"],
            row["symbols_regex_cnt"],
            row["symbols_dep_cnt"],
            row["symbols_dep_keys"],
            row["symbols_dep_preview"],
        ])
        fill = _STRAT_COLOUR.get(row["strategy"], _RED)
        for cell in ws_eq[ws_eq.max_row]:
            cell.fill = fill

    for col, width in zip("ABCDEFGHIJ", [14, 6, 7, 40, 60, 20, 17, 16, 30, 60]):
        ws_eq.column_dimensions[col].width = width

    # ── Sheet 3: None Found (filtered view) ───────────────────────────────────
    ws_none = wb.create_sheet("NoneFound")
    ws_none.append(["arxiv_id", "source", "eq_num", "latex", "symbols_dep_keys"])
    for cell in ws_none[1]:
        cell.fill = _HEADER
        cell.font = _BOLD

    for row in audit_rows:
        if row["strategy"] == "none":
            ws_none.append([
                row["arxiv_id"],
                row["source"],
                row["eq_num"],
                row["latex"],
                row["symbols_dep_keys"],
            ])
            for cell in ws_none[ws_none.max_row]:
                cell.fill = _RED

    for col, width in zip("ABCDE", [14, 6, 7, 50, 40]):
        ws_none.column_dimensions[col].width = width

    # ── save Excel ────────────────────────────────────────────────────────────
    xlsx_path = EXPORT_DIR / f"meaning_audit_{stamp}.xlsx"
    wb.save(xlsx_path)
    print(f"  Excel  -> {xlsx_path}")

    # ── save JSON audit ───────────────────────────────────────────────────────
    audit_json = {
        "meta": {
            "generated":          stamp,
            "papers":             len(paper_summary),
            "total_equations":    total_equations,
            "meanings_found":     meanings_found,
            "meanings_none":      total_equations - meanings_found,
            "strategy_citation":     strategy_counts["citation_match"],
            "strategy_gazetteer":    strategy_counts["gazetteer_match"],
            "strategy_definitional": strategy_counts["definitional_match"],
            "strategy_tfidf":        strategy_counts["tfidf_rank"],
            "strategy_none":         strategy_counts["none"],
        },
        "papers": {},
    }

    for row in audit_rows:
        pid = row["arxiv_id"]
        if pid not in audit_json["papers"]:
            audit_json["papers"][pid] = {
                "source":    paper_summary[pid]["source"],
                "equations": {},
            }
        audit_json["papers"][pid]["equations"][row["eq_num"]] = {
            "latex":               row["latex"],
            "meaning":             row["meaning"],
            "strategy":            row["strategy"],
            "symbols_regex_cnt":   row["symbols_regex_cnt"],
            "symbols_dep_cnt":     row["symbols_dep_cnt"],
            "symbols_dep_keys":    row["symbols_dep_keys"].split(", ") if row["symbols_dep_keys"] else [],
            "symbols_dep_preview": row["symbols_dep_preview"],
        }

    json_path = EXPORT_DIR / f"meaning_audit_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False)
    print(f"  JSON   -> {json_path}")


if __name__ == "__main__":
    main()
