"""
scripts/measure.py

Post-pipeline quality measurement for eq_final.json.

Prints a structured report covering:
  - Corpus counts (papers, equations, source breakdown)
  - Per-field empty rates (meaning, symbols, relations)
  - Symbol garbage rate (single-char or pure-operator definitions)
  - Meaning render-noise rate (artifact tokens in meaning strings)
  - Relations format validity (grade/description spec compliance)
  - Strategy distribution (which audit signals fired for meaning/symbols)
  - Equation count distribution (papers with 0/1…7 equations)

Targets
-------
  symbol garbage   < 2 %
  meaning noise    < 3 %
  relations valid  100 %
  equations total  350–356

Usage
-----
    python project/scripts/measure.py [path/to/eq_final.json]

Defaults to data/output/eq_final.json relative to the project root.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── paths ─────────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPTS_DIR.parent
DEFAULT_PATH  = _PROJECT_DIR / "data" / "output" / "eq_final.json"

# ── constants ─────────────────────────────────────────────────────────────────
VALID_GRADES = frozenset({"none", "strong", "potential"})

# Artifact tokens that should not appear in cleaned meaning strings.
# Must mirror all patterns that _clean_context() in meaning.py strips.
_ARTIFACT_RE = re.compile(
    r'\bitalic[-_]\S+'                                    # italic_x and italic-x
    r'|\bstart_[A-Z]\w*'                                  # LaTeXML start_ROW
    r'|\bend_[A-Z]\w*'                                    # LaTeXML end_ARRAY
    r'|\broad_[A-Za-z_]+'                                 # LaTeXML road_*
    r'|\bcaligraphic_\S+'                                 # caligraphic_W etc.
    r'|\bbold_[a-z]\S*'                                   # bold_italic_P etc.
    r'|[\U0001D400-\U0001D7FF]'                           # Unicode math block
    r'|[_^]\{[^{}]*\}'                                    # _{t}, ^{2} subscript
    r'|\\[a-zA-Z]+(?:\{[^{}]*\})*'                       # \rho, \mathcal{A}
    r'|\b(?:subscript|superscript|postsubscript|postsuperscript)\b'  # bare words
    r'|\xa0'                                              # non-breaking space
)

# Patterns that indicate a "garbage" symbol definition:
#   - Single alphabetic character (too short to be meaningful)
#   - Pure operator / punctuation
#   - Starts with a backslash (LaTeX command leaked into definition)
_GARBAGE_DEF_RE = re.compile(
    r'^[a-zA-Z]$'            # single letter
    r'|^[\\^_{}()\[\]|]+$'  # pure LaTeX structure
    r'|^\\.+'                # starts with backslash
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(num: int, denom: int) -> str:
    return f"{100.0 * num / denom:.1f}%" if denom else "n/a"


def _bar(frac: float, width: int = 30) -> str:
    filled = round(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _audit_strategies(audit_trail: Dict[str, str]) -> List[str]:
    """Return list of strategy labels recorded in an audit trail."""
    strats = []
    for key, val in audit_trail.items():
        if key == "meaning":
            m = re.search(r'strategy=(\S+)', val)
            if m:
                strats.append("meaning:" + m.group(1))
        elif key.startswith("symbol") and not key.startswith("symbol_reject"):
            strats.append("symbol:regex")
        elif key == "symbol_dep":
            strats.append("symbol:dep")
    return strats


# ── main measurement ──────────────────────────────────────────────────────────

def measure(data: Dict) -> None:
    # ── corpus counts ─────────────────────────────────────────────────────────
    n_papers    = len(data)
    n_equations = sum(len(eqs) for eqs in data.values())
    n_empty     = sum(1 for eqs in data.values() if not eqs)

    # Equation count distribution
    eq_count_dist: Counter = Counter(len(eqs) for eqs in data.values())

    print("=" * 60)
    print("EQUATIONS KNOWLEDGE GRAPH — PIPELINE QUALITY REPORT")
    print("=" * 60)
    print(f"\nCorpus")
    print(f"  Papers processed      : {n_papers}")
    print(f"  Papers with 0 eqs     : {n_empty}")
    print(f"  Equations total       : {n_equations}")
    if n_equations < 350:
        print(f"  ⚠  BELOW target 350 (got {n_equations})")
    elif n_equations > 356:
        print(f"  ⚠  ABOVE target 356 (got {n_equations})")
    else:
        print(f"  ✓  Within target range 350–356")

    print(f"\nEqs per paper distribution:")
    for cnt in sorted(eq_count_dist):
        papers = eq_count_dist[cnt]
        print(f"  {cnt:>2} eqs : {papers:>4} papers  {_bar(papers / n_papers, 20)}")

    # ── per-field empty rates ─────────────────────────────────────────────────
    n_meaning_empty  = 0
    n_symbols_empty  = 0
    n_relations_none = 0   # relations where ALL pairs are grade "none"
    n_pairs_total    = 0
    n_pairs_nonnone  = 0

    for arxiv_id, eqs in data.items():
        for eq_num, entry in eqs.items():
            meaning = entry.get("meaning", "")
            symbols = entry.get("symbols", {})
            rels    = entry.get("relations", {})

            if not meaning:
                n_meaning_empty += 1
            if not symbols:
                n_symbols_empty += 1

            eq_rels_all_none = all(
                isinstance(r, dict) and r.get("grade", "none") == "none"
                for r in rels.values()
            ) if rels else True
            if eq_rels_all_none:
                n_relations_none += 1

            for other_eq, rel in rels.items():
                n_pairs_total += 1
                if isinstance(rel, dict) and rel.get("grade", "none") != "none":
                    n_pairs_nonnone += 1

    print(f"\nPer-field empty / coverage rates  (N = {n_equations} equations)")
    me_frac = n_meaning_empty / n_equations if n_equations else 0
    sy_frac = n_symbols_empty / n_equations if n_equations else 0
    rel_frac = n_relations_none / n_equations if n_equations else 0
    print(f"  meaning empty         : {n_meaning_empty:>5} / {n_equations}  "
          f"= {_pct(n_meaning_empty, n_equations):<7}  {_bar(me_frac)}")
    print(f"  symbols empty         : {n_symbols_empty:>5} / {n_equations}  "
          f"= {_pct(n_symbols_empty, n_equations):<7}  {_bar(sy_frac)}")
    print(f"  rels all-none         : {n_relations_none:>5} / {n_equations}  "
          f"= {_pct(n_relations_none, n_equations):<7}  {_bar(rel_frac)}")
    print(f"  relation pairs total  : {n_pairs_total}")
    print(f"  pairs with signal     : {n_pairs_nonnone}  "
          f"({_pct(n_pairs_nonnone, n_pairs_total)})")

    # ── symbol garbage rate ───────────────────────────────────────────────────
    n_sym_total   = 0
    n_sym_garbage = 0
    garbage_examples: List[Tuple[str, str]] = []

    for arxiv_id, eqs in data.items():
        for eq_num, entry in eqs.items():
            for sym, defn in entry.get("symbols", {}).items():
                n_sym_total += 1
                if _GARBAGE_DEF_RE.match(defn.strip()):
                    n_sym_garbage += 1
                    if len(garbage_examples) < 5:
                        garbage_examples.append((sym, defn))

    garbage_frac = n_sym_garbage / n_sym_total if n_sym_total else 0
    flag = "✓" if garbage_frac < 0.02 else "⚠ TARGET < 2 %"
    print(f"\nSymbol garbage rate  (N = {n_sym_total} symbols)")
    print(f"  garbage defs          : {n_sym_garbage:>5} / {n_sym_total}  "
          f"= {_pct(n_sym_garbage, n_sym_total):<7}  {flag}")
    if garbage_examples:
        print("  examples:")
        for sym, defn in garbage_examples:
            print(f"    sym={sym!r:>12}  def={defn!r}")

    # ── meaning render-noise rate ─────────────────────────────────────────────
    n_meaning_total = 0
    n_meaning_noisy = 0
    noise_examples: List[str] = []

    for arxiv_id, eqs in data.items():
        for eq_num, entry in eqs.items():
            meaning = entry.get("meaning", "")
            if not meaning:
                continue
            n_meaning_total += 1
            if _ARTIFACT_RE.search(meaning):
                n_meaning_noisy += 1
                if len(noise_examples) < 3:
                    noise_examples.append(meaning[:80])

    noise_frac = n_meaning_noisy / n_meaning_total if n_meaning_total else 0
    flag = "✓" if noise_frac < 0.03 else "⚠ TARGET < 3 %"
    print(f"\nMeaning render-noise rate  (N = {n_meaning_total} non-empty meanings)")
    print(f"  noisy meanings        : {n_meaning_noisy:>5} / {n_meaning_total}  "
          f"= {_pct(n_meaning_noisy, n_meaning_total):<7}  {flag}")
    if noise_examples:
        print("  examples:")
        for ex in noise_examples:
            print(f"    {ex!r}")

    # ── relations format validity ─────────────────────────────────────────────
    n_rel_valid   = 0
    n_rel_invalid = 0
    rel_errors: List[str] = []

    for arxiv_id, eqs in data.items():
        all_eq_nums = list(eqs.keys())
        for eq_num, entry in eqs.items():
            rels = entry.get("relations", {})
            expected = {n for n in all_eq_nums if n != eq_num}
            present  = set(rels.keys())
            for n in expected - present:
                n_rel_invalid += 1
                if len(rel_errors) < 3:
                    rel_errors.append(
                        f"{arxiv_id}/{eq_num}: missing relation to {n!r}"
                    )
            for other_eq, rel in rels.items():
                if not isinstance(rel, dict):
                    n_rel_invalid += 1
                    continue
                grade = rel.get("grade", "")
                desc  = rel.get("description", "")
                if grade in VALID_GRADES and isinstance(desc, str):
                    n_rel_valid += 1
                else:
                    n_rel_invalid += 1
                    if len(rel_errors) < 3:
                        rel_errors.append(
                            f"{arxiv_id}/{eq_num}~{other_eq}: "
                            f"grade={grade!r}"
                        )

    n_rel_checked = n_rel_valid + n_rel_invalid
    flag = "✓" if n_rel_invalid == 0 else f"⚠  {n_rel_invalid} violations"
    print(f"\nRelations format validity  (N = {n_rel_checked} relation entries)")
    print(f"  valid entries         : {n_rel_valid:>6} / {n_rel_checked}  {flag}")
    if rel_errors:
        print("  first errors:")
        for err in rel_errors:
            print(f"    {err}")

    # ── grade distribution ────────────────────────────────────────────────────
    grade_dist: Counter = Counter()
    for arxiv_id, eqs in data.items():
        for eq_num, entry in eqs.items():
            for rel in entry.get("relations", {}).values():
                if isinstance(rel, dict):
                    grade_dist[rel.get("grade", "?")] += 1

    print(f"\nRelation grade distribution  (N = {sum(grade_dist.values())} pairs)")
    for grade in ("strong", "potential", "none"):
        n   = grade_dist.get(grade, 0)
        tot = sum(grade_dist.values())
        print(f"  {grade:<12} : {n:>6}  ({_pct(n, tot)})")

    # ── meaning strategy distribution ────────────────────────────────────────
    strat_dist: Counter = Counter()
    for arxiv_id, eqs in data.items():
        for eq_num, entry in eqs.items():
            trail = entry.get("audit-trail", {})
            if isinstance(trail, dict):
                for key, val in trail.items():
                    if key == "meaning":
                        m = re.search(r'strategy=(\S+)', val)
                        if m:
                            strat_dist[m.group(1)] += 1
                            break

    if strat_dist:
        strat_total = sum(strat_dist.values())
        print(f"\nMeaning strategy distribution  (N = {strat_total} equations attempted)")
        print(f"  ({n_meaning_total} returned non-empty meaning; "
              f"{strat_total - n_meaning_total} scored below threshold)")
        for strat, cnt in strat_dist.most_common():
            print(f"  {strat:<24} : {cnt:>5}  ({_pct(cnt, strat_total)})")

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {path} ...")
    data = _load(path)
    measure(data)


if __name__ == "__main__":
    main()
