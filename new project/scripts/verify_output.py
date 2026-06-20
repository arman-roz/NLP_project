"""
scripts/verify_output.py

One-shot check: is this eq_final.json the new spec-compliant output or the old baseline?

Prints PASS (new format) or FAIL (old format) with a one-line reason.

Usage
-----
    python project/scripts/verify_output.py [path/to/eq_final.json]
"""

import json
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_PATH  = _SCRIPTS_DIR.parent / "data" / "output" / "eq_final.json"


def verify(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    checks = {
        "relations_spec_format": 0,   # {grade, description} pairs
        "relations_old_format":  0,   # list values (cites / shares_symbol_with)
        "strategy_weighted":     0,   # audit-trail has strategy=weighted
        "strategy_old":          0,   # audit-trail has strategy=tfidf_rank etc.
        "symbol_reject_logged":  0,   # symbol_reject key present anywhere
        "symbol_source_mathml":  0,   # MathML <mi> in symbol_source audit key
    }

    for eqs in data.values():
        for entry in eqs.values():
            rels  = entry.get("relations", {})
            trail = entry.get("audit-trail", {}) or {}

            for v in rels.values():
                if isinstance(v, dict) and "grade" in v:
                    checks["relations_spec_format"] += 1
                elif isinstance(v, list):
                    checks["relations_old_format"] += 1

            meaning_audit = trail.get("meaning", "")
            if "strategy=weighted" in meaning_audit:
                checks["strategy_weighted"] += 1
            elif re.search(r'strategy=(tfidf|gazetteer|citation)', meaning_audit):
                checks["strategy_old"] += 1

            for k in trail:
                if k.startswith("symbol_reject"):
                    checks["symbol_reject_logged"] += 1
                    break
            if "MathML" in trail.get("symbol_source", ""):
                checks["symbol_source_mathml"] += 1

    old_signals = checks["relations_old_format"] + checks["strategy_old"]
    new_signals = (
        checks["relations_spec_format"]
        + checks["strategy_weighted"]
        + checks["symbol_reject_logged"]
        + checks["symbol_source_mathml"]
    )

    print(f"File: {path}")
    print(f"  relations spec-format pairs : {checks['relations_spec_format']}")
    print(f"  relations old-format lists  : {checks['relations_old_format']}")
    print(f"  strategy=weighted           : {checks['strategy_weighted']}")
    print(f"  strategy=old (tfidf/gaz)    : {checks['strategy_old']}")
    print(f"  symbol_reject logged        : {checks['symbol_reject_logged']}")
    print(f"  MathML <mi> symbol source   : {checks['symbol_source_mathml']}")
    print()

    if old_signals > 0 and new_signals == 0:
        print("RESULT: FAIL — this is the OLD baseline output (submit this = hard schema failure)")
    elif old_signals > 0:
        print("RESULT: MIXED — old and new format signals both present; investigate")
    else:
        print("RESULT: PASS — this is the NEW spec-compliant output (safe to submit)")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    verify(path)
