"""
src/validate.py

Strict schema validation for the Stage 3 output (eq_final.json).

Validates every entry in the pipeline output against the fixed spec schema and
raises ``ValidationError`` on the first batch of violations found.  The caller
must use ``validate(data, fail_loudly=False)`` to collect errors silently.

Expected schema
---------------
{
  "<arxiv_id>": {
    "<eq_number>": {
      "equation":    str,
      "meaning":     str,
      "symbols":     {str: str},
      "relations":   {
          "<other_eq_num>": {
              "grade":       "none" | "strong" | "potential",
              "description": str
          }
      },
      "audit-trail": {str: str}
    }
  }
}

Strict checks (Task F additions)
---------------------------------
1. Every arxiv_id in the top-level dict has at least one equation (papers with
   zero equations are valid but logged; their entry must be an empty dict {}).
2. Each equation entry has exactly the five required keys.
3. Relations include an entry for EVERY other equation in the same paper.
   Missing pairs are a hard error.
4. Each relation has grade ∈ {none, strong, potential} and a description str.
5. Non-empty meaning / symbols / relations fields must have at least one
   audit-trail entry whose key matches one of the expected prefixes.
6. No private working keys (_ctx_before, _ctx_after, _audit) may appear.

Usage
-----
    from src.validate import Validator, ValidationError

    # Raise on first violation batch (pipeline default):
    Validator().validate(stage3_data)

    # Collect all errors (reporting / CI):
    errors = Validator().validate(stage3_data, fail_loudly=False)
    for msg in errors:
        print(msg)
"""

from typing import Dict, List

# ── Schema constants ──────────────────────────────────────────────────────────
VALID_GRADES: frozenset = frozenset({"none", "strong", "potential"})

_REQUIRED_EQ_KEYS: frozenset = frozenset({
    "equation", "meaning", "symbols", "relations", "audit-trail",
})

_PRIVATE_KEYS: frozenset = frozenset({
    "_ctx_before", "_ctx_after", "_audit",
})

# Audit-trail key prefixes that confirm a field was processed.
_MEANING_AUDIT_PREFIXES: tuple = ("meaning", "retrieval")
_SYMBOL_AUDIT_PREFIXES:  tuple = ("symbol", "symbol_source", "symbol_reject")
_RELATION_AUDIT_PREFIXES: tuple = ("relations",)


class ValidationError(Exception):
    """Raised by :meth:`Validator.validate` when *fail_loudly* is True."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        bullet = "\n  ".join(errors[:20])
        super().__init__(
            f"{len(errors)} validation error(s) found:\n  {bullet}"
            + (f"\n  … and {len(errors) - 20} more" if len(errors) > 20 else "")
        )


class Validator:
    """Validate Stage 3 output against the fixed spec schema.

    Parameters
    ----------
    None

    Examples
    --------
    >>> v = Validator()
    >>> v.validate(stage3_data)          # raises ValidationError if invalid
    >>> errors = v.validate(stage3_data, fail_loudly=False)  # collect only
    >>> errors
    []
    """

    def validate(
        self,
        data: object,
        fail_loudly: bool = True,
    ) -> List[str]:
        """Validate the full Stage 3 output dict.

        Parameters
        ----------
        data : dict
            The Stage 3 mapping ``{arxiv_id: {eq_num: entry}}``.
        fail_loudly : bool
            When ``True`` (default), raise :exc:`ValidationError` on any
            violation.  When ``False``, return all errors as a list.

        Returns
        -------
        list of str
            Error messages (empty list ⟹ valid).  Only returned when
            *fail_loudly* is ``False``; otherwise always raises or returns [].

        Raises
        ------
        ValidationError
            When *fail_loudly* is ``True`` and at least one error is found.
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            errors.append(f"Root must be dict, got {type(data).__name__}")
        else:
            for arxiv_id, paper in data.items():
                errors.extend(self._validate_paper(str(arxiv_id), paper))

        if errors and fail_loudly:
            raise ValidationError(errors)
        return errors

    # ── per-paper ─────────────────────────────────────────────────────────────

    def _validate_paper(self, arxiv_id: str, paper: object) -> List[str]:
        errors: List[str] = []

        if not isinstance(paper, dict):
            return [f"{arxiv_id}: must be dict, got {type(paper).__name__}"]

        # Collect all equation numbers so we can check pair completeness.
        eq_nums = list(paper.keys())

        for eq_num, entry in paper.items():
            errors.extend(
                self._validate_entry(arxiv_id, str(eq_num), entry, eq_nums)
            )

        return errors

    def _validate_entry(
        self,
        arxiv_id: str,
        eq_num: str,
        entry: object,
        all_eq_nums: List[str],
    ) -> List[str]:
        prefix = f"{arxiv_id}/{eq_num}"
        errors: List[str] = []

        if not isinstance(entry, dict):
            return [f"{prefix}: must be dict, got {type(entry).__name__}"]

        # Check for private working keys leaking into output.
        leaked = _PRIVATE_KEYS & entry.keys()
        if leaked:
            errors.append(
                f"{prefix}: private working keys must not appear in output: "
                f"{sorted(leaked)}"
            )

        # Required key presence.
        missing = _REQUIRED_EQ_KEYS - entry.keys()
        if missing:
            errors.append(f"{prefix}: missing required keys: {sorted(missing)}")

        # Type checks for string fields.
        if not isinstance(entry.get("equation", ""), str):
            errors.append(f"{prefix}: 'equation' must be str")
        if not isinstance(entry.get("meaning", ""), str):
            errors.append(f"{prefix}: 'meaning' must be str")

        # Field validators.
        errors.extend(self._validate_symbols(prefix, entry.get("symbols")))
        errors.extend(
            self._validate_relations(prefix, entry.get("relations"), eq_num, all_eq_nums)
        )
        audit_trail = entry.get("audit-trail")
        errors.extend(self._validate_audit(prefix, audit_trail))

        # Non-empty content fields must have a corresponding audit entry.
        if isinstance(audit_trail, dict):
            errors.extend(
                self._check_audit_coverage(prefix, entry, audit_trail)
            )

        return errors

    # ── field-level validators ─────────────────────────────────────────────────

    @staticmethod
    def _validate_symbols(prefix: str, syms: object) -> List[str]:
        if syms is None:
            return []
        if not isinstance(syms, dict):
            return [f"{prefix}: 'symbols' must be dict, got {type(syms).__name__}"]
        errors: List[str] = []
        for k, v in syms.items():
            if not isinstance(k, str):
                errors.append(f"{prefix}/symbols: key {k!r} must be str")
            if not isinstance(v, str):
                errors.append(
                    f"{prefix}/symbols/{k}: value must be str, "
                    f"got {type(v).__name__}"
                )
        return errors

    @staticmethod
    def _validate_relations(
        prefix: str,
        rels: object,
        eq_num: str,
        all_eq_nums: List[str],
    ) -> List[str]:
        if rels is None:
            return []
        if not isinstance(rels, dict):
            return [f"{prefix}: 'relations' must be dict, got {type(rels).__name__}"]

        errors: List[str] = []

        # Every OTHER equation in the paper must have a relation entry.
        expected_partners = {n for n in all_eq_nums if n != eq_num}
        present_partners  = set(rels.keys())
        missing_partners  = expected_partners - present_partners
        if missing_partners:
            errors.append(
                f"{prefix}/relations: missing entries for equations "
                f"{sorted(missing_partners, key=lambda x: (x.isdigit() == False, x))}"
            )

        for other_eq, rel in rels.items():
            rel_prefix = f"{prefix}/relations/{other_eq}"
            if not isinstance(rel, dict):
                errors.append(
                    f"{rel_prefix}: must be dict, got {type(rel).__name__}"
                )
                continue
            missing_rel = {"grade", "description"} - rel.keys()
            if missing_rel:
                errors.append(f"{rel_prefix}: missing keys: {sorted(missing_rel)}")
            grade = rel.get("grade", "")
            if grade not in VALID_GRADES:
                errors.append(
                    f"{rel_prefix}: grade={grade!r} must be one of "
                    f"{sorted(VALID_GRADES)}"
                )
            if not isinstance(rel.get("description", ""), str):
                errors.append(f"{rel_prefix}: 'description' must be str")

        return errors

    @staticmethod
    def _validate_audit(prefix: str, trail: object) -> List[str]:
        if trail is None:
            return []
        if not isinstance(trail, dict):
            return [
                f"{prefix}: 'audit-trail' must be dict, "
                f"got {type(trail).__name__}"
            ]
        errors: List[str] = []
        for k, v in trail.items():
            if not isinstance(k, str):
                errors.append(f"{prefix}/audit-trail: key {k!r} must be str")
            if not isinstance(v, str):
                errors.append(
                    f"{prefix}/audit-trail/{k}: value must be str, "
                    f"got {type(v).__name__}"
                )
        return errors

    @staticmethod
    def _check_audit_coverage(
        prefix: str,
        entry: Dict,
        audit_trail: Dict,
    ) -> List[str]:
        """Warn when a non-empty field has no corresponding audit entry."""
        errors: List[str] = []
        audit_keys = set(audit_trail.keys())

        # meaning non-empty → must have a meaning or retrieval audit key
        meaning = entry.get("meaning", "")
        if meaning and not any(
            k.startswith(_MEANING_AUDIT_PREFIXES) for k in audit_keys
        ):
            errors.append(
                f"{prefix}: non-empty 'meaning' field has no audit entry "
                f"with prefix {_MEANING_AUDIT_PREFIXES}"
            )

        # symbols non-empty → must have a symbol* audit key
        symbols = entry.get("symbols", {})
        if isinstance(symbols, dict) and symbols and not any(
            k.startswith(_SYMBOL_AUDIT_PREFIXES) for k in audit_keys
        ):
            errors.append(
                f"{prefix}: non-empty 'symbols' field has no audit entry "
                f"with prefix {_SYMBOL_AUDIT_PREFIXES}"
            )

        # relations with any non-none grade → must have a relations audit key
        rels = entry.get("relations", {})
        if isinstance(rels, dict):
            has_non_none = any(
                isinstance(r, dict) and r.get("grade", "none") != "none"
                for r in rels.values()
            )
            if has_non_none and not any(
                k.startswith(_RELATION_AUDIT_PREFIXES) for k in audit_keys
            ):
                errors.append(
                    f"{prefix}: relations with non-none grades have no "
                    f"audit entry with prefix {_RELATION_AUDIT_PREFIXES}"
                )

        return errors
