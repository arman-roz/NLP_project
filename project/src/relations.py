"""
src/relations.py

Stage 3 (partial): extract relations between equations in the same paper.

Two relation types are produced, both purely from the paper's own text —
no text generation, no external lookups, no LLM calls.

Relation types
--------------
cites
    Equation numbers explicitly referenced in this equation's surrounding
    context text.  Detected by regex matching patterns such as
    "Eq. (1)", "eq. (2)", "Eqs. (1)-(3)".  Only numbers that actually
    exist as equations in the same paper are kept (noise filter).

    Example:
        Context: "Using the result from Eq. (1), we obtain..."
        → {"cites": ["1"]}

shares_symbol_with
    Other equations in the same paper whose extracted symbol set overlaps
    with this equation's symbol set by at least one key.  Computed after
    all equations in the paper are processed (requires a second pass).

    Example:
        Equation (1) uses symbols {"H", "omega", "rho"}
        Equation (3) uses symbols {"H", "kappa"}
        → equation (1): {"shares_symbol_with": ["3"]}
        → equation (3): {"shares_symbol_with": ["1"]}

Why these two relation types
-----------------------------
Both relations are fully extractive and verifiable from the source text:

- cites: the paper's authors write "Eq. (N)" explicitly; no inference
  needed.  In a knowledge graph, this edge represents a logical dependency
  between equations.

- shares_symbol_with: a shared symbol key means both equations operate
  in the same mathematical vocabulary.  This edge captures the semantic
  coupling between equations that reuse the same variable — essential for
  tracing how a quantity evolves across a paper's derivation.

Output shape per equation (inside the relations field)
------------------------------------------------------
{
  "cites":             ["1", "3"],
  "shares_symbol_with": ["2", "5"]
}

Empty lists indicate no relation of that type was found — the keys are
always present so downstream consumers do not need to check for their
existence.
"""

import logging
import re
from typing import Dict, List

from .audit import AuditTrail

logger = logging.getLogger(__name__)

# Matches explicit equation references in prose:
#   "Eq. (1)"  "eq. (2)"  "Eqs. (1)-(3)"  "Equation (A.2)"
# Group 1: number from "Eq./Eqs./Equation (N)" form
# Group 2: bare "(N)" form — only when surrounded by non-paren context
_EQ_REF_RE = re.compile(
    r"[Ee]quations?\s*\(\s*([\dA-Za-z]+(?:[.\-][\dA-Za-z]+)*)\s*\)"
    r"|[Ee]qs?\.?\s*\(\s*([\dA-Za-z]+(?:[.\-][\dA-Za-z]+)*)\s*\)"
    r"|(?<!\()\(\s*([\dA-Za-z]+(?:[.\-][\dA-Za-z]+)*)\s*\)(?!\s*\))"
)


class RelationExtractor:
    """Extract relations between equations in the same paper.

    Usage pattern
    -------------
    Call :meth:`extract` once per equation to get citation relations.
    Call :meth:`compute_shared_symbols` once after all equations in a
    paper have been processed (meaning + symbols extracted) to fill in
    the ``shares_symbol_with`` field.

    Examples
    --------
    >>> re = RelationExtractor()
    >>> rels = re.extract("2", ctx_b, ctx_a, ["1", "2", "3"], audit)
    >>> rels
    {"cites": ["1"], "shares_symbol_with": []}
    """

    def extract(
        self,
        eq_num: str,
        context_before: str,
        context_after: str,
        paper_eq_nums: List[str],
        audit: AuditTrail,
    ) -> Dict:
        """Find equation numbers cited in the surrounding context.

        Parameters
        ----------
        eq_num : str
            The current equation's own number (excluded from results).
        context_before : str
            Prose text immediately before the equation.
        context_after : str
            Prose text immediately after the equation.
        paper_eq_nums : list of str
            All equation numbers found in this paper — used to validate
            that a cited number is a real equation, not a figure or table.
        audit : AuditTrail
            Receives one log entry if any citations are found.

        Returns
        -------
        dict
            ``{"cites": [...], "shares_symbol_with": []}``
            ``shares_symbol_with`` is always empty here; it is filled in
            later by :meth:`compute_shared_symbols`.
        """
        context = f"{context_before} {context_after}"
        cited: List[str] = []

        for m in _EQ_REF_RE.finditer(context):
            num = m.group(1) or m.group(2) or m.group(3)
            if num and num != eq_num and num in paper_eq_nums and num not in cited:
                cited.append(num)

        if cited:
            audit.log("relations_cites", f"eq ({eq_num}) cites: {cited}")
        else:
            audit.log("relations_cites", f"eq ({eq_num}): no citations found in context")

        return {"cites": cited, "shares_symbol_with": []}

    def compute_shared_symbols(
        self, paper_equations: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """Fill the ``shares_symbol_with`` relation for every equation.

        Must be called after all equations in the paper have been
        processed so that every equation has a populated ``symbols`` dict.

        Parameters
        ----------
        paper_equations : dict
            ``{eq_num: {"symbols": {sym_key: definition, ...}, ...}}``
            as built up during Stage 3 processing.

        Returns
        -------
        dict
            ``{eq_num: [list_of_other_eq_nums_sharing_a_symbol]}``
            Empty list for equations with no symbols or no overlap.
        """
        # build symbol-key sets per equation
        sym_sets: Dict[str, set] = {
            eq_num: set(data.get("symbols", {}).keys())
            for eq_num, data in paper_equations.items()
        }

        shared: Dict[str, List[str]] = {}
        eq_nums = list(sym_sets.keys())

        for eq_num in eq_nums:
            my_syms = sym_sets[eq_num]
            if not my_syms:
                shared[eq_num] = []
                continue
            others = [
                other for other in eq_nums
                if other != eq_num and bool(my_syms & sym_sets[other])
            ]
            shared[eq_num] = others

            if others:
                logger.debug(
                    "eq (%s) shares symbols with: %s", eq_num, others
                )

        return shared
