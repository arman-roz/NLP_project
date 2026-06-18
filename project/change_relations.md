# Relations Module — Design and Implementation Notes

This document covers the design decisions, analysis, and implementation of
`src/relations.py` — the module responsible for identifying relationships
between equations within the same paper.

---

## What the relations field contains

The `relations` field for each equation is a dict with two keys, both of
which are always present (empty list if no relation was found):

```json
{
  "cites":              ["1", "3"],
  "shares_symbol_with": ["2", "5"]
}
```

All relation values are equation number strings that correspond to real
equations in the same paper.

---

## Why extraction only (no generation)

Same constraint as `meaning.py` and `symbols.py`: the project rules forbid
text generation at runtime.  Both relation types are derived purely from the
paper's own text and the symbol sets produced by `symbols.py`.  No inference,
no LLM calls, no external resources.

---

## Relation type 1 — `cites`

### Definition

An equation `(N)` cites another equation `(M)` if the text immediately
surrounding `(N)` (context_before or context_after) contains an explicit
reference to `(M)` such as:

- `"Eq. (1)"` or `"eq. (1)"`
- `"Eqs. (1)-(3)"` or `"Equation (A.2)"`
- A bare `"(M)"` form when not part of a larger parenthesised expression

### Detection method

A single compiled regex with three alternation branches handles all the
written forms found in physics papers:

```
[Ee]quations?\s*\(\s*(number)\s*\)
[Ee]qs?\.?\s*\(\s*(number)\s*\)
(?<!\()\(\s*(number)\s*\)(?!\s*\))   ← bare form, not inside double parens
```

where `number` matches patterns like `1`, `A.2`, `S1`, `1a`.

### Noise filter

Only numbers that appear in the paper's own equation set are kept.  This
filters out figure numbers, table numbers, and section references that
accidentally match the same pattern (e.g. a paper that writes
"see Table (3)").  If the matched number is not in the paper's equation
list, it is silently discarded.

### Why this relation matters for a knowledge graph

`cites` edges represent **logical dependencies** between equations.  In a
derivation, if the proof of equation (3) explicitly invokes equation (1),
that is a directed dependency edge in the knowledge graph:
`(3) → cites → (1)`.  These edges allow graph traversal to answer questions
like "which earlier equations does this result depend on?" — a core
knowledge-graph query that would require reading the full paper without it.

### Example

```
Context after eq (2):
"The first term of Equation (2) is the deflection signal and corresponds
to the amplified displacement of Eq. (1)."

→ eq (2) cites: ["1"]
```

---

## Relation type 2 — `shares_symbol_with`

### Definition

Two equations share a symbol if they have at least one common key in their
`symbols` dicts.  Symbol keys are the LaTeX command names without backslash
(e.g. `"H"`, `"omega"`, `"rho"`), as produced by `symbols.py`.

### Computation method

`shares_symbol_with` requires all equations in a paper to be fully annotated
first (meaning + symbols extracted).  It is computed in a second pass over
the paper's equations:

```python
for eq_num in paper_equations:
    my_keys  = set(paper_equations[eq_num]["symbols"].keys())
    overlaps = [
        other for other in paper_equations
        if other != eq_num
        and bool(my_keys & set(paper_equations[other]["symbols"].keys()))
    ]
    paper_equations[eq_num]["relations"]["shares_symbol_with"] = overlaps
```

The relation is symmetric: if equation (1) shares a symbol with equation (3),
equation (3) also lists equation (1).

### Why this relation matters for a knowledge graph

`shares_symbol_with` edges represent **semantic coupling** between equations.
When two equations both use `omega` (frequency) or `rho` (density matrix),
they operate in the same mathematical vocabulary.  In a knowledge graph, these
edges cluster equations by the physical quantities they involve, allowing
queries like "which equations describe the same observable?" or "which
equations together form a complete system for variable `H`?"

This relation is particularly valuable for identifying equation systems
(e.g. coupled differential equations) even when the paper does not
explicitly label them as a system.

### Example

```
Equation (1): symbols = {"H": "Hamiltonian", "omega": "frequency"}
Equation (2): symbols = {"H": "Hamiltonian", "kappa": "decay rate"}
Equation (5): symbols = {"omega": "resonance frequency", "kappa": "loss rate"}

→ eq (1) shares_symbol_with: ["2", "5"]
→ eq (2) shares_symbol_with: ["1", "5"]
→ eq (5) shares_symbol_with: ["1", "2"]
```

---

## Two-pass design in main.py

The two relation types are computed at different points in the pipeline:

**Pass 1 (per equation):** `extract()` is called immediately after context
extraction.  It scans the context text for citation patterns and returns
`{"cites": [...], "shares_symbol_with": []}`.

**Pass 2 (per paper):** `compute_shared_symbols()` is called after all
equations in the paper have been processed.  It fills in the
`shares_symbol_with` list for every equation and updates the relations
dict in place.

This two-pass design is necessary because `shares_symbol_with` depends on
the symbols dicts of all other equations in the paper — information that does
not exist until the full paper has been processed.

---

## Audit trail entries

The `cites` relation is logged whether or not citations are found:

```
relations_cites: eq (2) cites: ['1']
relations_cites: eq (3): no citations found in context
```

`shares_symbol_with` is not logged to the audit trail (it is derived
deterministically from the symbols dicts, which are already audited by
`symbols.py`).

---

## Design decisions

### Why not include more relation types?

Several other relation types were considered and rejected:

**Derivation cue words** (e.g. "therefore", "it follows that", "substituting
into"): These words appear frequently in physics papers but rarely have a
consistent enough form to be matched reliably without NLP generation.
A regex approach would produce many false positives.

**Cross-paper symbol sharing**: Sharing a symbol key like `H` across different
papers is not meaningful — `H` is the Hamiltonian in one paper and the
Heaviside function in another.  Within-paper sharing is meaningful because
the paper's own definitions constrain what each symbol means.

**Equation type classification** (e.g. "this is a Hamiltonian", "this is a
master equation"): This would require text generation or a pre-trained
classifier, both of which are forbidden.

### Why use symbol keys (not full definitions) for overlap?

Symbol keys (`"H"`, `"omega"`) are more reliable than definitions
(`"total Hamiltonian"` vs `"Hamiltonian of the system"`) for detecting
shared variables.  Two equations using the same letter in the same paper
almost certainly refer to the same physical quantity, even if the extracted
definitions differ slightly in wording.  Using keys avoids false negatives
from minor phrasing differences.

### Why filter `cites` against the paper's own equation list?

Physics papers often contain numbered items other than equations: figures,
tables, algorithms, and sections.  Without filtering, a reference to
"Table (3)" or "Figure (2)" would appear as a false equation citation.
The filter keeps only numbers that correspond to actually extracted equations
in the paper.

---

## Limitations

- **cites only captures explicit references**: If an author writes "plugging
  the previous result into..." without naming an equation number, the relation
  is not detected.  The `cites` relation captures formal cross-references only.

- **shares_symbol_with depends on symbol extraction quality**: If `symbols.py`
  misses a symbol definition (e.g. due to PDF extraction quality or a
  non-standard definition pattern), the overlap computation will not see that
  symbol and may miss a sharing relation.

- **No cross-paper relations**: Both relation types are within-paper only.
  Cross-paper equations knowledge graphs require entity linking (matching
  variable names across papers), which is outside the scope of this project.

- **Symmetric but not transitive**: `shares_symbol_with` is symmetric
  (if (1) shares with (3), then (3) shares with (1)) but not explicitly
  transitive (if (1) shares with (3) and (3) shares with (5), that does not
  automatically mean (1) shares with (5) — that is only true if there is
  a direct symbol overlap between (1) and (5)).
