# Symbols Module — Design and Implementation Notes

This document covers the design decisions, analysis, and implementation of `src/symbols.py` — the module responsible for identifying mathematical symbols in each equation and finding their definitions in the surrounding text.

---

## What the symbols field contains

The `symbols` field for each equation is a dict mapping each meaningful symbol (key: LaTeX command name without backslash, or bare letter) to a short phrase describing what that symbol represents, extracted verbatim from the paper:

```json
{
  "H":     "total Hamiltonian of the system",
  "omega": "resonance frequency",
  "psi":   "quantum state of the field"
}
```

Only symbols that have a definition found in the surrounding text are included.  Symbols with no definition are silently omitted — the field is never padded with guesses.

---

## Why extraction only (no generation)

Same constraint as `meaning.py`: the project rules forbid text generation.  Every description string must be a phrase copied verbatim from a sentence the paper's authors wrote.  The audit trail records which pattern matched and which phrase was extracted.

---

## Two runnable modes (for report comparison)

The `SymbolExtractor` takes a `mode` parameter:

| Mode | What it does |
|---|---|
| `"regex"` | Regex definitional patterns only |
| `"regex+dep"` | Regex first; spaCy dependency parsing as fallback for misses |

The two modes are designed to be compared:

```python
se_regex  = SymbolExtractor(mode="regex")
se_full   = SymbolExtractor(mode="regex+dep")

syms_r = se_regex.extract(latex, ctx_before, ctx_after, audit)
syms_d = se_full.extract(latex, ctx_before, ctx_after, audit)

# symbols found only by dependency parsing
extra = {k: v for k, v in syms_d.items() if k not in syms_r}
```

Running both modes on the full dataset and counting `len(extra)` across all equations gives a direct measure of how many additional definitions dependency parsing recovers.  This comparison belongs in the report.

---

## Step 1 — LaTeX symbol tokeniser

### What is a "meaningful symbol"

Not every token in a LaTeX equation is a symbol worth defining.  The tokeniser applies three filters:

**Keep:**
- `\alpha`, `\beta`, … `\omega`, `\Gamma` … `\Omega` — Greek letters (appear in `_GREEK_TEXT`)
- `\hbar` — reduced Planck constant
- Bare uppercase letters: `H`, `E`, `F`, `G`, `L`, `N`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
- Bare lowercase letters not in the skip set: `a`, `b`, `c`, `e`, `f`, `g`, `h`, `p`, `q`, `r`, `s`, `t`, `u`, `v`, `w`, `x`, `y`, `z`

**Skip (structural commands):**
`\frac`, `\begin`, `\end`, `\left`, `\right`, `\text`, `\mathcal`, `\mathbf`, `\hat`, `\sqrt`, `\sum`, `\int`, and ~60 others defined in `_SKIP_COMMANDS`.

**Skip (operator commands):**
`\cdot`, `\times`, `\leq`, `\partial`, `\nabla`, `\rightarrow`, `\dagger`, and all standard operators.

**Skip (index letters):**
`d` (differential), `i`, `j`, `k`, `l`, `m`, `n` (almost always indices, rarely worth defining).

### Output key format

- LaTeX command → key without backslash: `\alpha` → `"alpha"`, `\hbar` → `"hbar"`
- Bare letter → key is the letter itself: `H` → `"H"`, `E` → `"E"`

---

## Step 2 — Definition search

### Regex patterns (five templates)

Patterns are tried in order.  The first match whose captured group is 1–15 words long is used.

| Pattern name | Template | Example match |
|---|---|---|
| `where_is` | `where {SYM} is/denotes/represents …` | "where H is the total Hamiltonian" |
| `let_be` | `let {SYM} be …` | "let ψ be the initial state" |
| `sym_is` | `{SYM} is/denotes … the …` | "E denotes the energy eigenvalue" |
| `the_noun_sym` | `the <noun phrase> {SYM}` | "the coupling constant J" |
| `sym_the` | `{SYM}, the …` | "ρ, the density matrix" |

The `{SYM}` placeholder is expanded at match time to a regex fragment that matches all text forms of the symbol:
- Greek: `\alpha` → matches `α`, `alpha`, `$\alpha$`, `$alpha$`
- Single letter: `H` → matches `H`, `$H$`

This handles the common physics writing style of switching between Unicode and dollar-sign math notation within the same sentence.

### Sanity filter on captured definitions

Captured groups are checked for length: 1–15 words.  This rejects:
- Single-word captures that are too vague (e.g. "state" alone)
- Run-on captures that grabbed too many trailing words (a symptom of a missing sentence-ending punctuation)

### spaCy dependency parsing (regex+dep mode only)

When regex fails for a symbol, spaCy parses each sentence in the context window.  Inline math (`$...$`, `\cmd{...}`) is replaced with the placeholder `SYM` before parsing so spaCy sees clean English.

Two dependency patterns are used:

**Pattern A — nsubj + copula:**
```
X → nsubj → is → attr/xcomp
      ↑
  "X is the coupling strength"
```
The token matching the symbol is a nominal subject whose head verb lemmatises to "be", and the head has an attribute or complement child.

**Pattern B — appositive:**
```
X ← appos ← noun phrase
      ↑
  "the Hamiltonian H" or "H, the Hamiltonian"
```
The token is in an appositive relationship and the subtree of the appositive head gives the description.

Both patterns log to the audit trail with the key `symbol_dep` so dependency-parsed results are distinguishable from regex results in post-processing.

---

## Audit trail entries

Every symbol found logs one entry:

```
symbol: sym='alpha', definition='resonance frequency'
```

If dependency parsing was used as fallback:
```
symbol_dep: sym='H', dep=nsubj+attr, phrase='total Hamiltonian'
```

Symbols for which no definition was found produce no audit entry (silent omission).

---

## Design decisions

### Why skip common index letters (i, j, k, n, d)?

In quantum physics papers, these letters appear almost exclusively as summation indices, tensor indices, or differentials (`dn`, `dk`).  Searching for "where i is…" produces false matches like "where it is shown…".  Skipping them avoids noisy output.  If a paper uses `i` to mean current density (unusual), the definition simply won't appear — which is correct behaviour for a precision-oriented tool.

### Why 1–15 words for captured definitions?

- **Too short (0 words):** regex matched but captured nothing useful.
- **1 word:** acceptable — "where m is the mass" captures "mass".
- **15 words:** a phrase this long is likely to have grabbed trailing prose due to a missed sentence boundary.  Cutting at 15 keeps the output readable without a full sentence parser.

### Why not use the full sentence as the definition?

The full sentence often contains the equation itself, other symbols, and surrounding context that is not the definition of this specific symbol.  Extracting just the definitional phrase (the captured group) gives a cleaner, more focused output.

---

## Limitations

- **Symbol collision:** if `H` appears in a sentence like "let H be the Hamiltonian" but the paper also uses `H` as the Heaviside function in a different section, the first match wins.  The context window limits this risk — only sentences in the immediate neighbourhood of the equation are searched.
- **Non-standard notation:** some papers define symbols in tables or figure captions, which are not in the context window.  These definitions will be missed.
- **Multi-letter variables:** symbols like `QFI`, `SNR`, `NV` are not extracted by the current tokeniser (only single letters and Greek commands are kept).  This is a deliberate conservative choice to avoid false positives.
- **PDF papers:** context text from PDF is lower quality (no paragraph structure, possible OCR artefacts).  Regex patterns are less likely to match cleanly.  Dependency parsing is also less reliable on fragmented PDF text.
