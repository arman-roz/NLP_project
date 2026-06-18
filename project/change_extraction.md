# Extraction Module — Change Log

This document records every bug found in `src/extraction.py`, how we found it, what the old code was doing, why it failed, and exactly what we changed and why it now works.

---

## HTML Extraction Overhaul

### How we found the bugs

We ran `test_pipeline.py` on 30 papers and got `test_output.json`. Auditing that file revealed three classes of problem:

- Some papers had fewer equations than expected (equation numbers were skipped in the output).
- One paper had an equation group labelled `(3)` that appeared nowhere in the output at all.
- Many equations started with `\displaystyle` or contained `%\n` inside them.

We then read the actual cached HTML files from `project/data/cache/` to find the root causes rather than guessing.

---

### Bug A — `ltx_tag_equationgroup` spans were never searched

**Affected paper:** `2506.07618` — equation `(3)` completely missing from output.

#### What the old code did

```python
tag_spans = soup.find_all("span", class_="ltx_tag_equation")
```

It searched only for spans whose class list contained `ltx_tag_equation`. Every span found was then scoped to its parent `<tr>` to find the `<math>` element on the same row.

#### What the actual HTML contained

```html
<td ... rowspan="4">
  <span class="ltx_tag ltx_tag_equationgroup ltx_align_right">(3)</span>
</td>
```

For equation groups where all rows share a single label (a `\begin{align}` block with one number for all lines), LaTeXML uses `ltx_tag_equationgroup` instead of `ltx_tag_equation`. The span's class list does NOT contain `ltx_tag_equation`, so `find_all("span", class_="ltx_tag_equation")` never found it.

We surveyed all 21 cached HTML files:

```
1040 spans with class ltx_tag_equation
  33 spans with class ltx_tag_equationgroup
```

33 equation group spans were being silently skipped.

#### What we changed

We replaced the entire bottom-up span search with a **top-down table search**. Instead of hunting for number spans and trying to find the math around them, we now walk every `<table class="ltx_eqn_table">` in document order and handle three sub-cases:

```python
eq_tables = soup.find_all("table", class_="ltx_eqn_table")

for table in eq_tables:
    table_classes = table.get("class", [])

    if "ltx_equationgroup" in table_classes:
        grp_tag = table.find("span", class_="ltx_tag_equationgroup")
        if grp_tag:
            self._record_group_equation(table, grp_tag, equations, audit)
        else:
            for tr in table.find_all("tr"):
                self._record_row_equation(tr, table, equations, audit)
    else:
        self._record_standalone_equation(table, equations, audit)
```

For a shared-label group (`_record_group_equation`), we collect the `<annotation>` LaTeX from **every row** in the table and join them with `\\`:

```python
math_elems = table.find_all("math")
parts = [self._clean_latex(self._get_latex_from_math(m)[0]) for m in math_elems]
latex = " \\\\\n".join(parts)
```

This correctly captures the full multi-line system as one equation entry.

---

### Bug B — `rowspan="0"` placed the number in a different row from the math

**Affected paper:** `2404.19140` — equation `(2)` missing from output.

#### What the old code did

```python
scope = self._get_html_scope(tag_span)   # returned parent <tr>
math_elem = scope.find("math")           # looked for <math> only in that <tr>
if math_elem is None:
    continue                             # silently skipped the equation
```

`_get_html_scope` found the parent `<tr>` of the `ltx_tag_equation` span and returned it as the scope. Then `scope.find("math")` looked for a `<math>` element inside that row. If none was found the equation was dropped.

#### What the actual HTML contained

Inspecting the cached HTML for paper `2404.19140`:

```
table id="A2.EGx2"  (class: ltx_equationgroup ltx_eqn_eqnarray ltx_eqn_table)
  tbody id="S2.Ex2"
    <tr>  ← math element is HERE  (H_k = matrix)
  tbody id="S2.E2"
    <tr>  ← ltx_tag_equation "(2)" is HERE, no math
```

LaTeXML placed the equation number in a `<td rowspan="0">` inside a **separate `<tbody>`** from the one containing the actual `<math>` element. The `rowspan="0"` visually spans the number across the rows above it, but the HTML puts the `<td>` in its own tbody below the content.

When the old code scoped to the `<tr>` containing the `(2)` tag, that row had no `<math>`, so it silently skipped equation `(2)`.

We confirmed this with a direct inspection script:

```
tbody id="S2.Ex2" → row with math=True,  has_eq_tag=False
tbody id="S2.E2"  → row with math=False, has_eq_tag=True   ← eq(2) tag is here
```

#### What we changed

In `_record_standalone_equation`, we now search for math at **table level**, not row level:

```python
math_elem = table.find("math")
```

Since a `table.ltx_equation` always contains exactly one equation, `table.find("math")` always finds the right element regardless of which row holds the number tag. The rowspan layout becomes irrelevant.

For individually-numbered rows in equation groups (`_record_row_equation`), we added a **backwards row search** as a fallback:

```python
math_elem = tr.find("math")
if math_elem is None:
    # rowspan issue: walk backwards through all rows in the table
    # to find the nearest preceding row that contains math
    all_rows = table.find_all("tr")
    tr_idx = next((i for i, r in enumerate(all_rows) if r is tr), -1)
    for i in range(tr_idx - 1, -1, -1):
        m = all_rows[i].find("math")
        if m:
            math_elem = m
            break
```

We search **backwards** (not forwards, not at tbody level) because the number row is always placed at the **end** of the equation it labels. The nearest preceding row with math is always the correct content. This also correctly handles multi-equation group tables — each number row will find its own preceding content row, not someone else's.

---

### Bug C — `\displaystyle` and `%\n` artefacts in extracted LaTeX

**Affected papers:** widespread across most papers using align/eqnarray environments.

**Examples from `test_output.json`:**
```
\displaystyle H                        ← eq (1) of 2404.19140
\displaystyle\eta_{aa}                 ← eq (3) of 2506.16300
I_{\rm in}(x%\n,y)                    ← eq (3) of 2401.13506
```

#### Why these appeared

LaTeXML copies the original LaTeX verbatim into the `<annotation encoding="application/x-tex">` tag. Inside an `align` or `eqnarray` environment, every cell gets `\displaystyle` prepended by LaTeXML to force display-math sizing. It is a rendering detail, not part of the equation.

`%\n` is a standard LaTeX line-continuation trick: `%` starts a comment that runs to end-of-line, so `x%\n,y` in the source means `x,y` with the newline suppressed. Authors use it to break long equations across source lines. It is invisible in the rendered output but was copied verbatim into the annotation.

#### What we changed

We added a `_clean_latex` static method that is called on every extracted LaTeX string before it is stored:

```python
@staticmethod
def _clean_latex(latex: str) -> str:
    latex = re.sub(r"^\\displaystyle\s*", "", latex.strip())
    latex = latex.replace("%\n", "")
    return latex.strip()
```

- `^\\displaystyle\s*` — strips the prefix only when it appears at the start of the string (so it does not accidentally strip `\displaystyle` from inside a longer expression).
- `%\n` replacement — removes the literal two-character sequence that LaTeX uses for source line continuations.

---

## PDF Extraction Fix

### How we found the bug

The `test_output.json` showed two papers falling back to PDF (`2502.03234` and `2504.11399`), both returning empty equation dicts `{}`. We read the raw lines extracted by PyMuPDF from both files.

---

### Bug D — Content-length filter rejected all equations in `2502.03234`

#### What the old code did

```python
match = _PDF_EQ_LINE_RE.match(line)
eq_content = match.group(1).strip()
eq_num     = match.group(2).strip()

if not eq_content or len(eq_content) < 2:
    continue   # ← this killed every equation in 2502.03234
```

The guard was intended to reject false positives like section headings that accidentally matched the `(N)` pattern. The threshold was `len < 2`, meaning any one-character content was dropped.

#### What the actual PDF contained

For `2502.03234`, every equation line looked like this:

```
'     ,                                   (1) '
```

The regex correctly found this line. `match.group(1).strip()` → `','`. Length = 1. Rejected.

**Why the content is only a comma:** This paper typsets its equations as positioned PDF glyphs (not as extractable Unicode text). PyMuPDF sees only whitespace where the math symbols should be — the glyphs are rendered as graphics or private-use Unicode that text extraction cannot recover. The terminating `,` or `.` is the only visible character because it is an ordinary ASCII punctuation mark placed at the end of the equation.

We verified this by printing every line of page 2 with its non-space character count:

```
line 43: len=53  nonspace=0   '                                                     '
line 44: len=1   nonspace=0   ' '
...
line 51: len=3   nonspace=0   '   '
line 52: len=45  nonspace=4   '     ,                                   (1) '
```

Lines 43–51 (the equation body) are pure whitespace. There is no text to collect.

For `2504.11399`, we searched all 31 pages for lines matching the `(N)` pattern and found zero matches. The paper uses unnumbered display equations — it genuinely has no equation numbers. Correct behaviour is zero equations returned.

#### What we changed

We removed the content-length filter entirely:

```python
# No content-length filter.
# Some PDFs render equation bodies as positioned glyphs that are invisible
# to text extraction — the number-terminator line then carries only ',' or '.'.
# We still record the equation: the number is the valuable datum.
```

We also improved the context extraction to look for the nearest **prose sentence** (a line with ≥ 4 space-separated words) rather than just the immediately adjacent raw lines:

```python
before_lines: List[str] = []
for i in range(line_idx - 1, max(0, line_idx - 10) - 1, -1):
    t = all_lines[i][0].strip()
    if t and len(t.split()) >= 4:
        before_lines.append(t)
        break
```

This means `context_before` now captures the sentence that introduces the equation (e.g. `"Let's consider the initial SMSV state"` for equation (1)), which is useful for the meaning and relations modules downstream.

#### Result after fix

| Paper | Before | After |
|---|---|---|
| `2502.03234` | 0 equations | 7 equations with correct numbers; content = `,` or `.` (honest — math is not text-extractable) |
| `2504.11399` | 0 equations | 0 equations (correct — paper has no numbered equations) |
| `2508.21253` | 7 equations | 7 equations (unchanged — inline format still works) |

---

## Survey of arXiv LaTeXML HTML equation structures

During the investigation we ran a survey across all 21 cached HTML files to understand every container type that exists. This informed the top-down design.

```
Container type                                          Count
──────────────────────────────────────────────────────────────
table.ltx_equation + ltx_tag_equation                   553   standalone numbered equation
table.ltx_equationgroup (eqnarray) + ltx_tag_equation   163   per-row numbered align group
table.ltx_equationgroup (align)    + ltx_tag_equation   123   per-row numbered align group
table.ltx_equationgroup (gather)   + ltx_tag_equation   108   per-row numbered gather group
table.ltx_equationgroup            + ltx_tag_equationgroup 33  whole-group shared number ← Bug A
table.ltx_equation  (unnumbered)                         34   no number tag, skip
```

All numbered equations live inside a `table.ltx_eqn_table`. The top-down approach iterates these tables and handles all six cases correctly.

---

## Spec Compliance: Output Schema

### Issue — `source` and `latex_method` were top-level keys beside `equation`

The project spec defines each equation entry as exactly:

```json
{
  "equation": "...",
  "meaning": "...",
  "symbols": {},
  "relations": {},
  "audit-trail": {}
}
```

Our initial output added two extra keys at the same level:

```json
{
  "equation": "...",
  "source": "html",
  "latex_method": "annotation",
  "meaning": "...",
  ...
}
```

These are not part of the spec. Both values were already being written into the `audit-trail` string (e.g. `"found eq (1), source=html, latex_method=annotation, latex_preview='...'"`) so no information was lost. We removed `source` and `latex_method` as top-level keys so the per-equation shape matches the spec exactly.

---

## Design Decision: Audit-Trail Duplicate Keys

### The problem

The spec's own example for `audit-trail` shows the same method name appearing more than once:

```
find_symbol: 'found a'
find_symbol: 'found phi'
```

A JSON object cannot have duplicate keys — if you write `{"find_symbol": "found a", "find_symbol": "found phi"}`, any JSON parser will silently drop the first entry. The spec example is therefore not valid JSON as written.

### Our decision

We resolved this in `src/audit.py` using a **numeric suffix** strategy: the first occurrence of a method name keeps the bare name; every subsequent occurrence gets `_1`, `_2`, etc.:

```
find_symbol:   'found a'
find_symbol_1: 'found phi'
```

This keeps the output valid JSON, preserves every log entry in insertion order, and is immediately readable without any post-processing. We explicitly document this decision in the `AuditTrail` class docstring because the spec says: *"if things are not specified, clearly state which decisions you made instead."*

---

## Dataset Observation: Single-Fragment Numbered Equations

During dataset review, several equations in paper `2401.13506` are single-symbol or single-expression fragments:

| Eq | LaTeX |
|----|-------|
| 3  | `I_{\rm ON}(x,y)` |
| 4  | `I_{\rm OFF}(x,y)` |
| 5  | `\mathcal{F}` |
| 7  | `I_{\rm ON}(x,y)` |

These are genuine numbered display equations in the paper — the authors define a symbol in a standalone display block and assign it an equation number so they can refer back to it ("as in eq. 3"). They are not extraction errors.

However, a one-symbol equation carries almost no standalone meaning without the surrounding prose. This is a real limitation of equation-only extraction:

- The `equation` field is correct and complete per the spec.
- The `meaning` field (filled by `meaning.py`) is where the surrounding-text context will be used to explain what the symbol represents.
- "The first 7 relevant equations" in the spec is interpreted as "first 7 **numbered** equations" — the equation number is the author's own signal that the formula is important enough to reference. We do not apply a secondary relevance filter.

---

## Crawl Compliance

### Base URL — `arxiv.org` not `export.arxiv.org`

`export.arxiv.org` is the OAI-PMH metadata API. It serves XML records (title, authors, abstract) and is designed for bulk metadata harvesting. The `/html/{id}` and `/pdf/{id}` endpoints that this project requires **do not exist** on `export.arxiv.org` — they are only available on `arxiv.org`. Using `export.arxiv.org` for HTML/PDF would fail with 404.

We fetch from `https://arxiv.org` which is correct for our endpoints.

### Rate limiting — 15-second crawl delay

`arxiv.org/robots.txt` contains:

```
User-agent: *
Crawl-delay: 15
```

`acquisition.py` enforces this exactly: `time.sleep(CRAWL_DELAY)` is called after **every** network request, including failed requests, so an error never shortens the gap. Cache hits make zero network calls and incur no sleep penalty.

### User-Agent

We send a descriptive `User-Agent` header:

```
OTH-NLP-EquationsKG/1.0 (Academic project, OTH Amberg-Weiden; quantum physics equations extraction; compliant with arxiv.org/robots.txt)
```

This identifies the crawler, its purpose, and its institution — consistent with polite-crawler convention and arXiv's expectation for academic harvesters.

### Audit proof

Every network request is logged in the `AuditTrail` for that paper:

```
fetch_html: HTTP 200, url=https://arxiv.org/html/2401.13506
```

Cache hits are also logged:

```
fetch_html: cache hit, 84321 bytes, file=2401.13506_html.html
```

This means the final JSON contains a verifiable record of exactly how each paper was fetched, which endpoint was used, and whether the rate limit would have applied.
