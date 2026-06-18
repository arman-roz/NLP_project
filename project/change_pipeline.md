# Pipeline Architecture — Three-Stage Design

This document describes the overall pipeline architecture: why a three-stage
approach was chosen, what each stage produces, how the stages are connected,
and how the design satisfies all project constraints.

---

## Motivation

The original single-pass pipeline parsed raw HTML, extracted equations, and
annotated them with meaning and symbols in one continuous execution.  When
something went wrong — a missing equation, garbled LaTeX, an empty meaning
field — there was no way to pinpoint which step failed without adding
print statements and re-running the entire pipeline.

The three-stage approach solves this by writing a checkpoint JSON file after
each major transformation.  Each checkpoint is human-readable and can be
inspected independently, fed to an LLM for pattern analysis, or used as
input for a specific stage without re-running earlier ones.

A secondary benefit: because each stage file is self-contained, debugging is
possible without opening the original multi-megabyte HTML files.

---

## Architecture Overview

```
paper_list_44.txt
       │
       ▼
┌─────────────────────────────────┐
│  Acquisition (acquisition.py)   │  /abs, /html, /pdf arXiv endpoints
│  Cache-first, 15s crawl delay   │  PDF fallback when HTML unavailable
└────────────┬────────────────────┘
             │  raw HTML bytes / PDF bytes
             ▼
┌─────────────────────────────────┐
│  Stage 1 — Fragment Extraction  │  fragmenter.py
│  eq_fragments.json              │
└────────────┬────────────────────┘
             │  list of raw HTML table strings per paper
             ▼
┌─────────────────────────────────┐
│  Stage 2 — Structured Extract.  │  extraction.py
│  eq_extracted.json              │
└────────────┬────────────────────┘
             │  equation + context text per paper
             │
             │  (corpus-wide IDF fitted here on all context texts)
             ▼
┌─────────────────────────────────┐
│  Stage 3 — Semantic Annotation  │  meaning.py + symbols.py + relations.py
│  eq_final.json  (spec output)   │
└─────────────────────────────────┘
```

---

## Stage 1 — Fragment Extraction (`eq_fragments.json`)

**Module:** `src/fragmenter.py` — `FragmentExtractor.extract()`

**Input:** Raw HTML bytes for one paper.

**What it does:** Parses the full HTML with BeautifulSoup and finds every
`<table class="ltx_eqn_table">` element.  LaTeXML (arXiv's HTML renderer)
wraps every numbered equation in this table structure, regardless of whether
the equation is standalone, part of an align block, or a multi-line group.
Each table is serialised back to an HTML string and stored alongside metadata
that characterises the table without requiring re-parsing.

**No equation cap is applied here.**  All tables in the paper are stored.
The 7-equation limit is enforced in Stage 2 so that Stage 1 gives a complete
picture of what the HTML actually contains — useful for investigating why
certain equations are not in the final output.

**Output shape:**

```json
{
  "2401.13506": {
    "source": "html",
    "fragments": [
      {
        "table_index":        0,
        "classes":            ["ltx_equation", "ltx_eqn_table"],
        "eq_num":             "(1)",
        "has_math":           true,
        "has_annotation":     true,
        "annotation_preview": "I_{\\rm OFF}(x,y)=...",
        "alttext_preview":    null,
        "html":               "<table class=\"ltx_eqn_table ...\">...</table>"
      }
    ]
  },
  "2502.03234": {
    "source": "pdf",
    "fragments": []
  }
}
```

**PDF papers** produce an empty `fragments` list.  Their equations are
extracted directly from PDF text in Stage 2 using PyMuPDF — there is no
HTML table structure to capture.

**Metadata fields explained:**

| Field | Purpose |
|---|---|
| `table_index` | Position in the document — helps trace missing equations |
| `classes` | Reveals equation type: `ltx_equation` (standalone) vs `ltx_equationgroup` (align/eqnarray) |
| `eq_num` | The printed number tag, e.g. `(1)` — taken directly from the HTML |
| `has_math` | Whether a `<math>` element is present inside the table |
| `has_annotation` | Whether the verbatim LaTeX annotation tag is present (best quality) |
| `annotation_preview` | First 120 chars of the annotation — readable without parsing |
| `alttext_preview` | First 120 chars of the alttext attribute, shown only when no annotation |
| `html` | Full raw HTML of the table — parseable, inspectable, feedable to an LLM |

**Debugging use:**
To investigate why a paper has fewer equations than expected, open
`eq_fragments.json`, find the paper ID, and count the `fragments` list.
If the count is lower than the expected number of equations, the HTML
structure differs from the three known patterns (standalone, row, group).
The raw `html` field makes the exact DOM structure immediately visible.

**Pattern analysis use:**
Feed the `fragments` list for a subset of papers to an LLM with a prompt
like "find all tables where `has_annotation` is false and describe what
alternative LaTeX source is present".  This requires no HTML parsing skills
and produces actionable observations about extraction quality across the corpus.

---

## Stage 2 — Structured Extraction (`eq_extracted.json`)

**Module:** `src/extraction.py` — `EquationExtractor.extract()`

**Input:** Raw fetch result dict (HTML bytes or PDF bytes from Fetcher / cache).

**What it does:** Applies the full extraction logic — HTML table routing
(standalone / row / group sub-cases), LaTeX retrieval (annotation → alttext →
MathML conversion), PDF line detection, context text collection — and stores
the structured result per equation.  The 7-equation cap and first-occurrence
deduplication are enforced here.

**Output shape:**

```json
{
  "2401.13506": {
    "1": {
      "equation":       "I_{\\rm OFF}(x,y)=\\left((\\delta a)^{2}+...",
      "source":         "html",
      "latex_method":   "annotation",
      "context_before": "It is either related to an intrinsic asymetry...",
      "context_after":  "The first term of Equation (2) is the deflection..."
    }
  }
}
```

**Fields:**

| Field | Values | Meaning |
|---|---|---|
| `equation` | LaTeX string | Verbatim from annotation tag, or alttext, or MathML conversion, or PDF text |
| `source` | `html`, `pdf` | Which arXiv endpoint provided the content |
| `latex_method` | `annotation`, `alttext`, `mathml`, `pdf_text` | Which method extracted the LaTeX |
| `context_before` | plain text | Up to two paragraph sentences immediately before the equation |
| `context_after` | plain text | Up to two paragraph sentences immediately after the equation |

**Note on `source` and `latex_method`:** These fields are present in Stage 2
because they describe the extraction process and are useful for quality
analysis.  In Stage 3 (the final spec output) they are moved inside the
`audit-trail` field — they do not appear at the top level, as required by the
output schema.

**Debugging use:**
If a symbol definition is wrong in Stage 3, check Stage 2 first.  If the
context text is correct, the bug is in `symbols.py`.  If the context text is
wrong, the bug is in `extraction.py`'s `_get_surrounding_text_html`.  If the
LaTeX itself is wrong, check `latex_method`: if it is `mathml` instead of
`annotation`, the annotation tag was absent and the lossy MathML fallback
was used.

---

## Stage 3 — Semantic Annotation (`eq_final.json`)

**Modules:** `src/meaning.py`, `src/symbols.py`, `src/relations.py`

**Input:** In-memory equations dict from Stage 2 (including context text and
per-equation audit trails, which are not written to the Stage 2 JSON).

**What it does:** Runs three annotation passes over every equation:

1. **Meaning** (`MeaningExtractor.extract`) — finds the single best sentence
   that explains what the equation represents, using three strategies in order:
   citation match → gazetteer → TF-IDF ranking.

2. **Symbols** (`SymbolExtractor.extract`) — for each meaningful symbol in the
   LaTeX, finds a verbatim definitional phrase from the surrounding text using
   regex patterns and (optionally) spaCy dependency parsing.

3. **Relations** (`RelationExtractor.extract` + `compute_shared_symbols`) —
   finds which other equations in the same paper are explicitly cited in this
   equation's context, and which equations share at least one symbol key.

**Output shape (spec-compliant):**

```json
{
  "2401.13506": {
    "1": {
      "equation":   "I_{\\rm OFF}(x,y)=\\left((\\delta a)^{2}+...",
      "meaning":    "It is either related to an intrinsic asymetry of the beamsplitter...",
      "symbols": {
        "I":     "intensity profile",
        "delta": "phase noise",
        "a":     "beamsplitter"
      },
      "relations": {
        "cites":              ["2"],
        "shares_symbol_with": ["2", "6"]
      },
      "audit-trail": {
        "extract_html":    "found eq (1), source=html, latex_method=annotation, ...",
        "meaning":         "tfidf_rank: sentence score=0.42",
        "symbol":          "sym='I', definition='intensity profile'",
        "relations_cites": "eq (1) cites: ['2']",
        "source":          "html",
        "latex_method":    "annotation"
      }
    }
  }
}
```

**Spec compliance:**
- `source` and `latex_method` appear inside `audit-trail` only — not at
  the top level of the equation entry.
- Top-level keys are exactly: `equation`, `meaning`, `symbols`, `relations`,
  `audit-trail`.
- `context_before` and `context_after` are used internally during annotation
  but are not written to Stage 3 output.

---

## Cache-First Fetch Strategy

On the first run the pipeline fetches all papers from arXiv via `Fetcher`,
which makes network requests and caches each response to `data/cache/`.
The 15-second crawl delay (as required by arxiv.org robots.txt) is applied
after every network request.

On subsequent runs, `main.py` checks the cache directory first:
- If `{arxiv_id}_html.html` exists and is non-empty → read HTML from disk
- Else if `{arxiv_id}_pdf.pdf` exists and is non-empty → read PDF from disk
- Else → call `Fetcher.fetch_paper()` for a network request

This avoids the penalty of a 404 network request (+ 15s sleep) on every
re-run for papers that have a cached PDF but no HTML.  On a 100-paper dataset
with 19 PDF-only papers, this saves approximately 19 × 15 = 285 seconds per
run after the first.

---

## Spec Constraints Checklist

| Constraint | Where enforced |
|---|---|
| First 7 numbered equations per paper | `extraction.py` MAX_EQUATIONS_PER_PAPER |
| Stop at 350 total, finish current paper | `main.py` post-paper check |
| 15s crawl delay | `acquisition.py` CRAWL_DELAY = 15 |
| Only /abs /html /pdf endpoints | `acquisition.py` — no /src or /e-print |
| Deduplicate to first occurrence | `acquisition.py` `read_paper_list` |
| No text generation | `meaning.py`, `symbols.py` — extractive only |
| output schema {equation, meaning, symbols, relations, audit-trail} | `main.py` Stage 3 output construction |
| source/latex_method inside audit-trail only | `main.py` — logged to audit, excluded from top-level |

---

## File Sizes (100 papers)

| File | Estimated size | Content |
|---|---|---|
| `eq_fragments.json` | 30–50 MB | Full HTML of all 1000+ equation tables |
| `eq_extracted.json` | 2–4 MB | LaTeX + context text for 503 equations |
| `eq_final.json` | 3–6 MB | Full spec output with meaning, symbols, relations, audit-trail |

`eq_fragments.json` is large because it stores full HTML including the
complete MathML tree for each equation.  This is intentional — the purpose of
Stage 1 is complete transparency into what the extractor saw.

---

## Limitations

- **Stage 1 size:** At 30–50 MB, `eq_fragments.json` is too large to open
  comfortably in a text editor.  Use Python's `json` module or VS Code's
  JSON viewer to inspect individual papers.

- **Stage 1 HTML-only:** PDF papers produce no Stage 1 fragments.  Their
  LaTeX quality cannot be inspected via the fragment file; use the `latex_method`
  field in Stage 2 (`pdf_text`) as an indicator that LaTeX quality is lower.

- **Re-run cost for Stage 3 only:** There is no mechanism to skip Stage 1 and
  Stage 2 and re-run only Stage 3 from the saved Stage 2 JSON.  `main.py`
  always runs all three stages.  For rapid re-iteration of meaning/symbols/
  relations logic, use the test scripts (`experiments/test_meaning_symbols.py`)
  which read from the extraction cache directly.
