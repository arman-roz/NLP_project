# NLP Project — Equations Knowledge Graph
**OTH Amberg-Weiden | Summer 2026 | Exam ID 44**  
**Deadline: 25.06.2026 12:00**

---

## What are we building?

We are building a Python pipeline that:
1. Downloads quantum physics papers from arXiv
2. Finds all **numbered equations** in each paper (e.g. equations labeled `(1)`, `(2)`, etc.)
3. For each equation, extracts:
   - The equation itself in LaTeX format
   - What the equation **means** (a short description)
   - What each **symbol** in the equation means
   - How this equation **relates** to every other equation in the same paper
   - An **audit trail** — a log of exactly which code method found each piece of data
4. Saves everything into one big **JSON file**

The goal is ~350–356 equations total, taking the first 7 numbered equations from each paper, processing papers in the exact order given in `paper_list_44.txt`.

---

## The one rule that changes everything

> **No text generation allowed.**

We cannot use ChatGPT, Claude, or any LLM to write the meanings or symbol definitions. Every piece of information must be **extracted** from the paper's own text using classical NLP techniques (regex, pattern matching, spaCy, TF-IDF). We can use SBERT (sentence embeddings) only for computing similarity scores — not for generating text.

This is the hardest constraint and it puts a ceiling on quality, especially for `meaning` and `symbols`. We must be honest about this in the report.

---

## Where does the data come from?

We download papers from `arxiv.org`. The site's `robots.txt` tells us what we are and are not allowed to fetch:

| Endpoint | Allowed? | What it gives us |
|---|---|---|
| `/abs/{id}` | ✅ Yes | Abstract page (title, abstract text) |
| `/html/{id}` | ✅ Yes | Full paper as HTML with MathML equations — **our primary source** |
| `/pdf/{id}` | ✅ Yes | PDF of the paper — **fallback** |
| `/src/{id}` or `/e-print/{id}` | ❌ No | Raw LaTeX source — **forbidden** |

The `robots.txt` also mandates a **15-second wait** between every request. We must respect this.

---

## Project folder structure

```
project/
├── data/
│   ├── cache/       ← every downloaded file is saved here (never re-downloaded)
│   └── output/      ← the final JSON dataset goes here
├── src/             ← all Python code lives here
│   ├── audit.py         step 1 — audit trail helper
│   ├── acquisition.py   step 2 — downloading papers
│   ├── extraction.py    step 3 — finding equations in HTML/PDF
│   ├── meaning.py       step 4 — figuring out what each equation means
│   ├── symbols.py       step 5 — finding what each symbol means
│   ├── relations.py     step 6 — finding how equations relate to each other
│   └── main.py          step 7 — puts it all together
├── gold/            ← a small hand-labeled test set for measuring quality
├── experiments/     ← scripts to test different approaches
├── requirements.txt ← Python package dependencies
└── NOTES.md         ← this file
```

---

## Step-by-step breakdown

---

### Step 1 — `audit.py` : The Audit Trail

**What is it?**  
Every time a piece of code extracts something, it writes a short log entry saying what it found and how. This log is stored per equation and ends up in the final JSON. The grader uses it to verify that we followed the rules.

**Why do we need it?**  
The spec requires it. It also proves that no LLM was called — every value in the JSON has a traceable source in the code.

**The tricky part:**  
The spec shows the audit trail as a Python dict, but its own example has duplicate keys (e.g. `find_symbol` appearing twice), which is invalid JSON. We solve this by appending a number to duplicate keys: `find_symbol`, `find_symbol_1`, `find_symbol_2`, etc.

**How the code works:**  
```python
trail = AuditTrail()
trail.log("fetch_html", "HTTP 200, cached, 84321 bytes")
trail.log("find_symbol", "found symbol: E")
trail.log("find_symbol", "found symbol: m")   # duplicate method name!
trail.to_dict()
# → {'fetch_html': 'HTTP 200, cached, 84321 bytes',
#    'find_symbol': 'found symbol: E',
#    'find_symbol_1': 'found symbol: m'}   ← auto-numbered
```

Internally entries are stored as a **list of (method, detail) tuples**. The `to_dict()` method converts them to a valid JSON dict only when needed.

---

### Step 2 — `acquisition.py` : Downloading Papers

**What does it do?**  
- Reads `paper_list_44.txt` and returns a clean list of arXiv IDs
- Downloads each paper, trying HTML first, PDF second
- Saves everything to `data/cache/` so re-runs are instant
- Waits 15 seconds after every real network request

**Two main components:**

#### `read_paper_list(path)`
Reads the file line by line, strips the `arXiv:` prefix from each line, removes duplicates (keeping only the first occurrence), and logs how many duplicates were dropped.

```
arXiv:2401.13506  →  '2401.13506'
arXiv:2401.13506  →  SKIPPED (duplicate)
arXiv:2411.11679  →  '2411.11679'
```

#### `Fetcher` class
The main downloader. Key methods:

- **`fetch_paper(arxiv_id, audit)`** — the one you call. Tries HTML, falls back to PDF, records the source.
- **`fetch_html(arxiv_id, audit)`** — tries `arxiv.org/html/{id}`
- **`fetch_pdf(arxiv_id, audit)`** — tries `arxiv.org/pdf/{id}`
- **`fetch_abs(arxiv_id, audit)`** — always fetches the abstract page
- **`_get(url, cache_path, audit, method)`** — the internal helper that does the actual HTTP call (or cache read)

**How caching works:**  
Before making any HTTP request, `_get()` checks if the file already exists in `data/cache/`. If yes, it reads from disk and returns immediately — no network call, no 15-second sleep. If no, it makes the request, sleeps 15 seconds, then saves the response to disk.

```
First run:   request → sleep(15s) → save to cache → return bytes
Second run:  read from cache → return bytes  (zero network calls!)
```

**Cache file naming:**  
```
data/cache/2401.13506_html.html   ← HTML version of paper 2401.13506
data/cache/2401.13506_pdf.pdf     ← PDF version of paper 2401.13506
data/cache/2401.13506_abs.html    ← abstract page of paper 2401.13506
```

**What `fetch_paper` returns:**  
```python
{
    'arxiv_id':    '2401.13506',
    'abs_content': b'<html>...',   # abstract page bytes
    'content':     b'<html>...',   # body (HTML or PDF bytes)
    'source':      'html',         # 'html', 'pdf', or 'none'
    'html_status': -1,             # -1 = cache hit, 200 = fresh, 404 = not found
    'pdf_status':  None,           # None = not attempted (html succeeded)
}
```

The `source` field is crucial — it ends up in every equation's audit trail to prove which allowed endpoint the data came from.

---

### Step 3 — `extraction.py` : Finding Equations *(next task)*

**What it will do:**  
Parse the HTML or PDF content and find all **numbered equations**.

In HTML, equations are inside `<math>` elements (MathML format) tagged with something like `class="ltx_equation"`. The number `(1)` appears nearby as text.

In PDF, we use PyMuPDF to extract text. Numbered equations appear as lines near patterns like `(1)`, `(2)`, etc. — but the equation content itself is messy Unicode (no clean LaTeX).

**Why HTML is better:**  
HTML gives us structured MathML that we can convert to a LaTeX-like string. PDF gives us rendered text — symbols like `∇` or `ψ` come through as Unicode characters, not LaTeX commands. We note this limitation honestly in the report.

---

### Step 4 — `meaning.py` : What does the equation mean? *(future task)*

**What it will do:**  
Scan the text around each equation for sentences that name or describe it.

**Approach — extractive, never generative:**  
Look for patterns like:
- `"is called the Schrödinger equation"` → meaning: `"Schrödinger equation"`
- `"where equation (1) defines the Hamiltonian"` → meaning: `"defines the Hamiltonian"`
- `"the wave function (3) satisfies"` → meaning: `"wave function"`

Also uses a **named-equation gazetteer** — a dictionary of known physics equation names (Schrödinger, Dirac, Maxwell, Hamiltonian, etc.) that we match against the surrounding text.

**Quality ceiling:**  
Without generation, if the paper's text never introduces the equation by name or description, we cannot infer a meaning. This is a fundamental limitation we document in the report.

---

### Step 5 — `symbols.py` : What does each symbol mean? *(future task)*

**What it will do:**  
For each symbol in an equation, find a definition in the paper's text.

**Approach:**  
Look for definitional patterns in the sentences near the equation:
- `"where $\psi$ is the wave function"` → `psi: wave function`
- `"$H$ denotes the Hamiltonian"` → `H: Hamiltonian`
- `"$E$ is the energy eigenvalue"` → `E: energy eigenvalue`

Uses **spaCy** for dependency parsing to handle more complex sentences, plus a physics symbol gazetteer for common cases like `\hbar` (reduced Planck constant), `c` (speed of light), etc.

---

### Step 6 — `relations.py` : How do equations relate to each other? *(future task)*

**What it will do:**  
For every pair of equations in the same paper, assign a relation:
- `"strong"` — a clear, definite relationship (with a description like `"special case"`, `"derived from"`, `"equivalent"`)
- `"potential"` — some possible relationship, not certain
- `"none"` — no discernible relationship

**Four signals used:**
1. **Cross-references** — text like `"substituting (2) into (3)"` or `"using Eq. (1)"` is a strong signal
2. **Shared symbols** — if two equations share many of the same symbols, they are likely related
3. **Derivation cue words** — words like `"therefore"`, `"hence"`, `"which gives"` between equations
4. **SBERT cosine similarity** — encode the surrounding text of both equations into vectors and measure their similarity; high similarity → potential/strong relation

---

### Step 7 — `main.py` : Putting it all together *(future task)*

**What it will do:**  
1. Read `paper_list_44.txt`
2. For each paper (in order), download it
3. Extract up to 7 numbered equations
4. For each equation, run meaning + symbol + relation extraction
5. Accumulate audit trail entries throughout
6. Stop when total equations ≥ 350 (but finish the current paper)
7. Write `data/output/dataset.json`

---

## The final JSON format

```json
{
  "2401.13506": {
    "1": {
      "equation": "E = \\hbar \\omega",
      "meaning": "Planck-Einstein relation",
      "symbols": {
        "hbar": "reduced Planck constant",
        "omega": "angular frequency"
      },
      "relations": {
        "2": {"grade": "strong", "description": "derived from"},
        "3": {"grade": "none",   "description": ""}
      },
      "audit-trail": {
        "fetch_html":          "cache hit, 84321 bytes, file=2401.13506_html.html",
        "fetch_paper":         "2401.13506: resolved via html (status=-1)",
        "extract_eq":          "found eq (1) via ltx_equation tag, source=html",
        "extract_meaning":     "gazetteer match: 'Planck-Einstein relation'",
        "find_symbol":         "found symbol: hbar",
        "extract_symbol_name": "pattern match 'hbar is the reduced Planck constant'",
        "find_symbol_1":       "found symbol: omega",
        "extract_symbol_name_1": "pattern match 'omega denotes angular frequency'",
        "classify_relation":   "eq(1)↔eq(2): cross-ref found '(2) into (1)', grade=strong"
      }
    }
  }
}
```

---

## Important numbers

| Item | Value |
|---|---|
| Target equations | 350–356 |
| Max equations per paper | 7 (first 7 numbered) |
| Crawl delay | 15 seconds per request |
| Paper list | `paper_list_44.txt` |
| Deadline | 2026-06-25 12:00 |

---

## What to be honest about in the report

1. **Equation LaTeX quality from HTML** — we get MathML converted to a LaTeX-like string, not the original source LaTeX (which is forbidden). It is close but may differ in formatting.
2. **Equation content from PDF** — we only get rendered Unicode text, not LaTeX. This is a fundamental limitation of using PDF without source access.
3. **Meaning quality** — purely extractive; if the paper never names the equation, we cannot infer a name. This is the lowest-quality field.
4. **Symbol quality** — depends on how well the paper's text follows "where X denotes Y" patterns. Heavily math-dense papers with minimal prose explanations will score poorly.
5. **Relations** — heuristic thresholds for none/potential/strong need calibration and should be presented with evidence.
