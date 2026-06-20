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

### Step 3 — `extraction.py` : Finding Equations ✅ DONE

**What it does:**  
Finds all numbered equations in a paper and extracts their LaTeX. Returns a dict keyed by equation number (e.g. `"1"`, `"A.2"`), with each entry containing the LaTeX, the source path used, and surrounding paragraph text for downstream use.

#### HTML path (primary)
1. BeautifulSoup finds every `<span class="ltx_tag_equation">` — each one marks a numbered equation
2. For each span, navigate up to the containing `<tr>` row (for equation arrays) or nearest `ltx_equation` container (for standalone equations)
3. Within that scope, find the `<math>` element
4. Try to get LaTeX in priority order:
   - `<annotation encoding="application/x-tex">` — the verbatim original LaTeX (**best**)
   - `alttext` attribute on `<math>` — equivalent copy, present when annotation is
   - Recursive MathML-to-LaTeX conversion — algorithmic fallback (**lossy**)

The MathML converter handles the most common elements: `msup` → `^{}`, `msub` → `_{}`, `mfrac` → `\frac{}{}`, `mover` → `\hat{}` / `\bar{}` etc., Greek letters, operators. Unknown elements fall back to concatenating child text.

#### PDF path (fallback)
PyMuPDF extracts text page by page. Numbered equations are identified by lines matching the pattern `"... equation content   (N)"` — i.e., text followed by whitespace followed by a right-margin number in parentheses. The equation content is the text to the left of the number. Quality is low (Unicode glyphs, not LaTeX), documented in audit trail.

#### Each equation dict contains (internally)
- `equation` — LaTeX string (goes into final JSON)
- `source` — `"html"` or `"pdf"` (goes into audit trail)
- `latex_method` — `"annotation"` / `"alttext"` / `"mathml"` / `"pdf_text"` (audit trail)
- `context_before`, `context_after` — surrounding text (used by meaning.py and symbols.py, stripped from final JSON)
- `_audit` — an `AuditTrail` instance that grows as each downstream module processes this equation

#### experiments/compare_extraction.py
A standalone script that scans the first few papers, extracts the first 10 equations found, and prints all three HTML methods side by side so you can judge which is cleanest for the report. Run with:
```
python experiments/compare_extraction.py
```
Output is printed to console and saved to `experiments/comparison_results.txt`.

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

## Tools & Technologies

This section explains every library and technique used in the pipeline — what it does, why we chose it, and what the alternative was. This feeds directly into the report's justification sections.

---

### HTTP & Web Scraping

#### `requests` (v2.32.3)
**What:** Standard Python HTTP library. Used to download pages from arXiv (`/abs`, `/html`, `/pdf`).  
**Why:** Simple, widely used, supports sessions (so we set the User-Agent once and reuse it). The `requests.Session` object is more efficient than making separate connections per request.  
**Alternative considered:** `httpx` (async HTTP) — rejected because async adds complexity we don't need; our pipeline is already rate-limited to one request per 15 seconds, so parallelism gives no benefit.

#### `BeautifulSoup4` + `lxml` (v4.12.3 + v5.3.0)
**What:** HTML/XML parser. Used to navigate the arXiv LaTeXML HTML structure and find equation elements, paragraph text, and MathML.  
**Why:** BeautifulSoup gives a clean Python API to walk the DOM tree. We use the `lxml` backend because it is faster than the built-in `html.parser` and handles malformed HTML better.  
**Alternative considered:** `lxml` directly (without BeautifulSoup) — more verbose API, harder to read. BeautifulSoup wraps it cleanly.

---

### Equation Extraction

#### MathML `<annotation>` tag / `alttext` attribute
**What:** arXiv's LaTeXML converter preserves the original LaTeX in two places: the `<annotation encoding="application/x-tex">` child element and the `alttext` attribute of the `<math>` element. Both contain the verbatim LaTeX the author wrote.  
**Why:** This is the cleanest possible LaTeX — no conversion loss. It is the first thing we check.  
**Why not LaTeX source directly:** The `/src` and `/e-print` endpoints are disallowed by `robots.txt` and explicitly confirmed off-limits by the professor.

#### Recursive MathML-to-LaTeX Converter (custom, in `extraction.py`)
**What:** A hand-written recursive function that walks MathML elements (`msup`, `msub`, `mfrac`, `mover`, `mi`, `mo`, etc.) and produces a LaTeX-like string. Uses lookup tables for Greek letters (α→`\alpha`) and operators (∂→`\partial`).  
**Why:** Needed as a fallback when neither the annotation tag nor alttext is available. Covers ~90% of common physics notation.  
**Limitation:** Lossy — MathML encodes rendered structure, not source intent. Custom macros, style choices (`\mathbf` vs `\boldsymbol`), and unusual constructs may come out differently from the original.  
**Alternative considered:** `latex2mathml` package (converts LaTeX→MathML, not the reverse). No mature, well-maintained MathML→LaTeX Python library exists, hence the custom implementation.

#### `PyMuPDF / fitz` (v1.25.5)
**What:** PDF text extraction library. Used on the PDF fallback path to extract text page by page via `page.get_text()`.  
**Why:** PyMuPDF is the fastest and most accurate open-source PDF text extractor for Python. Handles multi-column layouts better than `pdfminer`.  
**Limitation:** Returns rendered Unicode characters (`ψ`, `∇`), not LaTeX commands. Equation structure is lost. This is documented per-equation in the audit trail.  
**Alternative considered:** `pdfminer.six` — slower, more complex API, similar Unicode output. `pdfplumber` — good for tables but no advantage for this use case.

---

### NLP — Meaning & Symbol Extraction

#### Regex pattern matching (built-in `re`)
**What:** Regular expressions used to find definitional sentences around equations. Patterns like `"where X (is|denotes|represents) ..."`, `"(is|are) called the ..."`, `"known as ..."`.  
**Why:** Fast, zero dependencies, fully deterministic. Works well on well-structured academic prose which follows predictable definitional conventions.  
**Limitation:** Brittle on unusual sentence structures. Will miss definitions phrased in non-standard ways.  
**Alternative considered:** Full dependency parsing for all sentences — overkill for simple "where X is Y" patterns; regex is faster and more predictable here.

#### `spaCy` (v3.8.3, `en_core_web_sm` model)
**What:** Industrial-strength NLP library. Used for:
- Tokenisation and sentence splitting (to isolate the sentence containing each symbol reference)
- Dependency parsing (to extract the subject-verb-object structure of definitional sentences, e.g. "where *ψ* **is** the *wave function*")
- Named entity recognition (identifying proper noun equation names)

**Why:** spaCy's dependency parser reliably identifies the grammatical relationship between a symbol (`ψ`) and its definition (`wave function`) even in complex sentences where a simple regex would fail. The `en_core_web_sm` model is small and fast enough to run on every paragraph.  
**Alternative considered:** NLTK — older, slower, less accurate dependency parser. Transformers-based NER — would require an LLM-style model and risks crossing the "no generation" line.

#### Named-equation Gazetteer (custom dictionary)
**What:** A hand-curated dictionary of known physics equation names and their keywords. Examples: `"Schrödinger"` → `"Schrödinger equation"`, `"Hamiltonian"` → `"Hamiltonian"`, `"Maxwell"` → `"Maxwell's equations"`.  
**Why:** Many equations in quantum physics papers are never explicitly named in the surrounding text — authors assume the reader knows them. The gazetteer covers these well-known cases.  
**Why it's not cheating:** The gazetteer values come from our knowledge, not from reading the paper. We are classifying, not generating — the equation name is a label we assign based on keywords found in the paper's text.

#### Physics Symbol Gazetteer (custom dictionary)
**What:** A dictionary mapping common LaTeX symbols to their standard physics meanings. Examples: `hbar` → `"reduced Planck constant"`, `c` → `"speed of light"`, `epsilon_0` → `"vacuum permittivity"`.  
**Why:** Used as a high-confidence fallback when the paper's text does not define a symbol explicitly. Common in quantum physics where `ℏ` and `c` are used without definition.

---

### NLP — Relation Classification

#### TF-IDF Cosine Similarity (`scikit-learn`, v1.6.1)
**What:** TF-IDF (Term Frequency–Inverse Document Frequency) converts text (the surrounding context of each equation) into a vector of word-importance weights. Cosine similarity between two vectors measures how similar the surrounding contexts are.  
**Why chosen as baseline:** Purely classical, no model weights, fully deterministic, fast, and explainable. The IDF is computed over the full corpus of all extracted context texts (not just three sentences) so rare physics terms get proper weighting.  
**Limitation:** Treats text as a bag of words — word order and meaning are ignored. "The Hamiltonian *is not* the energy" and "The Hamiltonian *is* the energy" get the same vector.  
**Role in pipeline:** Primary classical signal for relation scoring; also serves as a reproducibility backstop if SBERT is questioned by the examiner.

#### SBERT — Sentence-BERT (`sentence-transformers`, v3.3.1)
**What:** A pre-trained neural model that encodes a sentence (or paragraph) into a fixed-length dense vector (embedding) that captures semantic meaning. We compute cosine similarity between the embeddings of two equations' surrounding text.  
**Why over plain TF-IDF:** SBERT captures meaning beyond keyword overlap. "The wave function satisfies" and "ψ obeys" get similar vectors in SBERT; TF-IDF would rate them as unrelated.  
**Why this is allowed:** SBERT is used purely as an **encoder** — it maps text to a vector. We never ask it to generate, complete, or summarise text. This is embedding/scoring, not prompting.  
**Model used:** `all-MiniLM-L6-v2` — small (80MB), fast, runs on CPU and GPU, good accuracy for semantic similarity tasks.  
**Alternative considered:** OpenAI embeddings API — external API call, forbidden. `word2vec` / `GloVe` — older, lower quality, no sentence-level encoding.

#### Cross-reference Pattern Matching
**What:** Regex patterns that detect when one equation explicitly refers to another in the paper's text. Examples: `"substituting (2) into (3)"`, `"using Eq. (1)"`, `"from equation (4) we get"`.  
**Why:** The most reliable signal for a **strong** relation. An explicit textual cross-reference is unambiguous evidence that the author considered the equations related.  
**How:** Search the context text of each equation for patterns citing other equation numbers from the same paper.

#### Shared Symbol Overlap
**What:** Count how many symbols appear in both equations (after extracting the symbol set from each LaTeX string). Normalise by the total unique symbols across both.  
**Why:** Equations that share many symbols (e.g. both use `H`, `ψ`, `E`) are very likely in the same derivation chain. A high overlap score is a strong signal for a **potential** or **strong** relation.

#### Derivation Cue Words
**What:** A curated list of words/phrases that indicate one equation leads to another: `"therefore"`, `"hence"`, `"which gives"`, `"it follows that"`, `"substituting"`, `"combining"`, `"this yields"`.  
**Why:** These words appear in the text between two equations when one is derived from the other. Detecting them in the context between two equation positions adds a strong signal.

#### Grade Thresholds (calibrated)
**What:** The three signals (cross-reference, symbol overlap, SBERT/TF-IDF similarity) are combined into a score. Thresholds determine the grade:
- Score above upper threshold → `"strong"` (with auto-description)
- Score between thresholds → `"potential"`
- Score below lower threshold → `"none"`

**Why calibrate rather than guess:** We run `experiments/` scripts on a manually labelled sample of equation pairs to find the threshold values that maximise precision/recall on the gold set.

---

### Infrastructure

| Tool | Version | Purpose |
|---|---|---|
| `requests` | 2.32.3 | HTTP downloads from arXiv |
| `beautifulsoup4` + `lxml` | 4.12.3 + 5.3.0 | HTML parsing, MathML traversal |
| `PyMuPDF (fitz)` | 1.25.5 | PDF text extraction |
| `spacy` + `en_core_web_sm` | 3.8.3 | Tokenisation, dependency parsing, NER |
| `sentence-transformers` | 3.3.1 | SBERT semantic embeddings (encoder only) |
| `scikit-learn` | 1.6.1 | TF-IDF vectorisation, cosine similarity |
| `numpy` | 1.26.4 | Array operations for similarity scoring |
| Python `re` | built-in | Regex pattern matching throughout |
| Python `json` | built-in | Reading/writing the output dataset |
| Python `logging` | built-in | Structured logs for debugging |

---

## What to be honest about in the report

1. **Equation LaTeX quality from HTML** — we get MathML converted to a LaTeX-like string, not the original source LaTeX (which is forbidden). It is close but may differ in formatting.
2. **Equation content from PDF** — we only get rendered Unicode text, not LaTeX. This is a fundamental limitation of using PDF without source access.
3. **Meaning quality** — purely extractive; if the paper never names the equation, we cannot infer a name. This is the lowest-quality field.
4. **Symbol quality** — depends on how well the paper's text follows "where X denotes Y" patterns. Heavily math-dense papers with minimal prose explanations will score poorly.
5. **Relations** — heuristic thresholds for none/potential/strong need calibration and should be presented with evidence.
