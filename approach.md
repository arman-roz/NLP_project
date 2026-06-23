# Approaches Tried — Equations Knowledge Graph (Exam ID 44)

This document records, point by point and from start to end, the two approaches
that have been built for the project, what each one does, why it was designed
that way, and where its limits are.

Both live as self-contained pipelines in the repo:

- **Approach 1 — `modified/`** : a streamlined, single-pass *structural* pipeline.
- **Approach 2 — `arxiv_rag/`** : a *hybrid-retrieval (RAG)* pipeline.

---

## 0. Task and the one hard constraint (recap)

- **Goal:** for every *numbered* equation in quantum-physics arXiv papers, extract
  (1) the equation in LaTeX, (2) a short `meaning`, (3) a `symbols` dictionary,
  (4) graded `relations` to every other equation in the paper
  (`none` / `potential` / `strong` with a description), and (5) an `audit-trail`.
- **Output:** one JSON file; arXiv ID as main key, equation number as sub-key.
- **Scale target:** ~350–356 equations; first 7 numbered equations per paper;
  papers in the exact order of `paper_list_44.txt`.
- **Sources:** arXiv `/abs`, `/html`, `/pdf` only (`/src`, `/e-print` forbidden);
  15 s sleep between requests; cache locally.
- **THE rule that shapes everything — NO TEXT GENERATION.** No LLM prompting.
  Every `meaning`, symbol definition, and relation description must be **extracted
  verbatim** from the paper or built from a **deterministic template**. Embedding
  / ranking models (MathBERT, BM25) are allowed only as *encoders/rankers that
  select evidence*, never to *produce* text.

Both approaches honour this; the difference is **where the evidence comes from**
(a local window vs. full-paper retrieval) and **how much machinery** is used.

---

## Approach 1 — `modified/` : streamlined structural pipeline

The current lead. It uses **document structure (MathML) + linguistic parses
(spaCy POS/dependencies)** plus a math embedding model used **only as an
encoder** for relation similarity. The core design principle, sharpened over
several iterations, is **no hand-written vocabulary**: no physics gazetteer, no
Greek-letter table, no verb gazetteer, no "generic word" blocklist. Decisions are
made from **part-of-speech tags, the dependency parse, and `unicodedata`**, so the
method generalises to papers whose terminology we have never seen. The only word
lists that remain are *domain-independent*: a fixed set of LaTeX/MathML command
names (markup, not physics) and a small set of structural "meta" nouns
(`equation`, `result`, `form`, …) that name a piece of writing rather than a
concept.

### 1.1 Architecture (`modified/src/`)
- `arxiv_html.py` — cache-first fetch (15 s delay) + equation extraction from
  arXiv LaTeXML HTML; inline LaTeX in the prose is rendered to readable Unicode
  with **`pylatexenc`** (so `I_{\rm OFF}` reads as `I_OFF`, not `I rm OFF`), while
  the `equation` field keeps verbatim LaTeX.
- `nlp_methods.py` — module-level grammar helpers (`_np_span`, `_phrase_tokens`,
  `_is_greek_letter_name`, `_content_token`) shared by the extractors, plus
  `TextTools`, `EmbeddingSimilarity`, `MeaningExtractor`, `SymbolExtractor`,
  `RelationExtractor`.
- `pipeline.py` — single orchestrator, one pass per paper.
- `common.py` — `AuditTrail` (auto-numbers duplicate method keys so the JSON
  stays valid), paper-list reader, `short()`.
- `output_check.py` — validates the exact spec schema before writing.
- `main.py` — CLI; defaults `--limit-papers 2`, `--max-equations-per-paper 7`,
  `--max-relation-edges 2`, `--relation-threshold 0.9`, embedding model
  `tbs17/MathBERT`.

### 1.2 Fetching
- `ArxivHtmlClient` requests only `https://arxiv.org/html/<id>`, sleeps 15 s after
  every real request, and caches the HTML under `data/cache/`. Re-runs read from
  cache with zero network calls. The endpoint and cache hit/miss are logged to the
  audit trail.

### 1.3 Equation extraction — structure only
- Find every `<table class="ltx_eqn_table">` in document order; handle both
  per-row numbered equations and table-level equations; dedup by number; cap at 7.
- LaTeX retrieval priority: `<annotation encoding="application/x-tex">` (verbatim)
  → `alttext` attribute → MathML text.
- **Symbols come structurally from MathML, not from vocab lists:** collect `<mi>`
  identifiers; `<mo>` operators and `<mn>` numbers are ignored *by structure*;
  `<msub>/<msubsup>` handled so a single-character subscript is treated as an
  index. Greek code points are mapped to LaTeX-style names (`𝜙` → `phi`).
  Standard math operators (`d`, `∂`, `∇`, `δ`, …) are excluded, as the spec
  allows.
- Local context windows (`before` / `after`) are built from the equation's
  paragraph plus nearest sibling paragraphs, with sentence de-duplication.

### 1.4 Meaning — a short *name*, by grammar (no gazetteer)
The `meaning` is a concise **name** for the equation (e.g. "degree of coherence",
"threshold power"), extracted verbatim — never a whole explanatory sentence. The
key fix over the first version is that names keep their prepositional complement:
a dependency-subtree walk (`_np_span`) extends a head noun with its `of`/`for`
complement, so "degree **of coherence**" is kept whole instead of being truncated
to "degree" by spaCy's base noun chunks. Selection runs in three tiers:

1. **Explicit naming** — an outright "called / known as X" or "X equation / law /
   theorem" label (trusted only in an introducing context).
2. **Nearest introducing clause** — the subject *or* object noun phrase of the
   clause closest to the equation. Which one is the named quantity is decided
   from the **parse, not a verb list**: a passive ("X is given by:") or copula
   ("X is …") clause → the subject; an active clause with a pronoun subject ("we
   define X") → the object. Selection is **proximity-first and deterministic**
   (the embedding ranker proved unreliable for short-phrase-vs-context choices and
   is deliberately *not* used to override proximity here). **Soft
   de-duplication** prefers a name an earlier equation has not already taken, so a
   paper's meanings stay varied, but reuses one when the clause offers no
   alternative.
3. **Fallback** — when no clause names the equation, nearby noun phrases are
   ranked by **MathBERT cosine** similarity to the local context (the genuinely
   ambiguous case where the encoder helps).
- Phrase cleanup is POS-driven: determiner/preposition/conjunction edges are
  trimmed by tag, bare single letters and non-ASCII glyphs are math and dropped,
  Greek-letter names are caught with `unicodedata`, markup tokens with the fixed
  LaTeX set, and a lone structural meta-noun is rejected. A valid name must
  contain a noun.

### 1.5 Symbols — dependency parses, paper-supported only
- Symbol candidates = cleaned MathML `<mi>` identifiers (subscripted forms kept as
  `base_subscript`); operators are excluded.
- For each symbol, find sentences that *mention* it (a regex that matches the name
  **and** its Unicode Greek form, e.g. `eta`↔`η`, built with `unicodedata` — no
  hand-written symbol table), then extract a definition from definitional grammar
  ("X is / denotes / represents …", the appositive "the ⟨phrase⟩ X") with a spaCy
  **dependency fallback** (subject→complement, appositive head).
- Definitions are **full noun phrases** (with `of`/`for` complements) and are
  validated by **POS**: the phrase must contain a noun and must **not** contain a
  verb/auxiliary (so a definition is a noun phrase, "efficiency of the detector",
  not a clause, "is large when …"); a lone meta-noun is rejected.
- **Strict "paper-supported" rule:** a definition is written only if a parse
  yields a supporting phrase from the arXiv text; otherwise the symbol is omitted
  (never padded with guesses). Duplicate-evidence single-letter symbols are
  skipped.

### 1.6 Relations — brute-force references + context similarity
Following the low-cost methods surveyed in the arXiv "derivation graph" study,
two model-free signals classify every ordered equation pair. The embedding model
is again **encoder-only** (`tbs17/MathBERT`, mean pooling, L2-normalised, cosine).
- **strong** — one equation's context **explicitly mentions** the other
  equation's number (the brute-force signal). The description is the connecting
  verb **lifted verbatim** from the citing sentence (the governing verb plus an
  attached preposition/particle, e.g. "given by", "reduces to"); a bare auxiliary
  is rejected and the fallback is "directly referenced". *No hand-written verb
  map.*
- **potential** — otherwise, the two equations' small extracted contexts are
  **embedded whole (no chunking)** and compared by cosine similarity, plus a
  +0.05 bonus per literally shared symbol; a score ≥ `--relation-threshold`
  (default 0.9) is a potential relation. The description lists the top shared
  content nouns (a simple bag-of-words overlap) or the shared symbols.
- **none** otherwise.
- **Edge cap:** keep all `strong` plus at most `--max-relation-edges` (default 2)
  `potential` edges per equation; the rest are downgraded to `none`. Every pair is
  still emitted with a grade (spec shape).

### 1.7 Audit + validation
- An `AuditTrail` per equation records each step (fetch, parse, equation
  extraction, the meaning tier + source sentence, each symbol definition +
  evidence, each relation grade + score, the edge limit). `output_check.check_dataset`
  enforces the exact schema (top-level keys, valid grades, complete relation keys)
  before the JSON is written.

### 1.8 Strengths / honest limits
- **Strengths:** essentially **no vocabulary lists** → not over-fit to known
  physics terms; works from MathML structure + grammar + Unicode that any
  quantum-physics paper shares; meaning is a short verbatim *name* with full
  `of`-complements; one pass, few dependencies, schema-validated, transparent
  audit trail.
- **Limits:** evidence is **local** to the equation's paragraph window, so a
  definition in a far-away "Notation" section is missed; the embedding encoder is
  reliable for relation similarity but was found *unreliable* for picking a short
  meaning, so meaning falls back to proximity; the potential threshold (0.9) and
  edge cap are heuristics that need calibration; soft de-duplication can push two
  equations that genuinely share a name onto different ones.

---

## Approach 2 — `arxiv_rag/` : hybrid-retrieval (RAG) pipeline

This version replaces the *local window* as the evidence source with **retrieval
over the whole paper**. Each equation's meaning/symbols/relations are drawn from
the top chunks returned by a **BM25 + MathBERT hybrid index**, addressing the
"definition is far from the equation" failure mode of the window approach.

### 2.1 Architecture (`arxiv_rag/src/`)
- `fetcher.py` — cache-first arXiv HTML fetch (15 s delay).
- `parser.py` — parse HTML into `EquationRecord`s (LaTeX, number, before/after,
  section, MathML symbols) and paragraph-level `TextChunk`s.
- `chunker.py` — turn records into `IndexedChunk`s for retrieval.
- `indexer.py` — `BM25Index`, `EmbeddingIndex` (MathBERT), and a fused
  `HybridIndex`.
- `retriever.py` — `HybridRetriever` with Reciprocal Rank Fusion + targeted
  retrieval helpers.
- `extractor.py` — `MeaningExtractor`, `SymbolExtractor`, `RelationExtractor` that
  consume retrieved chunks.
- `pipeline.py` / `main.py` — orchestration; defaults `--limit-papers 50`,
  `--max-equations-per-paper 7`, `--max-relation-edges 2`, model `tbs17/MathBERT`,
  target 350 equations (stop once reached).

### 2.2 Document model — chunking
- `EquationChunker.chunk_paper` builds **multiple chunk types per equation**:
  - an **equation chunk** = `before + latex + after` (the main context),
  - **symbol chunks** = symbol list + surrounding sentence,
  - **text chunks** = paragraph-level section text.
- Each `IndexedChunk` carries provenance (`paper_id`, `eq_num`, `chunk_type`,
  `latex`, `symbols`, `section`) so retrieval results stay attributable.

### 2.3 Hybrid index
- **BM25 (`rank_bm25`)** for exact-term matching, with a math-friendly tokeniser
  that combines word tokens *and* character bigrams (so notation like `H_0` and
  symbol fragments still match).
- **Embedding index** uses **MathBERT** (`AutoModel`, mean pooling, L2-normalised,
  batched) — encoder only.
- **`HybridIndex.search`** min-max normalises each score list and fuses them with
  configurable weights (`bm25_weight = emb_weight = 0.5`).
- **`HybridRetriever`** additionally offers **Reciprocal Rank Fusion** (RRF,
  `rrf_k = 60`) and helpers `retrieve_equation_context` /
  `retrieve_symbol_context` to pull chunks tied to one equation or one symbol.

### 2.4 Meaning — scored over retrieved chunks
- For each equation, query the index with `latex + context`, take the top-k chunks,
  split equation-type chunks into sentences, and score each sentence by: the
  **retrieval score** (×0.3), a source bonus (after > before), an
  **equation-reference** bonus (+2), a **symbol-overlap** bonus (+1.5), noun
  density, and a subject+verb structure bonus. Highest-scoring sentence wins.

### 2.5 Symbols — definition search over retrieved chunks
- For each symbol, scan retrieved chunks for sentences that mention it, score them
  by retrieval score + definitional patterns ("X is/denotes/represents…", "where X
  is…", "X characterizes/measures…") + noun density, then extract the definition
  phrase via those regex patterns with a spaCy dependency fallback
  (`attr`/`acomp`/`dobj` child subtree). Captures are cleaned and constrained to
  1–12 words; evidence sentences are not reused across symbols.

### 2.6 Relations — explicit refs + shared concepts
- `RelationExtractor` classifies every ordered pair:
  - **strong** if a retrieval-backed search finds an explicit reference to the
    other equation; the description is a **governing verb** mapped to a phrase
    ("derived from", "follows from", …).
  - **potential** via **Jaccard token overlap** of the two equations'
    meaning+context (≥ 0.3 and ≥ 2 shared content tokens); description lists the
    shared concepts.
  - **none** otherwise. A `--max-relation-edges` cap limits potential edges.

### 2.7 Strengths / honest limits
- **Strengths:** evidence is no longer window-bound — definitions and
  cross-references located anywhere in the paper can be retrieved; BM25+embedding
  fusion balances exact-term and semantic matching; provenance metadata keeps the
  audit trail attributable.
- **Limits:** more moving parts (index build, fusion weights, RRF parameter) and
  heavier runtime; retrieval quality depends on chunking; the audit trail is
  coarser than Approach 1's per-step trail; for a 7-equation-per-paper prototype
  the retrieval gain over a good local window is not always large.

---

## Side-by-side summary

| Aspect | Approach 1 — `modified/` | Approach 2 — `arxiv_rag/` |
|---|---|---|
| Evidence source | Local paragraph window per equation | Full-paper **BM25 + MathBERT** retrieval |
| Symbol source | MathML `<mi>` structure | MathML symbols + retrieved chunks |
| Meaning | short **name** via parse (proximity-first, soft de-dup) | linguistic scoring of retrieved sentences |
| Symbol defs | regex + spaCy dependency, POS-validated, paper-supported | regex + spaCy over retrieved chunks |
| Relations: strong | explicit reference → verb lifted verbatim | retrieval-backed reference → verb phrase |
| Relations: potential | context **cosine ≥ threshold** + shared-symbol bonus | **Jaccard** token overlap ≥ 0.3 |
| Embedding model | MathBERT (encoder only) | MathBERT (encoder) + BM25 |
| Hand-written word lists | none (only markup + meta-noun sets) | regex templates |
| Extra machinery | none (single pass) | chunker, hybrid index, retriever (RRF) |
| Audit trail | fine-grained per step | coarser, per stage |
| Best at | generalisation, simplicity, transparency | finding far-away evidence |

---

## Open items / next steps

- Calibrate thresholds on a small hand-labelled gold set and report
  precision/recall (Approach 1's relation threshold `0.9` + edge cap; Approach 2's
  fusion weights, RRF, Jaccard cut-off).
- Quantify symbol/meaning coverage across the full ~350-equation run for both and
  document success vs. failure cases.
- Optionally add a lightweight **BM25/TF-IDF sentence retriever for symbol
  definitions only** to Approach 1, to recover symbols defined far from the
  equation (improves recall without a vector DB — unnecessary at 7 equations/paper).
- Decide the submitted pipeline (`modified/` is the current lead).
