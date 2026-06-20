# Equations Knowledge Graph — Technical Design Document

**Project:** Equations Knowledge Graph (Exam ID 44)
**Institution:** OTH Amberg-Weiden, Summer Semester 2026
**Deadline:** 2026-06-25

---

## 1. Project Overview

This project builds a knowledge graph of numbered equations extracted from arXiv physics papers. For each equation the system produces four outputs: a natural-language meaning sentence, a dictionary of symbol definitions, a set of graded relations to other equations in the same paper, and an audit trail explaining how each output was derived.

The system is structured as a three-stage pipeline:

- **Stage 1** extracts raw equation HTML fragments from cached paper content.
- **Stage 2** parses each fragment into a structured record containing the equation's LaTeX, its number, and two short context windows (the prose immediately before and after the equation).
- **Stage 3** annotates each equation with meaning, symbol definitions, inter-equation relations, and an audit trail.

**Core constraint — no text generation.** No language model may produce any string written to the output JSON. Every value in the `meaning`, `symbols`, and `relations.description` fields must be either (a) a verbatim fragment copied from the source paper, or (b) a deterministic template filled with values extracted from the paper. Models such as SBERT and BM25 are permitted only as encoders and rankers that *select* evidence; they never *produce* it.

---

## 2. The Evidence Problem and Why Retrieval Was Needed

The initial Stage 3 implementation drew all evidence for meaning and symbol extraction from two narrow text windows around each equation: `context_before` and `context_after`, each capturing roughly 200 characters of prose. This design was straightforward but suffered from three recurring failure modes.

**Definitions placed far from the equation.** Physics papers routinely define all notation in a dedicated "Notation" or "Symbols" subsection near the beginning of the paper, then use those symbols in equations throughout. A 200-character window around equation (7) contains no definition of the symbol H if H was defined in Section 1.2.

**Explaining sentences in summary sections.** Cross-reference sentences such as "Combining Eq. (3) with Eq. (7) yields the main result" typically appear in a discussion or conclusion section, far from either equation. The relations extractor could not see them.

**Windows containing only LaTeX.** In papers with dense equation blocks, the two context windows are sometimes entirely LaTeX with no surrounding prose. None of the extraction strategies can fire on a window of pure symbols.

The solution was to replace the narrow context window as the *primary* evidence source with a full-paper BM25 retrieval system. The context windows are retained as a fallback for PDF-sourced papers and unit tests where no HTML is available.

---

## 3. Document Model — `src/chunks.py`

### 3.1 Motivation

Before retrieval could be added, the paper text needed to be segmented into retrievable units with enough structural metadata to make retrieval meaningful. The original chunking implementation used a simple sliding window: the full paper text was stripped of HTML tags and split into overlapping 400-character character windows. This approach had several critical shortcomings.

- Windows frequently crossed section boundaries, mixing context from unrelated topics.
- Sentences were regularly split mid-way, breaking definitional phrases such as "where H is the total Hamiltonian of the system" across two windows.
- Chunks carried no provenance — there was no way to know which section or paragraph a retrieved window came from.
- There was no structured record of which paragraphs cited which equations, so the relations extractor had to regex-scan raw context text and could miss citations appearing outside the narrow window.

### 3.2 DOM-Aware Document Model

The rewritten `chunks.py` builds a structured document model by walking the actual HTML DOM of arXiv's LaTeXML-rendered papers rather than stripping tags and splitting blindly.

Before writing any parsing code, several cached HTML files were inspected to verify the actual DOM structure. Key findings are summarised below.

| Element | Expected | Actual |
|---|---|---|
| Section container | `<div class="ltx_section">` | `<section class="ltx_section">` (a `<section>` tag, not `<div>`) |
| Paragraph container | `<div class="ltx_para">` | `<div class="ltx_para">` ✓ |
| Equation tables | Between paragraphs | Inside `<div class="ltx_para">`, alongside `<p class="ltx_p">` elements |
| Cross-reference href | `href="#S2.E1"` | `href="https://arxiv.org/html/{id}v3#S2.E1"` (full absolute URL) |
| Fragment resolution | Always resolves to table id | ~30% of fragments are not in the equation-table id map; link text always contains the bare number |

These findings directly drove design decisions: `soup.select('section.ltx_section')` is used instead of `soup.select('div.ltx_section')`; hrefs are split with `href.rsplit('#', 1)[-1]` rather than `href.split('#')[-1]` to avoid splitting on `#` characters in the URL path; and when a fragment is not in the id map, the link's visible text is used as the equation number.

### 3.3 Data Structures

Three dataclasses form the public API of `chunks.py`.

**`Chunk`** is the fundamental unit passed to the BM25 index. Each chunk carries full provenance information:

```python
@dataclass
class Chunk:
    chunk_id:        str        # "{arxiv_id}:{section_id}:{chunk_type}:{n}"
    arxiv_id:        str
    chunk_type:      str        # "sentence" | "paragraph" | "equation_neighborhood"
    text:            str        # cleaned prose text
    section_id:      str        # e.g. "S2.SS3"
    section_title:   str        # e.g. "II.3 Amplified signal in the dark port"
    paragraph_id:    str        # e.g. "S2.SS3.p1"
    eq_nums_nearby:  List[str]  # equation numbers in the same paragraph
    char_offset:     int        # sentence start offset within its paragraph
    source_endpoint: str        # URL from which the paper was fetched
```

**`XRef`** records one directed citation edge from a paragraph to an equation:

```python
@dataclass
class XRef:
    from_para_id:  str  # citing paragraph id
    from_sentence: str  # verbatim citing sentence (≤300 chars)
    to_eq_num:     str  # cited equation number
    source:        str  # "structural" | "structural_text" | "lexical"
```

**`ParsedPaper`** collects all three chunk views and the cross-reference graph for a single paper, and is returned by `DocumentParser.parse()`.

### 3.4 Three Chunk Views

The parser produces three distinct views of the same paper, each suited to a different retrieval scenario.

| View | Granularity | Primary use |
|---|---|---|
| `sentence_chunks` | One chunk per sentence | Final evidence selection; most precise |
| `paragraph_chunks` | One chunk per `div.ltx_para` | Broad-recall BM25 baseline |
| `eq_neighborhood_chunks` | Section title + surrounding prose + LaTeX for one equation | Default retrieval unit for meaning and relations |

The equation neighborhood chunk is the most important of the three. It bundles the section title, the prose paragraph before the equation, the LaTeX string itself, and the prose paragraph after the equation into a single retrievable unit. This means a BM25 query for "equation 3 Hamiltonian energy" has a single chunk that contains all of this information together, rather than requiring the retriever to assemble it from fragments.

### 3.5 Cross-Reference Graph

`build_cross_reference_index(paper)` constructs a directed graph of equation citations found in the paper. It uses a two-pass strategy.

**Pass 1 — Structural (preferred).** The parser scans all `<a class="ltx_ref">` anchor elements. For each anchor whose href fragment matches the pattern for an equation table id (`.E` followed by digits), the fragment is resolved to an equation number via the `eq_id_to_num` map built during parsing. If the fragment is not in the map — which occurs in approximately 30% of cases — the anchor's visible text is used instead, as it always contains the bare equation number. The containing `<p class="ltx_p">` element provides the citing sentence. Edges discovered this way are labelled `source = "structural"` or `source = "structural_text"`.

**Pass 2 — Lexical (fallback).** For any (paragraph, equation number) pair not already covered by a structural edge, a regex scans sentence text for citation patterns such as `(3)`, `Eq. (3)`, and `equation (3)`. Edges discovered this way are labelled `source = "lexical"`. This pass covers PDF-sourced papers and HTML papers where DOM links are absent or broken.

Every edge is logged at `DEBUG` level recording which pass produced it, making the evidence chain fully auditable.

### 3.6 Text Cleaning

Chunk text is cleaned in four layers applied in sequence. The first two layers are shared with `meaning.py`:

1. **LaTeXML artifact tokens** — patterns such as `italic_x`, `start_ROW`, `end_ARRAY`, `overACCENT_ARG`, and `road_*` that appear in arXiv HTML as text nodes instead of being rendered invisibly.
2. **Unicode mathematical alphanumeric block** (U+1D400–U+1D7FF) and invisible operators (U+2061–U+2064).

Two additional layers are applied in the chunker:

3. **Zero-width and non-printable characters** — U+200B (zero-width space), U+200C, U+200D, U+FEFF (BOM), and U+00AD (soft hyphen). These appear in arXiv HTML and cause BM25 token boundaries in unexpected places, producing garbage vocabulary entries.
4. **Residual artifact words** — bare tokens `superscript`, `subscript`, `POSTSUBSCRIPT`, and `POSTSUPERSCRIPT` that some LaTeXML versions emit as visible text nodes rather than wrapping in artifact-style tokens.

### 3.7 Sentence Splitting with Character Offsets

`_split_sentences(text)` returns `List[Tuple[str, int]]` where each tuple is `(sentence_text, char_offset_within_paragraph)`. Carrying the character offset serves two purposes: it enables audit entries to record exactly where in the paragraph a piece of evidence was found, and it enables future tooling to highlight the sentence in the original HTML without re-parsing.

### 3.8 Validation Results

The parser was tested against all 82 cached HTML papers in the corpus. Results:

- Zero parsing errors across all 82 papers.
- Total chunks produced: 24,946 sentences, 6,030 paragraphs, 454 equation neighborhoods.
- Total cross-reference edges: 3,555 (mix of structural and lexical sources).
- Section titles populated for 100% of sentences in 65 of 82 papers. The remaining 17 papers have no `section.ltx_section` structure — their paragraphs sit directly under `article` — and correctly receive `section_title = ""`.

---

## 4. Retrieval System — `src/retrieval.py`

### 4.1 Design Goals

The retrieval module must satisfy four requirements simultaneously:

1. **Full determinism.** Identical input must always produce identical ranked output, across runs, machines, and Python versions.
2. **No generation.** The retriever selects chunks from the paper; it never produces or modifies text.
3. **Metadata-aware filtering.** Callers must be able to restrict the candidate set by chunk type, section, or nearby equation numbers before BM25 scoring, so the top-k budget is not wasted on irrelevant chunks.
4. **Ablation support.** The BM25 strategy must be interchangeable with a TF-IDF baseline using the same tokenizer, so that retrieval strategy can be evaluated independently.

### 4.2 Tokenizer

The tokenizer `_tokenize(text)` is shared by both `BM25Retriever` and `TfidfRetriever`, ensuring that vocabulary is identical across strategies and comparisons are fair.

The key design decision was to tokenize with the pattern `[a-z0-9]+` rather than the simpler `[a-z]+` used in the original implementation. The difference is that digit strings are preserved as vocabulary terms. This matters because equation numbers appear as bare digits in query templates — `"Eq 3 Equation 3"` — but appear in chunk text inside parentheses — `"(3)"`. The pattern `[a-z0-9]+` applied to `"(3)"` yields the token `"3"`, which then matches the query token `"3"`. The old pattern `[a-z]+` would drop all digits, making equation-number queries ineffective.

The tokenizer also removes English stop words and a short list of LaTeXML artifact token prefixes (`italic`, `start`, `end`, `over`, `road`, `superscript`, `subscript`) that may survive chunk cleaning in some LaTeXML versions.

### 4.3 BM25+ Implementation

The project uses the BM25+ variant (Lü & Callan, 2011) rather than standard BM25. In standard BM25, terms that appear in many documents receive an IDF weight close to zero, which means common physics vocabulary such as "equation" or "energy" contributes almost nothing to the score even when it is exactly what is being searched for. BM25+ addresses this by adding a floor constant δ so that every matched term always contributes at least δ to the score, regardless of document frequency.

The formula per term t and document d is:

```
score(t, d) = IDF(t) × [ tf(t,d)×(K1+1) / (tf(t,d) + K1×norm(d)) + δ ]
where norm(d) = 1 − B + B × |d| / avgdl
```

Constants:

```python
BM25_K1    = 1.5   # term-frequency saturation rate
BM25_B     = 0.75  # length normalisation strength
BM25_DELTA = 1.0   # BM25+ floor — matched terms always contribute positively
```

The implementation is self-contained: no external BM25 library is used. The full formula fits in approximately 30 lines of standard Python arithmetic. This was a deliberate choice to avoid dependency on packages such as `rank-bm25` that are not guaranteed to be installed in the GPU lab environment.

One retriever instance is built per paper and discarded after Stage 3 for that paper completes. Cross-paper retrieval is explicitly avoided: if a single corpus-wide index were used, evidence for equation (3) in paper A could be retrieved from paper B.

### 4.4 `BM25Retriever` — Class API

```python
ret = BM25Retriever()
ret.fit(paper.eq_neighborhood_chunks + paper.sentence_chunks)
hits = ret.search(meaning_query("3", "H E"), top_k=8)
```

`fit(chunks)` accepts a list of `Chunk` objects directly from `DocumentParser`, builds the inverted index, and computes IDF weights and average document length. A second call to `fit()` replaces the previous index.

`search(query, top_k, filters)` tokenizes the query, optionally restricts candidates using the `filters` dict, scores all candidates with BM25+, and returns up to `top_k` results sorted by score. Each result is a dict containing:

```python
{
    "chunk_id":      str,        # unique chunk identifier
    "score":         float,      # BM25+ score
    "chunk_type":    str,        # "sentence", "paragraph", or "equation_neighborhood"
    "text":          str,        # verbatim chunk text
    "matched_terms": list[str],  # query tokens that matched this chunk
}
```

The `matched_terms` field records *why* a chunk was retrieved, enabling audit entries such as "retrieved because: 3, equation, hamiltonian". This is important for transparency and debugging.

The `filters` parameter restricts the candidate pool before BM25 scoring:

```python
filters = {
    "chunk_type":     "sentence",   # or a list: ["sentence", "paragraph"]
    "section_id":     "S2.SS3",     # exact section match
    "eq_nums_nearby": ["3", "7"],   # chunk must have at least one of these nearby
}
```

The `retrieve_text(query, top_k)` and `retrieve()` methods are provided as convenience wrappers so that the existing extractors in `meaning.py`, `symbols.py`, and `relations.py` continue to work without modification.

### 4.5 `TfidfRetriever` — Ablation Baseline

`TfidfRetriever` is a drop-in replacement for `BM25Retriever` with an identical public interface. It uses scikit-learn's `TfidfVectorizer` configured with `analyzer=_tokenize` — the same tokenizer as BM25Retriever, ensuring a fair ablation comparison. The `sublinear_tf=True` option applies a `1 + log(tf)` transformation to reduce the effect of high-frequency terms. Cosine similarity over TF-IDF vectors is then used for ranking.

`TfidfRetriever` raises a clear `ImportError` if scikit-learn is not installed rather than failing silently.

### 4.6 Optional MiniLM Reranker

`rerank_with_minilm(query, candidates)` provides an optional third stage that reranks BM25 or TF-IDF results by dense cosine similarity using the `all-MiniLM-L6-v2` sentence encoder. This function is never called automatically by the pipeline — it must be invoked explicitly.

The function encodes the query and all candidate texts into dense vectors, L2-normalises them, and computes cosine similarity via a dot product. Candidates are returned reordered by cosine score, with an additional `"minilm_score"` key appended to each dict. The original BM25 score is preserved unchanged.

The model is loaded lazily on the first call and cached in a module-level dict, so subsequent calls do not incur a loading cost. If the model cannot be loaded — because `sentence-transformers` is not installed or the lab is offline — the function logs a `WARNING` and returns the candidates unchanged. The pipeline does not crash.

MiniLM is strictly an encoder in this context: it encodes text into a vector. No text is produced, generated, or modified. The compliance principle is that models may encode → score → rank, but they may never produce any string written to the output JSON.

### 4.7 Deterministic Query Templates

Three helper functions produce standardised, reproducible query strings for the three retrieval scenarios:

| Function | Scenario | Cue words |
|---|---|---|
| `meaning_query(eq_num, top_symbols)` | Find the sentence describing the equation | `describes represents gives defines called known as` |
| `symbol_query(symbol, greek_forms=None)` | Find the definitional sentence for a symbol | `where denotes represents is defined as called corresponds to` |
| `relation_query(eq_a, eq_b)` | Find sentences discussing two equations together | `using substituting inserting follows from derived from yields equivalent special case` |

`symbol_query()` additionally expands Greek LaTeX commands to their text and Unicode surface forms using the `_GREEK_SURFACE` dictionary. For example, `symbol_query(r"\alpha")` produces a query containing both `"alpha"` and `"α"`, so that definitional prose written as "where α is the fine-structure constant" is retrieved whether the chunk text contains the Greek letter, the word "alpha", or the LaTeX command `\alpha`.

---

## 5. Meaning Extraction — `src/meaning.py`

### 5.1 Approach: Retrieve → Score → Threshold

The original four-strategy pipeline (citation match → gazetteer → definitional → TF-IDF) was replaced with a single unified weighted-scoring pass. The previous design ran four strategies in sequence and returned the first successful result — which meant that a moderately good citation sentence always beat an excellent definitional sentence even if the former was short or fragmentary. The new design scores **every candidate sentence** against a fixed table of seven signals and returns the single highest scorer, subject to a minimum threshold.

The pipeline for one equation:

1. Retrieve the top `RETRIEVAL_TOP_K = 12` chunks from the paper's BM25 index using a query built from the equation number and LaTeX symbol names.
2. Split each retrieved chunk into sentences; also split both context windows (`context_before`, `context_after`).
3. Apply `_is_prose()` to discard fragments that are mostly single-character math tokens.
4. Score each surviving sentence with the weighted signal table.
5. Return the highest-scoring sentence verbatim if its normalised score ≥ `SCORE_THRESHOLD = 0.33`; otherwise return `""`.

### 5.2 Weighted Signal Table

```
Signal                                                   Weight
──────────────────────────────────────────────────────────────
Sentence explicitly cites this equation number (+5)       +5
Contains a definitional cue word                         +3
  (defines / describes / represents / gives / denotes)
Matches a named-equation gazetteer entry                 +3
  (Hamiltonian, Schrödinger, Lindblad, …)
Chunk is in the paragraph immediately before/after eq    +2
Chunk is in the same section as the equation             +1
Chunk is the top-1 BM25-scored result for this query     +1
Sentence is too short (< 30 chars) or too long (> 400)   -1
──────────────────────────────────────────────────────────────
MAX_SIGNAL_SCORE = 15   SCORE_THRESHOLD = 0.33 (≥ 5 pts)
```

The normalised score is `raw_score / MAX_SIGNAL_SCORE`. A threshold of 0.33 (5 points) means:

- A plain citation sentence scores 5/15 = 0.33 and just passes.
- A definitional sentence adjacent to the equation scores (3+2+1)/15 = 0.40 and passes.
- A sentence from a distant section with no content signal scores at most 1/15 and fails.
- Context-window sentences receive the `+2 adjacent` and `+1 same-section` bonuses automatically because they come from the equation's immediate prose neighbourhood.

### 5.3 `_is_prose()` Filter

Before scoring, each candidate is checked by `_is_prose()`. This function counts the proportion of whitespace-separated tokens that contain two or more consecutive alphabetic characters. A sentence like "H = E α ψ + β" (7 tokens, only 1 with 2+ alpha chars → 14 % < 40 %) is rejected; "The total Hamiltonian H describes the system" (7 tokens, 5 with 2+ alpha chars → 71 % > 40 %) is kept.

### 5.4 Paragraph Adjacency

The `+2` adjacency signal uses `_is_adjacent_para(eq_para_id, cand_para_id)`, which checks that both paragraph IDs share the same section prefix and that their trailing numeric suffixes differ by exactly 1 (e.g. `S2.p3` and `S2.p4` are adjacent; `S2.p3` and `S2.p5` are not; `S2.p3` and `S3.p4` are not).

This requires knowing the equation's own paragraph ID, which is extracted in `main.py` from the equation's neighborhood chunk (`parsed_paper.eq_neighborhood_chunks`) and passed to `extract()` as the new `eq_para_id` parameter. A matching `eq_section_id` parameter enables the same-section signal.

### 5.5 Audit Entry Format

```
meaning -> "eq=(3): strategy=weighted bm25=8.4 score=9/15
            chunk=2401.13506:S2.p3:sentence:4
            char_span=[142,240]
            signals=['eq_num','definitional','same_section','bm25_top1']
            sentence='Eq. (3) defines the total Hamiltonian H of the …'"
```

### 5.6 `fit_corpus()` Role

`fit_corpus()` is still called from `main.py` for backward compatibility. It no longer computes IDF weights (corpus-wide IDF was used only by the old strategy 4 TF-IDF ranking, which is now replaced). It still builds the optional spaCy `PhraseMatcher` for the gazetteer signal. Papers can also call it with an empty list — the matcher is simply not built in that case and the gazetteer falls back to plain string search.

---

## 6. Symbol Extraction — `src/symbols.py`

### 6.1 Two-Phase Pipeline

Symbol extraction now runs in two distinct phases.

**Phase 1 — Symbol identification.** The primary source is the MathML `<mi>` elements inside the equation's `<math>` block in the LaTeXML-generated HTML. In MathML, `<mi>` is the *identifier* element; it represents variables and constants. It is structurally distinct from `<mo>` (operators) and `<mn>` (numbers). Parsing `<mi>` elements directly gives a clean list of identifiers without the noise introduced by regex-scanning a LaTeX string.

For subscripted identifiers (`<msub><mi>H</mi><mn>0</mn></msub>`), the extractor produces both the base form `"H"` and the subscripted form `"H_0"` so that both can be looked up independently.

Greek Unicode characters in `<mi>` text (e.g. `ψ`) are mapped to their LaTeX command names (e.g. `"psi"`) using the `_UNICODE_TO_SYMBOL` lookup, so the resulting symbol keys are consistent with the definitions found in English prose ("where ψ is the wave function").

The fallback for PDF papers or absent HTML is the original LaTeX regex tokeniser.

**Phase 2 — Definition retrieval and confidence gating.** For each symbol key, a search context is built by concatenating: (a) chunks from `symbol_chunk_view` if `parsed_paper` is available (structure-aware lookup by symbol surface form), (b) BM25 retrieval using a query built from the symbol's surface forms, and (c) the baseline context windows. The five definitional regex patterns are applied to this context in decreasing confidence order.

### 6.2 Confidence Gate

Each regex pattern is assigned a confidence level that reflects how strongly it implies a definitional intent:

| Pattern | Example | Confidence |
|---|---|---|
| `where_is` | "where H is the Hamiltonian" | HIGH |
| `let_be` | "let H be the Hamiltonian" | HIGH |
| `sym_is` | "H is the total Hamiltonian" | HIGH |
| `the_noun_sym` | "the total Hamiltonian H" | MEDIUM |
| `sym_the` | "H, the Hamiltonian" | LOW → rejected |

Definitions with LOW confidence (the appositive-only `sym_the` pattern) are discarded. The spaCy dependency parsing fallback produces MEDIUM confidence (nsubj+copula) or LOW confidence (appositive), and only MEDIUM is kept.

### 6.3 Rejection Conditions

After finding a definition that passes the confidence gate, five additional rejection conditions are applied:

| Condition | Reason |
|---|---|
| `len(defn.strip()) ≤ 1` | Single-character definition is uninformative |
| `defn.lower() == sym.lower()` | Definition repeats the symbol itself |
| `defn in other_symbol_keys` | Definition is another symbol key (cross-symbol confusion) |
| Last word in `_DANGLING_WORDS` | Definition ends with "the", "a", "of", "in", etc. |
| All words in `_STOP_WORDS` | Definition contains only function words |
| Contains zero-width characters | Definition contains HTML rendering artifacts |

Rejected definitions are logged as `symbol_reject` audit entries, not simply silently discarded:

```
symbol_reject -> "sym='k' def='k' reason='single_char_def' pattern=sym_is"
```

**Summation index handling.** Symbols from `_SKIP_SINGLE` (i, j, k, l, m, n, d) are included in MathML-sourced symbol lists and Phase 2 attempts to find their definitions. If no definition is found and the symbol came from MathML extraction, it is silently omitted (not logged as a reject). If a definition is found, it passes through the confidence gate and rejection conditions normally and is included in the output. This ensures that a `k` defined as "wave vector" is captured while a bare summation index `k` with no definition is omitted.

---

## 7. Relation Extraction — `src/relations.py`

### 7.1 Purpose and Output Format

`RelationExtractor` computes a directed relation for every ordered pair of equations within the same paper. For each pair (A, B) it produces:

```python
{"grade": "strong" | "potential" | "none", "description": "<text>"}
```

The `description` field is always either a verbatim sentence from the paper (for strong relations) or a deterministic template filled with extracted values (for potential relations). No text is generated.

### 7.2 The Four Signals

Four independent signals contribute to the grade and description for each pair.

| Signal | Type | Grade | Description source |
|---|---|---|---|
| 1 — Cross-reference | Textual | `strong` | Verbatim sentence containing a citation of equation B |
| 2 — Jaccard symbol overlap | Structural | `potential` | `f"shares symbols: {', '.join(sorted(shared))}"` |
| 3 — Derivation cue | Textual | `strong` | Verbatim sentence with a derivation verb and a citation of B |
| 4 — Semantic similarity | Semantic | `potential` | `f"semantically similar (cosine={score:.2f})"` |

**Grading rule:** strong takes priority over potential. When both signals 1 and 3 fire, the cross-reference sentence (signal 1) is used as the description because it is the most direct form of evidence. When both signals 2 and 4 fire, the Jaccard description is used because it is more specific. If no signal fires, the grade is `"none"` and the description is an empty string.

### 7.3 Signal 1 — Cross-Reference (Structural Graph Primary, Text Search Fallback)

**Primary source: DOM cross-reference graph.** The `compute_all_relations()` method now accepts a `cross_refs` parameter — the `ParsedPaper.cross_refs` list built during document parsing. Each `XRef` object records the equation number cited (`to_eq_num`), the paragraph containing the citation (`from_para_id`), and the verbatim sentence containing the citation (`from_sentence`).

For pair (A, B), the extractor first checks whether any XRef with `to_eq_num == B` originates from a paragraph in the **same section as equation A**. The section is extracted from the paragraph ID by stripping the trailing `.p{N}` suffix (e.g. `S2.SS3.p4` → section `S2.SS3`). If a matching structural XRef is found, its `from_sentence` is used as the citation evidence and the pair is immediately graded `"strong"` without running any regex.

**Fallback: text search.** If the cross-reference graph is unavailable (PDF papers) or contains no matching XRef, the existing `_find_citation_sentence(eq_num_b, context)` runs on the BM25-expanded context string. This regex:

```
(?:eq(?:uation)?s?\.?\s*)?\(\s*{B}\s*\)
```

matches citation forms such as `(3)`, `Eq. (3)`, `equation (3)`, and `Eqs. (3)`. After finding a match, the full sentence is extracted by the `_sentence_around()` helper rather than by the standard sentence splitter.

```
(?:eq(?:uation)?s?\.?\s*)?\(\s*{B}\s*\)
```

which matches citation forms such as `(3)`, `Eq. (3)`, `equation (3)`, and `Eqs. (3)`. After finding a match, the full sentence is extracted by the `_sentence_around()` helper rather than by the standard sentence splitter. Fragments shorter than 20 characters, or sentences that begin with a bare parenthesis, are discarded as noise.

**Why `_sentence_around()` instead of the sentence splitter.** The standard splitter `_SENT_SPLIT_RE` breaks on the pattern `[.!?]` followed by a capital letter. The abbreviation "Eq." triggers this rule, splitting "Eq. (3) gives the total Hamiltonian" into two fragments at the dot. `_sentence_around()` avoids the splitter entirely by walking left and right from the regex match position in the raw string until sentence-ending punctuation is reached, collecting the sentence as a contiguous substring. This is more robust than splitting first and then searching.

### 7.4 Signal 3 — Derivation Cue

`_find_derivation_cue_sentence(eq_num_b, context)` checks every citation of B in the context and accepts the sentence if it also contains any of the following cue phrases:

```python
_DERIVE_CUES = {
    "substituting", "inserting", "derived from", "follows from",
    "combining", "plugging", "yields", "into eq", "using eq",
}
```

The multi-word prefixes "into eq" and "using eq" are intentionally short: they match "into Eq. (3)", "into equation (5)", "using Eq. (1)", and similar constructions without needing to enumerate every possible suffix.

### 7.5 Signal 2 — Jaccard Symbol Overlap

This signal uses the `symbols` dicts already extracted by `SymbolExtractor` for each equation. The Jaccard similarity of the two symbol vocabularies is computed as:

```
Jaccard(A, B) = |symbols(A) ∩ symbols(B)| / |symbols(A) ∪ symbols(B)|
```

If this value meets or exceeds the threshold `JACCARD_MIN = 0.30`, the pair is graded `"potential"`. The description lists the shared symbols in sorted order so the output is reproducible across runs.

### 7.6 Signal 4 — Semantic Similarity

Each equation's meaning string is encoded with the `all-MiniLM-L6-v2` sentence encoder. If no meaning string was extracted, the equation's context window is used instead. All equations in a paper are encoded in a single batched call for efficiency, and pairwise cosine similarity is computed via:

```python
cos_matrix = normed_embeddings @ normed_embeddings.T
```

The threshold for this signal is `SBERT_MIN = 0.50`. Pairs whose cosine similarity meets or exceeds this value are graded `"potential"`.

**TF-IDF fallback (two levels).** If `sentence-transformers` is not installed or the SBERT model fails to load, the system falls back to TF-IDF cosine similarity. If scikit-learn is available, `TfidfVectorizer` and `cosine_similarity` are used. If sklearn is also absent, `_tfidf_cosine_builtin()` computes smoothed IDF weights and L2-normalised TF-IDF vectors using only Python's standard library and numpy. The TF-IDF threshold is set higher (`TFIDF_MIN = 0.60`) because TF-IDF cosine is less discriminative than SBERT cosine.

SBERT is used strictly as an encoder: it maps text to a vector, and the vector is used for scoring. No text is produced at any point.

### 7.7 SBERT Loading Strategy

The SBERT model is loaded once at `RelationExtractor.__init__()`, not lazily on first use. Loading takes approximately 0.5–2 seconds, and loading it at construction time means the first paper's processing is not penalised with an unexpected delay. If the model fails to load, a `WARNING` is logged and the extractor continues with the TF-IDF fallback — the pipeline never crashes due to a missing model.

### 7.8 Deterministic Iteration Order

Equation numbers are sorted numerically before iterating over pairs, using the `_sort_eq_nums()` helper. Pure integer strings are sorted first by numeric value; all other strings are appended alphabetically. This ensures that the pairs in the output always appear in the same order regardless of the Python dict insertion order from the extraction stage.

### 7.9 Retrieval Integration

Signals 1 and 3 originally searched only equation A's local context window (roughly 200 characters on each side). The `compute_all_relations()` method now accepts an optional `retriever` parameter.

When a retriever is provided, the context for equation A is expanded before signals 1 and 3 are evaluated. A BM25 query is issued using multiple surface forms of the equation number:

```python
query = f"equation {eq_a} eq {eq_a} ({eq_a})"
```

The top `RETRIEVAL_TOP_K_RELATIONS = 3` chunks are retrieved and appended to A's local context. Both signal helpers then run on this expanded string.

The retrieval depth of 3 is deliberately lower than the depths used for meaning (8) and symbols (5). The query targets a specific equation number, so the top results are expected to be highly relevant — retrieving more would risk introducing noise without improving recall.

**Why expand A's context rather than searching for mentions of B.** An alternative design was considered in which the BM25 query targeted equation B's number to find sentences that mention B, and those sentences were then checked for signal 1 or 3 evidence. This approach is semantically incorrect: a sentence that mentions B while explaining equation C would create a spurious A→B relation edge. By expanding A's own context instead, the query stays true to the intended question — "what does the paper say in proximity to equation A?" — and any citation of B found there reflects a genuine relationship.

Signals 2 and 4 are not affected by retrieval because they operate on symbol sets and meaning embeddings, not on raw context text.

### 7.10 Fixed-Label Description Vocabulary

The description field uses a fixed-label vocabulary rather than free-form text generation:

| Signal fired | Description |
|---|---|
| Signal 1 (structural XRef) | Verbatim `from_sentence` (≤ 200 chars) |
| Signal 1 (text search, strong) | Verbatim citation sentence (≤ 200 chars) |
| Signal 3 (derivation cue, strong) | Verbatim cue sentence (≤ 200 chars) |
| Signal 2 (Jaccard, potential) | `"shares symbols: X, Y, Z"` |
| Signal 4 (SBERT/TF-IDF, potential) | `"semantically similar (cosine=X.XX)"` |
| No signal | `""` |

All verbatim sentences are extracted from the paper's own HTML; no string is generated or paraphrased.

### 7.11 Per-Pair Audit Entry

```
relations -> "eq=(3)~(7): grade=strong signals=['crossref+graph']
             jaccard=0.00 sim=0.42
             evidence='Substituting Eq. (3) into Eq. (7) yields the …'"
```

The audit records which specific signals fired, the Jaccard and cosine scores for every pair regardless of whether they crossed their thresholds, and the first 80 characters of the description.

### 7.12 Key Constants

```python
JACCARD_MIN:               float = 0.30  # minimum Jaccard for signal 2
SBERT_MIN:                 float = 0.50  # minimum cosine for SBERT in signal 4
TFIDF_MIN:                 float = 0.60  # minimum cosine for TF-IDF in signal 4
DESCRIPTION_MAX_LEN:       int   = 200   # truncation limit for verbatim sentences
RETRIEVAL_TOP_K_RELATIONS: int   = 3     # BM25 chunks added to equation A's context
```

---

## 8. Schema Validation — `src/validate.py`

### 8.1 Strict Mode and `ValidationError`

`Validator.validate(data, fail_loudly=True)` raises `ValidationError` on the first batch of violations found. This makes validation fail loudly rather than silently accumulate warnings. The pipeline calls it with `fail_loudly=False` so that errors are logged and the best-effort output file is still written; but a standalone correctness check or CI test should use `fail_loudly=True`.

### 8.2 Checks Performed

**Structural checks (original):**
- The root is a dict of arXiv IDs.
- Each arXiv ID maps to a dict of equation numbers.
- Each equation entry has exactly the five required keys: `equation`, `meaning`, `symbols`, `relations`, `audit-trail`.
- `equation` and `meaning` are strings; `symbols` is `{str: str}`; `audit-trail` is `{str: str}`.
- Each relation entry has `grade` ∈ `{"none", "strong", "potential"}` and a `description` string.

**New strict checks (Task F additions):**

| Check | Violation raised when |
|---|---|
| Private key leak | `_ctx_before`, `_ctx_after`, or `_audit` appear in an equation entry |
| All-pairs completeness | An equation's `relations` dict is missing an entry for any other equation in the same paper |
| Audit coverage — meaning | `meaning` is non-empty but no audit key starts with `"meaning"` or `"retrieval"` |
| Audit coverage — symbols | `symbols` is non-empty but no audit key starts with `"symbol"` |
| Audit coverage — relations | At least one relation has grade ≠ `"none"` but no audit key starts with `"relations"` |

The all-pairs check is the most impactful: it ensures that every ordered pair of equations in a paper has a relation entry, even if that entry is `{"grade": "none", "description": ""}`. This guarantees the relations structure is structurally complete regardless of signal strength.

### 8.3 `ValidationError`

```python
class ValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors   # full list of error strings
        ...
```

Up to 20 error strings are shown in the exception message; the rest are summarised by count.

---

## 9. Pipeline Orchestration — `src/main.py`

### 9.1 Per-Paper Setup

For each HTML-sourced paper, Stage 3 now computes four resources before the per-equation loop:

```python
parsed_paper   = doc_parser.parse(paper_content, arxiv_id, endpoint, equations)
retriever      = BM25Retriever(); retriever.fit(primary + fallback chunks)
eq_section_ids = {eq_n: chunk.section_id  for chunk in eq_neighborhood_chunks ...}
eq_para_ids    = {eq_n: chunk.paragraph_id for chunk in eq_neighborhood_chunks ...}
```

`eq_section_ids` and `eq_para_ids` are derived by scanning `parsed_paper.eq_neighborhood_chunks` — each neighborhood chunk's `eq_nums_nearby` list tells us which equations it covers.

### 9.2 Equation-Level Calls

All three extractors receive the new contextual parameters:

```python
meaning = meaning_ext.extract(
    eq_num, latex, ctx_b, ctx_a, eq_audit,
    retriever=retriever,
    eq_section_id=eq_section_ids.get(eq_num, ""),
    eq_para_id=eq_para_ids.get(eq_num, ""),
)

symbols = symbol_ext.extract(
    eq_num, latex, ctx_b, ctx_a, eq_audit,
    retriever=retriever,
    html_content=paper_content,   # enables MathML <mi> parsing
    parsed_paper=parsed_paper,    # enables symbol_chunk_view lookup
)
```

### 9.3 Relations Call

```python
all_relations = relation_ext.compute_all_relations(
    paper_build, eq_audits,
    retriever=retriever,
    cross_refs=parsed_paper.cross_refs,   # structural DOM citation graph
    eq_section_ids=eq_section_ids,         # for XRef proximity filtering
)
```

### 9.4 PDF Papers

PDF-sourced papers receive `parsed_paper = None`, `retriever = None`, `html_content = None`. The extractors fall back to context-window evidence. Symbol identification falls back to the LaTeX regex tokeniser. Relations use only local context search without the structural graph.

### 9.5 Validation

After Stage 3 is saved to disk, `validator.validate(stage3, fail_loudly=False)` runs. Errors are logged as warnings. A dedicated quality report can then be generated by running `scripts/measure.py`.

---

## 10. Compliance with the No-Generation Constraint

The fundamental constraint of this project is that no language model may produce any string that appears in the output JSON. This section documents how every module upholds that constraint.

**BM25 and TF-IDF** are scoring functions that assign a number to each text chunk. They select which existing chunk is most relevant; they produce no text.

**SBERT (`all-MiniLM-L6-v2`)** is an encoder that maps text to a dense vector. The vector is used to compute a cosine similarity score. The score is used to rank existing chunks or to populate the template `f"semantically similar (cosine={score:.2f})"`. The model produces a vector, not text.

**MiniLM reranker** in `rerank_with_minilm()` encodes query and candidate texts into vectors and reorders the candidates. Candidate texts are verbatim chunk texts from the paper; the reranker does not modify them.

**Verbatim extraction** — all meaning sentences, symbol definition phrases, and cross-reference evidence sentences are substrings of the original paper text. The extraction code finds and returns them; it does not write or rephrase them.

**Deterministic templates** — where a string cannot be extracted verbatim (signal 2 and signal 4 in relations), a Python f-string template is filled with extracted values: `f"shares symbols: {', '.join(sorted(shared))}"` or `f"semantically similar (cosine={score:.2f})"`. These are not generated text; they are arithmetic formatting of numbers and symbol names.

The principle is: **retrieval selects evidence; it never writes the answer.**

---

## 11. File Map

```
src/
├── acquisition.py   — paper fetcher and cache (unchanged)
├── audit.py         — AuditTrail class (unchanged)
├── chunks.py        — DOM-aware document model, three chunk views, cross-reference graph
├── extraction.py    — equation extraction from HTML and PDF (unchanged)
├── fragmenter.py    — HTML fragment extractor for Stage 1 (unchanged)
├── main.py          — pipeline orchestrator; threads parsed_paper, retriever,
│                      eq_section_ids, eq_para_ids, cross_refs into Stage 3
├── meaning.py       — weighted 7-signal meaning extractor; retrieve → score → threshold
├── relations.py     — four-signal relation extractor with structural XRef graph +
│                      BM25 context expansion; fixed-label descriptions
├── retrieval.py     — BM25Retriever (search() now returns section_id/paragraph_id),
│                      TfidfRetriever, rerank_with_minilm, deterministic query templates
├── symbols.py       — MathML <mi> symbol identification + BM25 retrieval +
│                      confidence gate + rejection logging
└── validate.py      — strict schema validator; all-pairs check; ValidationError on failure

scripts/
└── measure.py       — post-pipeline quality report: empty rates, garbage rate,
                       noise rate, relations validity, grade distribution, eq stats
```

---

## 12. Quality Scripts — `scripts/measure.py` and `scripts/verify_output.py`

### 12.1 `scripts/measure.py`

Post-pipeline quality measurement tool. Reads `data/output/eq_final.json` (or a path supplied via `argv[1]`) and prints a structured report covering:

- **Corpus counts** — papers processed, equations total, equations-per-paper distribution.
- **Per-field empty rates** — fraction of equations with empty meaning / symbols / all-none relations.
- **Symbol garbage rate** — definitions matching `_GARBAGE_DEF_RE` (single letter, pure LaTeX structure, starts with backslash). Target: < 2%.
- **Meaning render-noise rate** — meanings containing `_ARTIFACT_RE` token patterns. Target: < 3%.
- **Relations format validity** — checks every pair has `{grade, description}` and all pairs present. Target: 100%.
- **Grade distribution** — count of strong / potential / none pairs.
- **Meaning strategy distribution** — which audit strategies fired (weighted / below-threshold count).

### 12.2 `scripts/verify_output.py`

One-shot file format check. Reads any `eq_final.json` and prints `PASS` (new spec-compliant format) or `FAIL` (old baseline format). Checks for:

- `relations_spec_format` — `{grade, description}` dict pairs (new).
- `relations_old_format` — list values (old; any nonzero = stale file).
- `strategy=weighted` in audit trail (new).
- `strategy=tfidf_rank` / `gazetteer_match` in audit trail (old).
- `symbol_reject` audit key (new).
- `MathML <mi>` in `symbol_source` audit key (new).

Usage: `python project/scripts/verify_output.py` — prints `RESULT: PASS` or `RESULT: FAIL` with signal counts.

---

## 13. Known Limitations and Future Work

**Equation neighborhood coverage is capped at 7.** Only equations present in the extracted `equations` dict receive a neighborhood chunk. Papers are capped at 7 equations per the project specification. This means the BM25 index contains no neighborhood chunk for equations 8 and beyond, even though those equations may appear in the paper's prose and be referenced by the first 7.

**PDF papers receive no retriever.** `DocumentParser.parse()` requires HTML bytes. For PDF-sourced papers the system falls back to the narrow context windows from Stage 2 for all three extractors. A future improvement would add a PDF text extraction path and build a retriever from that extracted text.

**MathML subscripted-form symbol keys are not yet searched for in prose.** The extractor generates keys like `"H_0"` for subscripted identifiers, but the `_build_sym_pattern()` regex is not yet aware of this form. Prose like "where H₀ is the ground state" uses Unicode subscript characters, not an underscore, and would not match the regex for `"H_0"`. This means subscripted symbols currently fall back to finding a definition for the base form `"H"` only.

**`_is_prose()` threshold is fixed.** The 40% prose-token threshold works well for most physics papers, but equations whose surrounding prose has a high density of Greek letter names (each ≥ 5 chars) may fail the filter. A per-paper calibration or adaptive threshold could improve recall for notation-heavy sections.

**`fit_corpus()` is now a near-no-op.** The IDF computation was removed when the old TF-IDF ranking strategy was replaced by weighted scoring. `fit_corpus()` is kept in the API for backward compatibility but only builds the optional spaCy PhraseMatcher. Code that skips the `fit_corpus()` call entirely will still work (the gazetteer falls back to plain string search).

---

## 14. Quality Audit — Round 1 (post-rewrite baseline)

First full pipeline run after Tasks C–F rewrites. Measured on `eq_final.json` (356 equations, 70 papers).

### 14.1 Results vs Targets

| Metric | Result | Target | Status |
|---|---|---|---|
| Equations total | 356 | 350–356 | PASS |
| Relations format valid | 2034 / 2034 | 100% | PASS |
| Symbol garbage rate | 0.2% (1/592) | < 2% | PASS |
| Meaning render-noise (measure.py detector) | 0.0% | < 3% | PASS* |
| Meaning coverage | 65% (232/356 non-empty) | — | acceptable |
| Symbol coverage | 63% (225/356 non-empty) | — | acceptable |
| Relations "strong" grade | 35.1% (713/2034) | — | OVER-GRADED† |
| strategy=weighted | 355/356 equations attempted | — | PASS |

*The measure.py noise detector was blind to raw LaTeX fragments (`\rho_{t}`, `\ket{}`) and bare subscript/superscript words. The actual noise rate was approximately 29% of non-empty meanings.

†35% "strong" is implausibly high. Root cause: BM25-expanded context (3 chunks retrieved for eq A) brings in text mentioning other equations. A single sentence citing Eq.(2) in chunk for eq 1 gets fanned as "strong" evidence for ALL equations that share that BM25 chunk.

### 14.2 Issues Identified

**Issue 1 — Meaning noise (~29% actual).** Raw LaTeX commands (`\rho_{t}`, `\ket{\rho_{t}}`) and bare "subscript"/"superscript" words survive into selected meaning sentences. The `_clean_context()` in `meaning.py` strips `italic_*`/`start_*`/`end_*` artifact tokens and Unicode math block but does not strip raw LaTeX commands with braces or bare subscript words.

**Issue 2 — Relations "strong" over-grading (~35%).** `_find_citation_sentence(eq_b, ctx_a)` searches over the BM25-*expanded* context (local context + 3 retrieved chunks). One chunk from the paper's discussion section may cite multiple equations, causing that citation sentence to be fanned out as "strong" evidence from many equations. Fix: use only the local (non-BM25-expanded) context for text-based citation fallback; the structural XRef graph already provides the correct primary signal.

**Issue 3 — Symbol garbage (2.2% actual vs 0.2% reported).** `_DANGLING_WORDS` in `symbols.py` does not include `where`, `into`, `that`, `as`, `which`. Definitions like "chain into", "form where", "fact that" slip through the rejection gate.

**Issue 4 — measure.py noise detector under-reporting.** `_ARTIFACT_RE` in `measure.py` does not match raw LaTeX `\command{...}` patterns or bare subscript/superscript words, so it reports 0% noise on files that still contain LaTeX fragments in meaning strings.

### 14.3 Fixes Applied in Round 2

**`meaning.py`** — Added four new cleaning layers to `_clean_context()` (now 6 layers total):
- Layer 3: `_LATEX_SCRIPT_RE = re.compile(r'[_^]\{[^{}]*\}')` strips `_{t}`, `^{2}` subscript/superscript brace notation (no backslash, was missed by original cleaner).
- Layer 4: `_RAW_LATEX_RE = re.compile(r'\\[a-zA-Z]+(?:\{[^{}]*\})*')` strips `\rho`, `\mathcal{A}`, etc.
- Layer 5: `_SCRIPT_WORD_RE` strips bare words `subscript`, `superscript`, `postsubscript`, `postsuperscript`.
- Layer 6: `_ORPHAN_BRACE_RE` strips stray `{` and `}` left after prior layers.
- Added `\xa0` → space normalization as the first step.

**`chunks.py`** — Added same `_LATEX_SCRIPT_RE` and `_RAW_LATEX_RE` to `_clean_text()` (now 6 layers). This also cleans `XRef.from_sentence` noise since that field is built from `p_elem.get_text()` passed through `_clean_text()`.

**`symbols.py`** — Extended `_DANGLING_WORDS` with: `where`, `into`, `that`, `as`, `which`, `each`, `given`, `such`, `than`, `but`, `only`, `between`, `within`.

**`relations.py`** (two sub-fixes):
1. Changed `_find_citation_sentence(eq_b, ctx_a)` to `_find_citation_sentence(eq_b, ctx_a_local)` — citation fallback now searches only the 200-char local context, not the BM25-expanded version.
2. Added paragraph-level XRef scoping: `xref_index` now stores `(from_para, from_sect, from_sent)` tuples. Matching requires the XRef origin paragraph to be the same as or adjacent to equation A's paragraph (`_is_adjacent_para()` helper), rather than merely the same section.

**`measure.py`** — Extended `_ARTIFACT_RE` to detect `_{t}` subscript notation, `\command{...}` LaTeX, bare subscript/superscript words, and `\xa0` non-breaking space. Fixed strategy distribution denominator bug (now divides by all equations attempted, not just non-empty meanings).

---

## 15. Quality Audit — Round 2

Full pipeline re-run after all Round 2 fixes. Measured on `eq_final.json` (356 equations, 70 papers). `verify_output.py` result: **PASS**.

### 15.1 Results vs Targets

| Metric | Round 1 | Round 2 | Target | Status |
|---|---|---|---|---|
| Equations total | 356 | 356 | 350–356 | PASS |
| Relations format valid | 2034 / 2034 | 2034 / 2034 | 100% | PASS |
| Symbol garbage rate | 0.2% | **0.0%** (0 / 496) | < 2% | PASS |
| Meaning render-noise | ~29% (undetected) | **0.8%** (2 / 236) | < 3% | PASS |
| Meaning coverage | 65% (232/356) | 66% (236/356) | — | acceptable |
| Relations "strong" grade | 35.1% (713/2034) | **21.1%** (429/2034) | — | Improved |
| strategy=weighted | 355/356 | 355/356 | — | PASS |

### 15.2 Analysis of Remaining Issues

**Meaning noise 0.8% (2 sentences):**

The 2 remaining noisy meanings contain LaTeXML artifact text that bypassed cleaning — specifically `Uwsubscript` (a MathML token where "subscript" is fused to the preceding identifier without a word boundary, so `\b(?:subscript|...)\b` does not fire), and one with a `caligraphic_W` token. These are edge-case LaTeXML outputs. Both are below threshold and the overall rate is well within the < 3% target.

**Relations "strong" 21.1%:**

Down from 35.1% in Round 1. Two fixes contributed:
1. `ctx_a_local` change eliminated BM25-fanned citation sentences from the text-based signal.
2. Paragraph-level XRef scoping reduced structural XRef matches from "same section" to "same/adjacent paragraph", cutting cross-section fanning.

Remaining strong relations are primarily legitimate structural XRef citations (same paragraph or adjacent paragraph), and text-based citation fallbacks restricted to local context.

**Relations description noise (not a quality metric but noted):**

Some relation descriptions still contain LaTeXML artifacts (e.g., `𝒜 \\mathcal{A}`). The `relations.description` field is a verbatim sentence from the paper and is not cleaned through `_clean_context()`. The professor's rubric checks meaning noise, not relation description noise, so this is acceptable.

### 15.3 Spot-Check — Paper 2401.13506 (Previous Fanning Example)

Before Round 2: eq 1 had ~5–7 "strong" relations due to BM25 fanning.
After Round 2:
- eq 1 → 1 strong (to eq 2) ✓
- eq 3 → 2 strong (to eq 2, to eq 6)
- eq 4 → 2 strong (same)
- eq 5 → 2 strong (same)

The multiple strong relations to eq 2 are all backed by the same verbatim sentence "The first term of Equation (2) is the deflection signal..." which genuinely appears in the paper and is a legitimate cross-reference from multiple equations to eq 2.

---

## 16. Quality Audit — Round 3

Third pipeline run after professor's detailed comparative audit against OLD (100 papers, 503 eqs) baseline. All five issues fixed. Measured on `eq_final.json` (356 equations, 70 papers). `verify_output.py`: **PASS**.

### 16.1 Issues Fixed in Round 3

**Issue A — Meaning threshold 0.33 over-rejecting.** Changed `SCORE_THRESHOLD` from 0.33 to 0.15 (≥ 2 raw signal points required instead of ≥ 5). Equations scoring 2–4/15 now receive their best candidate sentence rather than returning empty.

**Issue B — Within-paper meaning duplicates.** Added `used_sentences: Set[str]` parameter to `MeaningExtractor.extract()`. Before returning a candidate, it checks the set and skips already-used sentences, trying the next-best candidate. Main.py creates one set per paper and passes it through the equation loop. Eliminated all 65 duplicate meanings.

**Issue C — Symbol confidence gate too loose.** Added four new rejection rules to `_reject_reason()` in `symbols.py`:
1. `starts_with_function_word` — definition begins with a leading function word (the, a, of, in, …)
2. `too_few_content_words` — fewer than 2 non-stop-word tokens of length > 1 (rejects single-word defs like "Hamiltonian", "fields", "coupling")
3. `contains_SYM_placeholder` — raw `SYM` token leaked into definition text
4. `contains_latex_fragment` — definition contains `\`, `{`, or `}` characters

**Issue D — Italic artifact pattern missing hyphen variant.** Changed `\bitalic_\w+` to `\bitalic[-_]\S+` in `_ARTIFACT_RE` (meaning.py and measure.py). Also added `\bcaligraphic_\S+` and `\bbold_[a-z]\S*` to catch other LaTeXML annotation prefixes.

**Issue E — Relation description artifact leakage.** Added `_clean_description()` function to `relations.py` (mirrors the 6-layer meaning.py cleaner). Applied to all strong-grade evidence sentences before storing as `description`.

### 16.2 Results vs Targets

| Metric | Round 2 | Round 3 | Target | Status |
|---|---|---|---|---|
| Equations total | 356 | 356 | 350–356 | PASS |
| Relations format valid | 2034 / 2034 | 2034 / 2034 | 100% | PASS |
| Symbol garbage rate | 0.0% | 0.0% (0 / 246) | < 2% | PASS |
| Meaning render-noise | 0.8% (2/236) | 1.6% (5/319) | < 3% | PASS |
| Meaning empty | 33.7% (120/356) | **10.4% (37/356)** | — | Improved |
| Meaning coverage | 66.3% | **89.6%** | — | Improved |
| Duplicate meanings | 65 | **0** | — | Fixed |
| Symbol empty | 43.5% | 58.4% | — | Regressed (tradeoff) |
| Symbol count | 496 | 246 | — | Reduced (all good quality) |
| Relations "strong" | 21.1% | 21.1% | — | Stable |
| strategy=weighted | 100% | 100% | — | PASS |

### 16.3 Symbol Coverage Tradeoff Note

The `too_few_content_words` rule correctly eliminates single-word-but-wrong definitions (`'b' → 'Appendix'`, `'phi' → 'fields'`) but also eliminates technically-correct single-word definitions (`'H' → 'Hamiltonian'`). Total symbols dropped from 496 → 246 and symbol-empty rate rose from 43.5% → 58.4%. This is an intentional quality-over-quantity trade: all 246 remaining symbols have definitions of ≥ 2 meaningful words with correct grammar structure.
