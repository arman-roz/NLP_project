# Final Approach — Equations Knowledge Graph (Exam ID 44)

A complete, end-to-end explanation of the **techniques** and the **reasoning** behind the
submitted pipeline (`modified/`). This document deliberately does **not** walk through the
source code; it explains *what NLP ideas were used, why they were chosen, and how the system
thinks* at every stage, so it can be used as source material for the written report. NLP
terms you may want to cite are **in bold**.

---

## 0. The task in one sentence

For every **enumerated equation** in arbitrary quantum-physics arXiv papers, extract (1) the
equation as LaTeX, (2) a short **meaning**, (3) a **symbol dictionary**, (4) graded
**relations** to every other equation in the paper, and (5) an **audit trail**, and write it
all to one JSON file that is *graph-ready* (a **knowledge graph** of equations).

---

## 1. The two hard constraints that shaped every decision

Everything in the design follows from two rules in the task:

1. **No text generation / no prompting.** No LLM (ChatGPT/Claude) and no self-hosted language
   model may be *prompted to produce text*. Therefore every output string must be **extractive**
   — *lifted verbatim* from the paper — never **abstractive** (generated/paraphrased). This is
   the single most important constraint: it rules out summarisation models and turns the whole
   problem into **Information Extraction (IE)** rather than generation.
2. **Sources from arXiv only** (`/abs`, `/html`, `/pdf`; `/src` and `/e-print` forbidden), with
   **`robots.txt` compliance** and a polite **crawl delay**.

A self-imposed third principle (the "way of thinking") was added because it is what makes the
method *generalise* and *defensible*:

3. **No hand-written domain vocabulary.** No physics gazetteer, no Greek-letter table, no list
   of "definition verbs", no list of equation names. Decisions are made from **document
   structure** (HTML/MathML), **grammar** (the **dependency parse** and **part-of-speech
   tags**), and **Unicode properties** — features that *any* quantum-physics paper shares.
   A **gazetteer** (hand-built word list) would over-fit to the terms seen in these particular
   papers and fail on unseen ones; grammar does not.

> **Design philosophy in one line:** prefer *structure + grammar* over *vocabulary*, *extraction*
> over *generation*, *precision* over *recall* (omit rather than guess), and *transparency*
> (every value is traceable in the audit trail).

---

## 2. The pipeline at a glance

```
arXiv HTML  ─►  (1) Acquisition: polite fetch + local cache
            ─►  (2) Structural parsing: equations + LaTeX + MathML symbols
            ─►  (3) Text normalisation + sentence segmentation (context windows)
            ─►  (4) BM25 lexical retrieval index (over the paper's sentences)
            ─►  (5) MEANING   : grammatical-role noun-phrase extraction
            ─►  (6) SYMBOLS   : definitional-dependency extraction
            ─►  (7) RELATIONS : cross-reference + lexical-overlap graph
            ─►  (8) Audit trail (provenance for every value)
            ─►  (9) Schema validation ─► JSON knowledge graph
```

Each stage is a classic NLP sub-task: **web scraping**, **document parsing**, **sentence
segmentation**, **information retrieval**, **definition/term extraction**, **relation
extraction**, and **provenance/traceability**.

---

## 3. Stage 1 — Data acquisition (web scraping, done politely)

- **Source choice.** The **arXiv HTML** rendering (produced by **LaTeXML**) is used as the
  primary source because it exposes equations as **MathML** *and* keeps the original LaTeX in a
  machine-readable annotation. HTML is far richer than PDF text (which loses equation structure)
  and is allowed by `robots.txt`.
- **Politeness / ethics of scraping.** Requests honour `robots.txt`, send a descriptive
  **User-Agent**, and wait a fixed **crawl delay (15 s)** between live requests to avoid
  overloading the server — standard responsible-crawling practice.
- **Caching (idempotency).** Every downloaded page is written to a local cache. Re-runs read
  from disk with **zero network calls**, which makes the pipeline **deterministic**,
  reproducible, and fast to iterate on. Caching is also what lets the 15 s delay be paid only
  once.
- **Order and stopping rule.** Papers are processed in the *exact assigned order*; the run
  stops once the dataset reaches the target (~350) equations, but the paper that crosses the
  target is finished completely, as the spec requires.

---

## 4. Stage 2 — Structural equation extraction (parse the document, don't read it)

The key insight: an enumerated equation is a **structural object** in the HTML, not something
you need language to find.

- **DOM parsing.** The page is parsed into a **DOM tree** with an HTML parser (BeautifulSoup +
  lxml). Every numbered equation lives in a table element with the class `ltx_eqn_table` and
  carries an equation-number tag. Finding equations is therefore **tree pattern matching**, not
  reading — this is what makes it general across papers.
- **Numbered vs. unnumbered.** Only equations that carry a number tag are kept (the spec wants
  *enumerated* equations); display equations without a number are ignored *by structure*. The
  first seven per paper are kept.
- **LaTeX retrieval priority.** The equation string is taken in a fixed quality order:
  (1) the `<annotation encoding="application/x-tex">` node — the **verbatim original LaTeX**;
  (2) the `alttext` attribute; (3) the raw MathML text. This guarantees the cleanest possible
  `equation` field without ever touching the forbidden LaTeX source endpoint.
- **Symbols come from MathML, structurally.** Inside each equation, **MathML identifier nodes**
  (`<mi>`) are the variables; operator (`<mo>`) and number (`<mn>`) nodes are ignored *by
  element type*. So the symbol set is read off the **markup structure**, not guessed from text.
  - **Subscript handling.** A subscript that *names* a quantity (e.g. `P_th`, `η_D`, `A_eff`) is
    kept so that physically distinct symbols sharing a base letter stay separate; a meaningless
    index subscript (a bare digit, `x_1`) is dropped.
  - **Unicode normalisation.** Greek code points are mapped to LaTeX-style names (`𝜙 → phi`)
    using **`unicodedata`** and **NFKC normalisation**, so symbols are reported as the spec asks
    (`phi`, not the glyph) without any hand-written Greek table.
  - Standard operators (`d`, `∂`, `∇`, `δ`) are excluded, as the spec permits.
- **Local context windows.** For each equation, the surrounding prose is captured as a
  **before** window and an **after** window (the equation's paragraph plus nearest sibling
  paragraphs). These windows are the local evidence for meaning and symbol definitions.

---

## 5. Stage 3 — Text normalisation and sentence segmentation

Raw LaTeXML text is noisy (it interleaves Unicode, LaTeX, and italic-annotation copies of every
symbol). Cleaning this *before* any linguistic analysis is the highest-leverage fix.

- **Inline-math rendering.** Inline LaTeX in the prose is converted to readable Unicode with the
  **`pylatexenc`** library (so `I_{\rm OFF}` reads as `I_OFF`, not `I rm OFF`). A library is used
  rather than ad-hoc regex so the full LaTeX macro set is handled correctly. The `equation` field
  itself keeps the verbatim LaTeX — only the *prose* is rendered.
- **Markup-error removal.** LaTeXML emits `<span class="ltx_ERROR">` nodes for macros it cannot
  expand (e.g. an undefined `\textcolor`). These are dropped *by their structural error class*,
  so markup command names never leak into the extracted text — again, no word list needed.
- **Sentence segmentation (sentence boundary detection).** Text is split into sentences with a
  **rule-based sentencizer**. The punctuation set is extended to break on `:` and `;` as well as
  `.?!`, because an *introducing clause* such as "… is given by:" must be isolated from what
  follows — this directly improves meaning extraction.
- **Tokenisation, POS tagging, dependency parsing, lemmatisation** are all provided by the
  small **spaCy** statistical model (`en_core_web_sm`). spaCy is the workhorse: every later
  decision reads its **part-of-speech tags** and **dependency labels**.
- **Stop words** (function words like *the, of, is*) are taken from spaCy's default list and are
  used only to trim phrase edges — never to judge meaning.

---

## 6. Stage 4 — BM25 lexical retrieval (find evidence anywhere in the paper)

A definition is sometimes written far from its equation (e.g. in a "Notation" paragraph). To
recover it, the pipeline builds a small **information-retrieval (IR)** index over the paper's own
sentences.

- **BM25 (Okapi BM25).** Each sentence is a "document"; **BM25** ranks sentences against a query
  by **term frequency** weighted by **inverse document frequency** with length normalisation. It
  is the classic strong **lexical** ranking function — deterministic, fast, no model, no GPU.
- **Why BM25 is a good fit.** The evidence almost always shares an *exact term* with the query —
  the symbol name (`eta`), the equation number, or a concept noun — which is exactly what
  **lexical matching** rewards. This is **sparse retrieval** (term overlap), not **dense
  retrieval** (embeddings).
- **Math-aware tokenisation.** The BM25 tokeniser emits word tokens *and* **character bigrams**
  of short notation-like tokens, so subscripted symbols (`H_0`) still match prose mentions.
- **Role in the pipeline.** Retrieval only *selects candidate sentences*; the answer strings are
  still produced by the grammar/structure stages. The IR layer **never generates text** — it
  ranks evidence.

---

## 7. Stage 5 — MEANING extraction (the core: how each formula gets its name)

This is the question "what does this equation express?", answered **extractively**. The output
is a short **descriptive noun phrase** lifted verbatim — e.g. *"intensity profile in the dark
output of the Sagnac interferometer"*, *"threshold power for FWM parametric oscillation"*,
*"conditional mutual information"* — not a whole sentence and never generated.

The central idea: **the name of an equation is whatever the introducing sentence makes the
grammatical subject or object of its main clause.** So the problem is solved with **syntax**
(the **dependency parse**), not with a list of physics terms. Step by step:

**(a) Gather candidate sentences (proximity + retrieval).**
- The last one or two sentences of the *before* window (the introducing clause, usually ending
  in ":") and the first one or two of the *after* window.
- Plus any sentence that **explicitly cites this equation's number** (e.g. "Equation (2) is …"),
  found with the BM25 index. These are visited in **proximity order**, so the nearest
  introducing clause is preferred.

**(b) Tier 1 — explicit naming (a lexico-syntactic pattern).**
- If a sentence says *"… called / known as / termed / referred to as X …"*, the phrase X is
  lifted directly. This is a **definitional pattern** (a small, closed, language-general
  construction for naming — not domain vocabulary) and is the strongest possible signal.

**(c) Tier 2 — naming by grammatical role (the main mechanism).**
This is where the "way of thinking" matters. From the **dependency parse** of the introducing
sentence we identify the **predicate** (the **ROOT** verb and any verbs **conjoined** to it) and
then choose the naming noun phrase from **syntactic roles**, deciding from the parse — never from
a verb list:
- **Passive voice** ("X *is given / is defined / can be modeled* by/as …") or a **copular**
  clause ("X *is* the …") → the grammatical **subject** (`nsubj`/`nsubjpass`) is the named
  quantity. *Example:* "**The detected squeezing level** in such a cavity OPO can be modeled by …"
  → subject = *detected squeezing level*.
- **Active voice** ("we *define* X", "QTP *yields* X") → the **direct object** (`dobj`) /
  predicate complement is the named quantity, because the subject is just the agent. *Example:*
  "We define **the vectorization** of ρ …" → object = *vectorization*.
- **Expletive / pronoun subject** ("**it** is shown that …", "**one** obtains …") → the subject
  is meaningless, so we descend into the **complement clause** (`ccomp`) and take *its* subject.
  *Example:* "it is shown that **the intensity profile** … is given by:" → *intensity profile*.

This single rule is why the system stopped picking leading adverbials ("In the absence of …",
"For a bipartition AB, …"): those nouns sit inside an **adverbial/prepositional phrase**, not in
the main-clause subject/object slot, so the role-based selector skips them.

**(d) Grow the head noun into a full descriptive phrase (NP chunking via the parse).**
Once the **head noun** is chosen, it is expanded into its complete **noun phrase** by walking its
**dependency subtree**: keep determiners, **adjectival** (`amod`) and **compound** modifiers, and
**prepositional complements** (*of / for / in / on / between …*), but **stop at clause boundaries
and coordination**. This turns a bare head ("profile") into a full, *understandable* name
("intensity profile in the dark output of the Sagnac interferometer"). Crucially, the words come
out **in the author's original order** — the system only decides *where the phrase starts and
ends*, it never reorders or invents words (that is what keeps it **extractive**, not generative).

**(e) Validate the phrase by POS (reject clauses, keep noun phrases).**
A valid name must **contain a noun** and must **not be a clause in disguise** — i.e. it must not
contain a **finite verb** acting as a clause head. A **participle** used as a modifier
("*detected* squeezing level", a `VBN`/`VBG` tag with an `amod` relation) is allowed. The check
uses the **in-context POS tags** from the original parse, because re-parsing a phrase in
isolation can mis-tag a participle as a finite verb.

**(f) Acronym expansion (initialism resolution).**
If the chosen name is an all-caps **initialism** (e.g. *CMI*), the paper is searched for the
pattern "Long Form (ACRONYM)" and the name is replaced by the spelled-out long form by matching
**initials** — so *CMI* becomes *conditional mutual information*. General, no dictionary.

**(g) Soft de-duplication.**
Within a paper, a name an earlier equation already took is avoided when an alternative exists, so
the meanings of related equations stay *varied* rather than all collapsing to the same phrase.

> **In short, the meaning of each formula is the main-clause subject/object noun phrase of its
> introducing sentence, grown to include its modifiers and prepositional complements, validated as
> a noun phrase, and lifted verbatim.** This is **syntactic information extraction** — POS tagging
> + dependency parsing + NP chunking — not generation and not keyword matching.

---

## 8. Stage 6 — SYMBOL definition extraction (definition mining by dependency)

For each MathML symbol, find the noun phrase that *defines* it, or omit it.

- **Candidates from structure.** Symbols come from the equation's MathML identifiers (Stage 2),
  not from scanning text.
- **Surface-form matching.** A symbol named `eta` must also match the glyph `η` in the prose.
  These **surface variants** are generated with **`unicodedata`** (name ↔ Unicode Greek), so no
  hand-written symbol table is needed. Mentions are found with **word-boundary regular
  expressions**.
- **Clause splitting.** A sentence is split into **clauses** (at commas, semicolons, and cues
  like *where / with / which*) so a definition cannot leak across into a neighbouring symbol's
  clause — important for lists like "where η_D is the efficiency, η_path is the propagation
  efficiency, …".
- **Definitional dependency relations (the key idea).** Within the clause that mentions the
  symbol, the definition is chosen from the symbol token's **syntactic relation**, not from the
  nearest words:
  - the symbol is the **subject** of a copular verb → take the **predicate complement**
    ("η *is* **the efficiency of the detector**");
  - the symbol **modifies / is in apposition to** a noun → take that noun
    ("**the efficiency** η", "**the Hamiltonian** H").
  This is exactly **definition extraction** framed as **relation extraction** between a symbol
  and a noun phrase. A position-based "nearest noun chunk" fallback is used only if the
  dependency relation yields nothing.
- **Local first, retrieval second — and only when safe.** The local window is searched first; if
  nothing is found, BM25 retrieves whole-paper sentences mentioning the symbol. **But the
  whole-paper fallback is suppressed for bare single letters** (`H`, `i`, `e`): a single letter
  occurs all over a long paper, so retrieving a far sentence by it grabs an unrelated noun (the
  `H` in a chemical formula → "piranha"). Only **distinctive** symbols (multi-letter, or with a
  meaningful subscript) use whole-paper retrieval. This is a **precision-over-recall** decision
  based purely on the symbol's *specificity*, not on any vocabulary.
- **Omit over guess.** A definition is written **only** if the parse yields a supporting noun
  phrase from the paper; otherwise the symbol is left out. This honours the spec's preference for
  omission over a confident wrong answer (a **precision-oriented** policy).
- **POS validation.** As with meaning, a definition must be a **noun phrase** (contains a noun,
  no finite verb/auxiliary, not adverb-initial).

---

## 9. Stage 7 — RELATION extraction (building the graph)

Every *ordered* pair of equations in a paper is graded **none / potential / strong**, with a
score in **[0, 1]**. Two model-free **lexical** signals are used:

- **`strong` — explicit cross-reference (the strongest, unambiguous signal).** If one equation's
  context **explicitly cites the other equation's number** ("From Eq. (4) … (8)"), the author has
  stated the link. Score = 1.0. The **description** is the **connecting verb** *lifted verbatim*
  from the citing sentence (the verb nearest the citation, plus any attached particle/preposition
  — e.g. "given by", "reduces to"), found through the **dependency parse**; the fallback is
  "directly referenced". *No hand-written verb map.*
- **`potential` — lexical overlap (topical relatedness).** Otherwise the two equations are scored
  by how much their contexts share, using the **Jaccard similarity coefficient**
  (|intersection| / |union|) computed on two sets:
  - the **content nouns** of each context (a **bag-of-words** of lemmatised, stop-word-filtered
    nouns), and
  - the **symbol sets** of the two equations.
  The final score is an even blend of the two Jaccards, which keeps it bounded in **[0, 1]**.
  Above a threshold the pair is `potential`; its description lists the shared concepts/symbols.
- **`none`** otherwise.
- **Edge capping.** To avoid an over-connected graph, only the top few `potential` partners per
  equation are kept; the rest are downgraded to `none`. Every pair still appears with a grade, so
  the output is **graph-ready** (an explicit adjacency structure).

> **Why Jaccard and not cosine/embeddings?** **Cosine similarity** is just a formula for comparing
> two vectors; the "understanding" comes from **word embeddings** (e.g. word2vec, SBERT), which
> were deliberately excluded. On *word-count* vectors, cosine reduces to the same **lexical
> overlap** that Jaccard already measures. Staying lexical keeps the method **transparent and fully
> auditable** (no opaque vector space, every score traceable), at the cost of being blind to
> **synonymy/paraphrase** — an honest, documented trade-off.

---

## 10. Stage 8 — Audit trail (provenance and traceability)

- Every equation carries an **audit trail**: a dictionary whose keys are the *methods* that ran
  and whose values are short notes about *what each found and why* (which sentence gave the
  meaning, the evidence for each symbol definition, the grade and score of each relation, cache
  hits, etc.).
- This is **provenance / traceability**: it lets a grader verify that every string was
  **extracted** from arXiv (not generated), which is exactly how the "no generation" rule is
  *demonstrated*. Duplicate method keys are auto-numbered so the JSON stays valid.

---

## 11. Stage 9 — Validation and output

- Before writing, a **schema validator** enforces the exact required shape (top-level arXiv-ID
  keys, equation-number sub-keys, the five required fields, valid relation grades, a complete set
  of relation keys). This is a **data-contract / schema-validation** step that guarantees a
  well-formed deliverable.
- **Robustness.** Each paper is processed inside a safety net: one malformed paper cannot abort a
  multi-hour batch — it is logged and emitted as an empty dictionary, which the spec allows. This
  is **graceful degradation**.
- The result is a single **JSON** file: a **knowledge graph** with the arXiv ID as the main key,
  equation numbers as sub-keys, and relations forming the edges.

---

## 12. NLP techniques used (glossary for the report)

| Area | Techniques / terms used |
|---|---|
| Acquisition | web scraping, `robots.txt` compliance, crawl delay, caching, idempotency |
| Document parsing | DOM traversal, HTML/MathML parsing, tree pattern matching |
| Markup | LaTeX→Unicode rendering, Unicode **NFKC** normalisation, `unicodedata` |
| Pre-processing | **tokenisation**, **sentence boundary detection**, **stop-word** filtering, **lemmatisation** |
| Syntax | **POS tagging**, **dependency parsing**, **noun-phrase (NP) chunking**, head words, **voice** (active/passive), **copula**, **apposition**, **coordination**, **finite verb**, **participle** |
| Information Extraction | **definition/term extraction**, **relation extraction**, **acronym/initialism resolution**, lexico-syntactic patterns, **regular expressions** |
| Information Retrieval | **BM25 / Okapi BM25**, **TF-IDF** intuition, **inverted index**, sparse vs **dense retrieval**, character **n-grams** |
| Similarity | **Jaccard coefficient**, **bag-of-words**, set overlap; (contrast: **cosine similarity**, **word embeddings**) |
| Paradigm | **rule-based + statistical** NLP, **extractive vs abstractive**, **precision/recall** trade-off |
| Engineering | provenance/audit trail, schema validation, deterministic/reproducible runs, graceful degradation |

---

## 13. Why the approach generalises

- It relies on features **every English-language quantum-physics paper shares**: numbered
  equations live in the same HTML structure; symbols live in MathML; introducing sentences use
  the same **grammatical roles** (subject/object/copula); definitions use the same **syntactic
  relations**; cross-references cite equation numbers. None of this depends on *which* physics
  terms appear.
- Because there is **no gazetteer**, the method does not over-fit to the vocabulary of the papers
  it was tuned on — a paper full of unfamiliar terminology is handled the same way. This is the
  central **generalisation** argument and is supported by the audit trail (you can show the same
  methods firing across very different papers).

---

## 14. Honest limitations (for the critical-discussion section)

- **Extractive ceiling.** If a paper never states an equation's name or a symbol's definition in
  prose, no extraction method can recover it — output is empty by design (precision over recall).
- **Local evidence + lexical retrieval.** Evidence far from the equation and *paraphrased* (no
  shared terms) can be missed, because retrieval and relation scoring are **lexical**, not
  semantic — the deliberate consequence of dropping embeddings.
- **Parser errors.** On heavily math-dense or ungrammatical sentences the **dependency parse**
  can be wrong, which then misleads the role-based selectors; meanings/definitions degrade
  gracefully to weaker but still on-topic phrases.
- **Markup artefacts in old papers.** A few pre-2020 papers render LaTeX imperfectly in HTML
  (e.g. a colour macro fuses into the next word); structural error-node removal handles most but
  not all such cases.
- **Single-letter symbols** remain the hardest case (low specificity), which is why whole-paper
  retrieval is intentionally restricted for them.
- **Realistic accuracy.** This is an unsupervised, training-free, extraction-only system; the
  honest expectation is roughly **~50%+ field-level accuracy** that *scales* to tens of thousands
  of papers without per-paper tuning — the goal was a sound, general, auditable method, not a
  hand-polished result on a few papers.

---

## 15. One-paragraph summary (drop-in for the report intro)

> The system treats the task as **extractive information extraction** under a strict
> no-generation rule. Equations and their symbols are read **structurally** from arXiv's
> **HTML/MathML**; the surrounding prose is cleaned, **sentence-segmented**, and **dependency-
> parsed** with spaCy. An equation's **meaning** is the noun phrase that fills the **subject** or
> **object** role of its introducing clause (chosen by **voice** and **dependency labels**, grown
> through **NP chunking**, and lifted verbatim). Each **symbol** is defined by the noun it stands
> in a **definitional dependency** with. **Relations** are graded from **explicit equation-number
> cross-references** (strong) and **Jaccard lexical overlap** of shared symbols and context nouns
> (potential), producing a **[0, 1]**-scored, graph-ready output. A per-equation **audit trail**
> proves every value was extracted, not generated. The method uses **no physics gazetteer and no
> embeddings**, so it is transparent, fully auditable, and designed to **generalise** to arbitrary
> quantum-physics papers.
