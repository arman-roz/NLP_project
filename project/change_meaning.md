# Meaning Module — Design and Implementation Notes

This document covers the design decisions, analysis, and implementation of `src/meaning.py` — the module responsible for assigning a human-readable description to each extracted equation.

---

## What "meaning" means in this project

The `meaning` field for each equation is a single sentence taken verbatim from the paper that best describes what the equation represents.  It is never generated — it is always a sentence the paper's authors wrote.

---

## Why extraction, not generation

The project rules forbid text generation at runtime.  This means the `meaning` field may not be written by a language model.  Every string stored in `meaning` must be traceable to a specific sentence in the cached paper content.  The audit trail records exactly which sentence was chosen and why, so a grader can verify the source.

If no suitable sentence is found in the context window, the field is left as an empty string rather than filled with a model-generated guess.

---

## Why IDF must be computed over the whole corpus

TF-IDF ranks sentences by how much specific, informative content they carry.  The IDF component measures how rare each word is across the document collection — rare words score higher because they are more likely to be domain-specific descriptions rather than boilerplate.

**If IDF is computed per-paper or over the local context window only**, the problem collapses:

- A context window of 2–3 sentences contains each word roughly once.
- Almost all document-frequency counts are 1.
- All IDF values become the same constant.
- TF-IDF reduces to plain word count — length bias dominates and the ranking is meaningless.

**Corpus-wide IDF** tells the model that "Hamiltonian" appears in most quantum physics papers (low IDF — down-weighted, boilerplate for this corpus) while "superradiant" appears in only a few papers (high IDF — up-weighted, specific to this equation's context).  A sentence containing highly-corpus-specific words is more likely to be the one specifically describing *this* equation.

**Implementation:** `fit_corpus(texts)` is called once before any `extract()` call, with every `context_before` and `context_after` string from all equations in the full paper set.  After that, IDF values are fixed for the entire run.

---

## Three-strategy extraction pipeline

Strategies are tried in order.  The first one that returns a non-empty result wins.

### Strategy 1 — Equation citation match

Looks for a sentence in the context window that explicitly references this equation's number, e.g.:

```
"where Eq. (3) gives the total energy of the system"
"as shown in (3), the Hamiltonian reduces to..."
```

Pattern used:
```
(?:eq(?:uation)?s?\.?\s*)?\(\s*{N}\s*\)
```

This is the highest-confidence strategy because the authors themselves wrote a sentence that ties prose directly to the equation label.

**Logged as:** `strategy=citation_match`

### Strategy 2 — Named-equation gazetteer

Uses a hand-curated list of known equation names from quantum physics:

```
Schrödinger equation, Hamiltonian, Lindblad equation,
master equation, von Neumann equation, Fokker-Planck equation,
Bloch equation, Maxwell equation, Dirac equation,
Heisenberg equation, Liouville equation, density matrix,
partition function, Green's function, Jaynes-Cummings,
Rabi model, tight-binding, Bogoliubov, Wigner function, …
```

If any of these phrases appear in the context window, the sentence containing the match is returned.

When spaCy is installed (`en_core_web_sm`), a `PhraseMatcher` is used for efficient tokenisation-aware matching.  When spaCy is absent, the code falls back to case-insensitive substring search — same results, slightly less robust to punctuation variants.

**Logged as:** `strategy=gazetteer_match`

### Strategy 3 — TF-IDF sentence ranking

Scores every sentence in the context window by the mean IDF of its non-stop-word tokens.  The sentence whose words are on average the most corpus-specific is returned.

```
score(sentence) = mean( idf[word] for word in tokenise(sentence) )
```

Stop words (the, is, where, equation, fig, …) are excluded from scoring so they do not inflate the score of long boilerplate sentences.

**Logged as:** `strategy=tfidf_rank, score=X.XXX`

---

## Audit trail entries

Every call to `extract()` logs exactly one entry to the equation's `AuditTrail`:

| Strategy used | Audit entry format |
|---|---|
| Citation match | `eq (N): strategy=citation_match, sentence='...'` |
| Gazetteer | `eq (N): strategy=gazetteer_match, sentence='...'` |
| TF-IDF | `eq (N): strategy=tfidf_rank, score=X.XXX, sentence='...'` |
| Nothing found | `eq (N): no meaning sentence found` |
| Not fitted | `MeaningExtractor not fitted — call fit_corpus first` |

---

## Design decision — single sentence output

The spec says `meaning` is a string, not a list.  We return exactly one sentence — the best-ranked one.  Returning multiple sentences would exceed the field's intended scope and make automated evaluation harder.

---

## Limitations

- **Context window depth:** only `context_before` and `context_after` are searched (the two closest paragraph siblings to the equation in the HTML).  A description that appears three paragraphs away will not be found.
- **Sentence splitter:** uses a simple regex (`(?<=[.!?])\s+(?=[A-Z])`).  It can mis-split on abbreviations like "Fig. 2" or "Eq. (3)".  A full sentence tokeniser (e.g. NLTK Punkt) would be more robust but adds a dependency.
- **PDF papers:** context is extracted from raw PDF text lines, which may lack paragraph structure.  The TF-IDF fallback is most likely to be used for PDF-sourced equations.
