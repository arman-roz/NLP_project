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

## Context cleaning — root fix for all strategies

### The problem: two layers of noise in arXiv HTML context text

arXiv's HTML renderer (LaTeXML) concatenates three representations of every math element inline: Unicode character + LaTeX source + italic annotation text.  So the variable δy appears in context as:

```
δ ⁢ y 𝛿 𝑦 \delta y italic_δ italic_y
```

When fed directly to the sentence splitter, tokenizer, or gazetteer, this produces:
- Sentence fragments that start or end mid-equation rendering
- Noise tokens (`italic`, `postsubscript`, `start`, `end`) that inflate TF-IDF scores because they are rare as English words but common as LaTeXML artifacts
- Gazetteer matches on sentences that contain the artifact text of a Hamiltonian equation rather than a prose description

This noise affects all three strategies simultaneously.  Cleaning the context before any strategy runs is the highest-leverage fix.

### Two-layer cleaning strategy

**Layer 1 — artifact token removal**

The pattern `italic_\w+`, `start_[A-Z]\w*`, `end_[A-Z]\w*`, `over_ARG` identifies all LaTeXML HTML rendering metadata tokens.  These are stripped by regex substitution:

```python
_ARTIFACT_RE = re.compile(
    r'\bitalic_\w+'        # italic_H, italic_δ
    r'|\bstart_[A-Z]\w*'   # start_POSTSUBSCRIPT, start_ARG
    r'|\bend_[A-Z]\w*'     # end_POSTSUBSCRIPT, end_ARG
    r'|\bover[a-z]*_ARG\b' # over_ARG, overline_ARG
)
```

**Layer 2 — Unicode math alphanumeric removal**

Characters in Unicode block U+1D400–U+1D7FF (𝑎 𝑏 𝑐 𝐴 𝐵 𝛼 𝛿) are LaTeXML's text rendering of math italic variables.  They are the Unicode versions of the same symbols already present as LaTeX commands (\delta, \alpha).  Stripping them removes the duplication without losing content:

```python
_MATH_UNICODE_RE = re.compile(r'[\U0001D400-\U0001D7FF⁡-⁤]')
```

The range U+2061–U+2064 covers invisible operators (invisible times ⁢, invisible separator ⁣, function application ⁡) that LaTeXML inserts between adjacent math tokens and that produce no visible output.

**What is deliberately preserved**

LaTeX commands (`\delta`, `\frac`, `\omega`) are preserved by both patterns because they are `\` + ASCII — neither matches as an artifact token nor falls in the Unicode math block.  A sentence like "where `\delta y` is the displacement" remains intact after cleaning and is a valid meaning sentence.

Standard Greek Unicode letters (α β γ in the range U+0391–U+03C9) are also preserved.  These are real content characters that appear in prose.

---

## Four-strategy extraction pipeline

Strategies are tried in order.  The first one that returns a non-empty result wins.  All strategies operate on the cleaned context text.

### Strategy 1 — Equation citation match

Looks for a sentence that explicitly cites this equation's number, e.g.:

```
"The first term of Equation (2) is the deflection signal..."
"as shown in Eq. (3), the Hamiltonian reduces to..."
```

Pattern:
```
(?:eq(?:uation)?s?\.?\s*)?\(\s*{N}\s*\)
```

**Fragment filter (Fix 4):** A citation match is only accepted if the sentence is longer than 30 characters AND does not start with `(`.  This rejects fragments produced when the sentence splitter cuts at the equation number itself:

```
BAD:  "( 5 ), a quantum transformation can."    ← starts with (, too short
BAD:  "( 5 ) using the DMRG algorithm."         ← fragment tail
GOOD: "The first term of Equation (2) is..."    ← accepted
```

**Logged as:** `strategy=citation_match`

### Strategy 2 — Named-equation gazetteer

Uses a hand-curated list of 20 known equation names from quantum physics:

```
Schrödinger equation, Hamiltonian, Lindblad equation,
master equation, von Neumann equation, Fokker-Planck equation,
Bloch equation, Maxwell equation, Dirac equation,
Heisenberg equation, Liouville equation, density matrix,
partition function, Green's function, Jaynes-Cummings,
Rabi model, tight-binding, Bogoliubov, Wigner function, …
```

**Ordering fix (Fix 3):** `context_after` sentences are checked before `context_before` sentences.  In physics papers, equations are typically introduced with an overview sentence before the equation ("The Hamiltonian H describes the system:") and followed by the definitional sentence ("where H_0 is the free part and V is the interaction").  The sentence that directly follows the equation is almost always more specific to it than the introductory sentence that precedes it.

This ordering does not break in the edge case of equation arrays (align/eqnarray) where `context_after` is empty or contains another equation — the code naturally falls through to `context_before` when `sents_after` is empty.

When spaCy is installed, a `PhraseMatcher` is used for efficient case-insensitive matching.  Without spaCy, plain substring search is used.

**Logged as:** `strategy=gazetteer_match`

### Strategy 3 — Definitional sentence pattern

Catches sentences that explicitly define a symbol or expression using standard mathematical writing conventions:

| Pattern | Example |
|---|---|
| `where \S+ is/denotes/represents` | "where H is the total Hamiltonian" |
| `denotes` (standalone) | "where \omega denotes the frequency" |
| `defined as` | "is defined as the reduced density matrix" |
| `is defined by` | "which is defined by the partition function" |

This strategy runs after the gazetteer and catches definitional sentences in papers that do not contain a named-equation phrase.  It requires no symbol selection — the pattern matches any "where X is..." form regardless of which variable X is.

**Logged as:** `strategy=definitional_match`

### Strategy 4 — TF-IDF sentence ranking

Scores every sentence by the mean IDF of its non-stop-word tokens.  The sentence whose words are on average the most corpus-specific is returned.

```
score(sentence) = mean( idf[word] for word in tokenise(sentence) )
```

Because strategy 4 now operates on **cleaned** sentences, artifact tokens (`italic`, `postsubscript`, `start`, `end`, `operatorname`) cannot inflate scores.  These tokens were genuinely high-IDF in the dirty text (they are rare as English words) but meaningless as content indicators.  After cleaning, the highest-IDF tokens are real low-frequency physics words.

Stop words are extended to include common LaTeX artifact words (`italic`, `start`, `end`, `postsubscript`, `postsuperscript`, `arg`, `displaystyle`, `rm`, `bf`, `operatorname`) as a secondary safety net in case any residual artifacts survive the cleaning step.

**Logged as:** `strategy=tfidf_rank, score=X.XXX`

---

## Audit trail entries

| Strategy used | Audit entry format |
|---|---|
| Citation match | `eq (N): strategy=citation_match, sentence='...'` |
| Gazetteer | `eq (N): strategy=gazetteer_match, sentence='...'` |
| Definitional | `eq (N): strategy=definitional_match, sentence='...'` |
| TF-IDF | `eq (N): strategy=tfidf_rank, score=X.XXX, sentence='...'` |
| Nothing found | `eq (N): no meaning sentence found` |

---

## Design decisions

### Why four strategies instead of one unified approach?

Each strategy targets a different signal type.  Citation match exploits explicit authorial labelling — the strongest possible signal.  Gazetteer exploits domain knowledge about named equations.  Definitional pattern exploits standard mathematical writing conventions.  TF-IDF exploits corpus-level word frequency statistics.  In sequence, these cover progressively weaker but broader evidence.  A single approach would need to do all four things at once, which produces either over-fitting to named equations or under-sensitivity to citation labels.

### Why strip artifacts before IDF fitting, not just before extraction?

The IDF vocabulary is built from the same context texts used during extraction.  If artifact tokens are present during fitting, they receive high IDF values (they are rare as real English words).  During extraction they then inflate the scores of noisy sentences.  Cleaning before fitting keeps the vocabulary clean so IDF values reflect genuine word rarity.

### Why preserve LaTeX commands in cleaned text?

A sentence like "where `\delta y` is the displacement" is a valid, high-quality meaning sentence.  If LaTeX commands were stripped, this sentence would lose its symbol content and might fall below the minimum-length threshold or score lower in TF-IDF.  More importantly, the definitional pattern `where \S+ is` matches `\delta` as the symbol identifier — preserving the command lets the pattern fire correctly.

---

## Limitations

- **Context window depth:** only `context_before` and `context_after` are searched (the two closest paragraph siblings to the equation in the HTML).  A description that appears three paragraphs away will not be found.
- **Shared context window:** for consecutive equations in an align block, equations (3), (4), (5) may share the same `context_before` text and therefore receive the same TF-IDF winner.  This is a structural limitation of the context extraction in `extraction.py`, not a scoring error.
- **Sentence splitter fragility:** the regex `(?<=[.!?])\s+(?=[A-Z\(])` misses splits after "Eq." or "Ref." (period inside abbreviation) and before lowercase conjunctions.  Some "sentences" fed to scoring are actually 2–3 merged sentences.
- **PDF papers:** context is extracted from raw PDF text lines, which lack paragraph structure.  Strategy 4 (TF-IDF) is most likely to be used for PDF-sourced equations, with lower sentence quality than HTML sources.
