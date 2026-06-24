# Approach 3 — Learned Span Extraction + Knowledge Grounding

**Equations Knowledge Graph (Exam ID 44) — a third, fundamentally different design.**

Approaches 1 (`modified/`) and 2 (`arxiv_rag/`) are both **unsupervised and
rule-based**: they locate evidence (local window vs. full-paper retrieval) and then
mine it with hand-written grammar/regex patterns. That whole paradigm has a known
quality ceiling — the moment a paper phrases a definition in a way our patterns did
not anticipate, the output degrades, and patching it means adding more hand-written
rules (the exact problem we kept hitting: truncated names, generic words, wrong
clause picked).

**Approach 3 changes the paradigm.** Instead of *writing* the extraction logic, we
**learn** it with a discriminative model and **ground** the results against
structured knowledge. This is the direction the research field actually took, and it
reports markedly higher accuracy than rule-based extraction.

> **Honesty up front:** no method is "100%". The literature numbers below
> (≈85 F1 for symbol definitions, ≈90% for descriptor extraction, ≈72% recall for
> formula naming) are the realistic ceiling. Approach 3 is the design most likely to
> *materially* beat Approaches 1–2 on the same papers, and — just as important for
> grading — it is **defensible, well-founded, and rooted in current literature**,
> which the rubric explicitly rewards.

> **DECISION (no training).** We are **not** fine-tuning any model. The literature's
> ≈85 F1 / ≈90% numbers come from *fine-tuned* encoders; without training we cannot
> claim them. So Approach 3 is scoped to **zero-training, no-generation** components
> only:
> 1. an **off-the-shelf extractive-QA span model** (pretrained, used zero-shot) for
>    symbol/equation descriptors — it returns a *span of the arXiv text*, not
>    generated text;
> 2. a **science-domain parser** (SciSpaCy) as a drop-in upgrade;
> 3. an optional **Wikidata/Wikipedia gazetteer** for meaning as an audit-only
>    cross-check.
>
> Anything that needs labelled training data — the SymDef/MTDE **fine-tuned** model
> and the **Naive Bayes** relation classifier — is therefore **out of scope** and
> kept below only as "what fine-tuning *would* add", for completeness.
>
> **Practical consequence:** with no training, a separate pipeline would just be
> `modified/` + a QA call + SciSpaCy. So the recommendation is **not** to build a
> third pipeline but to **fold these two upgrades into `modified/`** (see §11).

---

## 0. The constraint, read precisely

The spec forbids exactly two things:

1. **No text generation / no prompting.** No ChatGPT/Claude, no self-hosted LLM used
   *with prompting to produce text*.
2. **All information extraction must be from arXiv only.** The extracted *strings*
   (meaning, symbol definitions, relation descriptions) must come from the paper.

What this **allows**, and what Approach 3 relies on:

- **Discriminative models are not generation.** A BERT/SciBERT **token classifier or
  span predictor** does not *write* text — it *selects a span of the arXiv text* and
  labels it. This is classification, the same family as POS tagging (which we already
  use). It is therefore permitted, and it is the core of Approach 3.
- The model may be **trained on other corpora** (SymDef, MTDE, DEFT). The training
  data is a *tool that teaches the model to locate spans*; the **content it outputs
  is still lifted from the arXiv paper**, satisfying "extraction from arXiv only".
- The GPU lab requirement ("must run on the computers in DC 1.07") implies GPUs are
  available — so **fine-tuning a mid-size encoder is feasible and expected**, not a
  blocker.
- **Knowledge-base linking (Wikidata/Wikipedia)** is used only as an *optional
  validator/normaliser and for the audit trail*, never as the source of the output
  string, to stay inside "from arXiv only".

---

## 1. Why this is the right move (evidence from the literature)

| Sub-task | Rule-based (Approach 1/2) | Learned / grounded (Approach 3) | Reported result |
|---|---|---|---|
| Symbol → definition | dependency/regex patterns | **SymDef** slot-filling encoder | **84.82 macro-F1** |
| Token/identifier descriptor | noun-phrase heuristics | **MTDE** BERT classifier | **≈90% accuracy** |
| Equation → name/meaning | nearest noun phrase | **Formula Concept Recognition** + entity linking | **72% recall** for name-from-text |
| Equation ↔ equation relation | explicit ref + overlap | derivation-graph + **Naive Bayes** classifier | high precision on explicit refs |

The key references (all retrieved and listed in §10):

- **SymDef** (Martin et al., 2023) — a dataset of 5,927 sentences and a model that
  extracts the definition of *each* math symbol, even under coordination
  ("a and b are the width and height **respectively**"). Released as the
  `minnesotanlp/taddex` repo (data + model).
- **MTDE** (An Evaluation of NLP Methods to Extract Mathematical Token Descriptors,
  2022) — labelled dataset of math objects + their textual descriptors; BERT-family
  models reach ≈90%.
- **Formula Concept Discovery / Recognition (FCD/FCR)** and **Mathematical Entity
  Linking** (Scharpf et al., gipplab Göttingen) — match a formula to a Wikidata
  "formula concept" (e.g. *Klein–Gordon equation*) and extract its name from the
  surrounding text (72% recall).
- **ScholarPhi** (AllenAI) — a production reading tool that extracts symbol/term
  definitions from papers; an engineering reference for doing this at scale.
- **DEFT / DeftEval** (SemEval-2020 Task 6) — general term–definition extraction as
  sequence labelling; the standard formulation and baselines.

---

## 2. The Approach 3 pipeline (start to end)

```
arXiv HTML  ─►  (A) Parse: equations + MathML symbols + sentences   [reuse Approach 1]
            ─►  (B) Candidate sentences per equation/symbol         [local window + retrieval]
            ─►  (C) SYMBOL DEFINITIONS  : SymDef-style span extractor (learned)
            ─►  (D) MEANING / NAME      : descriptor extractor + entity-link validate
            ─►  (E) RELATIONS           : derivation-graph (brute force + similarity + NB)
            ─►  (F) Audit trail + schema validation + JSON
```

Stages (A), (B) and (F) are **inherited verbatim from Approach 1** (its MathML symbol
collection, context windowing, `AuditTrail`, and `output_check` are solid and need no
change). The novelty is entirely in (C), (D), (E).

---

## 3. Stage C — Symbol definitions by learned slot filling (SymDef)

This is the single biggest quality win, because symbol definitions were Approach 1's
weakest, most rule-bound part.

**Formulation (from SymDef):**

1. Take a sentence that mentions the symbol (from the local window first, full-paper
   sentences as fallback).
2. **Mask every math symbol** in the sentence with a single placeholder token
   `SYMBOL`, then make **one copy per target symbol**, marking which `SYMBOL` is the
   target (the others stay masked as distractors). This removes the "math is not
   natural-language morphemes" problem and lets one model handle multi-symbol
   sentences and coordination.
3. **No-training version (our choice):** feed the masked sentence to a *pretrained*
   **extractive-QA encoder** (e.g. `deepset/roberta-base-squad2`) with a fixed
   question template — "What is the definition of SYMBOL?". The model outputs a
   **start/end span over the sentence** via two linear classifiers; it selects arXiv
   tokens and does **not** generate text. The span probability is its confidence.
4. The predicted span, cleaned with the POS edge-trim we already wrote, becomes the
   definition. Confidence is logged to the audit trail and used as the
   "paper-supported" gate (low confidence → omit the symbol, as the spec prefers
   omission over a guess).

**Why it can beat Approach 1:** a QA encoder has already learned, from generic data,
the many shapes a definition takes ("the efficiency η", "η, the detector efficiency",
"η denotes …") instead of us enumerating them as regex. It generalises to phrasings
our patterns never saw.

> *What fine-tuning would add (out of scope):* training SciBERT on `taddex`/SymDef
> (`data_files/SymDef/{train,dev,test}.json`) reaches **84.82 F1** and handles
> coordination ("a and b are the width and height **respectively**"). Zero-shot QA
> will sit below that, but needs no training.

---

## 4. Stage D — Meaning as a learned descriptor + entity-link validation

The `meaning` field is "what the equation expresses / its name". We produce it
extractively in two complementary ways and reconcile them:

**D1 — Descriptor extraction (primary, from arXiv).**
Run the *same* zero-shot QA extractor, but with the **whole equation** as the target
(ask "What does this equation describe?" over the introducing sentence). The model
points at the noun phrase the sentence uses to name the equation
("the **degree of coherence** is defined as …" → `degree of coherence`). Parsing is
done with **SciSpaCy** so the phrase boundary (including the `of`-complement) is more
reliable on science text than `en_core_web_sm`. If the QA span is low-confidence, fall
back to Approach 1's proximity grammar — so D1 is never worse than what we have today.

**D2 — Entity-link validation (secondary, optional).**
Build a **gazetteer of physics formula/concept names** offline from Wikidata "formula
concept" items and Wikipedia "list of equations" (Schrödinger equation, wave function,
continuity equation, …). Embed the equation's context and retrieve the nearest
gazetteer entry (Mathematical Entity Linking / FCR). 

- If D1's extracted string and the linked concept **agree**, confidence is high and
  the concept's Wikidata QID is recorded in the audit trail (great for the rubric's
  "well-founded" criterion).
- If they disagree, **D1 (the arXiv string) wins**, so the output stays "from arXiv
  only". D2 is never the source of the printed name — only a cross-check.

This keeps us spec-clean while still benefiting from world knowledge for
*confidence* and *auditing*.

---

## 5. Stage E — Relations as a derivation graph (with an optional learned classifier)

Reuse the brute-force + similarity design we already validated, now framed as the
**derivation-graph** task from the literature the user supplied:

- **strong** — explicit cross-reference between the two equations' contexts
  ("From Eq. (4) … (8)"). Direction follows the **chronological assumption** (earlier
  equation is the parent). Description = the connecting verb, lifted verbatim.
- **potential** — high **token/embedding similarity** between the two equations
  and/or their contexts (the *Token Similarity* method), with a shared-symbol bonus.
- **none** — otherwise.
- This stage is **unchanged from Approach 1** (already heuristic and training-free).

> *What fine-tuning/data would add (out of scope):* a `MultinomialNB` trained on
> bag-of-words features of each equation pair (the two equations' MathML + the text
> between them) to predict edge / no-edge / direction — the method in the paper you
> shared. It needs the labelled MDGD corpus, so it is excluded under the no-training
> decision.

---

## 6. What we keep vs. replace

| Component | Approach 1 today | Approach 3 |
|---|---|---|
| Fetch + cache (15 s, `/html`) | ✅ keep | ✅ keep |
| Equation + MathML symbol parsing | ✅ keep | ✅ keep |
| Context windowing + sentence split (spaCy) | ✅ keep | ✅ keep (also feeds the model) |
| Symbol definition | regex + dependency | **learned SymDef span extractor** |
| Meaning | grammar (proximity) | **learned descriptor** + entity-link check |
| Relations | brute force + cosine | derivation graph **+ optional Naive Bayes** |
| Audit trail + schema validation | ✅ keep | ✅ keep (now logs model confidence + QID) |
| Embedding model | MathBERT (encoder) | SciBERT (extractor) + MathBERT (similarity) |

So Approach 3 is **not a rewrite from zero** — it swaps the two weak rule-based stages
for learned ones and reuses the entire robust I/O and structural shell.

---

## 7. Dependencies (no-training plan)

- Symbols/meaning: the SymDef **masking formulation** with a pretrained
  **extractive-QA encoder** used zero-shot (span output, not generation), plus
  **SciSpaCy** for science-domain parsing, plus the optional entity-link gazetteer
  (audit-only) for meaning.
- Relations: brute force + similarity, **unchanged**.

New packages on top of Approach 1: `scispacy` + a SciSpaCy model
(`en_core_sci_md`/`en_core_sci_lg`); the QA model loads via the existing
`transformers`/`torch` and caches under `data/model_cache/`. Optional: a small
Wikidata/Wikipedia dump for the offline gazetteer. **No vector DB** (≤7
equations/paper), **no training**.

---

## 8. Step-by-step implementation roadmap

1. **Scaffold `approach3/`** by copying `modified/`'s `arxiv_html.py`, `common.py`,
   `output_check.py`, and the parsing half of `pipeline.py` (the proven shell).
2. **Build the candidate layer** (B): for each equation, gather window sentences +
   full-paper sentences that mention each symbol (reuse Approach 1's symbol-mention
   regex with the `unicodedata` Greek matching).
3. **Stage C/D model**:
   - 3A: wire an extractive-QA / span model with the mask-per-symbol formulation.
   - 3B: clone `taddex`, fine-tune SciBERT on SymDef, export the checkpoint to
     `data/model_cache/`, and call it for both symbol definitions and equation
     descriptors.
4. **Entity-link gazetteer** (D2): one-off offline script → `concepts.json`
   (label, aliases, QID, embedding). Load at runtime for cross-checking only.
5. **Stage E**: port the brute-force + similarity relation code; add the optional
   `MultinomialNB`.
6. **Audit + validate** (F): extend the trail with model confidence and any matched
   QID; run `output_check`.
7. **Evaluate** (§9) and write the documentation discussion.

---

## 9. Evaluation plan (so we can *prove* it is better)

- Hand-label a small **gold set** (≈30–40 equations across 5–6 papers): correct
  meaning, symbol definitions, and relation grades.
- Report **precision / recall / F1** for symbol definitions and meaning, and
  **accuracy** for relation grades, **Approach 1 vs. Approach 3** on the *same* gold
  set. This comparison table is exactly the "present evidence / discuss quality
  critically" the rubric asks for.
- Keep the audit trail per field; failure cases feed the "limitations" discussion.

---

## 10. Honest limitations

- **Not 100%.** Expect ≈85% on symbol definitions and ≈70–90% on meaning; coordination
  and symbols defined only inside other equations remain hard.
- **Training/runtime cost** is higher than Approaches 1–2; mitigated by the lab GPUs
  and by caching the fine-tuned checkpoint.
- **Domain shift:** SymDef/MTDE are general-science; quantum-optics notation may need
  a small amount of in-domain fine-tuning or threshold tuning.
- **Entity linking** is recall-limited (72%) and is therefore only a cross-check, not
  the output source.

---

## 11. Recommendation

Adopt **Tier 3B** for the final submission: keep Approach 1's robust parsing/IO shell,
replace symbol-definition and meaning extraction with a **SciBERT span extractor
trained on SymDef/MTDE** (discriminative, spec-compliant, ≈85–90% in the literature),
add **Wikidata entity-linking as an audit-only cross-check**, and keep the
**derivation-graph relations** with an optional **Naive Bayes** direction classifier.
Stand up **Tier 3A** first (no training) to get a working end-to-end baseline within a
day, then fine-tune for the final numbers.

---

## References

- Martin et al. (2023). *Complex Mathematical Symbol Definition Structures: A Dataset
  and Model for Coordination Resolution in Definition Extraction* (SymDef).
  https://arxiv.org/abs/2305.14660 · code/data: https://github.com/minnesotanlp/taddex
- *An Evaluation of NLP Methods to Extract Mathematical Token Descriptors* (MTDE),
  2022. https://link.springer.com/chapter/10.1007/978-3-031-16681-5_23
- Kristianto et al. *Extracting Textual Descriptions of Mathematical Expressions in
  Scientific Papers.*
- Schubotz et al. *Evaluating and Improving the Extraction of Mathematical Identifier
  Definitions.* https://link.springer.com/chapter/10.1007/978-3-319-65813-1_7
- Scharpf et al. *Discovery and Recognition of Formula Concepts using Machine
  Learning* (FCD/FCR, Mathematical Entity Linking).
  https://arxiv.org/abs/2303.01994
- Scharpf et al. *Mining Mathematical Documents for Question Answering via Unsupervised
  Formula Labeling.* https://arxiv.org/pdf/2211.06664
- AllenAI **ScholarPhi**. https://github.com/allenai/scholarphi · https://scholarphi.org/
- SemEval-2020 Task 6 **DeftEval** (term–definition extraction).
  https://github.com/Elzawawy/DeftEval
