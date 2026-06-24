# Equation KG Prototype

Small prototype for the NLP project. It processes the assigned paper list in order and writes a JSON file for the first two papers by default.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The spaCy model is required because meaning extraction uses POS tags and noun phrases, and symbol-definition extraction uses dependency parsing.

## Method

The extraction uses document structure and the spaCy dependency parse instead of physics or LaTeX vocabulary lists. **No embedding model and no text generation are used** — every string is lifted from the paper, and evidence is found with BM25 lexical retrieval (`rank_bm25`) only.

- inline LaTeX in the surrounding prose is converted to readable Unicode with `pylatexenc` (so `I_{\rm OFF}` reads as `I_OFF`, not `I rm OFF`); the `equation` field keeps verbatim LaTeX
- the `meaning` is a short descriptive phrase taken by *grammatical role* — the subject of a passive/copular introduction ("X is given by"), the object of an active one ("we define X"), or an explicit "called X" — never a whole sentence
- symbol candidates come directly from MathML `<mi>` identifiers; operators and numbers are ignored by structure, and standard math operators (∇, ∂, δ, d) are excluded as the spec allows
- symbol definitions are the noun the symbol stands in a definitional dependency with ("eta is the EFFICIENCY", "the EFFICIENCY eta"); a name like `eta` is matched to the Unicode `η` via `unicodedata` (no hand-written symbol tables). The local window is searched first, then BM25 over the whole paper for symbols specific enough to match unambiguously
- relations are graded `strong`/`potential`/`none` with a score in [0, 1]: `strong` (score 1) when one context explicitly cites the other equation, `potential` from shared symbols and context nouns; potential edges are capped per equation

A symbol definition is written only when the parse yields a supporting phrase from the arXiv text; otherwise the symbol is omitted rather than guessed.

## Run

```bash
python -m src.main
```

Papers are processed in `paper_list_44.txt` order until the dataset reaches 350 equations (the paper crossing 350 is finished in full, per spec). Output is written to `modified/data/output/sample_2_papers.json`.

The code uses only `https://arxiv.org/html/<id>`, waits 15 seconds after each network request, and caches downloaded HTML in `modified/data/cache`, so re-runs are offline.

## Useful options

```bash
python -m src.main --limit-papers 2 --target-equations 0   # quick 2-paper sample
python -m src.main --target-equations 350                  # full prototype dataset
```
