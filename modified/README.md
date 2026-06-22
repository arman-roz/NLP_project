# Equation KG Prototype

Small prototype for the NLP project. It processes the assigned paper list in order and writes a JSON file for the first two papers by default.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The spaCy model is required because meaning extraction uses POS tags and noun phrases, and symbol-definition extraction uses dependency parsing.

## Method

The extraction method uses document structure and linguistic parses instead of physics or LaTeX vocabulary lists:

- explicit equation references are the strongest signal
- meaning sentences are scored with spaCy POS tags and noun phrases
- symbol candidates come directly from MathML `<mi>` identifiers; `<mo>` operators and `<mn>` numbers are ignored by structure
- single-character identifiers in MathML subscript positions are treated as indices
- symbol definitions are extracted from dependency parses of sentences that mention the symbol
- strong relation descriptions are verb phrases extracted from the citation sentence itself
- potential relation descriptions are shared noun phrases from the two equation contexts
- potential edges are capped per equation to reduce overgeneration

The default embedding model is `tbs17/MathBERT`. It is used only as an encoder for similarity/ranking, not for generation or prompting. Transformer files are cached under `modified/data/model_cache`.

Symbols are not filled from built-in physics knowledge. The code writes a symbol definition only when a dependency parse can extract a supporting phrase from the arXiv text.

## Run first two papers

```bash
python -m src.main
```

Output:

```text
modified/data/output/sample_2_papers.json
```

The code uses only `https://arxiv.org/html/<id>`, waits 15 seconds after network requests, and caches downloaded HTML in `modified/data/cache`.

## Useful options

```bash
python -m src.main --limit-papers 2
python -m src.main --limit-papers 5 --max-relation-edges 2
```
