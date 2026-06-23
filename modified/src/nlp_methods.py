"""NLP methods for meanings, symbols, and equation relations."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import spacy

from .common import AuditTrail, short


@lru_cache(maxsize=1)
def _load_spacy():
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "spaCy model en_core_web_sm is required. "
            "Install it with: python -m spacy download en_core_web_sm"
        )
    return spacy.load("en_core_web_sm")


class TextTools:
    """Sentence, token, POS, dependency, and lemma helpers."""

    def __init__(self) -> None:
        self.nlp = _load_spacy()
        self.sent_nlp = spacy.blank("en")
        self.sent_nlp.add_pipe("sentencizer")
        self.has_parser = "parser" in self.nlp.pipe_names
        self.stop_words = set(self.nlp.Defaults.stop_words)

    def sentences(self, text: str) -> List[str]:
        """Return clean sentence strings."""

        cleaned = self.clean(text)
        if not cleaned:
            return []
        doc = self.sent_nlp(cleaned)
        return [sent.text.strip() for sent in doc.sents if self._good_sentence(sent.text)]

    def tokens(self, text: str, keep: Iterable[str] = ()) -> List[str]:
        """Return normalized non-stopword lemmas."""

        keep_set = {item.lower() for item in keep}
        doc = self.nlp(self.clean(text))
        out: List[str] = []
        for token in doc:
            raw = token.text.lower()
            lemma = token.lemma_.lower() if token.lemma_ else raw
            if token.is_space or token.is_punct or token.like_num:
                continue
            if raw in self.stop_words and raw not in keep_set:
                continue
            if len(lemma) >= 2 and re.search(r"[a-z]", lemma):
                out.append(lemma)
        return out

    def doc(self, text: str):
        """Return a spaCy document."""

        return self.nlp(self.clean(text))

    def raw_doc(self, text: str):
        """Return a spaCy document without TeX-oriented cleanup."""

        return self.nlp(text.replace("\xa0", " "))

    @staticmethod
    def clean(text: str) -> str:
        """Normalize paper text while preserving LaTeX command names as words."""

        text = text.replace("\xa0", " ")
        text = re.sub(r"\\([A-Za-z]+)", r" \1 ", text)
        text = re.sub(r"[{}_^$]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _good_sentence(sentence: str) -> bool:
        words = re.findall(r"[A-Za-z]{2,}", sentence)
        return len(words) >= 4 and 20 <= len(sentence) <= 500


class EmbeddingSimilarity:
    """Transformer encoder wrapper used only for similarity, never generation.
    Uses AutoModel with mean pooling for proper sentence embeddings.
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized vectors using mean pooling."""
        if not texts:
            return np.zeros((0, 1), dtype=float)
        model, tokenizer = self._load()
        import torch
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def pairwise(self, labels: List[str], texts: List[str]) -> Dict[Tuple[str, str], float]:
        if len(labels) < 2:
            return {}
        if len(labels) != len(texts):
            raise ValueError("labels and texts must have the same length")
        vectors = self.encode(texts)
        out: Dict[Tuple[str, str], float] = {}
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if left != right:
                    out[(left, right)] = float(np.dot(vectors[i], vectors[j]))
        return out

    def rank(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        vectors = self.encode([query] + candidates)
        query_vec = vectors[0]
        return [float(np.dot(query_vec, vec)) for vec in vectors[1:]]

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir))
            self._model = AutoModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir))
            self._model.eval()
        return self._model, self._tokenizer


# ---------------------------------------------------------------------------
# Grammar-driven noun-phrase helpers (shared by meaning and symbol extraction)
#
# These replace hand-written word lists with the spaCy parse + Unicode data, so
# they generalize to papers whose vocabulary we have never seen. A "name" or a
# symbol "definition" is a noun phrase: its head noun, the modifiers in front of
# it, and any "of"/"for" complement after it ("degree of coherence", not just
# "degree"). Determiners, prepositions and conjunctions are identified by their
# part-of-speech tag, not by being on a list of specific English words.
# ---------------------------------------------------------------------------

# Dependency labels of children that begin a *new clause* and so must not be
# pulled into a noun phrase (relative/adverbial/complement clauses, etc.).
_CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "csubj", "parataxis"}
# Prepositions that introduce a genuine post-nominal complement worth keeping
# ("degree of coherence", "density of states"). Others (by/from/with/...) tend
# to start a separate adjunct, so they are dropped to keep the name compact.
_KEEP_PREPS = {"of", "for"}
# Edge part-of-speech tags to trim off the start/end of a phrase (articles,
# prepositions, conjunctions, particles, auxiliaries, pronouns, punctuation).
_EDGE_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "AUX", "PRON"}
# Structural "meta" nouns that name a piece of writing rather than a physics
# concept. Domain-independent (they occur in any paper, not in the unseen physics
# vocabulary), so rejecting one as a *lone* name/definition does not hurt
# generalization; specific multi-word phrases never trigger this guard. This is
# the only content-word list the extractors keep -- everything else is decided
# from POS tags, the dependency parse, and the embedding ranker.
_META_NOUNS = {
    "equation", "result", "form", "case", "value", "term", "expression",
    "quantity", "parameter", "function", "system", "example", "section",
    "figure", "table", "paper", "method", "approach", "number", "order",
    "part", "side", "way", "set", "thing", "one",
}
# LaTeX/MathML command leftovers (a fixed, finite set of markup tokens -- not
# domain vocabulary). Greek letter names are handled separately via unicodedata.
_LATEX_CMDS = {
    "rm", "mathrm", "text", "textrm", "cal", "mathcal", "mathbb", "mathbbm",
    "mathfrak", "hat", "bar", "tilde", "vec", "dot", "operatorname", "langle",
    "rangle", "ll", "gg", "approx", "ldots", "cdots", "prime", "dagger",
    "partial", "nabla", "sum", "int", "prod", "frac", "sqrt", "left", "right",
    "big", "mathop", "ordinarycolon", "coloneqq", "displaystyle",
}


@lru_cache(maxsize=4096)
def _is_greek_letter_name(word: str) -> bool:
    """True if ``word`` is the spelled-out name of a Greek letter (eta, phi, ...).

    Uses ``unicodedata`` to ask whether a Greek code point with this name exists,
    instead of hard-coding the alphabet, so it covers every Greek letter name.
    """

    upper = word.upper()
    for template in ("GREEK SMALL LETTER {}", "GREEK CAPITAL LETTER {}"):
        try:
            unicodedata.lookup(template.format(upper))
            return True
        except KeyError:
            continue
    return False


def _content_token(token) -> bool:
    """True if a token is a real word (not a math symbol or markup leftover)."""

    word = token.text
    if len(word) < 2 or not re.search(r"[A-Za-z]", word):
        return False  # bare single letters are math symbols, not words
    if any(ord(ch) > 127 for ch in word):
        return False  # non-ASCII math glyphs (η, ∇, ...)
    low = word.lower()
    return low not in _LATEX_CMDS and not _is_greek_letter_name(low)


def _np_span(head):
    """Token span of the noun phrase headed by ``head``.

    Walks the head noun's dependency subtree, keeping determiners, adjectival,
    compound, numeric and possessive modifiers plus "of"/"for" complements, but
    stopping at clause boundaries, appositions and coordinations. This yields
    full names like "degree of coherence" that spaCy's base noun chunks would
    truncate to "degree".
    """

    keep = {head.i}

    def walk(token):
        for child in token.children:
            dep = child.dep_
            if dep in _CLAUSE_DEPS or dep in {"punct", "cc", "conj", "appos"}:
                continue
            if dep == "prep" and child.text.lower() not in _KEEP_PREPS:
                continue
            keep.add(child.i)
            walk(child)

    walk(head)
    # Keep only the contiguous run around the head (pruned children leave gaps).
    start = end = head.i
    while start - 1 in keep:
        start -= 1
    while end + 1 in keep:
        end += 1
    return head.doc[start:end + 1]


def _phrase_tokens(tokens, stop_words) -> List:
    """Filter to content words and trim determiner/preposition edges by POS."""

    kept = [tok for tok in tokens if not tok.is_space and _content_token(tok)]
    while kept and (kept[0].pos_ in _EDGE_POS or kept[0].lower_ in stop_words):
        kept.pop(0)
    while kept and (kept[-1].pos_ in _EDGE_POS or kept[-1].lower_ in stop_words):
        kept.pop()
    return kept


class MeaningExtractor:
    """Extracts a short, precise *name* for each equation.

    The ``meaning`` field is meant to be a concise description of what the
    equation expresses or its name (e.g. "wave function", "threshold power",
    "Newton's third law") -- not a whole explanatory sentence. The name is
    always extracted verbatim from the paper text (never generated): it is the
    noun phrase the equation is introduced with, an explicit "called/known as"
    label, or a named-equation phrase ("X equation/law/theorem").
    """

    _CALLED_RE = re.compile(
        r"(?:called|known\s+as|termed|named|dubbed|referred\s+to\s+as|"
        r"so-?called|we\s+call|which\s+we\s+call)\s+(?:the\s+|a\s+|an\s+)?"
        r"(.+?)(?:[,.;:()\[\]]|$)",
        re.IGNORECASE,
    )
    _NAMED_EQ_RE = re.compile(
        r"\b([A-Z][A-Za-z'’.\-]+(?:\s+[A-Za-z'’\-]+){0,3}\s+"
        r"(?:equation|law|theorem|relation|formula|principle|rule|identity|"
        r"inequality|distribution|model|ansatz|transformation|effect|"
        r"hamiltonian|lagrangian))\b"
    )

    def __init__(self, text: TextTools, similarity: "EmbeddingSimilarity") -> None:
        self.text = text
        self.similarity = similarity

    def extract(
        self,
        eq_num: str,
        before: str,
        after: str,
        audit: AuditTrail,
        equation_symbols: Optional[List[str]] = None,
        used: Optional[set] = None,
    ) -> str:
        """Return a short name/description for the equation.

        The name is produced extractively in three tiers. First an explicit
        "called/known as X" or "X equation/law" label is trusted outright.
        Otherwise candidate noun phrases are collected from each clause near the
        equation -- *both* the subject and the object, without any verb list --
        and the embedding encoder ranks them by cosine similarity to the
        equation's local context, so the phrase most central to what the equation
        is about wins. No text is generated; the encoder only *ranks* phrases
        lifted verbatim from the paper.

        Parameters
        ----------
        eq_num : str
            The equation number (used to find sentences citing this equation).
        before, after : str
            Prose context immediately before/after the equation.
        audit : AuditTrail
            Trail to record which strategy and sentence produced the name.
        equation_symbols : list[str], optional
            Unused for naming; kept for interface compatibility.
        used : set[str], optional
            Names already assigned to earlier equations in the same paper. The
            ranker prefers a different name so equations do not all collapse onto
            one repeated meaning (it still reuses a name if no alternative fits).

        Returns
        -------
        str
            A concise extracted name (e.g. "degree of coherence"), or "" if no
            usable name is found in the local context.
        """

        candidates = self._ordered_candidates(eq_num, before, after)
        if not candidates:
            audit.add("meaning", "no local prose sentence")
            return ""
        local_context = f"{before} {after}".strip()
        used = used or set()

        # Tier 1: explicit naming -- "called/known as X" or "X equation/law".
        # Named-equation matches are only trusted in an introducing context
        # (the sentence cites the equation or directly precedes it) to avoid
        # picking up an unrelated named theorem mentioned in passing.
        for sentence, _ in candidates:
            phrase = self._name_from_called(sentence)
            if not phrase and self._is_intro_context(eq_num, sentence):
                phrase = self._named_equation(sentence)
            if phrase:
                audit.add("meaning", f"named: {phrase} <= {short(sentence)}")
                return phrase

        # Tier 2: the noun phrase of the clause nearest the equation (subject vs.
        # object chosen from the parse). Sentences are already ordered nearest
        # first, so this is proximity-driven and deterministic -- the reliable
        # signal for a name. Soft de-duplication prefers a name not yet used by an
        # earlier equation but reuses one if the clause offers no alternative.
        clause_pairs = [
            (self._name_from_clause(sentence, prefer_last), sentence)
            for sentence, prefer_last in candidates
        ]
        clause_pairs = [(phrase, sentence) for phrase, sentence in clause_pairs if phrase]
        for phrase, sentence in clause_pairs:
            if phrase not in used:
                audit.add("meaning", f"clause: {phrase} <= {short(sentence)}")
                return phrase
        if clause_pairs:
            audit.add("meaning", f"clause(reused): {clause_pairs[0][0]} <= {short(clause_pairs[0][1])}")
            return clause_pairs[0][0]

        # Tier 3: no introducing clause -- fall back to nearby noun phrases, here
        # ranked by embedding similarity to the local context (the ambiguous case
        # where the encoder genuinely helps choose the most relevant phrase).
        best = self._rank_best(self._fallback_candidates(candidates), local_context, used)
        if best:
            audit.add("meaning", f"fallback: {best[0]} <= {short(best[1])}")
            return best[0]

        audit.add("meaning", f"no name found in {len(candidates)} sentences")
        return ""

    def _rank_best(self, pairs: List[tuple], context: str, used: set) -> Optional[tuple]:
        """Pick the (phrase, sentence) whose phrase best fits the local context.

        Candidates are de-duplicated keeping their first (nearest) occurrence and
        ranked by cosine similarity to the context (nearest clause breaks near
        ties, cosine rounded to 2dp). A name already used by an earlier equation
        is only chosen if no unused candidate is available, which keeps a paper's
        meanings varied without inventing anything.
        """

        unique: List[tuple] = []
        seen: set[str] = set()
        for phrase, sentence in pairs:
            if phrase not in seen:
                seen.add(phrase)
                unique.append((phrase, sentence))
        if not unique:
            return None
        if len(unique) == 1 and unique[0][0] not in used:
            return unique[0]

        if len(unique) == 1 or not context:
            scores = [1.0] * len(unique)
        else:
            scores = self.similarity.rank(context, [phrase for phrase, _ in unique])
        order = sorted(
            range(len(unique)), key=lambda i: (round(scores[i], 2), -i), reverse=True
        )
        for index in order:
            if unique[index][0] not in used:
                return unique[index]
        return unique[order[0]]

    def _ordered_candidates(self, eq_num: str, before: str, after: str) -> List[tuple]:
        """Order context sentences by how likely they name the equation.

        Returns ``(sentence, prefer_last)`` pairs. The sentence immediately
        before the equation is usually the introducing sentence ("... is given
        by:"), so it comes first, followed by the sentence immediately after,
        then any sentence citing the equation number, then the remaining
        context. ``prefer_last`` is True for ``before`` sentences (the equation
        follows their last clause) and False for ``after`` sentences.
        """

        before_sents = self.text.sentences(before)
        after_sents = self.text.sentences(after)

        ordered: List[tuple] = []
        if before_sents:
            ordered.append((before_sents[-1], True))
        if after_sents:
            ordered.append((after_sents[0], False))
        ordered.extend((s, True) for s in before_sents if self._cites(eq_num, s))
        ordered.extend((s, False) for s in after_sents if self._cites(eq_num, s))
        ordered.extend((s, True) for s in reversed(before_sents[:-1]))
        ordered.extend((s, False) for s in after_sents[1:])

        seen: set[str] = set()
        out: List[tuple] = []
        for sentence, prefer_last in ordered:
            if sentence not in seen:
                seen.add(sentence)
                out.append((sentence, prefer_last))
        return out

    @staticmethod
    def _cites(eq_num: str, sentence: str) -> bool:
        """Return True if the sentence explicitly cites this equation number."""

        return bool(
            re.search(
                rf"(?:eq(?:uation)?s?\.?\s*)?\(\s*{re.escape(eq_num)}\s*\)",
                sentence,
                re.IGNORECASE,
            )
        )

    def _is_intro_context(self, eq_num: str, sentence: str) -> bool:
        """True if the sentence looks like it introduces this equation."""

        return self._cites(eq_num, sentence) or sentence.rstrip().endswith(":")

    def _name_from_called(self, sentence: str) -> str:
        """Extract a name from an explicit 'called/known as X' construction."""

        match = self._CALLED_RE.search(self.text.clean(sentence))
        if not match:
            return ""
        phrase = self._clean_phrase(match.group(1))
        return phrase if self._valid_phrase(phrase) else ""

    def _named_equation(self, sentence: str) -> str:
        """Extract a named equation like 'Schrodinger equation' or 'X law'."""

        match = self._NAMED_EQ_RE.search(self.text.clean(sentence))
        if not match:
            return ""
        phrase = self._clean_phrase(match.group(1))
        # Keep multi-word names; a lone keyword ("equation") is not a name.
        words = phrase.split()
        return phrase if len(words) >= 2 and self._valid_phrase(phrase) else ""

    def _name_from_clause(self, sentence: str, prefer_last: bool = True) -> str:
        """The noun phrase the equation's nearest clause describes.

        Which of the subject/object is the named quantity is decided from the
        *parse*, not a verb list: in a passive ("X is given by:") or copula
        ("X is ...") clause the subject is the quantity; in an active clause with
        a pronoun subject ("we define X") the object is. Otherwise the subject is
        tried first, then the object. Verbs are visited nearest-the-equation
        first (``prefer_last`` for ``before`` text). Returns one phrase, or "".
        """

        doc = self._parse_doc(sentence)
        verbs = [
            token for token in doc
            if token.pos_ in {"VERB", "AUX"} and self._subject(token) is not None
        ]
        verbs.sort(key=lambda token: token.i, reverse=prefer_last)

        for verb in verbs:
            subject = self._subject(verb)
            complement = self._complement(verb)
            passive = any(child.dep_ in {"nsubjpass", "auxpass"} for child in verb.children)
            copula = verb.lemma_ == "be" or any(
                child.dep_ in {"attr", "acomp"} for child in verb.children
            )
            pronoun_subject = subject is not None and subject.pos_ == "PRON"
            order = (
                [complement, subject]
                if (not passive and not copula and pronoun_subject)
                else [subject, complement]
            )
            for token in order:
                if token is None or self._is_reference(token):
                    continue
                phrase = self._phrase_from_head(token)
                if self._valid_phrase(phrase):
                    return phrase
        return ""

    def _phrase_from_head(self, head) -> str:
        """Build the full noun-phrase name (with 'of'/'for' complements) of a head.

        Unlike a base noun chunk, this keeps the prepositional complement so a
        head like "degree" becomes "degree of coherence". Returns "" if the head
        is not a noun or the phrase has no real content word.
        """

        if head is None or head.pos_ not in {"NOUN", "PROPN"}:
            return ""
        tokens = _phrase_tokens(_np_span(head), self.text.stop_words)
        return " ".join(tok.text for tok in tokens[:8])

    def _fallback_candidates(self, candidates: List[tuple]) -> List[tuple]:
        """Collect specific noun phrases from the nearest sentences for ranking."""

        pairs: List[tuple] = []
        for sentence, _ in candidates[:4]:
            for chunk in self._noun_chunks(self._parse_doc(sentence)):
                if self._is_reference(chunk.root):
                    continue
                phrase = self._phrase_from_head(chunk.root)
                if self._valid_phrase(phrase):
                    pairs.append((phrase, sentence))
        return pairs

    def _parse_doc(self, sentence: str):
        """Parse a sentence with inline math stripped so the prose parses cleanly.

        Inline math identifiers (``I OFF`` from ``I_{\\rm OFF}``, single symbols,
        Greek letters) confuse the dependency parser and break subject detection,
        so they are removed before parsing. Only used for naming the equation;
        symbol extraction keeps the original tokens.
        """

        cleaned = re.sub(r"\([^)]*\)", " ", sentence)
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        cleaned = cleaned.replace("�", " ")
        # symbol + all-caps/numeric subscript, e.g. "I OFF", "M AB", "S op"
        cleaned = re.sub(r"(?<![A-Za-z])[A-Za-z]\s+[A-Z0-9]{2,}(?![A-Za-z])", " ", cleaned)
        # remaining standalone single-letter symbols and non-ASCII math glyphs
        cleaned = re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", " ", cleaned)
        cleaned = re.sub(r"[^\x00-\x7f]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return self.text.doc(cleaned)

    @staticmethod
    def _subject(verb):
        for child in verb.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return child
        return None

    @staticmethod
    def _complement(verb):
        """Return the nominal complement/object of a verb, if any."""

        for dep in ("attr", "oprd", "dobj", "obj"):
            for child in verb.children:
                if child.dep_ == dep and child.pos_ in {"NOUN", "PROPN"}:
                    return child
        for child in verb.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj" and grandchild.pos_ in {"NOUN", "PROPN"}:
                        return grandchild
        return None

    @staticmethod
    def _is_reference(token) -> bool:
        """True if a noun-phrase head is a pronoun/citation rather than a name.

        Decided from the parse, not a word list: pronouns/demonstratives
        ("it", "this") are tagged ``PRON``, and an equation citation
        ("Equation (2)") has the head lemma "eq"/"equation".
        """

        if token is None:
            return True
        if token.pos_ == "PRON":
            return True
        return token.lemma_.lower() in {"eq", "equation"}

    def _clean_phrase(self, text: str) -> str:
        """Reduce a regex-captured string to a compact, math-free name (<=8 words).

        Used by the 'called X' / 'X equation' strategies, which yield a raw string
        rather than a parse node. The string is re-parsed so the same POS-based
        trimming and content-word filtering as the grammar path apply.
        """

        text = text.replace("�", " ")
        text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
        text = re.sub(r"\\[A-Za-z]+", " ", text)
        text = re.sub(r"[{}_^$\\]", " ", text)
        doc = self.text.doc(text)
        tokens = _phrase_tokens(list(doc), self.text.stop_words)
        return " ".join(tok.text for tok in tokens[:8])

    def _valid_phrase(self, phrase: str) -> bool:
        """A phrase is a usable name if it reads as a specific noun phrase.

        Decided from POS: it must contain a noun (so bare adjectives/verbs are
        rejected) and a single bare structural meta-noun ("result", "form") is
        rejected. Specific multi-word names always pass.
        """

        if len(phrase) < 3:
            return False
        content = [token for token in self.text.doc(phrase) if token.is_alpha]
        if not content:
            return False
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in content):
            return False
        if len(content) == 1 and content[0].lower_ in _META_NOUNS:
            return False
        return any(len(token.text) >= 3 for token in content)

    @staticmethod
    def _noun_chunks(doc) -> List:
        try:
            return list(doc.noun_chunks)
        except ValueError:
            return []


class SymbolExtractor:
    """Extracts symbols, then only keeps paper-supported definitions."""

    def __init__(self, text: TextTools) -> None:
        self.text = text

    def extract(
        self,
        mathml_symbols: List[str],
        local_sentences: List[str],
        paper_sentences: List[str],
        audit: AuditTrail,
    ) -> tuple[Dict[str, str], List[str]]:
        """Return paper-supported symbol definitions and the raw candidates.

        Symbols come from the equation's MathML identifiers. A definition is
        written only when the surrounding text actually defines the symbol
        ("where X is the ...", "the ... X"); symbols without textual support are
        omitted rather than guessed. No physics vocabulary is hard-coded -- only
        generic definitional grammar and Unicode letter names are used.
        """

        symbols = self._symbols(mathml_symbols)
        audit.add("symbol_candidates", ", ".join(symbols) if symbols else "none")

        definitions: Dict[str, str] = {}
        used: set[str] = set()
        for symbol in symbols:
            found = self._best_definition(symbol, local_sentences)
            if found is None:
                audit.add("symbol_definition", f"{symbol}: not found in paper text")
                continue
            definition, evidence = found
            base = symbol.split("_")[0]
            if definition.lower() in used and len(base) == 1:
                audit.add("symbol_definition", f"{symbol}: skipped duplicate '{definition}'")
                continue
            definitions[symbol] = definition
            used.add(definition.lower())
            audit.add("symbol_definition", f"{symbol}: {definition} | {short(evidence)}")
        return definitions, symbols

    # Verbs/phrases that introduce a symbol's definition in scientific prose.
    _DEFINING = (
        r"(?:is|are|was|were|be|denotes?|denote|represents?|represent|"
        r"stands?\s+for|measures?|describes?|characteri[sz]es?|quantif(?:ies|y)|"
        r"gives?|corresponds?\s+to|refers?\s+to|equals?|defined\s+as|denoted\s+by)"
    )

    # Standard mathematical operators. The assignment explicitly excludes these
    # from the symbols dict ("Mathematical standard operators (+, -, ∇, ...) are
    # not required to be explained"); the differential/variation operators below
    # act *on* variables rather than being variables themselves.
    _OPERATORS = {"d", "delta", "Delta", "partial", "nabla", "mathrm", "rm", "mathcal"}

    def _best_definition(self, symbol: str, sentences: List[str]) -> Optional[Tuple[str, str]]:
        """Find the best textual definition for a symbol among the sentences."""

        pattern = self._symbol_regex(symbol)
        best: Optional[Tuple[float, str, str]] = None
        for sentence in sentences:
            for match in pattern.finditer(sentence):
                definition = self._predicate_after(sentence, match.end())
                if not definition:
                    definition = self._appositive_before(sentence, match.start())
                if not definition:
                    continue
                score = self._def_score(definition, sentence)
                if best is None or score > best[0]:
                    best = (score, definition, sentence)
        if best is None:
            best = self._dep_best(symbol, sentences, pattern)
        if best is None:
            return None
        return best[1], best[2]

    def _symbol_regex(self, symbol: str) -> "re.Pattern":
        """Regex matching any surface form of the symbol (name or Unicode)."""

        variants = self._surface_variants(symbol.split("_")[0])
        alternation = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
        return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])")

    @staticmethod
    def _surface_variants(base: str) -> set:
        """Surface forms of a symbol: its name plus Unicode Greek equivalents.

        Uses ``unicodedata`` (a library, not a hand-written table) so that a
        MathML name like ``eta`` matches the Unicode ``η``/``Η`` in the prose.
        """

        variants = {base}
        upper = base.upper()
        for template in ("GREEK SMALL LETTER {}", "GREEK CAPITAL LETTER {}", "GREEK {} SYMBOL"):
            try:
                variants.add(unicodedata.lookup(template.format(upper)))
            except KeyError:
                pass
        return {variant for variant in variants if variant}

    def _predicate_after(self, sentence: str, pos: int) -> str:
        """Match 'SYM (subscript)? is/denotes/... the <definition>'."""

        tail = sentence[pos:]
        match = re.match(
            rf"\s*(?:[A-Za-z0-9]+\s+)?(?:\([^)]*\)\s*)?{self._DEFINING}\s+"
            r"(?:the|a|an|its|their|some)?\s*(.+?)"
            r"(?:[,.;:]|\sand\s|\swhere\s|\swith\s|\swhich\s|\sgiven\s|$)",
            tail,
            re.IGNORECASE,
        )
        if not match:
            return ""
        definition = self._clean_def(match.group(1))
        return definition if self._valid_def(definition) else ""

    def _appositive_before(self, sentence: str, start: int) -> str:
        """Match a tight appositive noun phrase right before the symbol.

        Only fires for a short ``the <noun phrase> SYM`` with no intervening
        clause or punctuation, so it cannot reach across a comma into a previous
        symbol's definition ("... an arbitrary field, rho ...").
        """

        words = sentence[:start].split()
        # Only look back a few tokens: a real appositive is adjacent to the symbol.
        for index in range(len(words) - 1, max(-1, len(words) - 6), -1):
            if words[index].lower() in {"the", "a", "an"}:
                phrase_words = words[index + 1:]
                if not 1 <= len(phrase_words) <= 4:
                    return ""
                if any(re.search(r"[,;:.]", word) for word in phrase_words):
                    return ""
                definition = self._clean_def(" ".join(phrase_words))
                return definition if self._valid_def(definition) else ""
        return ""

    @staticmethod
    def _def_score(definition: str, sentence: str) -> float:
        """Prefer compact phrases and sentences with explicit 'where' definitions."""

        word_count = len(definition.split())
        score = 1.0 if 2 <= word_count <= 6 else 0.3
        if re.search(r"\bwhere\b", sentence, re.IGNORECASE):
            score += 0.5
        return score

    def _dep_best(self, symbol: str, sentences: List[str], pattern: "re.Pattern") -> Optional[Tuple[float, str, str]]:
        """Dependency-parse fallback for definitions the regex patterns miss."""

        variants = {variant.lower() for variant in self._surface_variants(symbol.split("_")[0])}
        best: Optional[Tuple[float, str, str]] = None
        for sentence in sentences:
            if not pattern.search(sentence):
                continue
            for token in self.text.doc(sentence):
                if token.text.lower() not in variants:
                    continue
                definition = self._clean_def(self._dep_phrase(token))
                if not self._valid_def(definition):
                    continue
                score = self._def_score(definition, sentence)
                if best is None or score > best[0]:
                    best = (score, definition, sentence)
        return best

    def _dep_phrase(self, token) -> str:
        """Definition phrase implied by a symbol token's syntactic role."""

        if token.dep_ in {"nsubj", "nsubjpass"}:
            for child in token.head.children:
                if child is token:
                    continue
                if child.dep_ in {"attr", "oprd", "dobj", "obj"} and child.pos_ in {"NOUN", "PROPN"}:
                    return self._chunk_text(child)
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj" and grandchild.pos_ in {"NOUN", "PROPN"}:
                            return self._chunk_text(grandchild)
        if token.dep_ in {"appos", "compound", "nmod", "dep"} and token.head.pos_ in {"NOUN", "PROPN"}:
            return self._chunk_text(token.head)
        return ""

    def _chunk_text(self, token) -> str:
        """The full noun phrase headed by a token, including 'of'/'for' complements.

        Uses the shared grammar walk so a head like "degree" keeps its complement
        ("degree of coherence") instead of being truncated to the base chunk.
        """

        tokens = _phrase_tokens(_np_span(token), self.text.stop_words)
        return " ".join(item.text for item in tokens[:8])

    @staticmethod
    def _noun_chunks(doc) -> List:
        try:
            return list(doc.noun_chunks)
        except ValueError:
            return []

    def _symbols(self, mathml_symbols: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in mathml_symbols:
            symbol = raw.strip()
            if not self._usable_symbol(symbol):
                continue
            if symbol and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
        return out[:22]

    def _usable_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        parts = symbol.split("_")
        base = parts[0]
        if base in self._OPERATORS:
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", base):
            return False
        if len(parts) > 1 and not re.fullmatch(r"[A-Za-z][A-Za-z0-9 ]*", parts[1]):
            return False
        return True

    def _clean_def(self, text: str) -> str:
        """Reduce a captured phrase to a compact, math-free definition (<=8 words).

        Re-parses the captured string and keeps content words with POS-based edge
        trimming (the same path meaning extraction uses), so markup leftovers,
        bare symbols and Greek-letter names are dropped without any hand-written
        vocabulary list.
        """

        text = text.replace("�", " ")
        text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
        text = re.sub(r"\\[A-Za-z]+", " ", text)
        text = re.sub(r"[{}_^$\\]", " ", text)
        doc = self.text.doc(text)
        tokens = _phrase_tokens(list(doc), self.text.stop_words)
        return " ".join(token.text for token in tokens[:8])

    def _valid_def(self, phrase: str) -> bool:
        """A definition is usable if it reads as a specific noun phrase.

        Decided from POS, not word lists: a definition is a noun phrase
        ("efficiency of the detector", "bond dimension"), so it must contain a
        noun and must not contain a verb/auxiliary (which would make it a clause
        like "is large when ..."). A lone structural meta-noun is rejected.
        """

        if len(phrase) < 3:
            return False
        content = [token for token in self.text.doc(phrase) if token.is_alpha]
        if not content:
            return False
        if any(token.pos_ in {"VERB", "AUX"} for token in content):
            return False
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in content):
            return False
        if len(content) == 1 and content[0].lower_ in _META_NOUNS:
            return False
        return any(len(token.text) >= 3 for token in content)


class RelationExtractor:
    """Brute-force relations: explicit references (strong) and context similarity.

    Two low-cost, model-free signals classify each ordered equation pair, after
    the approach surveyed by Bishop et al. (arXiv "derivation graph" study):

    * **Brute force** -- if one equation's context explicitly mentions the other
      equation's number, that is a clear (``strong``) relation.
    * **Context similarity** -- otherwise, the two equations' small extracted
      contexts are embedded *whole* (no chunking) and compared by cosine
      similarity, with a bonus for symbols they literally share; a high score is
      a ``potential`` relation. This replaces the paper's character-level token
      overlap with a semantic comparison on the surrounding prose.

    The embedding model is used only as an encoder for cosine similarity, never
    for text generation.
    """

    def __init__(
        self,
        text: TextTools,
        similarity: "EmbeddingSimilarity",
        max_edges: int = 2,
        threshold: float = 0.9,
    ) -> None:
        self.text = text
        self.similarity = similarity
        self.max_edges = max_edges
        self.threshold = threshold

    def extract(self, equations: Dict[str, Dict], audits: Dict[str, AuditTrail]) -> Dict[str, Dict]:
        """Classify every ordered pair as strong, potential, or none.

        The relations dictionary for an equation contains an entry for *every*
        other equation in the paper, as the specification requires.
        """

        numbers = self._sort_numbers(list(equations))
        # Embed each equation's small extracted context once -- as a whole, not
        # chunked -- then compare contexts pairwise by cosine similarity.
        contexts = {number: self._relation_text(equations[number]) for number in numbers}
        semantic = self.similarity.pairwise(numbers, [contexts[number] for number in numbers])
        context_sentences = {number: self.text.sentences(contexts[number]) for number in numbers}
        bag_of_words = {number: self._content_lemmas(contexts[number]) for number in numbers}
        symbol_sets = {number: self._symbol_set(equations[number]) for number in numbers}

        out: Dict[str, Dict] = {}
        for left in numbers:
            scored: List[Tuple[float, str, str, str]] = []
            for right in numbers:
                if left == right:
                    continue
                grade, desc, score = self._classify(
                    left, right, context_sentences, semantic.get((left, right), 0.0),
                    bag_of_words, symbol_sets,
                )
                scored.append((score, right, grade, desc))

            potential = sorted([item for item in scored if item[2] == "potential"], reverse=True)
            keep_potential = {right for _, right, _, _ in potential[: self.max_edges]}
            n_strong = sum(1 for item in scored if item[2] == "strong")
            rels: Dict[str, Dict[str, str]] = {}
            for score, right, grade, desc in scored:
                final_grade, final_desc = grade, desc
                if grade == "potential" and right not in keep_potential:
                    final_grade, final_desc = "none", ""
                rels[right] = {"grade": final_grade, "description": final_desc}
                audits[left].add(
                    "relation",
                    f"({left})->({right}) {final_grade} score={score:.2f} {final_desc}".rstrip(),
                )
            out[left] = rels
            audits[left].add(
                "edge_limit",
                f"kept {n_strong} strong and at most {self.max_edges} potential edges",
            )
        return out

    def _classify(
        self,
        left: str,
        right: str,
        context_sentences: Dict[str, List[str]],
        semantic: float,
        bag_of_words: Dict[str, List[str]],
        symbol_sets: Dict[str, set],
    ) -> Tuple[str, str, float]:
        """Grade one ordered pair: brute-force reference first, else similarity."""

        # Strong: an explicit textual cross-reference in either context.
        phrase = (
            self._explicit_reference_phrase(context_sentences[left], right)
            or self._explicit_reference_phrase(context_sentences[right], left)
        )
        if phrase:
            return "strong", phrase, 1.0 + semantic

        # Potential: high context similarity, boosted by literally shared symbols.
        shared_symbols = symbol_sets[left] & symbol_sets[right]
        score = semantic + 0.05 * len(shared_symbols)
        if score >= self.threshold:
            return "potential", self._overlap_description(left, right, bag_of_words, shared_symbols), score
        return "none", "", semantic

    @staticmethod
    def _overlap_description(left: str, right: str, bag_of_words: Dict[str, List[str]], shared_symbols: set) -> str:
        """Describe a potential edge by the concepts/symbols the two contexts share."""

        right_set = set(bag_of_words[right])
        shared_terms = [word for word in bag_of_words[left] if word in right_set]
        if shared_terms:
            return "shared concepts: " + ", ".join(shared_terms[:3])
        if shared_symbols:
            return "shared symbols: " + ", ".join(sorted(shared_symbols)[:3])
        return "similar context"

    def _content_lemmas(self, text: str) -> List[str]:
        """Ordered, de-duplicated noun lemmas of a context (a simple bag of words)."""

        out: List[str] = []
        seen: set[str] = set()
        for token in self.text.doc(text):
            if token.pos_ not in {"NOUN", "PROPN"} or not token.is_alpha or token.is_stop:
                continue
            lemma = token.lemma_.lower()
            if len(lemma) >= 3 and lemma not in seen and lemma not in _META_NOUNS:
                seen.add(lemma)
                out.append(lemma)
        return out

    @staticmethod
    def _symbol_set(entry: Dict) -> set:
        """Base symbols of an equation (for the shared-symbol similarity bonus)."""

        return {str(symbol).split("_")[0] for symbol in entry.get("_raw_symbols", [])}

    def _explicit_reference_phrase(self, sentences: List[str], eq_num: str) -> str:
        for sentence in sentences:
            for match in self._reference_matches(sentence, eq_num):
                phrase = self._extract_relation_phrase(sentence, match, eq_num)
                if phrase:
                    return phrase
        return ""

    @staticmethod
    def _reference_matches(sentence: str, eq_num: str) -> List[re.Match]:
        patterns = [
            re.compile(rf"\beq(?:uation)?s?\.?\s*[\(\[\s]?\s*{re.escape(eq_num)}\s*[\)\]\s]?", re.IGNORECASE),
            re.compile(rf"[\(\[]\s*{re.escape(eq_num)}\s*[\)\]]", re.IGNORECASE),
        ]
        out: List[re.Match] = []
        seen: set[Tuple[int, int]] = set()
        for pattern in patterns:
            for match in pattern.finditer(sentence):
                span = match.span()
                if span not in seen:
                    seen.add(span)
                    out.append(match)
        return sorted(out, key=lambda item: item.start())

    def _extract_relation_phrase(self, sentence: str, match: re.Match, eq_num: str) -> str:
        doc = self.text.raw_doc(sentence)
        start, end = match.span()
        reference_tokens = self._tokens_overlapping(doc, start, end)
        anchor = self._reference_anchor(reference_tokens, eq_num)
        verb = self._governing_verb(anchor) if anchor is not None else None
        if verb is None:
            verb = self._nearest_verb(doc, start)
        if verb is not None:
            phrase = self._verb_phrase(verb, start, end)
            if phrase:
                return phrase
        # An explicit citation is always a strong relation; if no relation verb
        # or clean nearby phrase is found, fall back to a generic description
        # rather than an uninformative lemma like "be" or "hat".
        return self._near_reference_phrase(doc, start, end) or "directly referenced"

    @staticmethod
    def _tokens_overlapping(doc, start: int, end: int) -> List:
        return [token for token in doc if token.idx < end and token.idx + len(token.text) > start]

    @staticmethod
    def _reference_anchor(tokens: List, eq_num: str):
        for token in tokens:
            if token.text.strip("()[] .") == eq_num:
                return token
        return tokens[-1] if tokens else None

    @staticmethod
    def _governing_verb(token):
        current = token
        seen = set()
        while current is not None and current.i not in seen:
            seen.add(current.i)
            if current.pos_ in {"VERB", "AUX"}:
                return current
            if current.head is current:
                break
            current = current.head
        return None

    @staticmethod
    def _nearest_verb(doc, index: int):
        verbs = [token for token in doc if token.pos_ in {"VERB", "AUX"}]
        if not verbs:
            return None
        return min(verbs, key=lambda token: min(abs(token.idx - index), abs(token.idx + len(token.text) - index)))

    def _verb_phrase(self, verb, start: int, end: int) -> str:
        """Relation description lifted verbatim from the connecting verb.

        The verb that governs the cross-reference is the relation cue ("given by",
        "reduces to", "derived from"). A bare auxiliary ("is", "are") carries no
        relational meaning and is rejected. An attached preposition/particle is
        appended so the cue reads naturally -- all taken from the text, with no
        hand-written verb mapping.
        """

        if verb.pos_ == "AUX":
            return ""
        if not verb.lemma_.isalpha() or len(verb.lemma_) < 3:
            return ""
        parts = [verb.text]
        for child in verb.children:
            if child.dep_ in {"prep", "prt"} and child.idx > verb.idx:
                parts.append(child.text)
                break
        return " ".join(parts).lower()

    def _near_reference_phrase(self, doc, start: int, end: int) -> str:
        reference_tokens = self._tokens_overlapping(doc, start, end)
        if not reference_tokens:
            return ""
        first = max(reference_tokens[0].i - 3, 0)
        last = min(reference_tokens[-1].i + 4, len(doc))
        tokens = self._description_tokens(doc[first:last], start, end)
        return self._clean_description(" ".join(token.text for token in tokens[:8]))

    @staticmethod
    def _description_tokens(tokens: Iterable, start: int, end: int) -> List:
        out = []
        seen = set()
        for token in sorted(tokens, key=lambda item: item.i):
            if token.i in seen:
                continue
            seen.add(token.i)
            overlaps_reference = token.idx < end and token.idx + len(token.text) > start
            if overlaps_reference or token.is_space or token.is_punct or token.like_num:
                continue
            out.append(token)
        return out

    @staticmethod
    def _clean_description(text: str) -> str:
        text = text.replace("�", " ")
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\beq(?:uation)?s?\.?\s*[\(\[\s]?\s*[A-Za-z0-9.\-]+\s*[\)\]\s]?", " ", text, flags=re.IGNORECASE)
        # Keep only real word tokens; drop single letters, digits and math glyphs
        # so a citation snippet like "Ĉ a i i" collapses to nothing.
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        while words and words[0].lower() in {"the", "a", "an", "this", "these", "those", "that", "is", "are", "of"}:
            words.pop(0)
        return " ".join(words[:8])

    @staticmethod
    def _relation_text(entry: Dict) -> str:
        parts = [
            entry.get("meaning", ""),
            entry.get("_before", ""),
            entry.get("_after", ""),
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _sort_numbers(numbers: List[str]) -> List[str]:
        def key(value: str):
            return (0, int(value), "") if value.isdigit() else (1, 0, value)

        return sorted(numbers, key=key)
