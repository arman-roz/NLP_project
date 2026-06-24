"""Meaning, symbol and relation extraction with spaCy + BM25 (no embeddings).

Every string is lifted from the paper, never generated. Decisions are made from
the dependency parse, POS tags and ``unicodedata`` rather than physics word
lists, so the approach carries over to papers we have not seen. The only fixed
vocabulary is grammatical (dependency/POS labels) and a few LaTeX command names.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List, Optional, Tuple

import spacy

from .common import short

# Grammar labels and LaTeX leftovers only -- never domain vocabulary.
_CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "csubj", "parataxis"}
_EDGE_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "AUX", "PRON"}
_FINITE_VERBS = {"VBZ", "VBP", "VBD", "VB", "MD"}
_NAME_PREPS = {"of", "for", "in", "on", "between", "across", "within", "over", "per"}
_DEF_PREPS = {"of", "for"}
_MARKUP = {"rm", "mathrm", "text", "mathcal", "mathbb", "hat", "bar", "vec", "dot",
           "operatorname", "frac", "sqrt", "left", "right", "displaystyle", "partial"}


@lru_cache(maxsize=1)
def _nlp():
    """Load the small English spaCy model (POS tagger + dependency parser)."""
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError("missing model: python -m spacy download en_core_web_sm")
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=4096)
def _is_greek(word: str) -> bool:
    """True if ``word`` spells out a Greek letter (eta, phi, ...)."""
    for tmpl in ("GREEK SMALL LETTER {}", "GREEK CAPITAL LETTER {}"):
        try:
            unicodedata.lookup(tmpl.format(word.upper()))
            return True
        except KeyError:
            pass
    return False


@lru_cache(maxsize=4096)
def _variants(base: str) -> frozenset:
    """A symbol's name plus its Unicode Greek forms, so ``eta`` also matches η."""
    out = {base}
    for tmpl in ("GREEK SMALL LETTER {}", "GREEK CAPITAL LETTER {}", "GREEK {} SYMBOL"):
        try:
            out.add(unicodedata.lookup(tmpl.format(base.upper())))
        except KeyError:
            pass
    return frozenset(out)


def _is_word(tok) -> bool:
    """True if a token is a real word, not a one-letter symbol, glyph or markup."""
    w = tok.text
    if len(w) < 2 or not w.isascii() or not re.search(r"[A-Za-z]", w):
        return False
    return w.lower() not in _MARKUP and not _is_greek(w.lower())


def _noun_chunks(doc) -> List:
    try:
        return list(doc.noun_chunks)
    except ValueError:
        return []


def _np_span(head, preps):
    """Contiguous noun-phrase span around ``head`` (modifiers and ``preps``)."""
    keep = {head.i}

    def walk(tok):
        for c in tok.children:
            if c.dep_ in _CLAUSE_DEPS or c.dep_ in {"punct", "cc", "conj", "appos"}:
                continue
            if c.dep_ == "prep" and c.text.lower() not in preps:
                continue
            keep.add(c.i)
            walk(c)

    walk(head)
    start = end = head.i
    while start - 1 in keep:
        start -= 1
    while end + 1 in keep:
        end += 1
    return head.doc[start:end + 1]


def _phrase(head, stop, preps=_DEF_PREPS, limit=8) -> str:
    """Short noun-phrase string headed by ``head`` ('' if it is not noun-headed)."""
    if head is None or head.pos_ not in {"NOUN", "PROPN"}:
        return ""
    toks = [t for t in _np_span(head, preps) if not t.is_space and _is_word(t)]
    while toks and (toks[0].pos_ in _EDGE_POS or toks[0].lower_ in stop):
        toks.pop(0)
    while toks and (toks[-1].pos_ in _EDGE_POS or toks[-1].lower_ in stop):
        toks.pop()
    words = []
    for t in toks[:limit]:
        if not words or words[-1].lower() != t.text.lower():
            words.append(t.text)
    return " ".join(words)


def _is_name(toks) -> bool:
    """A name has a noun and is not a clause (a participle modifier is allowed)."""
    words = [t for t in toks if t.is_alpha]
    if not words or not any(t.pos_ in {"NOUN", "PROPN"} for t in words):
        return False
    return not any(t.tag_ in _FINITE_VERBS and t.dep_ not in
                   {"amod", "compound", "nmod", "npadvmod"} for t in words)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class TextTools:
    """spaCy-backed sentence splitting, cleaning and parsing."""

    def __init__(self) -> None:
        self.nlp = _nlp()
        self.sent = spacy.blank("en")
        self.sent.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", ";", ":", "…"]})
        self.stop_words = set(self.nlp.Defaults.stop_words)

    def sentences(self, text: str) -> List[str]:
        """Clean, reasonably long sentences from a block of paper text."""
        text = self.clean(text)
        out = []
        for span in (self.sent(text).sents if text else []):
            s = span.text.strip()
            if len(re.findall(r"[A-Za-z]{2,}", s)) >= 4 and 20 <= len(s) <= 500:
                out.append(s)
        return out

    def doc(self, text: str):
        return self.nlp(self.clean(text))

    def raw_doc(self, text: str):
        return self.nlp(text.replace("\xa0", " "))

    @staticmethod
    def clean(text: str) -> str:
        """Normalise text while keeping LaTeX command names as readable words."""
        text = re.sub(r"\\([A-Za-z]+)", r" \1 ", text.replace("\xa0", " "))
        return re.sub(r"\s+", " ", re.sub(r"[{}_^$]", " ", text)).strip()


class MeaningExtractor:
    """Name each equation with a short descriptive phrase taken from the prose.

    The name is the noun phrase filling the clause's naming role -- the subject of
    a passive/copular introduction ("X is given by"), the object of an active one
    ("we define X") -- or an explicit "called X". Roles come from the parse, so no
    verb or noun list is used.
    """

    _CALLED = re.compile(
        r"(?:called|known as|termed|named|dubbed|referred to as)\s+(?:the |an? )?(.+?)(?:[,.;:()\[\]]|$)",
        re.I,
    )

    def __init__(self, text: TextTools) -> None:
        self.text = text

    def extract(self, eq_num, before, after, audit, used=None,
                retriever=None, paper_sentences=None) -> str:
        """Return a short name for the equation, or ``""`` if none is found."""
        used = used if used is not None else set()
        sentences = self._candidate_sentences(eq_num, before, after, retriever)

        for sentence in sentences:                       # explicit "called X"
            match = self._CALLED.search(self.text.clean(sentence))
            name = self._tidy(match.group(1)) if match else ""
            if name:
                name = self._expand(name, paper_sentences)
                used.add(name.split()[-1].lower())
                audit.add("meaning", f"named: {name}")
                return name

        candidates = self._phrases(sentences)            # naming-role phrase
        chosen = next((c for c in candidates if c[0].split()[-1].lower() not in used),
                      candidates[0] if candidates else None)
        if chosen is None:
            audit.add("meaning", "no candidate")
            return ""
        phrase = self._expand(chosen[0], paper_sentences)
        used.add(phrase.split()[-1].lower())
        audit.add("meaning", f"{phrase} <= {short(chosen[1])}")
        return phrase

    def _candidate_sentences(self, eq_num, before, after, retriever) -> List[str]:
        """Introducing sentences nearest the equation, then any that cite it."""
        out, seen = [], set()

        def add(sentence):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                out.append(sentence)

        before_s, after_s = self.text.sentences(before), self.text.sentences(after)
        for sentence in before_s[-2:][::-1]:
            add(sentence)
        for sentence in after_s[:2]:
            add(sentence)
        if retriever is not None:
            query = " ".join(p for p in [f"equation {eq_num}", before, after] if p)
            for i, _ in retriever.retrieve(query, k=4):
                if self._cites(eq_num, retriever.sentences[i]):
                    add(retriever.sentences[i])
        return out

    def _phrases(self, sentences) -> List[Tuple[str, str]]:
        """(phrase, sentence) candidates, naming-role heads before plain chunks."""
        out, seen = [], set()
        for sentence in sentences:
            doc = self._parse(sentence)
            heads = self._naming_heads(doc) + [c.root for c in _noun_chunks(doc)]
            for head in heads:
                if head.pos_ not in {"NOUN", "PROPN"}:
                    continue
                toks = [t for t in _np_span(head, _NAME_PREPS) if not t.is_space and _is_word(t)]
                phrase = _phrase(head, self.text.stop_words, _NAME_PREPS, limit=12)
                if phrase and phrase.lower() not in seen and _is_name(toks):
                    seen.add(phrase.lower())
                    out.append((phrase, sentence))
        return out

    @staticmethod
    def _naming_heads(doc) -> List:
        """Clause heads that name the equation, chosen from dependency roles only."""

        def noun(t):
            return t is not None and t.pos_ in {"NOUN", "PROPN"}

        def child(t, deps):
            return next((c for c in t.children if c.dep_ in deps), None) if t else None

        preds = [t for t in doc if t.dep_ == "ROOT"]
        preds += [c for p in list(preds) for c in p.children
                  if c.dep_ == "conj" and c.pos_ in {"VERB", "AUX"}]
        heads = []
        for pred in preds:
            subj, attr = child(pred, {"nsubj", "nsubjpass"}), child(pred, {"attr", "oprd"})
            dobj, comp = child(pred, {"dobj"}), child(pred, {"ccomp", "xcomp"})
            passive = any(c.dep_ == "auxpass" for c in pred.children)
            order = [subj, attr, dobj] if (passive or attr) else [dobj, attr, subj]
            if not noun(subj) and comp is not None:       # "it is shown that <clause>"
                order = [comp if noun(comp) else child(comp, {"nsubj", "nsubjpass", "dobj", "attr"})] + order
            heads += [t for t in order if noun(t) and t not in heads]
        return heads

    def _parse(self, sentence):
        """Parse a sentence with inline math removed so the prose parses cleanly."""
        s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", sentence)
        s = re.sub(r"(?<![A-Za-z])[A-Za-z]\s+[A-Z0-9]{2,}(?![A-Za-z])", " ", s)   # "I OFF"
        s = re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])|[^\x00-\x7f]+", " ", s)    # lone letters, glyphs
        return self.text.doc(s)

    def _expand(self, name, paper_sentences) -> str:
        """Expand an all-caps acronym to the long form the paper spells out."""
        if not re.fullmatch(r"[A-Z]{2,6}", name):
            return name
        pattern = re.compile(r"((?:[A-Za-z][\w'\-]*\s+){1,9})\(\s*" + re.escape(name) + r"\s*\)")
        skip = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with"}
        target = name.lower()
        for sentence in paper_sentences or []:
            for match in pattern.finditer(sentence):
                words = [w for w in match.group(1).split() if w[:1].isalpha() and w.lower() not in skip]
                initials = [w[0].lower() for w in words]
                for i in range(len(words) - len(target) + 1):
                    if "".join(initials[i:i + len(target)]) == target:
                        return " ".join(words[i:i + len(target)])
        return name

    def _tidy(self, text) -> str:
        """Reduce a captured 'called X' string to a clean noun phrase."""
        text = re.sub(r"\\[A-Za-z]+|[{}_^$\\]|[^\x00-\x7f]+", " ", text)
        toks = [t for t in self.text.doc(text) if not t.is_space and _is_word(t)]
        while toks and toks[0].pos_ in _EDGE_POS:
            toks.pop(0)
        while toks and toks[-1].pos_ in _EDGE_POS:
            toks.pop()
        return " ".join(t.text for t in toks[:8]) if _is_name(toks) else ""

    @staticmethod
    def _cites(eq_num, sentence) -> bool:
        return bool(re.search(rf"(?:eq(?:uation)?s?\.?\s*)?\(\s*{re.escape(eq_num)}\s*\)", sentence, re.I))


class SymbolExtractor:
    """Define each MathML symbol from a paper sentence, or omit it.

    A definition is the noun the symbol stands in a definitional relation with --
    the complement when the symbol is a subject ("eta is the EFFICIENCY"), or the
    noun it modifies ("the EFFICIENCY eta"). The local window is searched first;
    only distinctive symbols (not bare single letters) fall back to BM25 over the
    whole paper, since a lone letter matches unrelated text. Symbols without
    textual support are omitted, never guessed.
    """

    _OPERATORS = {"d", "delta", "Delta", "partial", "nabla", "mathrm", "rm", "mathcal"}
    _SPLIT = re.compile(r"[,;:]|\b(?:where|with|which|wherein)\b", re.I)

    def __init__(self, text: TextTools) -> None:
        self.text = text

    def extract(self, mathml_symbols, local_sentences, retriever, audit):
        """Return ``{symbol: definition}`` for paper-supported symbols only."""
        symbols = self._symbols(mathml_symbols)
        audit.add("symbol_candidates", ", ".join(symbols) or "none")
        definitions, used = {}, set()
        for symbol in symbols:
            found = self._define(symbol, local_sentences, retriever)
            if not found:
                audit.add("symbol_definition", f"{symbol}: not found")
                continue
            phrase, evidence = found
            if len(symbol.split("_")[0]) == 1 and phrase.lower() in used:
                audit.add("symbol_definition", f"{symbol}: duplicate skipped")
                continue
            definitions[symbol] = phrase
            used.add(phrase.lower())
            audit.add("symbol_definition", f"{symbol}: {phrase} | {short(evidence)}")
        return definitions, symbols

    def _define(self, symbol, local_sentences, retriever):
        """First valid definition: local window, then BM25 for specific symbols."""
        base, _, sub = symbol.partition("_")
        variants = _variants(base)
        mention = re.compile(rf"(?<![A-Za-z])(?:{'|'.join(re.escape(v) for v in variants)})(?![A-Za-z])")
        sub_re = re.compile(rf"(?<![A-Za-z]){re.escape(sub)}(?![A-Za-z])") if sub and sub.isalpha() else None
        found = self._scan(local_sentences, variants, mention, sub_re)
        if not found and retriever is not None and (len(base) >= 2 or sub_re is not None):
            allowed = set(retriever.sentences_matching(lambda s: bool(mention.search(s))))
            ranked = [retriever.sentences[i] for i, _ in retriever.retrieve(base, k=8, allowed=allowed)]
            found = self._scan(ranked, variants, mention, sub_re)
        return found

    def _scan(self, sentences, variants, mention, sub_re):
        """First (phrase, sentence) whose clause mentions and defines the symbol."""
        for sentence in sentences:
            stripped = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", sentence)
            for clause in self._SPLIT.split(stripped):
                clause = clause.strip()
                if not clause or not mention.search(clause) or (sub_re and not sub_re.search(clause)):
                    continue
                for phrase in self._candidates(clause, variants):
                    return phrase, sentence
        return None

    def _candidates(self, clause, variants) -> List[str]:
        """Definition phrases for the symbol, definitional relation before nearest."""
        doc = self.text.raw_doc(clause)
        low = {v.lower() for v in variants}
        marks = [t for t in doc if t.text.lower() in low]
        if not marks:
            return []
        chunks = _noun_chunks(doc)
        out = []
        for tok in marks:
            after = sorted((c for c in chunks if c.start > tok.i), key=lambda c: c.start)
            before = sorted((c for c in chunks if c.end <= tok.i), key=lambda c: -c.end)
            for head in [self._def_head(tok)] + [c.root for c in after + before][:5]:
                if head is None or head.pos_ == "PRON":
                    continue
                phrase = _phrase(head, self.text.stop_words)
                if phrase and self._valid(phrase):
                    out.append(phrase)
                    break
        return out

    @staticmethod
    def _def_head(tok):
        """Defining noun for a symbol, from its dependency relation (or None)."""
        head = tok.head
        if head is tok:
            return None
        if tok.dep_ in {"nsubj", "nsubjpass"} and head.pos_ in {"VERB", "AUX"}:
            comp = next((c for c in head.children if c.dep_ in {"attr", "acomp", "dobj", "oprd"}), None)
            return comp if comp is not None and comp.pos_ in {"NOUN", "PROPN"} else None
        if head.pos_ in {"NOUN", "PROPN"} and tok.dep_ in \
                {"appos", "nmod", "compound", "conj", "dep", "npadvmod", "nummod", "flat"}:
            return head
        return None

    def _valid(self, phrase) -> bool:
        """A definition is a noun phrase, not a clause and not adverb-initial."""
        toks = [t for t in self.text.doc(phrase) if t.is_alpha]
        return (bool(toks) and toks[0].pos_ != "ADV"
                and any(t.pos_ in {"NOUN", "PROPN"} for t in toks)
                and not any(t.tag_ in _FINITE_VERBS or t.pos_ == "AUX" for t in toks))

    def _symbols(self, mathml_symbols) -> List[str]:
        """Usable MathML identifiers (operators and non-identifiers dropped)."""
        out, seen = [], set()
        for raw in mathml_symbols:
            symbol = raw.strip()
            base, _, sub = symbol.partition("_")
            if (symbol and symbol not in seen and base not in self._OPERATORS
                    and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", base)
                    and (not sub or re.fullmatch(r"[A-Za-z0-9]+", sub))):
                seen.add(symbol)
                out.append(symbol)
        return out[:22]


class RelationExtractor:
    """Grade every ordered equation pair none/potential/strong with a [0,1] score.

    ``strong`` (score 1) -- one equation's context explicitly cites the other's
    number; ``potential`` -- enough shared symbols and context nouns, scored as an
    even blend of two Jaccard overlaps so the value stays in [0,1]; the top
    ``max_edges`` potential partners per equation are kept, the rest ``none``.
    """

    def __init__(self, text: TextTools, max_edges: int = 2, threshold: float = 0.12) -> None:
        self.text = text
        self.max_edges = max_edges
        self.threshold = threshold

    def extract(self, equations, audits):
        """Classify every ordered pair; emit an entry for every other equation."""
        nums = sorted(equations, key=lambda v: (0, int(v), "") if v.isdigit() else (1, 0, v))
        sentences = {n: self.text.sentences(self._context(equations[n])) for n in nums}
        nouns = {n: self._nouns(self._context(equations[n])) for n in nums}
        symbols = {n: {str(s).split("_")[0] for s in equations[n].get("_raw_symbols", [])} for n in nums}

        out = {}
        for left in nums:
            scored = [self._pair(left, right, sentences, nouns, symbols) for right in nums if right != left]
            keep = {r for _, r, g, _ in sorted((x for x in scored if x[2] == "potential"), reverse=True)[:self.max_edges]}
            rels = {}
            for score, right, grade, desc in scored:
                if grade == "potential" and right not in keep:
                    grade, desc = "none", ""
                rels[right] = {"grade": grade, "description": desc}
                audits[left].add("relation", f"({left})->({right}) {grade} {score:.2f} {desc}".rstrip())
            out[left] = rels
        return out

    def _pair(self, left, right, sentences, nouns, symbols):
        """One ordered pair: explicit citation first, else a shared-content score."""
        desc = self._reference(sentences[left], right) or self._reference(sentences[right], left)
        if desc:
            return 1.0, right, "strong", desc
        score = 0.5 * _jaccard(set(nouns[left]), set(nouns[right])) + 0.5 * _jaccard(symbols[left], symbols[right])
        if score >= self.threshold:
            shared = [w for w in nouns[left] if w in set(nouns[right])][:3] or sorted(symbols[left] & symbols[right])[:3]
            return score, right, "potential", "shares " + ", ".join(shared) if shared else "similar context"
        return score, right, "none", ""

    def _reference(self, sentences, num) -> str:
        """Connecting verb lifted from a sentence that cites equation ``num``."""
        pattern = re.compile(
            rf"\beq(?:uation)?s?\.?\s*[\(\[]?\s*{re.escape(num)}\s*[\)\]]?|\(\s*{re.escape(num)}\s*\)", re.I)
        for sentence in sentences:
            match = pattern.search(sentence)
            if not match:
                continue
            verbs = [t for t in self.text.raw_doc(sentence)
                     if t.pos_ == "VERB" and t.lemma_.isalpha() and len(t.lemma_) > 2]
            if not verbs:
                return "directly referenced"
            verb = min(verbs, key=lambda t: abs(t.idx - match.start()))
            particle = next((c.text for c in verb.children if c.dep_ in {"prep", "prt"} and c.idx > verb.idx), "")
            return (verb.text + (" " + particle if particle else "")).lower()
        return ""

    @staticmethod
    def _context(entry) -> str:
        return " ".join(p for p in [entry.get("meaning", ""), entry.get("_before", ""), entry.get("_after", "")] if p)

    def _nouns(self, text) -> List[str]:
        """Ordered, de-duplicated content-noun lemmas of a context."""
        out, seen = [], set()
        for tok in self.text.doc(text):
            if tok.pos_ in {"NOUN", "PROPN"} and tok.is_alpha and not tok.is_stop:
                lemma = tok.lemma_.lower()
                if len(lemma) >= 3 and lemma not in seen:
                    seen.add(lemma)
                    out.append(lemma)
        return out
