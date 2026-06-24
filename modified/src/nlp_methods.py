"""NLP methods for equation meanings, symbol definitions, and relations.

All textual output is *extracted* from the arXiv paper -- never generated. The
transformer (MathBERT) is used only as an encoder for cosine similarity. Symbol
candidates come from the equation's MathML structure; meanings and symbol
definitions are noun phrases lifted from the surrounding prose using the spaCy
parse, POS tags and ``unicodedata`` rather than any hand-written physics
vocabulary, so the method generalises to unseen quantum-physics terminology.
"""

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
    """Load the small English spaCy model (POS tagger + dependency parser)."""

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "spaCy model en_core_web_sm is required. "
            "Install it with: python -m spacy download en_core_web_sm"
        )
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Domain-independent constants. The only word lists kept are *markup* (LaTeX
# command names) and structural "meta" nouns -- never physics vocabulary -- so
# the extractors do not over-fit to terms seen in these particular papers.
# ---------------------------------------------------------------------------

# Dependency labels of children that begin a *new clause* (relative/adverbial/
# complement clauses) and so must not be pulled into a noun phrase.
_CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "csubj", "parataxis"}
# Prepositions that introduce a genuine post-nominal complement worth keeping
# ("degree of coherence", "density of states"). Others tend to start an adjunct.
_KEEP_PREPS = {"of", "for"}
# Edge POS tags trimmed off the start/end of a phrase (articles, prepositions,
# conjunctions, particles, auxiliaries, pronouns, punctuation).
_EDGE_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "AUX", "PRON"}
# Structural nouns that name a piece of writing rather than a physics concept.
# Rejecting one as a *lone* name/definition is domain-independent and safe.
_META_NOUNS = {
    "equation", "result", "form", "case", "value", "term", "expression",
    "quantity", "parameter", "function", "system", "example", "section",
    "figure", "table", "paper", "method", "approach", "number", "order",
    "part", "side", "way", "set", "thing", "one",
}
# LaTeX/MathML command leftovers (markup, not domain vocabulary). Greek letter
# names are handled separately via ``unicodedata``.
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

    Uses ``unicodedata`` to ask whether a Greek code point with this name exists
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


@lru_cache(maxsize=4096)
def _surface_variants(base: str) -> frozenset:
    """Surface forms of a symbol: its name plus Unicode Greek equivalents.

    Uses ``unicodedata`` (a library, not a hand-written table) so a MathML name
    like ``eta`` also matches the Unicode ``η``/``Η`` written in the prose.
    """

    variants = {base}
    upper = base.upper()
    for template in ("GREEK SMALL LETTER {}", "GREEK CAPITAL LETTER {}", "GREEK {} SYMBOL"):
        try:
            variants.add(unicodedata.lookup(template.format(upper)))
        except KeyError:
            pass
    return frozenset(v for v in variants if v)


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

    Walks the head noun's subtree, keeping determiners, adjectival, compound,
    numeric and possessive modifiers plus ``of``/``for`` complements, but
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


def _phrase_from_text(text: str, tools: "TextTools") -> str:
    """Reduce a raw captured string to a compact, math-free phrase (<=8 words).

    Strips LaTeX/markup and bracketed material, re-parses the remainder, then
    keeps content words with POS-based edge trimming. Shared by the meaning and
    symbol extractors so the same cleaning rule applies everywhere.
    """

    text = text.replace("�", " ")
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}_^$\\]", " ", text)
    tokens = _phrase_tokens(list(tools.doc(text)), tools.stop_words)
    return " ".join(token.text for token in tokens[:8])


class TextTools:
    """Sentence, token, POS, dependency, and lemma helpers."""

    def __init__(self) -> None:
        self.nlp = _load_spacy()
        # A blank pipeline with a sentencizer that also breaks on ":" and ";"
        # so an introducing clause like "... is given by:" is isolated from the
        # following sentence (important for picking the right meaning).
        self.sent_nlp = spacy.blank("en")
        self.sent_nlp.add_pipe(
            "sentencizer", config={"punct_chars": [".", "!", "?", ";", ":", "…"]}
        )
        self.stop_words = set(self.nlp.Defaults.stop_words)

    def sentences(self, text: str) -> List[str]:
        """Return clean, sufficiently long sentence strings."""

        cleaned = self.clean(text)
        if not cleaned:
            return []
        doc = self.sent_nlp(cleaned)
        return [s.text.strip() for s in doc.sents if self._good_sentence(s.text)]

    def tokens(self, text: str, keep: Iterable[str] = ()) -> List[str]:
        """Return normalised non-stopword lemmas (a simple bag of words)."""

        keep_set = {item.lower() for item in keep}
        out: List[str] = []
        for token in self.nlp(self.clean(text)):
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
        """Return a spaCy document over TeX-cleaned text."""

        return self.nlp(self.clean(text))

    def raw_doc(self, text: str):
        """Return a spaCy document without TeX cleanup (keeps char offsets)."""

        return self.nlp(text.replace("\xa0", " "))

    @staticmethod
    def clean(text: str) -> str:
        """Normalise paper text while preserving LaTeX command names as words."""

        text = text.replace("\xa0", " ")
        text = re.sub(r"\\([A-Za-z]+)", r" \1 ", text)
        text = re.sub(r"[{}_^$]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _good_sentence(sentence: str) -> bool:
        words = re.findall(r"[A-Za-z]{2,}", sentence)
        return len(words) >= 4 and 20 <= len(sentence) <= 500


class EmbeddingSimilarity:
    """Transformer encoder used only for similarity, never for generation.

    Uses ``AutoModel`` with mean pooling and L2 normalisation to produce
    sentence embeddings, then compares them by cosine (dot product).
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalised mean-pooled vectors."""

        if not texts:
            return np.zeros((0, 1), dtype=float)
        model, tokenizer = self._load()
        import torch

        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state
            summed = torch.sum(token_embeddings * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            embeddings = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return embeddings.cpu().numpy()

    def pairwise(self, labels: List[str], texts: List[str]) -> Dict[Tuple[str, str], float]:
        """Cosine similarity for every ordered pair of labelled texts."""

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
        """Cosine similarity of each candidate to the query."""

        if not candidates:
            return []
        vectors = self.encode([query] + candidates)
        query_vec = vectors[0]
        return [float(np.dot(query_vec, vec)) for vec in vectors[1:]]

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir)
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir)
            )
            self._model.eval()
        return self._model, self._tokenizer


class MeaningExtractor:
    """Extracts a short, precise *name* for each equation.

    The ``meaning`` field is a concise description / name of what the equation
    expresses (e.g. "wave function", "threshold power"), not a whole sentence.
    The name is always lifted verbatim from the paper: an explicit "called/known
    as X" or "X equation/law" label, or the noun phrase the introducing clause
    describes. An acronym (e.g. "CMI") is expanded to the long form the paper
    itself spells out ("conditional mutual information (CMI)").
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
        paper_sentences: Optional[List[str]] = None,
    ) -> str:
        """Return a short name/description for the equation.

        Parameters
        ----------
        eq_num : str
            Equation number (used to find sentences citing this equation).
        before, after : str
            Prose context immediately before/after the equation.
        audit : AuditTrail
            Trail recording which strategy and sentence produced the name.
        equation_symbols : list of str, optional
            The equation's symbols; sentences that mention a *distinctive* one
            (a named/Greek symbol) are prioritised as the introducing clause.
        used : set of str, optional
            Names already taken by earlier equations; a different name is
            preferred so a paper's meanings do not all collapse onto one.
        paper_sentences : list of str, optional
            Whole-paper sentences, used only to expand an acronym to its long
            form when the paper defines one.

        Returns
        -------
        str
            A concise extracted name, or ``""`` if none is found nearby.
        """

        candidates = self._ordered_candidates(eq_num, before, after, equation_symbols)
        if not candidates:
            audit.add("meaning", "no local prose sentence")
            return ""
        local_context = f"{before} {after}".strip()
        used = used or set()

        name = self._select_name(eq_num, candidates, local_context, used, audit)
        if not name:
            audit.add("meaning", f"no name found in {len(candidates)} sentences")
            return ""
        expanded = self._expand_acronym(name, paper_sentences)
        if expanded != name:
            audit.add("meaning", f"expanded acronym {name} -> {expanded}")
        return expanded

    def _select_name(self, eq_num, candidates, local_context, used, audit) -> str:
        """Run the three naming tiers and return the chosen name (or "")."""

        # Tier 1: an explicit "called/known as X" or "X equation/law" label
        # (the latter only when the sentence is introducing this equation).
        for sentence, _ in candidates:
            phrase = self._name_from_called(sentence)
            if not phrase and self._is_intro_context(eq_num, sentence):
                phrase = self._named_equation(sentence)
            if phrase:
                audit.add("meaning", f"named: {phrase} <= {short(sentence)}")
                return phrase

        # Tier 2: the noun phrase of the clause nearest the equation (subject vs.
        # object chosen from the parse). Proximity-first and deterministic; soft
        # de-duplication prefers a name an earlier equation has not taken.
        clause_pairs = [
            (self._name_from_clause(sentence, prefer_last), sentence)
            for sentence, prefer_last in candidates
        ]
        clause_pairs = [(p, s) for p, s in clause_pairs if p]
        for phrase, sentence in clause_pairs:
            if phrase not in used:
                audit.add("meaning", f"clause: {phrase} <= {short(sentence)}")
                return phrase
        if clause_pairs:
            audit.add("meaning", f"clause(reused): {clause_pairs[0][0]} <= {short(clause_pairs[0][1])}")
            return clause_pairs[0][0]

        # Tier 3: no introducing clause -- rank nearby noun phrases by embedding
        # similarity to the local context (the genuinely ambiguous case).
        best = self._rank_best(self._fallback_candidates(candidates), local_context, used)
        if best:
            audit.add("meaning", f"fallback: {best[0]} <= {short(best[1])}")
            return best[0]
        return ""

    def _expand_acronym(self, name: str, paper_sentences: Optional[List[str]]) -> str:
        """Expand a short acronym name to the long form the paper defines.

        Looks for "<long form> (ACRONYM)" anywhere in the paper and, if the long
        form parses to a valid multi-word noun phrase, returns it. Keeps the
        output "from arXiv only" -- the expansion is the paper's own wording.
        """

        if not (name.isupper() and re.fullmatch(r"[A-Z]{2,6}", name)):
            return name
        # Capture the words just before "(ACRONYM)" and keep the run whose
        # initials spell the acronym ("conditional mutual information" for CMI),
        # which strips introducing verbs/articles the surrounding clause adds.
        pattern = re.compile(r"((?:[A-Za-z][\w'\-]*\s+){1,9})\(\s*" + re.escape(name) + r"\s*\)")
        for sentence in paper_sentences or []:
            for match in pattern.finditer(sentence):
                phrase = self._long_form_by_initials(match.group(1).split(), name.lower())
                if phrase and self._valid_phrase(phrase):
                    return phrase
        return name

    @staticmethod
    def _long_form_by_initials(words: List[str], letters: str) -> str:
        """Return the contiguous content-word run whose initials spell ``letters``.

        Articles/prepositions are skipped (acronyms omit them); the run nearest
        the acronym is preferred. Returns "" if no run matches.
        """

        skip = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with"}
        kept = [w for w in words if w[:1].isalpha() and w.lower() not in skip]
        initials = [w[0].lower() for w in kept]
        match = ""
        for start in range(len(kept) - len(letters) + 1):
            if "".join(initials[start:start + len(letters)]) == letters:
                match = " ".join(kept[start:start + len(letters)])  # keep rightmost
        return match

    def _rank_best(self, pairs: List[tuple], context: str, used: set) -> Optional[tuple]:
        """Pick the (phrase, sentence) whose phrase best fits the local context."""

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
        order = sorted(range(len(unique)), key=lambda i: (round(scores[i], 2), -i), reverse=True)
        for index in order:
            if unique[index][0] not in used:
                return unique[index]
        return unique[order[0]]

    def _ordered_candidates(
        self, eq_num: str, before: str, after: str, symbols: Optional[List[str]]
    ) -> List[tuple]:
        """Order context sentences by how likely they name the equation.

        Returns ``(sentence, prefer_last)`` pairs: the sentence immediately
        before the equation first (usually "... is given by:"), then the one
        after, then sentences citing the equation number or mentioning one of
        its distinctive symbols, then the rest. ``prefer_last`` is True for
        ``before`` sentences (the equation follows their last clause).
        """

        before_sents = self.text.sentences(before)
        after_sents = self.text.sentences(after)
        surfaces = self._distinctive_surfaces(symbols)

        ordered: List[tuple] = []
        if before_sents:
            ordered.append((before_sents[-1], True))
        if after_sents:
            ordered.append((after_sents[0], False))
        ordered.extend((s, True) for s in before_sents if self._cites(eq_num, s))
        ordered.extend((s, False) for s in after_sents if self._cites(eq_num, s))
        ordered.extend((s, True) for s in before_sents if self._mentions(s, surfaces))
        ordered.extend((s, False) for s in after_sents if self._mentions(s, surfaces))
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
    def _distinctive_surfaces(symbols: Optional[List[str]]) -> set:
        """Lower-cased surface forms of named/Greek symbols (single letters skipped)."""

        surfaces: set = set()
        for symbol in symbols or []:
            base = symbol.split("_")[0]
            if len(base) > 1 or _is_greek_letter_name(base.lower()):
                surfaces.update(v.lower() for v in _surface_variants(base))
        return surfaces

    @staticmethod
    def _mentions(sentence: str, surfaces: set) -> bool:
        low = sentence.lower()
        return any(surface in low for surface in surfaces)

    @staticmethod
    def _cites(eq_num: str, sentence: str) -> bool:
        """True if the sentence explicitly cites this equation number."""

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
        phrase = _phrase_from_text(match.group(1), self.text)
        return phrase if self._valid_phrase(phrase) else ""

    def _named_equation(self, sentence: str) -> str:
        """Extract a named equation like 'Schrodinger equation' or 'X law'."""

        match = self._NAMED_EQ_RE.search(self.text.clean(sentence))
        if not match:
            return ""
        phrase = _phrase_from_text(match.group(1), self.text)
        return phrase if len(phrase.split()) >= 2 and self._valid_phrase(phrase) else ""

    def _name_from_clause(self, sentence: str, prefer_last: bool = True) -> str:
        """The noun phrase the equation's nearest clause describes.

        Which of the subject/object is the named quantity is decided from the
        *parse*, not a verb list: a passive ("X is given by:") or copula
        ("X is ...") clause -> the subject; an active clause with a pronoun
        subject ("we define X") -> the object. Verbs are visited nearest the
        equation first. Returns one phrase, or "".
        """

        doc = self._parse_doc(sentence)
        verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"} and self._subject(t) is not None]
        verbs.sort(key=lambda token: token.i, reverse=prefer_last)

        for verb in verbs:
            subject = self._subject(verb)
            complement = self._complement(verb)
            passive = any(c.dep_ in {"nsubjpass", "auxpass"} for c in verb.children)
            copula = verb.lemma_ == "be" or any(c.dep_ in {"attr", "acomp"} for c in verb.children)
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
        """Full noun-phrase name (with 'of'/'for' complements) of a head noun."""

        if head is None or head.pos_ not in {"NOUN", "PROPN"}:
            return ""
        tokens = _phrase_tokens(_np_span(head), self.text.stop_words)
        return " ".join(tok.text for tok in tokens[:8])

    def _fallback_candidates(self, candidates: List[tuple]) -> List[tuple]:
        """Collect specific noun phrases from the nearest sentences for ranking."""

        pairs: List[tuple] = []
        for sentence, _ in candidates[:4]:
            doc = self._parse_doc(sentence)
            for chunk in self._noun_chunks(doc):
                if self._is_reference(chunk.root):
                    continue
                phrase = self._phrase_from_head(chunk.root)
                if self._valid_phrase(phrase):
                    pairs.append((phrase, sentence))
        return pairs

    def _parse_doc(self, sentence: str):
        """Parse a sentence with inline math stripped so the prose parses cleanly."""

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
        """True if a head is a pronoun/citation rather than a name (from the parse)."""

        if token is None or token.pos_ == "PRON":
            return True
        return token.lemma_.lower() in {"eq", "equation"}

    def _valid_phrase(self, phrase: str) -> bool:
        """A usable name reads as a specific noun phrase (POS-decided)."""

        if len(phrase) < 3:
            return False
        content = [t for t in self.text.doc(phrase) if t.is_alpha]
        if not content or not any(t.pos_ in {"NOUN", "PROPN"} for t in content):
            return False
        if len(content) == 1 and content[0].lower_ in _META_NOUNS:
            return False
        return any(len(t.text) >= 3 for t in content)

    @staticmethod
    def _noun_chunks(doc) -> List:
        try:
            return list(doc.noun_chunks)
        except ValueError:
            return []


class SymbolExtractor:
    """Extracts symbols from MathML and their paper-supported definitions.

    A definition is written only when the surrounding prose actually defines the
    symbol; symbols without textual support are omitted, never guessed. The
    search is *clause-bounded*: each sentence is split on commas/semicolons and
    on ``where``/``with`` so a definition cannot leak across into a neighbouring
    symbol's clause (the main failure mode of an unbounded regex). No physics
    vocabulary is hard-coded -- only generic definitional grammar and Unicode.
    """

    # Verbs/phrases that introduce a symbol's definition in scientific prose.
    _DEFINING = (
        r"(?:is|are|was|were|be|denotes?|represents?|stands?\s+for|measures?|"
        r"describes?|characteri[sz]es?|quantif(?:ies|y)|equals?|defined\s+as|"
        r"denoted\s+by|gives?|corresponds?\s+to|refers?\s+to)"
    )
    # Standard operators excluded from the symbols dict (the spec exempts them).
    _OPERATORS = {"d", "delta", "Delta", "partial", "nabla", "mathrm", "rm", "mathcal"}
    # A captured definition is cut at the first subordinator so a trailing
    # relative/adverbial clause does not contaminate the noun phrase.
    _SUBORDINATORS = re.compile(
        r"\b(?:that|which|where|when|while|whose|whom|since|because|so|thus|"
        r"hence|if|though|although|whereas)\b",
        re.IGNORECASE,
    )
    # Points at which a sentence is split into clauses.
    _CLAUSE_SPLIT = re.compile(r"[,;:]|\b(?:where|with|which|wherein)\b", re.IGNORECASE)

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

        The local window is searched first; for a *distinctive* symbol (a named
        or subscripted one, which cannot be confused with an English word) the
        whole paper is searched as a fallback to recover definitions placed far
        from the equation.
        """

        symbols = self._symbols(mathml_symbols)
        audit.add("symbol_candidates", ", ".join(symbols) if symbols else "none")

        definitions: Dict[str, str] = {}
        used: set[str] = set()
        for symbol in symbols:
            found = self._best_definition(symbol, local_sentences)
            # Whole-paper fallback for named/Greek symbols only, and only from an
            # explicit "where X is ..." definition. Single Latin letters are kept
            # local because far from the equation they collide with English words
            # and with other symbols' subscripts ("c" inside "tau_c").
            if found is None and self._distinctive(symbol):
                found = self._best_definition(symbol, paper_sentences, require_where=True)
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

    def _best_definition(
        self, symbol: str, sentences: List[str], require_where: bool = False
    ) -> Optional[Tuple[str, str]]:
        """Best clause-bounded definition for a symbol among the sentences.

        When ``require_where`` is set, only sentences containing an explicit
        "where" definition are considered (used for the high-precision
        whole-paper fallback).
        """

        base, _, sub = symbol.partition("_")
        pattern = self._mention_regex(_surface_variants(base))
        bare_single = len(base) == 1 and base.isascii() and base.islower()
        # When the symbol has an alphabetic subscript (eta_D), require that
        # subscript in the clause so eta_D and eta_path are not confused.
        sub_token = (
            re.compile(rf"(?<![A-Za-z]){re.escape(sub)}(?![A-Za-z])")
            if sub and sub.isalpha()
            else None
        )

        best: Optional[Tuple[float, str, str]] = None
        for sentence in sentences:
            if require_where and not re.search(r"\bwhere\b", sentence, re.IGNORECASE):
                continue
            for clause in self._clauses(sentence):
                match = pattern.search(clause)
                if not match:
                    continue
                if sub_token and not sub_token.search(clause):
                    continue
                definition = self._predicate(clause, match.end(), sub)
                if not definition and not bare_single:
                    definition = self._appositive(clause, match.start())
                if not definition:
                    continue
                score = self._score(definition, sentence)
                if best is None or score > best[0]:
                    best = (score, definition, sentence)
        return (best[1], best[2]) if best else None

    def _clauses(self, sentence: str) -> List[str]:
        """Split a sentence into clauses, dropping bracketed citations first.

        Removing ``[...]`` and ``(...)`` first stops citation commas
        (``[46, 31, 47]``) and parentheticals from creating spurious clause
        breaks before the real comma-separated definition list is split.
        """

        cleaned = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", sentence)
        return [part.strip() for part in self._CLAUSE_SPLIT.split(cleaned) if part.strip()]

    def _predicate(self, clause: str, pos: int, sub: str = "") -> str:
        """Definition from a 'SYM (sub)? is/denotes/... (the) <noun phrase>' clause.

        Only the symbol's own subscript may sit between the symbol and the verb
        ("eta D is ..."); an arbitrary word is *not* skipped, so a single letter
        inside another token ("delta y max is ...") cannot grab a definition.
        """

        tail = clause[pos:]
        # Allow only the symbol's kept subscript or a short (<=2 char) index that
        # was dropped from the key but still printed in the prose ("tau c is ...").
        # A longer word ("delta y max is ...") is not skipped, so a single letter
        # inside another token cannot capture a definition.
        skip = rf"(?:{re.escape(sub)}\s+)?" if sub else r"(?:[A-Za-z0-9]{1,2}\s+)?"
        match = re.match(
            rf"\s*{skip}{self._DEFINING}\s+"
            r"(?:the|a|an|its|their|some)?\s*(.+)$",
            tail,
            re.IGNORECASE,
        )
        if not match:
            return ""
        phrase = self._SUBORDINATORS.split(match.group(1))[0]
        phrase = _phrase_from_text(phrase, self.text)
        return phrase if self._valid_def(phrase) else ""

    def _appositive(self, clause: str, start: int) -> str:
        """Definition from a tight 'the <noun phrase> SYM' appositive.

        Anchored to the symbol (the phrase must end right at it), so it cannot
        reach back across a comma into another symbol's definition.
        """

        match = re.search(r"(?:^|\b)(?:the|a|an)\s+([A-Za-z][A-Za-z\- ]{2,40}?)\s*$", clause[:start], re.IGNORECASE)
        if not match:
            return ""
        phrase = _phrase_from_text(match.group(1), self.text)
        return phrase if self._valid_def(phrase) else ""

    @staticmethod
    def _score(definition: str, sentence: str) -> float:
        """Prefer compact phrases and sentences with an explicit 'where' definition."""

        score = 1.0 if 2 <= len(definition.split()) <= 6 else 0.4
        if re.search(r"\bwhere\b", sentence, re.IGNORECASE):
            score += 0.5
        return score

    def _mention_regex(self, variants: frozenset) -> "re.Pattern":
        """Word-boundary regex matching any surface form of the symbol."""

        alternation = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
        return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])")

    @staticmethod
    def _distinctive(symbol: str) -> bool:
        """True if a symbol can be searched paper-wide without word collisions.

        Only named/Greek symbols (a multi-letter base such as ``theta`` or a
        Greek-letter name) qualify; single Latin letters do not, because far
        from the equation they match ordinary words and other symbols' subscripts.
        """

        base = symbol.split("_")[0]
        return len(base) > 1 or _is_greek_letter_name(base.lower())

    def _symbols(self, mathml_symbols: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in mathml_symbols:
            symbol = raw.strip()
            if self._usable_symbol(symbol) and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
        return out[:22]

    def _usable_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        base, _, sub = symbol.partition("_")
        if base in self._OPERATORS:
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", base):
            return False
        if sub and not re.fullmatch(r"[A-Za-z0-9]+", sub):
            return False
        return True

    # Finite-verb tags that mark a clause ("X equals Y", "is large when ...").
    # Gerunds/participles (VBG/VBN) are *allowed*: they act as noun modifiers in
    # "overcoupling coefficient", "loaded quality factor", "damping rate".
    _FINITE_VERB_TAGS = {"VBZ", "VBP", "VBD", "VB", "MD"}

    def _valid_def(self, phrase: str) -> bool:
        """A usable definition reads as a noun-headed noun phrase (POS-decided).

        It must contain a noun, must not contain a finite verb/auxiliary (which
        would make it a clause, "is large when ..."), must not begin with an
        adverb ("inversely proportional ..."), and the head noun must appear
        early so adjectival predicates ("proportional to the FSR ...") are
        rejected. Gerund/participle modifiers are kept (see ``_FINITE_VERB_TAGS``).
        """

        if len(phrase) < 3:
            return False
        content = [t for t in self.text.doc(phrase) if t.is_alpha]
        if not content:
            return False
        if content[0].pos_ == "ADV":
            return False
        if any(t.pos_ == "AUX" or t.tag_ in self._FINITE_VERB_TAGS for t in content):
            return False
        noun_positions = [i for i, t in enumerate(content) if t.pos_ in {"NOUN", "PROPN"}]
        if not noun_positions or noun_positions[0] > 2:
            return False
        if len(content) == 1 and content[0].lower_ in _META_NOUNS:
            return False
        return any(len(t.text) >= 3 for t in content)


class RelationExtractor:
    """Grades every ordered equation pair as strong, potential, or none.

    The classification concept uses two model-free signals plus an encoder:

    * **strong** -- one equation's context explicitly cites the other's number
      (an unambiguous, paper-stated link). The description is the connecting
      verb lifted verbatim ("given by", "reduces to"), else "directly referenced".
    * **potential** -- no explicit citation, but the two equations are topically
      linked. Topicality is a combined score of MathBERT context cosine, shared
      symbols, and shared context nouns (Jaccard). To avoid a brittle absolute
      cutoff, the highest-scoring ``max_edges`` partners above a modest floor are
      kept as potential; the rest are ``none``.

    The embedding model is used only as an encoder for cosine similarity.
    """

    def __init__(
        self,
        text: TextTools,
        similarity: "EmbeddingSimilarity",
        max_edges: int = 2,
        threshold: float = 0.62,
    ) -> None:
        self.text = text
        self.similarity = similarity
        self.max_edges = max_edges
        self.threshold = threshold

    def extract(self, equations: Dict[str, Dict], audits: Dict[str, AuditTrail]) -> Dict[str, Dict]:
        """Classify every ordered pair; emit an entry for every other equation."""

        numbers = self._sort_numbers(list(equations))
        contexts = {n: self._relation_text(equations[n]) for n in numbers}
        cosine = self.similarity.pairwise(numbers, [contexts[n] for n in numbers])
        ctx_sents = {n: self.text.sentences(contexts[n]) for n in numbers}
        nouns = {n: self._content_lemmas(contexts[n]) for n in numbers}
        symbols = {n: self._symbol_set(equations[n]) for n in numbers}

        out: Dict[str, Dict] = {}
        for left in numbers:
            scored: List[Tuple[float, str, str, str]] = []
            for right in numbers:
                if left == right:
                    continue
                grade, desc, score = self._classify(
                    left, right, ctx_sents, cosine.get((left, right), 0.0), nouns, symbols
                )
                scored.append((score, right, grade, desc))

            potential = sorted([x for x in scored if x[2] == "potential"], reverse=True)
            keep = {right for _, right, _, _ in potential[: self.max_edges]}
            n_strong = sum(1 for x in scored if x[2] == "strong")
            rels: Dict[str, Dict[str, str]] = {}
            for score, right, grade, desc in scored:
                if grade == "potential" and right not in keep:
                    grade, desc = "none", ""
                rels[right] = {"grade": grade, "description": desc}
                audits[left].add("relation", f"({left})->({right}) {grade} score={score:.2f} {desc}".rstrip())
            out[left] = rels
            audits[left].add("edge_limit", f"kept {n_strong} strong, <= {self.max_edges} potential")
        return out

    def _classify(self, left, right, ctx_sents, cosine, nouns, symbols) -> Tuple[str, str, float]:
        """Grade one ordered pair: explicit reference first, else topical score."""

        phrase = (
            self._explicit_reference_phrase(ctx_sents[left], right)
            or self._explicit_reference_phrase(ctx_sents[right], left)
        )
        if phrase:
            return "strong", phrase, 2.0 + cosine

        shared = symbols[left] & symbols[right]
        jaccard = self._jaccard(set(nouns[left]), set(nouns[right]))
        score = cosine + 0.1 * len(shared) + 0.2 * jaccard
        if score >= self.threshold:
            return "potential", self._describe(left, right, nouns, shared), score
        return "none", "", score

    @staticmethod
    def _jaccard(left: set, right: set) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _describe(left: str, right: str, nouns: Dict[str, List[str]], shared: set) -> str:
        """Describe a potential edge by the concepts/symbols the contexts share."""

        right_set = set(nouns[right])
        common = [word for word in nouns[left] if word in right_set]
        if common:
            return "shared concepts: " + ", ".join(common[:3])
        if shared:
            return "shares symbols: " + ", ".join(sorted(shared)[:3])
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
        """Base symbols of an equation (for the shared-symbol bonus)."""

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
                if match.span() not in seen:
                    seen.add(match.span())
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
            phrase = self._verb_phrase(verb)
            if phrase:
                return phrase
        # An explicit citation is always a strong relation; fall back to a
        # generic description rather than an uninformative lemma.
        return self._near_reference_phrase(doc, start, end) or "directly referenced"

    @staticmethod
    def _tokens_overlapping(doc, start: int, end: int) -> List:
        return [t for t in doc if t.idx < end and t.idx + len(t.text) > start]

    @staticmethod
    def _reference_anchor(tokens: List, eq_num: str):
        for token in tokens:
            if token.text.strip("()[] .") == eq_num:
                return token
        return tokens[-1] if tokens else None

    @staticmethod
    def _governing_verb(token):
        current, seen = token, set()
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
        verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
        if not verbs:
            return None
        return min(verbs, key=lambda t: min(abs(t.idx - index), abs(t.idx + len(t.text) - index)))

    def _verb_phrase(self, verb) -> str:
        """Relation description lifted verbatim from the connecting verb.

        A bare auxiliary ("is", "are") carries no relational meaning and is
        rejected; an attached preposition/particle is appended ("given by",
        "reduces to"). No hand-written verb mapping.
        """

        if verb.pos_ == "AUX" or not verb.lemma_.isalpha() or len(verb.lemma_) < 3:
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
        return self._clean_description(" ".join(t.text for t in tokens[:8]))

    @staticmethod
    def _description_tokens(tokens: Iterable, start: int, end: int) -> List:
        out, seen = [], set()
        for token in sorted(tokens, key=lambda item: item.i):
            if token.i in seen:
                continue
            seen.add(token.i)
            overlaps = token.idx < end and token.idx + len(token.text) > start
            if overlaps or token.is_space or token.is_punct or token.like_num:
                continue
            out.append(token)
        return out

    @staticmethod
    def _clean_description(text: str) -> str:
        text = text.replace("�", " ")
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\beq(?:uation)?s?\.?\s*[\(\[\s]?\s*[A-Za-z0-9.\-]+\s*[\)\]\s]?", " ", text, flags=re.IGNORECASE)
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        while words and words[0].lower() in {"the", "a", "an", "this", "these", "those", "that", "is", "are", "of"}:
            words.pop(0)
        return " ".join(words[:8])

    @staticmethod
    def _relation_text(entry: Dict) -> str:
        parts = [entry.get("meaning", ""), entry.get("_before", ""), entry.get("_after", "")]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _sort_numbers(numbers: List[str]) -> List[str]:
        def key(value: str):
            return (0, int(value), "") if value.isdigit() else (1, 0, value)

        return sorted(numbers, key=key)
