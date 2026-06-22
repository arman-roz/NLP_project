"""NLP methods for meanings, symbols, and equation relations."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from html.entities import codepoint2name
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


class MeaningExtractor:
    """Selects an extractive meaning sentence from the local equation window."""

    def __init__(self, text: TextTools) -> None:
        self.text = text

    def extract(self, eq_num: str, before: str, after: str, audit: AuditTrail, equation_symbols: Optional[List[str]] = None) -> str:
        """Choose a sentence from before/after the equation."""

        candidates = [(sent, "after") for sent in self.text.sentences(after)]
        candidates.extend((sent, "before") for sent in self.text.sentences(before))
        if not candidates:
            audit.add("meaning", "no local prose sentence")
            return ""

        scored = [(self._score(eq_num, sentence, source, equation_symbols), sentence, source) for sentence, source in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        score, sentence, source = scored[0]
        # Always return the best sentence, even if score is low (never leave empty)
        if score < 1.0:
            audit.add("meaning", f"weak best sentence score={score:.2f}: {short(sentence)}")
        else:
            audit.add("meaning", f"{source} sentence score={score:.2f}: {short(sentence)}")
        return sentence

    def _score(self, eq_num: str, sentence: str, source: str, equation_symbols: Optional[List[str]] = None) -> float:
        score = 1.0 if source == "after" else 0.7
        has_equation_ref = self._has_equation_reference(eq_num, sentence)
        if has_equation_ref:
            score += 2.0
        score += self._linguistic_score(eq_num, sentence, has_equation_ref)
        if 40 <= len(sentence) <= 240:
            score += 0.6
        if equation_symbols:
            sentence_tokens = set(self.text.tokens(sentence))
            eq_symbol_set = set(sym.lower() for sym in equation_symbols)
            if sentence_tokens & eq_symbol_set:
                score += 1.5
        return score

    @staticmethod
    def _has_equation_reference(eq_num: str, sentence: str) -> bool:
        return bool(
            re.search(
                rf"(?:eq(?:uation)?s?\.?\s*)?\(\s*{re.escape(eq_num)}\s*\)",
                sentence,
                re.IGNORECASE,
            )
        )

    def _linguistic_score(self, eq_num: str, sentence: str, has_equation_ref: bool) -> float:
        """Reward explanatory sentence structure without domain vocabulary lists."""

        doc = self.text.doc(sentence)
        lexical_tokens = [token for token in doc if token.is_alpha]
        if not lexical_tokens:
            return 0.0

        noun_tokens = [token for token in lexical_tokens if token.pos_ in {"NOUN", "PROPN"}]
        noun_density = len(noun_tokens) / len(lexical_tokens)
        score = 0.0
        if len(noun_tokens) >= 2 and noun_density >= 0.28:
            score += 0.5

        noun_chunks = self._noun_chunks(doc)
        if noun_chunks:
            score += 0.4
        if has_equation_ref and self._noun_phrase_near_equation(eq_num, sentence, noun_chunks):
            score += 0.5
        if self._has_predicate_structure(doc):
            score += 1.0
        return score

    @staticmethod
    def _has_predicate_structure(doc) -> bool:
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            has_subject = any(child.dep_ in {"nsubj", "nsubjpass", "expl"} for child in token.children)
            has_complement = any(
                child.dep_ in {"attr", "acomp", "dobj", "obj", "oprd", "pobj", "pcomp", "prep"}
                for child in token.children
            )
            if has_subject and has_complement:
                return True
        return False

    @staticmethod
    def _noun_chunks(doc) -> List:
        try:
            return [
                chunk
                for chunk in doc.noun_chunks
                if any(token.pos_ in {"NOUN", "PROPN"} for token in chunk)
            ]
        except ValueError:
            return []

    @staticmethod
    def _noun_phrase_near_equation(eq_num: str, sentence: str, noun_chunks: List) -> bool:
        match = re.search(
            rf"(?:eq(?:uation)?s?\.?\s*)?\(\s*{re.escape(eq_num)}\s*\)",
            sentence,
            re.IGNORECASE,
        )
        if not match:
            return False
        start, end = match.span()
        return any(min(abs(chunk.end_char - start), abs(chunk.start_char - end)) <= 80 for chunk in noun_chunks)


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
        """Return supported definitions and raw symbol candidates.

        No definition is generated from built-in physics knowledge. A symbol is
        written to the JSON only when a definition-like sentence exists in the
        arXiv text.
        """

        symbols = self._symbols(mathml_symbols)
        audit.add("symbol_candidates", ", ".join(symbols) if symbols else "none")

        definitions: Dict[str, str] = {}
        used_evidence: set[str] = set()
        for symbol in symbols:
            found = self._best_definition(symbol, local_sentences)
            if found is None:
                audit.add("symbol_definition", f"{symbol}: not found in paper text")
                continue
            definition, evidence, score = found
            if evidence in used_evidence and len(symbol) == 1:
                audit.add("symbol_definition", f"{symbol}: skipped duplicate evidence")
                continue
            definitions[symbol] = definition
            used_evidence.add(evidence)
            audit.add("symbol_definition", f"{symbol}: {definition} | score={score:.2f} | {short(evidence)}")
        return definitions, symbols

    def _best_definition(
        self,
        symbol: str,
        local_sentences: List[str],
    ) -> Optional[Tuple[str, str, float]]:
        candidates = self._candidate_sentences(symbol, local_sentences)
        if not candidates:
            return None

        parsed: List[Tuple[float, str, str]] = []
        for sentence, base_score in candidates:
            extracted = self._definition_from_sentence(symbol, sentence)
            if extracted is None:
                continue
            definition = extracted
            score = base_score + self._definition_score(definition)
            parsed.append((score, definition, sentence))

        if not parsed:
            return None
        parsed.sort(key=lambda item: item[0], reverse=True)
        score, definition, evidence = parsed[0]
        return definition, evidence, score

    @staticmethod
    def _definition_score(definition: str) -> float:
        """Prefer compact noun-like phrases without using domain vocabularies."""

        words = re.findall(r"[A-Za-z][A-Za-z\-]*", definition)
        if 2 <= len(words) <= 8:
            return 0.5
        return 0.0

    def _candidate_sentences(
        self,
        symbol: str,
        local_sentences: List[str],
    ) -> List[Tuple[str, float]]:
        candidates: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for sentence in local_sentences:
            if sentence not in seen and self._sentence_mentions_symbol(symbol, sentence):
                seen.add(sentence)
                candidates.append((sentence, 3.0))
        return candidates[:30]

    def _sentence_mentions_symbol(self, symbol: str, sentence: str) -> bool:
        return self._symbol_match(symbol, sentence) is not None

    def _definition_from_sentence(self, symbol: str, sentence: str) -> Optional[str]:
        pattern_definition = self._pattern_definition(symbol, sentence)
        if pattern_definition:
            return pattern_definition
        doc = self.text.doc(sentence)
        for token in doc:
            if not self._is_symbol_token(token, symbol):
                continue
            for extractor in (self._predicate_definition, self._appositive_definition, self._chunk_definition):
                definition = extractor(token, symbol)
                if definition:
                    return definition
        return None

    def _pattern_definition(self, symbol: str, sentence: str) -> Optional[str]:
        match = self._symbol_match(symbol, sentence)
        if match is None:
            return None

        predicate_definition = self._predicate_pattern_definition(match, sentence)
        if predicate_definition:
            return predicate_definition

        after = sentence[match.end():]
        after_match = re.match(
            r"\s*(?:(?:rm|mathrm|text)\s+)?(?:[A-Za-z0-9]+)?\s*(?:\([^)]{0,40}\))?\s*"
            r"(?:is|are|denotes?|represents?|corresponds\s+to)\s+"
            r"(?:the\s+|a\s+|an\s+)?(.+?)(?:[,.;:]|$)",
            after,
            re.IGNORECASE,
        )
        if after_match:
            definition = self._clean_definition(after_match.group(1))
            if self._usable_definition(definition):
                return definition

        return None

    def _predicate_pattern_definition(self, match: re.Match, sentence: str) -> Optional[str]:
        tail = sentence[match.start():]
        predicate_match = re.match(
            r"[A-Za-z0-9_\\\s]{1,40}?\s+"
            r"(?:characterizes?|measures?|describes?|gives?|yields?|produces?|is\s+related\s+to)\s+"
            r"(?:the\s+|a\s+|an\s+)?(.+?)(?:[,.;:]|$)",
            self.text.clean(tail),
            re.IGNORECASE,
        )
        if not predicate_match:
            return None
        definition = self._clean_definition(predicate_match.group(1))
        return definition if self._usable_definition(definition) else None

    def _symbol_match(self, symbol: str, sentence: str) -> Optional[re.Match]:
        return re.search(self._symbol_pattern(symbol), self.text.clean(sentence), re.IGNORECASE)

    @staticmethod
    def _symbol_pattern(symbol: str) -> str:
        parts = symbol.split("_")
        if len(parts) == 1:
            part = re.escape(parts[0])
            return rf"(?<![A-Za-z]){part}(?![A-Za-z])"
        left = re.escape(parts[0])
        right = re.escape(parts[1])
        spacer = r"(?:\s+(?:rm|mathrm|text))?\s+"
        return rf"(?<![A-Za-z]){left}{spacer}{right}(?![A-Za-z])"

    def _predicate_definition(self, token, symbol: str) -> Optional[str]:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            return None
        predicate = token.head
        for child in predicate.children:
            if child is token or child.dep_ in {"nsubj", "nsubjpass", "aux", "auxpass", "neg", "punct"}:
                continue
            definition = self._definition_phrase(child, symbol)
            if definition:
                return definition
        return None

    def _appositive_definition(self, token, symbol: str) -> Optional[str]:
        if token.dep_ not in {"appos", "nmod", "compound"}:
            return None
        if token.head.pos_ not in {"NOUN", "PROPN"}:
            return None
        return self._definition_phrase(token.head, symbol)

    def _chunk_definition(self, token, symbol: str) -> Optional[str]:
        for chunk in self._noun_chunks(token.doc):
            if token.i < chunk.start or token.i >= chunk.end:
                continue
            words = [item.text for item in chunk if not self._is_symbol_token(item, symbol)]
            definition = self._clean_definition(" ".join(words))
            if self._usable_definition(definition):
                return definition
        return None

    def _definition_phrase(self, token, symbol: str) -> Optional[str]:
        if token.pos_ == "ADP":
            for child in token.children:
                phrase = self._definition_phrase(child, symbol)
                if phrase:
                    return phrase
            return None
        chunk = self._noun_chunk_for(token)
        source = chunk if chunk is not None else token.subtree
        words = [item.text for item in source if not self._is_symbol_token(item, symbol)]
        definition = self._clean_definition(" ".join(words))
        if self._usable_definition(definition):
            return definition
        return None

    def _noun_chunk_for(self, token):
        for chunk in self._noun_chunks(token.doc):
            if chunk.start <= token.i < chunk.end:
                return chunk
        return None

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

    @staticmethod
    def _usable_symbol(symbol: str) -> bool:
        if not symbol:
            return False
        parts = symbol.split("_")
        base = parts[0]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", base):
            return False
        if len(parts) > 1 and not re.fullmatch(r"[A-Za-z][A-Za-z0-9 ]*", parts[1]):
            return False
        return True

    def _is_symbol_token(self, token, symbol: str) -> bool:
        if "_" in symbol:
            return False
        return _canonical_token(token.text).casefold() == symbol.casefold()

    @staticmethod
    def _clean_definition(text: str) -> str:
        text = re.sub(r"\([^)]{0,40}\)", " ", text)
        text = re.sub(r"\\[A-Za-z]+|[{}_$]", " ", text)
        return re.sub(r"\s+", " ", text).strip(" ,;:.[]()")

    @staticmethod
    def _usable_definition(definition: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", definition)
        if not 1 <= len(words) <= 12:
            return False
        bad_edges = {"where", "with", "then", "that", "and", "or", "as", "by", "to", "when"}
        if words[0].lower() in bad_edges or words[-1].lower() in bad_edges:
            return False
        lowered = " ".join(word.lower() for word in words)
        bad_phrases = {
            "similar",
            "lower",
            "higher",
            "times lower",
            "times higher",
            "delta",
            "sig",
            "ref",
            "measured",
            "the measured",
            "appendix a",
            "psi",
        }
        if lowered in bad_phrases:
            return False
        if lowered.startswith((
            "inversely proportional",
            "proportional to",
            "same ",
            "the same ",
        )):
            return False
        if "mathrm" in lowered or "rangle" in lowered or "langle" in lowered:
            return False
        if re.search(r"\d", definition):
            return False
        return True


def _canonical_token(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw.strip().strip("\\"))
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)
    if len(text) == 1:
        return _unicode_letter_name(text) or text
    return text


def _unicode_letter_name(char: str) -> str:
    entity_name = codepoint2name.get(ord(char), "")
    if entity_name and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", entity_name):
        return entity_name
    name = unicodedata.name(char, "")
    match = re.fullmatch(r"GREEK (SMALL|CAPITAL) LETTER ([A-Z]+)(?: .*)?", name)
    if match:
        normalized = match.group(2).lower()
        return normalized.capitalize() if match.group(1) == "CAPITAL" else normalized
    match = re.fullmatch(r"GREEK ([A-Z]+) SYMBOL", name)
    return match.group(1).lower() if match else ""


class RelationExtractor:
    """Relation extraction from explicit references and embedded context similarity."""

    def __init__(self, text: TextTools, similarity: EmbeddingSimilarity, max_edges: int = 2) -> None:
        self.text = text
        self.similarity = similarity
        self.max_edges = max_edges

    def extract(self, equations: Dict[str, Dict], audits: Dict[str, AuditTrail]) -> Dict[str, Dict]:
        """Classify every ordered pair as strong, potential, or none."""

        numbers = self._sort_numbers(list(equations))
        relation_texts = [self._relation_text(equations[number]) for number in numbers]
        semantic = self.similarity.pairwise(numbers, relation_texts)
        context_sentences = {
            number: self.text.sentences(f"{equations[number].get('_before', '')} {equations[number].get('_after', '')}")
            for number in numbers
        }

        scored: Dict[str, List[Tuple[float, str, str, str]]] = {number: [] for number in numbers}
        for left in numbers:
            for right in numbers:
                if left == right:
                    continue
                grade, desc, score = self._classify(
                    left,
                    right,
                    context_sentences,
                    semantic[(left, right)],
                )
                scored[left].append((score, right, grade, desc))

        out: Dict[str, Dict] = {}
        for left, items in scored.items():
            strong = [item for item in items if item[2] == "strong"]
            potential = sorted([item for item in items if item[2] == "potential"], reverse=True)
            keep_potential = {right for _, right, _, _ in potential[: self.max_edges]}
            rels: Dict[str, Dict[str, str]] = {}
            for score, right, grade, desc in items:
                final_grade = grade
                final_desc = desc
                if grade == "potential" and right not in keep_potential:
                    final_grade = "none"
                    final_desc = ""
                rels[right] = {"grade": final_grade, "description": final_desc}
                audits[left].add("relation", f"({left})->({right}) {final_grade} score={score:.2f} {final_desc}")
            out[left] = rels
            audits[left].add("edge_limit", f"kept {len(strong)} strong and at most {self.max_edges} potential edges")
        return out

    def _classify(
        self,
        left: str,
        right: str,
        context_sentences: Dict[str, List[str]],
        semantic: float,
    ) -> Tuple[str, str, float]:
        target_phrase = self._explicit_reference_phrase(context_sentences[right], left)
        source_phrase = self._explicit_reference_phrase(context_sentences[left], right)
        if target_phrase:
            return "strong", target_phrase, 1.0 + semantic
        if source_phrase:
            return "strong", source_phrase, 1.0 + semantic

        shared_concept = self._shared_concept(context_sentences[left], context_sentences[right])
        if shared_concept and semantic >= 0.85:
            return "potential", shared_concept, semantic
        return "none", "", semantic

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
        return self._near_reference_phrase(doc, start, end)

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
        lemma = verb.lemma_.lower()
        VERB_MAP = {
            "yield": "yields", "obtain": "obtained from", "derive": "derived from",
            "define": "definition", "simplify": "simplification", "use": "used in",
            "give": "gives", "follow": "follows from", "substitute": "substituted into",
            "reduce": "reduces to", "generalize": "generalizes", "specialize": "specializes",
            "prove": "proves", "show": "shows", "imply": "implies", "result": "results in",
            "lead": "leads to", "express": "expresses", "represent": "represents",
            "describe": "describes", "characterize": "characterizes", "govern": "governs",
            "satisfy": "satisfies", "determine": "determines", "give": "gives",
        }
        return VERB_MAP.get(lemma, lemma)

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
        text = re.sub(r"\beq(?:uation)?s?\.?\s*[\(\[\s]?\s*[A-Za-z0-9.\-]+\s*[\)\]\s]?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,;:.()[]")
        return " ".join(text.split()[:8])

    def _shared_concept(self, left_sentences: List[str], right_sentences: List[str]) -> str:
        left_phrases = self._context_noun_phrases(left_sentences)
        right_phrases = self._context_noun_phrases(right_sentences)
        if not left_phrases or not right_phrases:
            return ""
        right_by_key = {phrase["key"]: phrase for phrase in right_phrases}
        exact = [
            phrase
            for phrase in left_phrases
            if phrase["key"] in right_by_key
        ]
        if exact:
            best = max(exact, key=lambda phrase: (len(phrase["key"]), len(phrase["surface"])))
            return best["surface"]

        best_surface = ""
        best_score = 0.0
        for left_phrase in left_phrases:
            left_key = set(left_phrase["key"])
            for right_phrase in right_phrases:
                right_key = set(right_phrase["key"])
                union = left_key | right_key
                if not union:
                    continue
                score = len(left_key & right_key) / len(union)
                if score > best_score:
                    best_score = score
                    best_surface = left_phrase["surface"]
        return best_surface if best_score >= 0.5 else ""

    def _context_noun_phrases(self, sentences: List[str]) -> List[Dict[str, object]]:
        doc = self.text.doc(" ".join(sentences))
        phrases: List[Dict[str, object]] = []
        try:
            chunks = list(doc.noun_chunks)
        except ValueError:
            return phrases
        seen: set[Tuple[str, ...]] = set()
        for chunk in chunks:
            surface = self._clean_description(chunk.text)
            if not surface or re.search(r"\beq(?:uation)?s?\.?\b", surface, re.IGNORECASE):
                continue
            key = tuple(
                token.lemma_.casefold()
                for token in chunk
                if token.is_alpha and not token.is_stop
            )
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            phrases.append({"surface": surface, "key": key})
        return phrases

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
