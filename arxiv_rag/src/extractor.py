"""Extract symbols, meanings, and relations from retrieved chunks."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import spacy

from .chunker import IndexedChunk
from .retriever import HybridRetriever, RetrievalResult


def _load_spacy():
    """Load spaCy model with lazy initialization."""
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "spaCy model en_core_web_sm required. "
            "Install with: python -m spacy download en_core_web_sm"
        )
    return spacy.load("en_core_web_sm")


class SymbolExtractor:
    """Extract symbol definitions from retrieved context chunks.

    Uses pattern matching and dependency parsing to find
    definitions of symbols in surrounding text.

    Parameters
    ----------
    nlp : spacy.Language
        spaCy language model.
    """

    def __init__(self, nlp=None) -> None:
        self.nlp = nlp or _load_spacy()
        self.stop_words = set(self.nlp.Defaults.stop_words)

    def extract_definitions(
        self, symbols: List[str], chunks: List[RetrievalResult]
    ) -> Dict[str, str]:
        """Extract definitions for each symbol from retrieved chunks.

        Parameters
        ----------
        symbols : list[str]
            Symbol names to find definitions for.
        chunks : list[RetrievalResult]
            Retrieved context chunks.

        Returns
        -------
        dict[str, str]
            Symbol -> definition mapping.
        """
        definitions: Dict[str, str] = {}
        used_evidence: set = set()

        for symbol in symbols:
            best = self._find_best_definition(symbol, chunks, used_evidence)
            if best:
                definitions[symbol] = best

        return definitions

    def _find_best_definition(
        self, symbol: str, chunks: List[RetrievalResult], used_evidence: set
    ) -> Optional[str]:
        """Find the best definition for a symbol from chunks."""
        candidates: List[Tuple[float, str]] = []

        for result in chunks:
            text = result.chunk.text
            if symbol.lower() not in text.lower():
                continue
            sentences = self._split_sentences(text)
            for sent in sentences:
                if symbol.lower() in sent.lower():
                    score = self._definition_score(symbol, sent, result.score)
                    if score > 0:
                        candidates.append((score, sent))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, sent in candidates:
            if sent not in used_evidence:
                used_evidence.add(sent)
                return self._extract_definition_phrase(symbol, sent)

        return None

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r"(?<=[.!?:])\s+", text) if len(s.strip()) > 15]

    def _definition_score(self, symbol: str, sentence: str, retrieval_score: float) -> float:
        """Score a sentence as a potential definition for a symbol."""
        score = retrieval_score * 0.5

        patterns = [
            rf"{re.escape(symbol)}\s+(?:is|are|denotes?|represents?|corresponds?\s+to)\s+",
            rf"(?:the|a|an)\s+.*{re.escape(symbol)}\s+(?:is|are|denotes?|represents?)\s+",
            rf"{re.escape(symbol)}\s+(?:characterizes?|measures?|describes?|gives?)\s+",
            rf"(?:where|with)\s+{re.escape(symbol)}\s+(?:is|are|being)\s+",
        ]
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 3.0
                break

        doc = self.nlp(sentence)
        noun_tokens = [t for t in doc if t.pos_ in {"NOUN", "PROPN"} and t.is_alpha]
        if len(noun_tokens) >= 2:
            score += 0.5

        return score

    def _extract_definition_phrase(self, symbol: str, sentence: str) -> str:
        """Extract the definition phrase from a sentence."""
        patterns = [
            rf"{re.escape(symbol)}\s+(?:is|are|denotes?|represents?|corresponds?\s+to)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[,.;:]|$)",
            rf"(?:the|a|an)\s+\w+\s+{re.escape(symbol)}\s+(?:is|are|denotes?)\s+(.+?)(?:[,.;:]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                defn = match.group(1).strip()
                defn = re.sub(r"\([^)]{0,40}\)", " ", defn)
                defn = re.sub(r"\\[A-Za-z]+|[{}_$^]", " ", defn)
                defn = re.sub(r"\s+", " ", defn).strip(" ,;:.[]()")
                words = re.findall(r"[A-Za-z][A-Za-z\-]*", defn)
                if 1 <= len(words) <= 12:
                    return defn

        doc = self.nlp(sentence)
        for token in doc:
            if token.text.lower() == symbol.lower():
                for child in token.children:
                    if child.dep_ in {"attr", "acomp", "dobj", "obj"}:
                        phrase = " ".join(t.text for t in child.subtree if t.text.lower() != symbol.lower())
                        phrase = re.sub(r"\s+", " ", phrase).strip(" ,;:.()")
                        if 1 <= len(phrase.split()) <= 12:
                            return phrase

        return sentence[:120].strip(" ,;:.()")


class MeaningExtractor:
    """Extract equation meanings from retrieved context.

    Uses linguistic scoring to select the most explanatory
    sentence describing what an equation expresses.

    Parameters
    ----------
    nlp : spacy.Language
        spaCy language model.
    """

    def __init__(self, nlp=None) -> None:
        self.nlp = nlp or _load_spacy()

    def extract_meaning(
        self,
        eq_num: str,
        latex: str,
        chunks: List[RetrievalResult],
        equation_symbols: List[str] = None,
    ) -> str:
        """Extract the best meaning sentence for an equation.

        Parameters
        ----------
        eq_num : str
            Equation number.
        latex : str
            LaTeX source of the equation.
        chunks : list[RetrievalResult]
            Retrieved context chunks.
        equation_symbols : list[str], optional
            Symbols from the equation for overlap scoring.

        Returns
        -------
        str
            Best meaning sentence.
        """
        candidates: List[Tuple[float, str, str]] = []

        for result in chunks:
            text = result.chunk.text
            if result.chunk.chunk_type in {"equation_full", "equation_before", "equation_after"}:
                sentences = self._split_sentences(text)
                for sent in sentences:
                    if len(sent) < 20:
                        continue
                    source = result.chunk.chunk_type
                    score = self._score_sentence(eq_num, sent, source, result.score, equation_symbols)
                    candidates.append((score, sent, source))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _score_sentence(
        self,
        eq_num: str,
        sentence: str,
        source: str,
        retrieval_score: float,
        equation_symbols: List[str] = None,
    ) -> float:
        """Score a sentence as a potential meaning."""
        score = retrieval_score * 0.3

        if source == "equation_after":
            score += 1.0
        elif source == "equation_before":
            score += 0.7

        if re.search(rf"\(\s*{re.escape(eq_num)}\s*\)", sentence):
            score += 2.0

        if equation_symbols:
            sent_tokens = set(t.lower() for t in re.findall(r"[a-zA-Z]+", sentence))
            eq_syms = set(s.lower() for s in equation_symbols)
            overlap = sent_tokens & eq_syms
            if overlap:
                score += 1.5

        doc = self.nlp(sentence)
        nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"} and t.is_alpha]
        if len(nouns) >= 2 and len(nouns) / max(len(doc), 1) >= 0.25:
            score += 0.5

        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)
        has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
        if has_verb and has_subj:
            score += 0.8

        if 40 <= len(sentence) <= 240:
            score += 0.4

        return score

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r"(?<=[.!?:])\s+", text) if len(s.strip()) > 15]


class RelationExtractor:
    """Extract relations between equations from retrieved context.

    Uses explicit equation references and embedding similarity
    to classify relations as strong, potential, or none.

    Parameters
    ----------
    retriever : HybridRetriever
        The hybrid retriever for context search.
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def extract_relations(
        self, paper_id: str, equations: Dict[str, Dict], max_edges: int = 2
    ) -> Dict[str, Dict]:
        """Extract relations for all equations in a paper.

        Parameters
        ----------
        paper_id : str
            Paper identifier.
        equations : dict
            Dictionary of equation numbers to their data.
        max_edges : int
            Maximum potential edges per equation.

        Returns
        -------
        dict[str, dict]
            Equation number -> {other_eq: {grade, description}} mapping.
        """
        eq_numbers = sorted(
            equations.keys(),
            key=lambda x: (0, int(x), "") if x.isdigit() else (1, 0, x),
        )

        relation_map: Dict[str, Dict] = {}
        for left in eq_numbers:
            relation_map[left] = {}
            for right in eq_numbers:
                if left == right:
                    continue
                grade, desc = self._classify_pair(
                    paper_id, left, right, equations
                )
                relation_map[left][right] = {"grade": grade, "description": desc}

        return relation_map

    def _classify_pair(
        self, paper_id: str, left: str, right: str, equations: Dict[str, Dict]
    ) -> Tuple[str, str]:
        """Classify the relation between two equations."""
        strong = self._check_explicit_reference(paper_id, left, right)
        if strong:
            return "strong", strong

        strong_rev = self._check_explicit_reference(paper_id, right, left)
        if strong_rev:
            return "strong", strong_rev

        potential = self._check_shared_concept(paper_id, left, right, equations)
        if potential:
            return "potential", potential

        return "none", ""

    def _check_explicit_reference(
        self, paper_id: str, source_eq: str, target_eq: str
    ) -> str:
        """Check if target equation explicitly references source equation."""
        query = f"equation ({source_eq}) used in derive obtain"
        results = self.retriever.retrieve_equation_context(
            paper_id, target_eq, query, k=5
        )

        for result in results:
            text = result.chunk.text
            patterns = [
                rf"eq(?:uation)?\.?\s*\(\s*{re.escape(source_eq)}\s*\)",
                rf"\(\s*{re.escape(source_eq)}\s*\)",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    verb = self._find_governing_verb(text, match.start())
                    if verb:
                        return verb

        return ""

    def _find_governing_verb(self, text: str, position: int) -> str:
        """Find the governing verb near a position in text."""
        before = text[:position]
        after = text[position:position+200]
        combined = before + " " + after

        verb_map = {
            "yield": "yields", "obtain": "obtained from", "derive": "derived from",
            "define": "definition", "simplify": "simplification", "use": "used in",
            "give": "gives", "follow": "follows from", "substitute": "substituted into",
            "reduce": "reduces to", "generalize": "generalizes", "specialize": "specializes",
            "prove": "proves", "show": "shows", "imply": "implies", "result": "results in",
            "lead": "leads to", "express": "expresses", "represent": "represents",
            "describe": "describes", "characterize": "characterizes", "govern": "governs",
            "satisfy": "satisfies", "determine": "determines",
        }

        doc = self.nlp(combined)
        verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]

        best_verb = None
        best_dist = float("inf")
        for verb in verbs:
            if verb.idx <= position:
                dist = position - verb.idx
                if dist < best_dist:
                    best_dist = dist
                    best_verb = verb

        if best_verb and best_dist < 80:
            lemma = best_verb.lemma_.lower()
            return verb_map.get(lemma, lemma)

        return ""

    def _check_shared_concept(
        self, paper_id: str, left: str, right: str, equations: Dict[str, Dict]
    ) -> str:
        """Check for shared concepts between two equations."""
        left_data = equations.get(left, {})
        right_data = equations.get(right, {})

        left_text = f"{left_data.get('meaning', '')} {left_data.get('_before', '')} {left_data.get('_after', '')}"
        right_text = f"{right_data.get('meaning', '')} {right_data.get('_before', '')} {right_data.get('_after', '')}"

        left_tokens = set(t.lower() for t in re.findall(r"[a-zA-Z]+", left_text))
        right_tokens = set(t.lower() for t in re.findall(r"[a-zA-Z]+", right_text))

        left_tokens -= {"the", "a", "an", "is", "are", "was", "were", "that", "this", "with", "from", "for", "in", "on", "to"}
        right_tokens -= left_tokens

        union = left_tokens | right_tokens
        if not union:
            return ""

        intersection = left_tokens & right_tokens
        jaccard = len(intersection) / len(union)

        if jaccard >= 0.3 and len(intersection) >= 2:
            shared = sorted(intersection, key=lambda x: len(x), reverse=True)
            return "shared concepts: " + ", ".join(shared[:3])

        return ""
