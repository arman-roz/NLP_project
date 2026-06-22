"""Create equation-centered chunks for indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .parser import EquationRecord, TextChunk


@dataclass
class IndexedChunk:
    """A chunk ready for indexing with all metadata."""

    chunk_id: str
    paper_id: str
    eq_num: str
    chunk_type: str
    text: str
    latex: str = ""
    symbols: List[str] = field(default_factory=list)
    section: str = ""
    metadata: Dict = field(default_factory=dict)


class EquationChunker:
    """Create multiple chunk types from equation records.

    Generates three chunk types per equation:
    - **equation**: before + equation + after (the main chunk)
    - **symbol**: symbol list + surrounding sentence
    - **context**: section text with equation reference

    Parameters
    ----------
    max_context_tokens : int
        Maximum characters for context text.
    """

    def __init__(self, max_context_tokens: int = 512) -> None:
        self.max_context_tokens = max_context_tokens

    def chunk_paper(
        self, equations: List[EquationRecord], text_chunks: List[TextChunk]
    ) -> List[IndexedChunk]:
        """Create all chunks for a paper.

        Parameters
        ----------
        equations : list[EquationRecord]
            Extracted equations from the paper.
        text_chunks : list[TextChunk]
            Paragraph-level text chunks.

        Returns
        -------
        list[IndexedChunk]
            All chunks ready for indexing.
        """
        chunks: List[IndexedChunk] = []

        for eq in equations:
            chunks.extend(self._equation_chunks(eq))
            chunks.extend(self._symbol_chunks(eq))

        for tc in text_chunks:
            chunks.append(self._text_chunk(tc))

        return chunks

    def _equation_chunks(self, eq: EquationRecord) -> List[IndexedChunk]:
        """Create equation-centered chunks."""
        chunks: List[IndexedChunk] = []

        full_text = f"{eq.before} {eq.latex} {eq.after}".strip()
        if full_text:
            chunks.append(IndexedChunk(
                chunk_id=f"{eq.paper_id}_eq{eq.eq_num}_full",
                paper_id=eq.paper_id,
                eq_num=eq.eq_num,
                chunk_type="equation_full",
                text=full_text[:self.max_context_tokens],
                latex=eq.latex,
                symbols=eq.mathml_symbols,
                section=eq.section,
                metadata={"context_type": "full_equation"},
            ))

        before = eq.before.strip()
        if before:
            chunks.append(IndexedChunk(
                chunk_id=f"{eq.paper_id}_eq{eq.eq_num}_before",
                paper_id=eq.paper_id,
                eq_num=eq.eq_num,
                chunk_type="equation_before",
                text=before[:self.max_context_tokens],
                latex=eq.latex,
                symbols=eq.mathml_symbols,
                section=eq.section,
                metadata={"context_type": "before_equation"},
            ))

        after = eq.after.strip()
        if after:
            chunks.append(IndexedChunk(
                chunk_id=f"{eq.paper_id}_eq{eq.eq_num}_after",
                paper_id=eq.paper_id,
                eq_num=eq.eq_num,
                chunk_type="equation_after",
                text=after[:self.max_context_tokens],
                latex=eq.latex,
                symbols=eq.mathml_symbols,
                section=eq.section,
                metadata={"context_type": "after_equation"},
            ))

        return chunks

    def _symbol_chunks(self, eq: EquationRecord) -> List[IndexedChunk]:
        """Create symbol-focused chunks."""
        chunks: List[IndexedChunk] = []

        if eq.mathml_symbols:
            symbol_text = f"symbols: {', '.join(eq.mathml_symbols)}"
            chunks.append(IndexedChunk(
                chunk_id=f"{eq.paper_id}_eq{eq.eq_num}_symlist",
                paper_id=eq.paper_id,
                eq_num=eq.eq_num,
                chunk_type="symbol_list",
                text=symbol_text,
                latex=eq.latex,
                symbols=eq.mathml_symbols,
                section=eq.section,
                metadata={"context_type": "symbol_list"},
            ))

        for sym in eq.mathml_symbols:
            sentences = self._find_symbol_sentences(sym, eq.before, eq.after)
            if sentences:
                chunks.append(IndexedChunk(
                    chunk_id=f"{eq.paper_id}_eq{eq.eq_num}_sym_{sym}",
                    paper_id=eq.paper_id,
                    eq_num=eq.eq_num,
                    chunk_type="symbol_usage",
                    text=f"{sym}: {sentences}",
                    latex=eq.latex,
                    symbols=[sym],
                    section=eq.section,
                    metadata={"context_type": "symbol_definition", "symbol": sym},
                ))

        return chunks

    def _text_chunk(self, tc: TextChunk) -> IndexedChunk:
        """Convert a text chunk to indexed chunk."""
        return IndexedChunk(
            chunk_id=f"{tc.paper_id}_{tc.chunk_id}",
            paper_id=tc.paper_id,
            eq_num="",
            chunk_type="paragraph",
            text=tc.text[:self.max_context_tokens],
            section=tc.section,
            metadata={"context_type": "paragraph"},
        )

    @staticmethod
    def _find_symbol_sentences(symbol: str, before: str, after: str) -> str:
        """Find sentences mentioning a symbol in context."""
        combined = f"{before} {after}"
        sentences = [s.strip() for s in combined.split(".") if len(s.strip()) > 20]
        matching = [s for s in sentences if symbol.lower() in s.lower()]
        return ". ".join(matching[:3]) if matching else ""
