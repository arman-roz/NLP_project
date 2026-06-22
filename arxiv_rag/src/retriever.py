"""Hybrid retrieval combining BM25 and embedding similarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chunker import IndexedChunk
from .indexer import HybridIndex


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    chunk_id: str
    score: float
    chunk: IndexedChunk
    rank: int = 0


class HybridRetriever:
    """Hybrid retriever using BM25 + embeddings with Reciprocal Rank Fusion.

    Combines exact-term matching (BM25) with semantic similarity
    (MathBERT embeddings) using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    index : HybridIndex
        The pre-built hybrid index.
    rrf_k : int
        RRF parameter (higher = less weight on top ranks).
    """

    def __init__(self, index: HybridIndex, rrf_k: int = 60) -> None:
        self.index = index
        self.rrf_k = rrf_k

    def search(
        self, query: str, k: int = 10, use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve top-k chunks for a query.

        Parameters
        ----------
        query : str
            The search query.
        k : int
            Number of results.
        use_rrf : bool
            If True, use Reciprocal Rank Fusion. Otherwise use weighted sum.

        Returns
        -------
        list[RetrievalResult]
            Ranked retrieval results.
        """
        if use_rrf:
            return self._rrf_search(query, k)
        return self._weighted_search(query, k)

    def _rrf_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Search using Reciprocal Rank Fusion."""
        bm25_hits = dict(self.index.bm25.search(query, k=k*3))
        emb_hits = dict(self.index.embeddings.search(query, k=k*3))

        all_ids = set(bm25_hits.keys()) | set(emb_hits.keys())
        rrf_scores: Dict[str, float] = {}

        bm25_ranked = sorted(bm25_hits.keys(), key=lambda x: bm25_hits[x], reverse=True)
        emb_ranked = sorted(emb_hits.keys(), key=lambda x: emb_hits[x], reverse=True)

        for rank, cid in enumerate(bm25_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, cid in enumerate(emb_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for rank, cid in enumerate(sorted_ids[:k]):
            if cid in self.index.chunk_map:
                results.append(RetrievalResult(
                    chunk_id=cid,
                    score=rrf_scores[cid],
                    chunk=self.index.chunk_map[cid],
                    rank=rank+1,
                ))
        return results

    def _weighted_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Search using weighted score fusion."""
        raw = self.index.search(query, k=k)
        results = []
        for rank, (cid, score, chunk) in enumerate(raw):
            results.append(RetrievalResult(
                chunk_id=cid,
                score=score,
                chunk=chunk,
                rank=rank+1,
            ))
        return results

    def retrieve_equation_context(
        self, paper_id: str, eq_num: str, query: str, k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve context chunks specific to one equation.

        Parameters
        ----------
        paper_id : str
            Paper identifier.
        eq_num : str
            Equation number.
        query : str
            Search query.
        k : int
            Number of results.

        Returns
        -------
        list[RetrievalResult]
            Filtered results for this equation.
        """
        all_results = self.search(query, k=k*3)
        eq_results = [
            r for r in all_results
            if r.chunk.paper_id == paper_id and r.chunk.eq_num == eq_num
        ]
        return eq_results[:k]

    def retrieve_symbol_context(
        self, symbol: str, paper_id: str = "", k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve context chunks mentioning a symbol.

        Parameters
        ----------
        symbol : str
            The symbol to search for.
        paper_id : str
            Optional paper filter.
        k : int
            Number of results.

        Returns
        -------
        list[RetrievalResult]
            Chunks containing the symbol.
        """
        query = f"definition of {symbol} symbol"
        all_results = self.search(query, k=k*3)
        sym_results = []
        for r in all_results:
            if paper_id and r.chunk.paper_id != paper_id:
                continue
            if symbol.lower() in r.chunk.text.lower() or symbol.lower() in [s.lower() for s in r.chunk.symbols]:
                sym_results.append(r)
        return sym_results[:k]
