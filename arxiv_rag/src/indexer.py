"""Build BM25 and embedding indices for hybrid retrieval."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from .chunker import IndexedChunk


class BM25Index:
    """BM25 index for exact-term retrieval.

    Tokenizes text at the character level for math-friendly matching
    and builds a BM25Okapi index.

    Parameters
    ----------
    chunks : list[IndexedChunk]
        Chunks to index.
    """

    def __init__(self, chunks: List[IndexedChunk]) -> None:
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.texts = [c.text for c in chunks]
        self.tokenized = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for top-k chunks by BM25 score.

        Parameters
        ----------
        query : str
            Search query string.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (chunk_id, score) pairs sorted by score descending.
        """
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def get_chunk(self, chunk_id: str) -> Optional[IndexedChunk]:
        """Retrieve chunk by ID."""
        idx_map = {c.chunk_id: c for c in self.chunks}
        return idx_map.get(chunk_id)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Character-level + word-level tokenization for math-aware matching."""
        import re
        text = text.lower()
        words = re.findall(r"[a-z0-9_]+", text)
        chars = [text[i:i+2] for i in range(len(text)-1) if text[i].isalpha()]
        return words + chars


class EmbeddingIndex:
    """Vector index using sentence-transformers or MathBERT for semantic search.

    Parameters
    ----------
    model_name : str
        HuggingFace model name for embeddings.
    cache_dir : Path
        Directory for caching model files.
    """

    def __init__(self, model_name: str = "tbs17/MathBERT", cache_dir: Path = Path("data/embeddings")) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._embeddings: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []

    def _load(self):
        """Lazy-load the transformer model."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir)
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir)
            )
            self._model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized embeddings using mean pooling.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.

        Returns
        -------
        np.ndarray
            Shape (n, d) normalized embedding matrix.
        """
        self._load()
        import torch

        if not texts:
            return np.zeros((0, 1), dtype=float)

        all_embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding embeddings", leave=False):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                outputs = self._model(**inputs)
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 1), dtype=float)

    def build_index(self, chunks: List[IndexedChunk]) -> None:
        """Build embedding index from chunks.

        Parameters
        ----------
        chunks : list[IndexedChunk]
            Chunks to embed and index.
        """
        self._chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        self._embeddings = self.encode(texts)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for top-k chunks by cosine similarity.

        Parameters
        ----------
        query : str
            Search query string.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (chunk_id, score) pairs sorted by score descending.
        """
        if self._embeddings is None or len(self._chunk_ids) == 0:
            return []

        query_vec = self.encode([query])[0]
        scores = np.dot(self._embeddings, query_vec)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_idx]


class HybridIndex:
    """Combined BM25 + embedding index for hybrid retrieval.

    Parameters
    ----------
    chunks : list[IndexedChunk]
        Chunks to index.
    model_name : str
        HuggingFace model for embeddings.
    cache_dir : Path
        Model cache directory.
    bm25_weight : float
        Weight for BM25 scores in fusion.
    emb_weight : float
        Weight for embedding scores in fusion.
    """

    def __init__(
        self,
        chunks: List[IndexedChunk],
        model_name: str = "tbs17/MathBERT",
        cache_dir: Path = Path("data/embeddings"),
        bm25_weight: float = 0.5,
        emb_weight: float = 0.5,
    ) -> None:
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.embeddings = EmbeddingIndex(model_name, cache_dir)
        self.bm25_weight = bm25_weight
        self.emb_weight = emb_weight

    def build(self) -> None:
        """Build both indices."""
        self.embeddings.build_index(self.chunks)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, IndexedChunk]]:
        """Hybrid search combining BM25 and embedding scores.

        Parameters
        ----------
        query : str
            Search query string.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float, IndexedChunk]]
            List of (chunk_id, combined_score, chunk) sorted by score descending.
        """
        bm25_hits = dict(self.bm25.search(query, k=k*2))
        emb_hits = dict(self.embeddings.search(query, k=k*2))

        all_ids = set(bm25_hits.keys()) | set(emb_hits.keys())
        fused: List[Tuple[str, float]] = []

        if bm25_hits:
            max_bm25 = max(bm25_hits.values()) if bm25_hits else 1.0
        else:
            max_bm25 = 1.0
        if emb_hits:
            max_emb = max(emb_hits.values()) if emb_hits else 1.0
        else:
            max_emb = 1.0

        for cid in all_ids:
            bm25_score = bm25_hits.get(cid, 0.0) / max_bm25
            emb_score = emb_hits.get(cid, 0.0) / max_emb
            combined = self.bm25_weight * bm25_score + self.emb_weight * emb_score
            if combined > 0.05:
                fused.append((cid, combined))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [
            (cid, score, self.chunk_map[cid])
            for cid, score in fused[:k]
            if cid in self.chunk_map
        ]

    def save(self, path: Path) -> None:
        """Save index metadata to disk."""
        meta = {
            "chunk_ids": self.bm25.chunk_ids,
            "bm25_weight": self.bm25_weight,
            "emb_weight": self.emb_weight,
        }
        (path / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load index metadata from disk."""
        meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
        self.bm25_weight = meta["bm25_weight"]
        self.emb_weight = meta["emb_weight"]
