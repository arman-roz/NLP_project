"""End-to-end extraction pipeline with hybrid retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .chunker import EquationChunker, IndexedChunk
from .extractor import MeaningExtractor, RelationExtractor, SymbolExtractor
from .fetcher import ArxivFetcher
from .indexer import HybridIndex
from .parser import ArxivParser, EquationRecord
from .retriever import HybridRetriever


class ExtractionPipeline:
    """Build JSON dataset from arXiv papers using hybrid retrieval.

    Pipeline stages:
    1. Fetch HTML pages (cache-first, robots.txt compliant)
    2. Parse HTML into equation records and text chunks
    3. Chunk equations into indexed records
    4. Build hybrid BM25 + embedding index
    5. Retrieve context for each equation
    6. Extract symbols, meanings, and relations
    7. Output structured JSON

    Parameters
    ----------
    paper_list : Path
        Path to file with arXiv IDs (one per line).
    cache_dir : Path
        Directory for caching HTML files.
    model_cache_dir : Path
        Directory for caching transformer models.
    output_path : Path
        Output JSON file path.
    limit_papers : int
        Maximum number of papers to process.
    max_equations_per_paper : int
        Maximum equations to extract per paper.
    sleep_seconds : float
        Seconds between HTTP requests.
    embedding_model : str
        HuggingFace model name for embeddings.
    max_relation_edges : int
        Maximum potential edges per equation.
    """

    def __init__(
        self,
        paper_list: Path,
        cache_dir: Path,
        model_cache_dir: Path,
        output_path: Path,
        limit_papers: int,
        max_equations_per_paper: int,
        sleep_seconds: float,
        embedding_model: str,
        max_relation_edges: int,
    ) -> None:
        self.paper_list = paper_list
        self.output_path = output_path
        self.limit_papers = limit_papers
        self.max_eq = max_equations_per_paper

        self.fetcher = ArxivFetcher(cache_dir, sleep_seconds)
        self.chunker = EquationChunker()
        self.meaning_extractor = MeaningExtractor()
        self.symbol_extractor = SymbolExtractor()
        self._embedding_model = embedding_model
        self._model_cache_dir = model_cache_dir
        self._max_relation_edges = max_relation_edges
        self._total_equations = 0
        self._target_equations = 350

    def run(self) -> Dict[str, Dict]:
        """Run the full extraction pipeline.

        Returns
        -------
        dict
            The final JSON dataset.
        """
        paper_ids = self._read_paper_list()[:self.limit_papers]
        all_chunks: List[IndexedChunk] = []
        paper_equations: Dict[str, List[EquationRecord]] = {}

        # Stage 1: Fetch and parse papers
        for arxiv_id in tqdm(paper_ids, desc="Fetching papers", unit="paper"):
            records = self._fetch_and_parse(arxiv_id)
            if records:
                paper_equations[arxiv_id] = records
                self._total_equations += len(records)
                if self._total_equations >= self._target_equations:
                    break

        print(f"\nCollected {self._total_equations} equations from {len(paper_equations)} papers")

        # Stage 2: Chunk equations with progress
        for arxiv_id, records in tqdm(paper_equations.items(), desc="Chunking papers", unit="paper"):
            parser = ArxivParser(self.fetcher.fetch(arxiv_id).html, arxiv_id)
            text_chunks = parser.text_chunks()
            chunks = self.chunker.chunk_paper(records, text_chunks)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks for indexing")

        # Stage 3: Build hybrid index with progress
        print("Building hybrid index...")
        index = HybridIndex(
            all_chunks,
            model_name=self._embedding_model,
            cache_dir=self._model_cache_dir,
        )
        index.build()

        retriever = HybridRetriever(index)
        relation_extractor = RelationExtractor(retriever)

        # Stage 4: Extract meanings, symbols, relations per equation
        dataset: Dict[str, Dict] = {}
        for arxiv_id, records in tqdm(paper_equations.items(), desc="Extracting", unit="paper"):
            dataset[arxiv_id] = self._process_paper(
                arxiv_id, records, retriever, relation_extractor, index
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved {len(dataset)} papers to {self.output_path}")
        return dataset

    def _fetch_and_parse(self, arxiv_id: str) -> List[EquationRecord]:
        """Fetch and parse a single paper."""
        page = self.fetcher.fetch(arxiv_id)
        if page.status_code != 200 or not page.html:
            return []
        parser = ArxivParser(page.html, arxiv_id)
        return parser.equations(self.max_eq)

    def _process_paper(
        self,
        arxiv_id: str,
        records: List[EquationRecord],
        retriever: HybridRetriever,
        relation_extractor: RelationExtractor,
        index: HybridIndex,
    ) -> Dict[str, Dict]:
        """Process one paper's equations."""
        equations_data: Dict[str, Dict] = {}

        for eq in tqdm(records, desc=f"Processing {arxiv_id}", unit="eq", leave=False):
            query = f"{eq.latex} {eq.context}"
            results = retriever.search(query, k=10)

            meaning = self.meaning_extractor.extract_meaning(
                eq.eq_num, eq.latex, results, eq.mathml_symbols
            )

            symbol_defs = self.symbol_extractor.extract_definitions(
                eq.mathml_symbols, results
            )

            equations_data[eq.eq_num] = {
                "equation": eq.latex,
                "meaning": meaning,
                "symbols": symbol_defs,
                "_before": eq.before,
                "_after": eq.after,
                "_section": eq.section,
            }

        relation_map = relation_extractor.extract_relations(
            arxiv_id, equations_data, self._max_relation_edges
        )

        return {
            eq_num: {
                "equation": data["equation"],
                "meaning": data["meaning"],
                "symbols": data["symbols"],
                "relations": relation_map.get(eq_num, {}),
                "audit-trail": self._build_audit(eq_num, data, relation_map),
            }
            for eq_num, data in equations_data.items()
        }

    def _build_audit(self, eq_num: str, data: Dict, relation_map: Dict) -> Dict[str, str]:
        """Build audit trail for one equation."""
        audit = {
            "fetch_html": f"Loaded HTML for paper",
            "parse_html": f"Found equation ({eq_num})",
            "chunking": f"Created chunks with symbols: {list(data['symbols'].keys())}",
            "retrieve_context": f"Retrieved top-10 chunks for hybrid search",
            "extract_meaning": f"Scored and selected: {data['meaning'][:100]}...",
        }
        for sym, defn in data["symbols"].items():
            audit[f"symbol_{sym}"] = f"{sym}: {defn}"

        rels = relation_map.get(eq_num, {})
        strong = [r for r, d in rels.items() if d.get("grade") == "strong"]
        potential = [r for r, d in rels.items() if d.get("grade") == "potential"]
        audit["relations"] = f"strong={strong}, potential={potential}"

        return audit

    def _read_paper_list(self) -> List[str]:
        """Read arXiv IDs from the paper list file."""
        import re
        ids: List[str] = []
        with self.paper_list.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(re.sub(r"(?i)^arxiv:", "", line))
        return ids
