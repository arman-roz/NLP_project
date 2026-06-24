"""End-to-end extraction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from .arxiv_html import ArxivHtmlClient, ArxivHtmlPaper
from .common import AuditTrail, paper_key, read_paper_list
from .nlp_methods import MeaningExtractor, RelationExtractor, SymbolExtractor, TextTools
from .output_check import check_dataset
from .retrieval import BM25Retriever


class ExtractionPipeline:
    """Build the JSON dataset from an assigned arXiv list.

    Meanings and symbol definitions are noun phrases lifted by the grammar/
    structure code; a per-paper BM25 retriever widens the symbol search to the
    whole paper; relations are graded lexically. No embedding model is used.
    """

    def __init__(
        self,
        paper_list: Path,
        cache_dir: Path,
        output_path: Path,
        limit_papers: int,
        max_equations_per_paper: int,
        sleep_seconds: float,
        max_relation_edges: int,
        relation_threshold: float,
        target_equations: int = 0,
    ) -> None:
        self.paper_list = paper_list
        self.output_path = output_path
        self.limit_papers = limit_papers
        self.max_equations_per_paper = max_equations_per_paper
        self.target_equations = target_equations
        self.client = ArxivHtmlClient(cache_dir, sleep_seconds)
        self.text = TextTools()
        self.meanings = MeaningExtractor(self.text)
        self.symbols = SymbolExtractor(self.text)
        self.relations = RelationExtractor(
            self.text, max_edges=max_relation_edges, threshold=relation_threshold
        )

    def run(self) -> Dict[str, Dict]:
        """Run extraction and write the JSON output.

        Papers are processed in the assigned order. If ``target_equations`` is
        set, processing stops once the dataset reaches that many equations, but
        the paper that crosses the target is still processed in full (per spec);
        otherwise the first ``limit_papers`` papers are processed.
        """

        paper_ids = read_paper_list(self.paper_list)
        if not self.target_equations:
            paper_ids = paper_ids[: self.limit_papers]

        dataset: Dict[str, Dict] = {}
        total_equations = 0
        for arxiv_id in tqdm(paper_ids, desc="papers", unit="paper"):
            # A single malformed paper must not abort a multi-hour batch run: on any
            # unexpected error, log it and emit an empty equation dict for that paper
            # (the spec allows an empty dict and still requires the paper's key).
            try:
                equations = self._process_paper(arxiv_id)
            except Exception:  # noqa: BLE001 - robustness backstop for the batch run
                logger.exception("failed to process %s; emitting empty equation dict", arxiv_id)
                equations = {}
            dataset[paper_key(arxiv_id)] = equations
            total_equations += len(equations)
            if self.target_equations and total_equations >= self.target_equations:
                break  # last paper is processed completely (per spec)

        check_dataset(dataset)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(_json_dump(dataset), encoding="utf-8")
        return dataset

    def _process_paper(self, arxiv_id: str) -> Dict:
        paper_audit = AuditTrail()
        page = self.client.fetch(arxiv_id, paper_audit)
        if page.status_code != 200 or not page.html:
            paper_audit.add("paper_status", f"no usable HTML, status={page.status_code}")
            return {}

        paper = ArxivHtmlPaper(page.html)
        paper_audit.add("parse_html", paper.title() or "title not found")
        paper_sentences = self.text.sentences(paper.paper_text())
        paper_audit.add("paper_sentences", f"{len(paper_sentences)} sentences available for retrieval")
        # One BM25 index over the whole paper, shared by every equation/symbol so
        # a definition far from the equation can still be retrieved.
        retriever = BM25Retriever(paper_sentences)
        equation_blocks = paper.equations(self.max_equations_per_paper, paper_audit)

        built: Dict[str, Dict] = {}
        audits: Dict[str, AuditTrail] = {}
        used_meanings: set[str] = set()
        for block in tqdm(equation_blocks, desc=arxiv_id, unit="eq", leave=False):
            audit = AuditTrail()
            audit.extend(paper_audit, prefix="paper_")
            audit.extend(block.audit)
            local_context = f"{block.before} {block.after}".strip()
            local_sentences = self.text.sentences(local_context)

            meaning = self.meanings.extract(
                block.number, block.before, block.after, audit,
                used=used_meanings, retriever=retriever, paper_sentences=paper_sentences,
            )
            symbol_defs, raw_symbols = self.symbols.extract(
                block.mathml_symbols,
                local_sentences,
                retriever,
                audit,
            )

            built[block.number] = {
                "equation": block.latex,
                "meaning": meaning,
                "symbols": symbol_defs,
                "_raw_symbols": raw_symbols,
                "_before": block.before,
                "_after": block.after,
            }
            audits[block.number] = audit

        relation_map = self.relations.extract(built, audits)
        return {
            number: {
                "equation": entry["equation"],
                "meaning": entry["meaning"],
                "symbols": entry["symbols"],
                "relations": relation_map.get(number, {}),
                "audit-trail": audits[number].as_dict(),
            }
            for number, entry in built.items()
        }


def _json_dump(data: Dict) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=False)
