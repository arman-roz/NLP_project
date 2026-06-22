"""End-to-end extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm

from .arxiv_html import ArxivHtmlClient, ArxivHtmlPaper
from .common import AuditTrail, paper_key, read_paper_list
from .nlp_methods import EmbeddingSimilarity, MeaningExtractor, RelationExtractor, SymbolExtractor, TextTools
from .output_check import check_dataset


class ExtractionPipeline:
    """Build the JSON dataset from an assigned arXiv list."""

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
        self.max_equations_per_paper = max_equations_per_paper
        self.client = ArxivHtmlClient(cache_dir, sleep_seconds)
        self.text = TextTools()
        self.similarity = EmbeddingSimilarity(embedding_model, model_cache_dir)
        self.meanings = MeaningExtractor(self.text)
        self.symbols = SymbolExtractor(self.text)
        self.relations = RelationExtractor(self.text, self.similarity, max_edges=max_relation_edges)

    def run(self) -> Dict[str, Dict]:
        """Run extraction and write the JSON output."""

        paper_ids = read_paper_list(self.paper_list)[: self.limit_papers]
        dataset: Dict[str, Dict] = {}
        for arxiv_id in tqdm(paper_ids, desc="papers", unit="paper"):
            dataset[paper_key(arxiv_id)] = self._process_paper(arxiv_id)

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
        paper_audit.add("paper_sentences", f"{len(paper_sentences)} sentences available for symbol definitions")
        equation_blocks = paper.equations(self.max_equations_per_paper, paper_audit)

        built: Dict[str, Dict] = {}
        audits: Dict[str, AuditTrail] = {}
        for block in tqdm(equation_blocks, desc=arxiv_id, unit="eq", leave=False):
            audit = AuditTrail()
            audit.extend(paper_audit, prefix="paper_")
            audit.extend(block.audit)
            local_context = f"{block.before} {block.after}"
            local_sentences = self.text.sentences(local_context)

            meaning = self.meanings.extract(block.number, block.before, block.after, audit, equation_symbols=block.mathml_symbols)
            symbol_defs, raw_symbols = self.symbols.extract(
                block.mathml_symbols,
                local_sentences,
                paper_sentences,
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
