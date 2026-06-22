"""Command line entry point for the arxiv_rag extraction pipeline."""

# run using cd this folder and then python -m src.main --limit-papers 10 --max-equations-per-paper 7 --output data/output/dataset.json

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import ExtractionPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent

DEFAULT_PAPER_LIST = REPO_DIR / "paper_list_44.txt"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache"
DEFAULT_MODEL_CACHE_DIR = PROJECT_DIR / "data" / "embeddings"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "output" / "dataset.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Run the arxiv_rag extraction pipeline."""
    args = parse_args()

    pipeline = ExtractionPipeline(
        paper_list=args.paper_list,
        cache_dir=args.cache_dir,
        model_cache_dir=args.model_cache_dir,
        output_path=args.output,
        limit_papers=args.limit_papers,
        max_equations_per_paper=args.max_equations_per_paper,
        sleep_seconds=args.sleep_seconds,
        embedding_model=args.embedding_model,
        max_relation_edges=args.max_relation_edges,
    )

    dataset = pipeline.run()

    total_eq = sum(len(eqs) for eqs in dataset.values())
    logging.info(
        "Saved %s with %d papers and %d equations",
        args.output,
        len(dataset),
        total_eq,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract equation knowledge graph from arXiv papers using hybrid retrieval."
    )
    parser.add_argument("--paper-list", type=Path, default=DEFAULT_PAPER_LIST)
    parser.add_argument("--limit-papers", type=int, default=50)
    parser.add_argument("--max-equations-per-paper", type=int, default=7)
    parser.add_argument("--max-relation-edges", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep-seconds", type=float, default=15.0)
    parser.add_argument("--embedding-model", default="tbs17/MathBERT")
    return parser.parse_args()


if __name__ == "__main__":
    main()
