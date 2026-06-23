"""Command line entry point for the equation knowledge graph prototype."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .arxiv_html import CRAWL_DELAY_SECONDS
from .pipeline import ExtractionPipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent

DEFAULT_PAPER_LIST = REPO_DIR / "paper_list_44.txt"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache"
DEFAULT_MODEL_CACHE_DIR = PROJECT_DIR / "data" / "model_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "output" / "sample_2_papers.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def main() -> None:
    """Run the extraction pipeline."""

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
        relation_threshold=args.relation_threshold,
    )
    dataset = pipeline.run()
    total_equations = sum(len(equations) for equations in dataset.values())
    logging.info("saved %s with %d papers and %d equations", args.output, len(dataset), total_equations)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Extract equation KG JSON from arXiv HTML.")
    parser.add_argument("--paper-list", type=Path, default=DEFAULT_PAPER_LIST)
    parser.add_argument("--limit-papers", type=int, default=2)
    parser.add_argument("--max-equations-per-paper", type=int, default=7)
    parser.add_argument("--max-relation-edges", type=int, default=2)
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=0.9,
        help="Cosine-similarity threshold above which two equation contexts form "
        "a 'potential' relation (calibrated for the MathBERT encoder).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep-seconds", type=float, default=CRAWL_DELAY_SECONDS)
    parser.add_argument("--embedding-model", default="tbs17/MathBERT")
    return parser.parse_args()


if __name__ == "__main__":
    main()
