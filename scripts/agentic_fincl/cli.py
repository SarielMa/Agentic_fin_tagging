from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic FinCL pipeline.")
    parser.add_argument("--memory", type=Path, default=Path("data/FinCL-eval-subset-clean-memory.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/FinCL-eval-subset-clean-test.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/us_gaap_2024_BM25.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agentic_fincl_initial"))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--memory-k", type=int, default=8)
    parser.add_argument("--memory-weight", type=float, default=0.10)
    parser.add_argument("--save-top-k", type=int, default=200)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--reranker", choices=["retrieval", "llama"], default="retrieval")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        memory_csv=args.memory,
        test_csv=args.test,
        taxonomy_jsonl=args.taxonomy,
        output_dir=args.output_dir,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        memory_k=args.memory_k,
        memory_weight=args.memory_weight,
        save_top_k=args.save_top_k,
        recall_k=tuple(args.recall_k),
        reranker=args.reranker,
        model=args.model,
        limit=args.limit,
    )


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(config_from_args(args))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
