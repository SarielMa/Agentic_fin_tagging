# CLI for the memoryless single-LLM testing baseline.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .single_llm import SingleLLMConfig, SingleLLMTester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the memoryless single-LLM test baseline.")
    parser.add_argument("--taxonomy", type=Path, default=Path("data/us_gaap_2024_BM25.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/FinCL-eval-subset-clean-test.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--limit", type=int, default=0, help="0 means use the full CSV.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SingleLLMConfig:
    return SingleLLMConfig(
        taxonomy_jsonl=args.taxonomy,
        test_csv=args.test,
        output_dir=args.output_dir,
        model=args.model,
        top_k=args.top_k,
        recall_k=tuple(args.recall_k),
        limit=args.limit,
    )


def main() -> None:
    args = parse_args()
    summary = SingleLLMTester(config_from_args(args)).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
