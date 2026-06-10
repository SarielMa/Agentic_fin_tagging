# CLI for running the minimal three-agent baseline modes.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import BaselineConfig, BaselinePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal three-agent FinAI tagging baseline.")
    parser.add_argument("--mode", choices=["offline", "online_gt", "online_wo_gt"], required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/us_gaap_2024_BM25.jsonl"))
    parser.add_argument("--memory-build", type=Path, default=Path("data/FinCL-eval-subset-clean-memory.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/FinCL-eval-subset-clean-test.csv"))
    parser.add_argument("--stream", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--validator-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--memory-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--limit", type=int, default=0, help="0 means use the full CSV.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        mode=args.mode,
        taxonomy_jsonl=args.taxonomy,
        output_dir=args.output_dir,
        memory_build_csv=args.memory_build,
        test_csv=args.test,
        stream_csv=args.stream,
        selector_model=args.selector_model,
        validator_model=args.validator_model,
        top_k=args.top_k,
        memory_k=args.memory_k,
        recall_k=tuple(args.recall_k),
        limit=args.limit,
    )


def main() -> None:
    args = parse_args()
    summary = BaselinePipeline(config_from_args(args)).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
