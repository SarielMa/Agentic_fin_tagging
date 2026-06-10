from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import AgenticExperiment, ExperimentConfig
from .retrieval import TAXONOMY_DOC_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-agent FinCL experiment.")
    parser.add_argument("--mode", choices=["offline", "online_with_gt", "online_without_gt"], required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/us_gaap_2024_BM25.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--memory-build", type=Path, default=Path("data/FinCL-eval-subset-clean-memory.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/FinCL-eval-subset-clean-test.csv"))
    parser.add_argument("--stream", type=Path, default=None)

    parser.add_argument("--selector-backend", choices=["retrieval", "llama"], default="retrieval")
    parser.add_argument("--selector-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--validator-backend", choices=["rule", "llama"], default="rule")
    parser.add_argument("--validator-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--table-evidence-backend", choices=["heuristic", "llama"], default="heuristic")
    parser.add_argument("--table-evidence-model", default="meta-llama/Llama-3.2-3B-Instruct")

    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rerank-k", type=int, default=200)
    parser.add_argument("--memory-k", type=int, default=8)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-model",
        default="",
        help="Optional local sentence-transformer model for dense retrieval; empty uses the SVD fallback.",
    )
    parser.add_argument("--taxonomy-doc-mode", choices=TAXONOMY_DOC_MODES, default="full")
    parser.add_argument("--memory-weight", type=float, default=0.10)
    parser.add_argument("--error-weight", type=float, default=0.05)
    parser.add_argument("--table-pattern-weight", type=float, default=0.05)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument(
        "--supervised-memory-iters",
        type=int,
        default=0,
        help="Post-prediction supervised selector refinement iterations used only to improve LTM writes.",
    )
    parser.add_argument("--save-top-k", type=int, default=200)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit per phase.")
    parser.add_argument("--resume-ltm", action="store_true", help="Resume from existing LTM records in --output-dir.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        mode=args.mode,
        taxonomy_jsonl=args.taxonomy,
        output_dir=args.output_dir,
        memory_build_csv=args.memory_build,
        test_csv=args.test,
        stream_csv=args.stream,
        selector_backend=args.selector_backend,
        selector_model=args.selector_model,
        validator_backend=args.validator_backend,
        validator_model=args.validator_model,
        table_evidence_backend=args.table_evidence_backend,
        table_evidence_model=args.table_evidence_model,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        memory_k=args.memory_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        dense_model=args.dense_model,
        taxonomy_doc_mode=args.taxonomy_doc_mode,
        memory_weight=args.memory_weight,
        error_weight=args.error_weight,
        table_pattern_weight=args.table_pattern_weight,
        max_iters=args.max_iters,
        supervised_memory_iters=args.supervised_memory_iters,
        save_top_k=args.save_top_k,
        recall_k=tuple(args.recall_k),
        limit=args.limit,
        resume_ltm=args.resume_ltm,
    )


def main() -> None:
    args = parse_args()
    summary = AgenticExperiment(config_from_args(args)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
