#!/usr/bin/env python3
"""Single-LLM FinCL tagging baseline.

This baseline is intentionally not agentic: it performs taxonomy retrieval,
asks one local Hugging Face causal LM to choose the best tag from the retrieved
candidates, and scores the final tag against the CSV answer column.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agentic_fincl.data import TaxonomyConcept, load_taxonomy  # noqa: E402
from agentic_fincl.evaluation import evaluate_single_llm_records, write_single_llm_breakdown  # noqa: E402
from agentic_fincl.evidence import build_evidence_builder  # noqa: E402
from agentic_fincl.rerankers import LlamaReranker  # noqa: E402
from agentic_fincl.retrieval import HybridTextIndex, TAXONOMY_DOC_MODES, normalize_scores, taxonomy_document  # noqa: E402


@dataclass(frozen=True)
class BaselineConfig:
    test_csv: Path
    taxonomy_jsonl: Path
    output_dir: Path
    model: str
    table_evidence_backend: str = "heuristic"
    table_evidence_model: str | None = None
    top_k: int = 200
    rerank_k: int = 200
    save_top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0
    bm25_weight: float = 1.0
    dense_weight: float = 0.0
    dense_model: str = ""
    taxonomy_doc_mode: str = "full"
    max_input_tokens: int = 12288
    max_new_tokens: int = 48


class TaxonomyOnlyRetriever:
    """Hybrid taxonomy retriever with no memory and no feedback loop."""

    def __init__(
        self,
        taxonomy: list[TaxonomyConcept],
        bm25_weight: float = 1.0,
        dense_weight: float = 0.0,
        dense_model: str = "",
        taxonomy_doc_mode: str = "full",
    ) -> None:
        self.taxonomy = taxonomy
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.by_type = self._index_concepts_by_type(taxonomy)
        docs = [taxonomy_document(concept, taxonomy_doc_mode) for concept in taxonomy]
        self.index = HybridTextIndex(docs, bm25_weight, dense_weight, dense_model)

    @staticmethod
    def _index_concepts_by_type(taxonomy: list[TaxonomyConcept]) -> dict[str, list[int]]:
        by_type: dict[str, list[int]] = {}
        for idx, concept in enumerate(taxonomy):
            by_type.setdefault(concept.entity_type, []).append(idx)
        return by_type

    def retrieve(self, entity: Any, entity_type: str, evidence: str, top_k: int) -> list[dict[str, Any]]:
        query = evidence
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        _, bm25, dense = self.index.scores(query)
        allowed_array = np.array(allowed, dtype=int)
        bm25_allowed = normalize_scores(bm25[allowed_array]) if allowed_array.size else np.zeros(0, dtype=float)
        dense_allowed = normalize_scores(dense[allowed_array]) if allowed_array.size else np.zeros(0, dtype=float)
        hybrid_allowed = self.bm25_weight * bm25_allowed + self.dense_weight * dense_allowed
        bm25_lookup = {idx: float(bm25_allowed[pos]) for pos, idx in enumerate(allowed)}
        dense_lookup = {idx: float(dense_allowed[pos]) for pos, idx in enumerate(allowed)}
        ranked = sorted(
            ((allowed[pos], float(score)) for pos, score in enumerate(hybrid_allowed)),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        return [
            {
                "rank": rank,
                "tag": self.taxonomy[idx].tag,
                "entity_type": self.taxonomy[idx].entity_type,
                "text": self.taxonomy[idx].text,
                "score": score,
                "bm25_score": bm25_lookup.get(idx, 0.0),
                "dense_score": dense_lookup.get(idx, 0.0),
                "memory_boost": 0.0,
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]


class SingleLLMBaseline:
    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = TaxonomyOnlyRetriever(
            self.taxonomy,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            dense_model=config.dense_model,
            taxonomy_doc_mode=config.taxonomy_doc_mode,
        )
        table_model = config.table_evidence_model or config.model
        self.evidence_builder = build_evidence_builder(config.table_evidence_backend, table_model)
        self.reranker = LlamaReranker(
            config.model,
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
        )

    def run(self) -> dict[str, Any]:
        df = pd.read_csv(self.config.test_csv)
        score = "answer" in df.columns
        phase_dir = self.config.output_dir / "single_llm"
        phase_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = phase_dir / "predictions.jsonl"

        records: list[dict[str, Any]] = []
        row_limit = min(len(df), self.config.limit) if self.config.limit else len(df)
        progress = tqdm(
            enumerate(df.itertuples(index=False), start=1),
            total=row_limit,
            desc="single_llm",
            unit="row",
            dynamic_ncols=True,
        )
        with predictions_path.open("w", encoding="utf-8") as f:
            for row_idx, row in progress:
                if self.config.limit and row_idx > self.config.limit:
                    break
                record = self._run_one(row_idx, row, score)
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.set_postfix(
                    correct=record["correct"],
                    prediction=record["prediction"]["Tag"][:40],
                    refresh=False,
                )

        metrics = evaluate_single_llm_records(
            records,
            score,
            self.config.recall_k,
            metadata={
                "model": self.config.model,
                "table_evidence_backend": self.config.table_evidence_backend,
                "table_evidence_model": (self.config.table_evidence_model or self.config.model)
                if self.config.table_evidence_backend == "llama"
                else None,
                "top_k": self.config.top_k,
                "rerank_k": self.config.rerank_k,
                "retrieval": "bm25" if self.config.dense_weight <= 0 else "hybrid_bm25_dense",
                "bm25_weight": self.config.bm25_weight,
                "dense_weight": self.config.dense_weight,
                "dense_model": (self.config.dense_model or "svd_fallback") if self.config.dense_weight > 0 else None,
                "taxonomy_doc_mode": self.config.taxonomy_doc_mode,
            },
        )
        metrics["predictions_path"] = str(predictions_path)
        with (phase_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_single_llm_breakdown(records, phase_dir, score)

        summary = {
            "mode": "single_llm_baseline",
            "primary_metric": {
                "phase": "single_llm",
                "num_examples": metrics.get("num_examples"),
                "tag_accuracy": metrics.get("tag_accuracy"),
            },
            "single_llm": metrics,
        }
        with (self.config.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _run_one(self, row_idx: int, row: Any, score: bool) -> dict[str, Any]:
        evidence = self.evidence_builder.build(row.context, row.category, row.entity, row.entity_type)
        candidates = self.retriever.retrieve(
            row.entity,
            row.entity_type,
            evidence,
            top_k=self.config.top_k,
        )
        candidate_tags = {candidate["tag"] for candidate in candidates}
        llm_tag = self.reranker.choose(
            row.entity,
            row.entity_type,
            evidence,
            candidates[: self.config.rerank_k],
        )
        final_tag = llm_tag if llm_tag in candidate_tags else (candidates[0]["tag"] if candidates else "")
        gold_tag = row.answer if score else None
        return {
            "row_index": row_idx - 1,
            "gold": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": gold_tag},
            "prediction": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": final_tag},
            "correct": final_tag == gold_tag if gold_tag is not None else None,
            "baseline": {
                "context_id": f"single_llm-{row_idx}",
                "category": row.category,
                "entity": {"value": str(row.entity), "type": row.entity_type},
                "evidence": evidence,
                "top_k": candidates[: self.config.save_top_k],
                "llm_selected_tag": llm_tag,
                "llm_selection_out_of_candidates": llm_tag not in candidate_tags,
            },
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-LLM FinCL tagging baseline.")
    parser.add_argument("--test", type=Path, default=REPO_ROOT / "data/FinCL-eval-subset-clean-test.csv")
    parser.add_argument("--taxonomy", type=Path, default=REPO_ROOT / "data/us_gaap_2024_BM25.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/single_llm_baseline")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--table-evidence-backend", choices=["heuristic", "llama"], default="heuristic")
    parser.add_argument("--table-evidence-model", default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rerank-k", type=int, default=200)
    parser.add_argument("--save-top-k", type=int, default=200)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-model",
        default="",
        help="Optional local sentence-transformer model for dense retrieval; empty uses the SVD fallback.",
    )
    parser.add_argument("--taxonomy-doc-mode", choices=TAXONOMY_DOC_MODES, default="full")
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        test_csv=args.test,
        taxonomy_jsonl=args.taxonomy,
        output_dir=args.output_dir,
        model=args.model,
        table_evidence_backend=args.table_evidence_backend,
        table_evidence_model=args.table_evidence_model,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        save_top_k=args.save_top_k,
        recall_k=tuple(args.recall_k),
        limit=args.limit,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        dense_model=args.dense_model,
        taxonomy_doc_mode=args.taxonomy_doc_mode,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    summary = SingleLLMBaseline(config_from_args(args)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
