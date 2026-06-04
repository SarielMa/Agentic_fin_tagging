#!/usr/bin/env python3
"""Single-LLM FinCL tagging baseline.

This baseline is intentionally not agentic: it performs taxonomy retrieval,
asks one local Hugging Face causal LM to choose the best tag from the retrieved
candidates, and scores the final tag against the CSV answer column.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agentic_fincl.data import TaxonomyConcept, load_taxonomy, tag_terms  # noqa: E402
from agentic_fincl.evidence import build_evidence_builder  # noqa: E402
from agentic_fincl.rerankers import LlamaReranker  # noqa: E402
from agentic_fincl.text_utils import build_query_text, normalize_space  # noqa: E402


@dataclass(frozen=True)
class BaselineConfig:
    test_csv: Path
    taxonomy_jsonl: Path
    output_dir: Path
    model: str
    table_evidence_backend: str = "heuristic"
    table_evidence_model: str | None = None
    top_k: int = 200
    rerank_k: int = 20
    save_top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0
    max_input_tokens: int = 4096
    max_new_tokens: int = 48


class TaxonomyOnlyRetriever:
    """TF-IDF taxonomy retriever with no memory and no feedback loop."""

    def __init__(self, taxonomy: list[TaxonomyConcept]) -> None:
        self.taxonomy = taxonomy
        self.by_type = self._index_concepts_by_type(taxonomy)
        docs = [
            normalize_space(f"{concept.text} {tag_terms(concept.tag)} {concept.entity_type}")
            for concept in taxonomy
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    @staticmethod
    def _index_concepts_by_type(taxonomy: list[TaxonomyConcept]) -> dict[str, list[int]]:
        by_type: dict[str, list[int]] = {}
        for idx, concept in enumerate(taxonomy):
            by_type.setdefault(concept.entity_type, []).append(idx)
        return by_type

    def retrieve(self, entity: Any, entity_type: str, evidence: str, top_k: int) -> list[dict[str, Any]]:
        query = build_query_text(entity, entity_type, evidence)
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        q_tax = self.vectorizer.transform([query])
        sims = cosine_similarity(q_tax, self.matrix[allowed]).ravel()
        ranked = sorted(
            ((allowed[pos], float(score)) for pos, score in enumerate(sims)),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            {
                "rank": rank,
                "tag": self.taxonomy[idx].tag,
                "entity_type": self.taxonomy[idx].entity_type,
                "text": self.taxonomy[idx].text,
                "score": score,
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]


class SingleLLMBaseline:
    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = TaxonomyOnlyRetriever(self.taxonomy)
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

        metrics = self._metrics(records, score)
        metrics["predictions_path"] = str(predictions_path)
        with (phase_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self._write_breakdown(records, phase_dir, score)

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

    def _metrics(self, records: list[dict[str, Any]], score: bool) -> dict[str, Any]:
        n = len(records)
        metrics: dict[str, Any] = {
            "num_examples": n,
            "model": self.config.model,
            "table_evidence_backend": self.config.table_evidence_backend,
            "table_evidence_model": self.config.table_evidence_model or self.config.model
            if self.config.table_evidence_backend == "llama"
            else None,
            "top_k": self.config.top_k,
            "rerank_k": self.config.rerank_k,
            "out_of_candidate_rate": (
                sum(1 for record in records if record["baseline"]["llm_selection_out_of_candidates"]) / n
                if n
                else math.nan
            ),
        }
        if not score:
            return metrics

        correct = sum(1 for record in records if record["correct"])
        recall_counts = {k: 0 for k in self.config.recall_k}
        for record in records:
            gold_tag = record["gold"]["Tag"]
            candidate_tags = [candidate["tag"] for candidate in record["baseline"]["top_k"]]
            for k in self.config.recall_k:
                recall_counts[k] += int(gold_tag in candidate_tags[:k])
        metrics["tag_accuracy"] = correct / n if n else math.nan
        metrics["recall_at_k"] = {str(k): recall_counts[k] / n if n else math.nan for k in self.config.recall_k}
        return metrics

    @staticmethod
    def _write_breakdown(records: list[dict[str, Any]], phase_dir: Path, score: bool) -> None:
        if not records or not score:
            return
        rows = []
        for record in records:
            candidates = [candidate["tag"] for candidate in record["baseline"]["top_k"]]
            rows.append(
                {
                    "category": record["baseline"]["category"],
                    "entity_type": record["gold"]["Type"],
                    "correct": record["correct"],
                    "recall20": record["gold"]["Tag"] in candidates[:20],
                    "recall50": record["gold"]["Tag"] in candidates[:50],
                    "recall100": record["gold"]["Tag"] in candidates[:100],
                    "recall200": record["gold"]["Tag"] in candidates[:200],
                }
            )
        df = pd.DataFrame(rows)
        breakdown = {
            "by_category": SingleLLMBaseline._group_breakdown(df, "category"),
            "by_entity_type": SingleLLMBaseline._group_breakdown(df, "entity_type"),
        }
        with (phase_dir / "breakdown.json").open("w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2)

    @staticmethod
    def _group_breakdown(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for key, group in df.groupby(column):
            result[str(key)] = {
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "recall20": float(group["recall20"].mean()),
                "recall50": float(group["recall50"].mean()),
                "recall100": float(group["recall100"].mean()),
                "recall200": float(group["recall200"].mean()),
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-LLM FinCL tagging baseline.")
    parser.add_argument("--test", type=Path, default=REPO_ROOT / "data/FinCL-eval-subset-clean-test.csv")
    parser.add_argument("--taxonomy", type=Path, default=REPO_ROOT / "data/us_gaap_2024_BM25.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/single_llm_baseline")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--table-evidence-backend", choices=["heuristic", "llama"], default="heuristic")
    parser.add_argument("--table-evidence-model", default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--save-top-k", type=int, default=200)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
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
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    summary = SingleLLMBaseline(config_from_args(args)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
