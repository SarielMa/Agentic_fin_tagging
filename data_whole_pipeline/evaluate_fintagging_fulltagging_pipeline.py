#!/usr/bin/env python3
"""Summarize fullTagging extraction-to-grounding metrics.

This evaluator does not replace the grounding evaluator. It adds a gold-entity
scope view so extractor misses are counted as failures before grounding.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from evaluate_fintagging_context_extraction import (
    normalize_datatype,
    normalize_numeric_entity,
)
from run_fintagging_grounding_baseline import normalize_tag


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_TEST = SCRIPT_DIR / "FinTagging_800_200_HF" / "data" / "test.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-test-parquet", type=Path, default=DEFAULT_ORIGINAL_TEST)
    parser.add_argument("--grounding-input-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, default=None)
    parser.add_argument("--rerank-predictions-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[10, 50, 200])
    return parser.parse_args()


def has_table_markup(text: Any) -> bool:
    return "<table" in str(text).lower()


def iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entry in value:
            if isinstance(entry, dict):
                yield entry


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold_entities(path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    gold: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        source_sample_idx = int(row.get("source_sample_idx", row_idx))
        context_id = int(row.get("context_id", source_sample_idx))
        input_type = "table" if has_table_markup(row.get("text", "")) else "text"
        for entity_idx, entity in enumerate(iter_records(row.get("numeric_entities", []))):
            value = normalize_numeric_entity(entity.get("value", entity.get("numeric_entity")))
            datatype = normalize_datatype(entity.get("type", entity.get("datatype")))
            concept = normalize_tag(entity.get("concept"))
            if value and datatype and concept:
                gold.append(
                    {
                        "source_sample_idx": source_sample_idx,
                        "context_id": context_id,
                        "input_type": input_type,
                        "entity_idx": int(entity_idx),
                        "numeric_entity": value,
                        "datatype": datatype,
                        "concept": concept,
                    }
                )
    return gold


def ranking_metric(ranking: list[str], gold_tag: str, top_ks: list[int]) -> dict[str, Any]:
    rank = None
    gold_tag = normalize_tag(gold_tag)
    for idx, tag in enumerate(ranking, start=1):
        if normalize_tag(tag) == gold_tag:
            rank = idx
            break
    row: dict[str, Any] = {
        "rank": rank,
        "accuracy": 1.0 if rank == 1 else 0.0,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }
    for top_k in top_ks:
        row[f"recall_at_{top_k}"] = 1.0 if rank is not None and rank <= top_k else 0.0
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n": len(rows)}
    for key in ["accuracy", "mrr", *(f"recall_at_{top_k}" for top_k in top_ks)]:
        metrics[key] = (
            round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
            if rows
            else 0.0
        )
    return metrics


def candidate_ranking(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    return [normalize_tag(candidate.get("tag")) for candidate in row.get("candidates", [])]


def rerank_ranking(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    return [normalize_tag(tag) for tag in row.get("final_ranking", []) if normalize_tag(tag)]


def matched_row_by_gold(
    grounding_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    by_gold: dict[tuple[int, int], dict[str, Any]] = {}
    for row in grounding_rows:
        source_sample_idx = int(row["source_sample_idx"])
        for entity_idx in row.get("source_entity_indices", []):
            by_gold[(source_sample_idx, int(entity_idx))] = row
    return by_gold


def summarize_scope(
    gold_entities: list[dict[str, Any]],
    by_gold: dict[tuple[int, int], dict[str, Any]],
    candidate_by_example: dict[int, dict[str, Any]],
    rerank_by_example: dict[int, dict[str, Any]],
    top_ks: list[int],
) -> dict[str, Any]:
    bm25_rows: list[dict[str, Any]] = []
    rerank_rows: list[dict[str, Any]] = []
    extraction_matched = 0
    rerank_available = bool(rerank_by_example)

    for gold in gold_entities:
        row = by_gold.get((int(gold["source_sample_idx"]), int(gold["entity_idx"])))
        if row is not None:
            extraction_matched += 1
            example_idx = int(row["example_idx"])
            bm25 = candidate_ranking(candidate_by_example.get(example_idx))
            rerank = rerank_ranking(rerank_by_example.get(example_idx)) if rerank_available else []
        else:
            bm25 = []
            rerank = []

        bm25_rows.append(ranking_metric(bm25, gold["concept"], top_ks))
        if rerank_available:
            rerank_rows.append(ranking_metric(rerank, gold["concept"], top_ks))

    summary: dict[str, Any] = {
        "n_gold_entities": len(gold_entities),
        "extraction_matched_gold_entities": extraction_matched,
        "extraction_recall_by_value_type": round(
            extraction_matched / len(gold_entities), 6
        )
        if gold_entities
        else 0.0,
        "bm25_gold_entity_scope": aggregate_metric_rows(bm25_rows, top_ks),
    }
    if rerank_available:
        summary["rerank_gold_entity_scope"] = aggregate_metric_rows(rerank_rows, top_ks)
    return summary


def main() -> None:
    args = parse_args()
    top_ks = sorted(set(args.top_ks))

    gold_entities = load_gold_entities(args.original_test_parquet, args.limit)
    grounding_rows = load_jsonl(args.grounding_input_jsonl)
    candidate_rows = load_jsonl(args.candidate_jsonl)
    rerank_rows = load_jsonl(args.rerank_predictions_jsonl)

    by_gold = matched_row_by_gold(grounding_rows)
    candidate_by_example = {int(row["example_idx"]): row for row in candidate_rows}
    rerank_by_example = {int(row["example_idx"]): row for row in rerank_rows}

    overall = summarize_scope(
        gold_entities,
        by_gold,
        candidate_by_example,
        rerank_by_example,
        top_ks,
    )

    by_input_type: dict[str, Any] = {}
    for input_type in sorted({row["input_type"] for row in gold_entities}):
        scoped_gold = [row for row in gold_entities if row["input_type"] == input_type]
        by_input_type[input_type] = summarize_scope(
            scoped_gold,
            by_gold,
            candidate_by_example,
            rerank_by_example,
            top_ks,
        )

    grounding_row_count = len(grounding_rows)
    rows_with_no_gold = sum(1 for row in grounding_rows if not row.get("ground_truth_concepts"))
    result = {
        "metric_scope": "gold entities from original FinTagging test parquet",
        "note": (
            "Grounding candidates/rerank are still evaluated with the original "
            "grounding logic; this file additionally counts extraction misses as "
            "zero-recall grounding failures."
        ),
        "original_test_parquet": str(args.original_test_parquet),
        "limit": args.limit,
        "grounding_input_jsonl": str(args.grounding_input_jsonl),
        "candidate_jsonl": str(args.candidate_jsonl) if args.candidate_jsonl else None,
        "rerank_predictions_jsonl": str(args.rerank_predictions_jsonl)
        if args.rerank_predictions_jsonl
        else None,
        "grounding_row_count": grounding_row_count,
        "grounding_rows_with_no_gold_match": rows_with_no_gold,
        "grounding_rows_with_no_gold_match_rate": round(rows_with_no_gold / grounding_row_count, 6)
        if grounding_row_count
        else 0.0,
        "gold_input_type_counts": dict(Counter(row["input_type"] for row in gold_entities)),
        **overall,
        "by_input_type": by_input_type,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
