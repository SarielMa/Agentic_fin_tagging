#!/usr/bin/env python3
"""Build grounding-style input from fullTagging extraction predictions.

The grounding query intentionally uses only:

1. extracted numeric entity,
2. extracted datatype,
3. the original source context.

Fine extraction contexts such as sentence_context, row_context, and column_context
are retained only as provenance fields and are not included in input_fields.
This keeps direct retrieval and one-pass grounding identical to the weak
grounding baselines, except that extracted entities replace gold entities.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from evaluate_fintagging_context_extraction import (
    entries_from_answer,
    entry_get_case_insensitive,
    load_prediction_rows,
    normalize_datatype,
    normalize_numeric_entity,
)
from generate_fintagging_fulltagging_extractions import has_table_markup
from run_fintagging_grounding_baseline import normalize_space, normalize_tag


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_TEST = SCRIPT_DIR / "FinTagging_800_200_HF" / "data" / "test.parquet"
DEFAULT_TEXT_PREDICTIONS = (
    SCRIPT_DIR
    / "runs_fintagging_text_context"
    / "llama3.3_70b_instruct"
    / "sft_3ep"
    / "predictions"
    / "test_predictions.jsonl"
)
DEFAULT_TABLE_PREDICTIONS = (
    SCRIPT_DIR
    / "runs_fintagging_table_context"
    / "llama3.3_70b_instruct"
    / "sft_3ep"
    / "predictions"
    / "test_predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-test-parquet", type=Path, default=DEFAULT_ORIGINAL_TEST)
    parser.add_argument(
        "--extraction-predictions",
        type=Path,
        default=None,
        help="Combined fullTagging extraction JSONL with input_type and prediction columns.",
    )
    parser.add_argument("--text-predictions", type=Path, default=DEFAULT_TEXT_PREDICTIONS)
    parser.add_argument("--table-predictions", type=Path, default=DEFAULT_TABLE_PREDICTIONS)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entry in value:
            if isinstance(entry, dict):
                yield entry


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def input_type_for_context(text: str) -> str:
    return "table" if has_table_markup(text) else "text"


def load_original_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        source_sample_idx = int(row.get("source_sample_idx", row_idx))
        context_id = int(row.get("context_id", source_sample_idx))
        original_context = str(row.get("text", ""))
        numeric_entities = list(iter_records(row.get("numeric_entities", [])))
        rows.append(
            {
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": str(row.get("split", "test")),
                "original_context": original_context,
                "input_type": input_type_for_context(original_context),
                "numeric_entities": numeric_entities,
            }
        )
    return rows


def gold_index_by_pair(
    numeric_entities: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity_idx, entity in enumerate(numeric_entities):
        value = entity.get("value", entity.get("numeric_entity"))
        entity_type = entity.get("type", entity.get("datatype"))
        concept = normalize_tag(entity.get("concept"))
        key = (normalize_numeric_entity(value), normalize_datatype(entity_type))
        if key[0] and key[1] and concept:
            by_pair[key].append(
                {
                    "entity_index": entity_idx,
                    "concept": concept,
                    "value": key[0],
                    "datatype": key[1],
                }
            )
    return by_pair


def load_combined_extraction_predictions(path: Path) -> list[dict[str, Any]]:
    rows = load_prediction_rows(path)
    normalized = []
    for row in rows:
        task = normalize_space(row.get("input_type"))
        if task not in {"text", "table"}:
            raise ValueError(f"Combined prediction row missing valid input_type: {row}")
        normalized.append({**row, "input_type": task})
    return normalized


def load_split_extraction_predictions(text_path: Path, table_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, path in (("text", text_path), ("table", table_path)):
        for row in load_prediction_rows(path):
            rows.append({**row, "input_type": task})
    return rows


def prediction_text(row: dict[str, Any]) -> Any:
    for key in ("prediction", "pred", "response", "generated_text", "output", "answer"):
        if key in row:
            return row[key]
    raise ValueError(f"Prediction row has no recognized prediction column: {sorted(row)}")


def extracted_pair(entry: dict[str, Any]) -> tuple[str, str] | None:
    numeric_entity = entry_get_case_insensitive(
        entry,
        {"numeric_entity", "numericentity", "entity", "value", "numeric_value", "number"},
    )
    datatype = entry_get_case_insensitive(entry, {"datatype", "data_type", "type"})
    value = normalize_numeric_entity(numeric_entity)
    entity_type = normalize_datatype(datatype)
    if not value or not entity_type:
        return None
    return value, entity_type


def provenance_fields(entry: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "table":
        return {
            "row_context": entry_get_case_insensitive(
                entry,
                {"row_context", "rowcontext", "row_header", "row", "row_label"},
            ),
            "column_context": entry_get_case_insensitive(
                entry,
                {
                    "column_context",
                    "columncontext",
                    "col_context",
                    "col_header",
                    "column_header",
                    "column",
                    "col",
                },
            ),
        }
    return {
        "sentence_context": entry_get_case_insensitive(
            entry,
            {"sentence_context", "sentencecontext", "sentence", "context"},
        )
    }


def input_group_key(input_type: str, input_fields: dict[str, Any]) -> str:
    return json.dumps(
        {"input_type": input_type, "input_fields": input_fields},
        sort_keys=True,
        ensure_ascii=False,
    )


def build_grounding_rows(
    original_rows: list[dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_by_idx = {int(row["source_sample_idx"]): row for row in original_rows}
    gold_by_source = {
        int(row["source_sample_idx"]): gold_index_by_pair(row["numeric_entities"])
        for row in original_rows
    }

    grouped: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    matched_gold_entity_indices: set[tuple[int, int]] = set()

    for pred_row in extraction_rows:
        source_sample_idx = int(pred_row["source_sample_idx"])
        task = normalize_space(pred_row.get("input_type"))
        original = original_by_idx.get(source_sample_idx)
        if original is None:
            stats["prediction_rows_without_original"] += 1
            continue
        if original["input_type"] != task:
            stats["prediction_rows_wrong_routed_type"] += 1
            continue

        entries, parse_ok = entries_from_answer(prediction_text(pred_row))
        stats["prediction_rows"] += 1
        stats[f"{task}_prediction_rows"] += 1
        stats["prediction_parse_ok"] += int(parse_ok)
        stats["raw_predicted_entries"] += len(entries)

        for pred_entry_idx, entry in enumerate(entries):
            pair = extracted_pair(entry)
            if pair is None:
                stats["invalid_predicted_entries"] += 1
                continue

            value, entity_type = pair
            input_fields = {
                "numeric_entity": value,
                "datatype": entity_type,
                "original_context": original["original_context"],
            }
            key = input_group_key(task, input_fields)
            gold_matches = gold_by_source[source_sample_idx].get(pair, [])
            gold_tags = ordered_unique(match["concept"] for match in gold_matches)
            gold_indices = [int(match["entity_index"]) for match in gold_matches]
            for entity_idx in gold_indices:
                matched_gold_entity_indices.add((source_sample_idx, entity_idx))

            if key not in grouped:
                grouped[key] = {
                    "source_sample_idx": source_sample_idx,
                    "context_id": original["context_id"],
                    "split": original["split"],
                    "input_type": task,
                    "input_fields": input_fields,
                    "input": json.dumps(input_fields, ensure_ascii=False),
                    "ground_truth_concepts": [],
                    "source_entity_indices": [],
                    "source_occurrence_count": 0,
                    "source_prediction_indices": [],
                    "source_prediction_provenance": [],
                }

            row = grouped[key]
            row["ground_truth_concepts"] = ordered_unique(row["ground_truth_concepts"] + gold_tags)
            row["source_entity_indices"] = sorted(set(row["source_entity_indices"]) | set(gold_indices))
            row["source_occurrence_count"] = len(row["source_entity_indices"])
            row["source_prediction_indices"].append(pred_entry_idx)
            row["source_prediction_provenance"].append(provenance_fields(entry, task))
            stats["valid_predicted_entries"] += 1
            stats[f"{task}_valid_predicted_entries"] += 1
            stats["matched_predicted_entries"] += int(bool(gold_tags))
            stats["unmatched_predicted_entries"] += int(not gold_tags)

    rows = list(grouped.values())
    rows.sort(
        key=lambda row: (
            int(row["source_sample_idx"]),
            row["input_type"],
            row["input_fields"]["numeric_entity"],
            row["input_fields"]["datatype"],
        )
    )
    for example_idx, row in enumerate(rows):
        row["example_idx"] = example_idx
        row["ground_truth_count"] = len(row["ground_truth_concepts"])
        row["output"] = json.dumps(row["ground_truth_concepts"], ensure_ascii=False)
        row["predicted_occurrence_count"] = len(row["source_prediction_indices"])

    total_gold_entities = sum(len(row["numeric_entities"]) for row in original_rows)
    metadata = {
        "input_format": "grounding JSONL generated from extractor predictions",
        "grounding_query_fields": ["numeric_entity", "datatype", "original_context"],
        "fine_context_fields_used_for_grounding": False,
        "original_sample_count": len(original_rows),
        "original_input_type_counts": {
            task: sum(1 for row in original_rows if row["input_type"] == task)
            for task in ("text", "table")
        },
        "gold_entity_count": total_gold_entities,
        "matched_gold_entity_count_by_value_type": len(matched_gold_entity_indices),
        "grounding_row_count": len(rows),
        "grounding_rows_with_no_gold_match": sum(
            1 for row in rows if not row["ground_truth_concepts"]
        ),
        "stats": {key: int(value) for key, value in sorted(stats.items())},
    }
    return rows, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    original_rows = load_original_rows(args.original_test_parquet, args.limit)
    if args.extraction_predictions is not None:
        extraction_rows = load_combined_extraction_predictions(args.extraction_predictions)
    else:
        extraction_rows = load_split_extraction_predictions(
            args.text_predictions,
            args.table_predictions,
        )
    if args.limit is not None:
        allowed = {int(row["source_sample_idx"]) for row in original_rows}
        extraction_rows = [
            row for row in extraction_rows if int(row["source_sample_idx"]) in allowed
        ]

    rows, metadata = build_grounding_rows(original_rows, extraction_rows)
    metadata.update(
        {
            "original_test_parquet": str(args.original_test_parquet),
            "extraction_predictions": str(args.extraction_predictions)
            if args.extraction_predictions is not None
            else None,
            "text_predictions": str(args.text_predictions)
            if args.extraction_predictions is None
            else None,
            "table_predictions": str(args.table_predictions)
            if args.extraction_predictions is None
            else None,
            "output_jsonl": str(args.output_jsonl),
        }
    )

    write_jsonl(args.output_jsonl, rows)
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
