#!/usr/bin/env python3
"""Evaluate FinTagging value/type extraction predictions without FinBen.

Gold answers are JSON arrays:

[
  {"numeric_entity": "62", "datatype": "monetaryItemType"}
]

The evaluator also accepts PV-style {"results": [...]} predictions for
robustness, but the SFT dataset built in this repo uses a plain JSON array.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk


DATATYPES = {
    "monetaryItemType",
    "percentItemType",
    "sharesItemType",
    "perShareItemType",
    "integerItemType",
}
DATATYPE_BY_LOWER = {item.lower(): item for item in DATATYPES}
PREDICTION_COLUMNS = ("prediction", "pred", "response", "generated_text", "output", "answer")
ID_COLUMNS = ("source_sample_idx", "context_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinTagging numeric entity/datatype pair extraction."
    )
    parser.add_argument(
        "--gold-dataset",
        default="FinTagging_800_200_value_type_sft_arrow",
        help="Arrow DatasetDict path created by build_fintagging_value_type_sft_dataset.py.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Gold split to evaluate. Default: %(default)s",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "JSONL/CSV/Parquet file with predictions. If omitted, gold answers "
            "are used as predictions for a sanity check."
        ),
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Prediction text column. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for aggregate metrics JSON.",
    )
    parser.add_argument(
        "--per-row-csv",
        default=None,
        help="Optional path for per-row evaluation CSV.",
    )
    return parser.parse_args()


def normalize_numeric_entity(value: Any) -> str:
    text = str(value).strip()
    text = text.strip('"').strip("'").strip()
    if text in {"-", "—", "–"}:
        return text

    paren_match = re.fullmatch(r"\(\s*(.*?)\s*\)", text)
    if paren_match:
        text = paren_match.group(1)

    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_datatype(value: Any) -> str:
    text = str(value).strip()
    return DATATYPE_BY_LOWER.get(text.lower(), text)


def extract_balanced_json(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def json_loads_robust(text: Any) -> tuple[Any, bool]:
    if isinstance(text, (list, dict)):
        return text, True
    if text is None:
        return [], False

    raw = str(text).strip()
    if not raw:
        return [], False

    for candidate in (raw, raw.replace("'", '"')):
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    for start_char, end_char in (("[", "]"), ("{", "}")):
        candidate = extract_balanced_json(raw, start_char, end_char)
        if candidate is None:
            continue
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            try:
                return json.loads(candidate.replace("'", '"')), True
            except json.JSONDecodeError:
                pass

    return [], False


def entries_from_json(value: Any) -> list[dict[str, Any]]:
    parsed, ok = json_loads_robust(value)
    if not ok:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("results", [])
    if not isinstance(parsed, list):
        return []
    return [entry for entry in parsed if isinstance(entry, dict)]


def entry_get_case_insensitive(entry: dict[str, Any], aliases: set[str]) -> Any:
    normalized = {str(key).lower().replace("-", "_").replace(" ", "_"): value for key, value in entry.items()}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def pair_counter(answer: Any) -> tuple[Counter[tuple[str, str]], bool, int]:
    parsed, parse_ok = json_loads_robust(answer)
    if isinstance(parsed, dict):
        parsed = parsed.get("results", [])
    if not isinstance(parsed, list):
        return Counter(), False, 0

    counter: Counter[tuple[str, str]] = Counter()
    valid_entry_count = 0
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        numeric_entity = entry_get_case_insensitive(
            entry,
            {"numeric_entity", "numericentity", "entity", "value", "numeric_value", "number"},
        )
        datatype = entry_get_case_insensitive(
            entry,
            {"datatype", "data_type", "type"},
        )
        if numeric_entity is None or datatype is None:
            continue
        pair = (normalize_numeric_entity(numeric_entity), normalize_datatype(datatype))
        counter[pair] += 1
        valid_entry_count += 1

    return counter, parse_ok, valid_entry_count


def prf(tp: int, pred_total: int, gold_total: int) -> dict[str, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            return [row for row in payload["predictions"] if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON prediction structure: {path}")
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported prediction file extension: {path.suffix}")


def choose_prediction_column(rows: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        return requested
    if not rows:
        raise ValueError("Prediction file is empty")
    columns = set(rows[0])
    for column in PREDICTION_COLUMNS:
        if column in columns:
            return column
    raise ValueError(
        f"Could not auto-detect prediction column. Available columns: {sorted(columns)}"
    )


def align_predictions(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    prediction_column: str,
) -> list[str]:
    if not prediction_rows:
        raise ValueError("Prediction file is empty")

    for id_column in ID_COLUMNS:
        if id_column in prediction_rows[0]:
            pred_by_id = {str(row[id_column]): row[prediction_column] for row in prediction_rows}
            missing = [row[id_column] for row in gold_rows if str(row[id_column]) not in pred_by_id]
            if missing:
                raise ValueError(
                    f"Predictions missing {len(missing)} gold rows by {id_column}; first missing: {missing[:5]}"
                )
            return [pred_by_id[str(row[id_column])] for row in gold_rows]

    if len(prediction_rows) != len(gold_rows):
        raise ValueError(
            f"Prediction row count {len(prediction_rows)} does not match gold row count {len(gold_rows)}"
        )
    return [row[prediction_column] for row in prediction_rows]


def evaluate(gold_rows: list[dict[str, Any]], predictions: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_tp = pair_pred_total = pair_gold_total = 0
    dtype_tp = dtype_pred_total = dtype_gold_total = 0
    entity_tp = entity_pred_total = entity_gold_total = 0
    exact_match_count = 0
    parse_success_count = 0
    valid_pred_entry_count = 0
    per_row: list[dict[str, Any]] = []

    for row, prediction in zip(gold_rows, predictions):
        gold_pairs, _, gold_valid_entries = pair_counter(row["answer"])
        pred_pairs, pred_parse_ok, pred_valid_entries = pair_counter(prediction)

        pair_matches = sum((gold_pairs & pred_pairs).values())
        pair_tp += pair_matches
        pair_pred_total += sum(pred_pairs.values())
        pair_gold_total += sum(gold_pairs.values())

        gold_dtypes = Counter(dtype for _, dtype in gold_pairs.elements())
        pred_dtypes = Counter(dtype for _, dtype in pred_pairs.elements())
        dtype_matches = sum((gold_dtypes & pred_dtypes).values())
        dtype_tp += dtype_matches
        dtype_pred_total += sum(pred_dtypes.values())
        dtype_gold_total += sum(gold_dtypes.values())

        gold_entities = Counter(entity for entity, _ in gold_pairs.elements())
        pred_entities = Counter(entity for entity, _ in pred_pairs.elements())
        entity_matches = sum((gold_entities & pred_entities).values())
        entity_tp += entity_matches
        entity_pred_total += sum(pred_entities.values())
        entity_gold_total += sum(gold_entities.values())

        exact = gold_pairs == pred_pairs
        exact_match_count += int(exact)
        parse_success_count += int(pred_parse_ok)
        valid_pred_entry_count += pred_valid_entries

        per_row.append(
            {
                "source_sample_idx": row["source_sample_idx"],
                "context_id": row["context_id"],
                "gold_entry_count": gold_valid_entries,
                "pred_entry_count": pred_valid_entries,
                "pair_matches": pair_matches,
                "datatype_matches": dtype_matches,
                "numeric_entity_matches": entity_matches,
                "json_parse_ok": pred_parse_ok,
                "exact_match": exact,
            }
        )

    n = len(gold_rows)
    metrics = {
        "sample_count": n,
        "json_parse_success_rate": round(parse_success_count / n, 6) if n else 0.0,
        "exact_row_match_rate": round(exact_match_count / n, 6) if n else 0.0,
        "gold_entry_count": pair_gold_total,
        "pred_entry_count": pair_pred_total,
        "valid_pred_entry_count": valid_pred_entry_count,
        "pair_exact": {
            **prf(pair_tp, pair_pred_total, pair_gold_total),
            "true_positive": pair_tp,
        },
        "datatype_only": {
            **prf(dtype_tp, dtype_pred_total, dtype_gold_total),
            "true_positive": dtype_tp,
        },
        "numeric_entity_only": {
            **prf(entity_tp, entity_pred_total, entity_gold_total),
            "true_positive": entity_tp,
        },
    }
    return metrics, per_row


def main() -> None:
    args = parse_args()
    ds = load_from_disk(args.gold_dataset)[args.split]
    gold_rows = [dict(row) for row in ds]

    if args.predictions is None:
        predictions = [row["answer"] for row in gold_rows]
        prediction_source = "gold_answers_sanity_check"
    else:
        prediction_rows = load_prediction_rows(Path(args.predictions))
        prediction_column = choose_prediction_column(prediction_rows, args.prediction_column)
        predictions = align_predictions(gold_rows, prediction_rows, prediction_column)
        prediction_source = args.predictions

    metrics, per_row = evaluate(gold_rows, predictions)
    metrics["gold_dataset"] = args.gold_dataset
    metrics["split"] = args.split
    metrics["prediction_source"] = prediction_source

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.per_row_csv:
        output_path = Path(args.per_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_row).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
