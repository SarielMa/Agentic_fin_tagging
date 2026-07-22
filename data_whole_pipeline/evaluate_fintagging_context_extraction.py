#!/usr/bin/env python3
"""Evaluate FinTagging table/text context extraction predictions.

Gold answers are JSON arrays.

Table task:
[
  {
    "numeric_entity": "62",
    "datatype": "monetaryItemType",
    "row_context": "Current | U.S. Federal",
    "column_context": "Provision for Income Taxes | 2024"
  }
]

Text task:
[
  {
    "numeric_entity": "250",
    "datatype": "monetaryItemType",
    "sentence_context": "On February 3, 2025, we repaid $ 250 million ..."
  }
]
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
        description="Evaluate FinTagging context extraction predictions."
    )
    parser.add_argument(
        "--gold-dataset",
        required=True,
        help="Arrow DatasetDict path for table or text context extraction SFT data.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["table", "text"],
        help="Evaluation task schema.",
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
        help="JSONL/CSV/Parquet prediction file. If omitted, gold answers are used.",
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Prediction text column. If omitted, common names are auto-detected.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--per-row-csv", default=None)
    return parser.parse_args()


def normalize_numeric_entity(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = text.strip('"').strip("'").strip()
    if text in {"-", "-", "-", "-"}:
        return "-"

    paren_match = re.fullmatch(r"\(\s*(.*?)\s*\)", text)
    if paren_match:
        text = paren_match.group(1)

    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2015", "-")
    text = text.replace("\u2212", "-")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_datatype(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return DATATYPE_BY_LOWER.get(text.lower(), text)


def normalize_context(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "na", "n/a"}:
        return None
    return re.sub(r"\s+", " ", text)


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


def entry_get_case_insensitive(entry: dict[str, Any], aliases: set[str]) -> Any:
    normalized = {
        str(key).lower().replace("-", "_").replace(" ", "_"): value
        for key, value in entry.items()
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def entries_from_answer(answer: Any) -> tuple[list[dict[str, Any]], bool]:
    parsed, parse_ok = json_loads_robust(answer)
    if isinstance(parsed, dict):
        parsed = parsed.get("results", [])
    if not isinstance(parsed, list):
        return [], False
    return [entry for entry in parsed if isinstance(entry, dict)], parse_ok


def normalized_entry_key(entry: dict[str, Any], task: str) -> tuple[Any, ...] | None:
    numeric_entity = entry_get_case_insensitive(
        entry,
        {"numeric_entity", "numericentity", "entity", "value", "numeric_value", "number"},
    )
    datatype = entry_get_case_insensitive(entry, {"datatype", "data_type", "type"})
    if numeric_entity is None or datatype is None:
        return None

    base = (normalize_numeric_entity(numeric_entity), normalize_datatype(datatype))
    if task == "table":
        row_context = entry_get_case_insensitive(
            entry,
            {"row_context", "rowcontext", "row_header", "row", "row_label"},
        )
        column_context = entry_get_case_insensitive(
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
        )
        return (*base, normalize_context(row_context), normalize_context(column_context))

    sentence_context = entry_get_case_insensitive(
        entry,
        {"sentence_context", "sentencecontext", "sentence", "context"},
    )
    if sentence_context is None:
        return None
    return (*base, normalize_context(sentence_context))


def key_counter(answer: Any, task: str) -> tuple[Counter[tuple[Any, ...]], bool, int]:
    entries, parse_ok = entries_from_answer(answer)
    counter: Counter[tuple[Any, ...]] = Counter()
    valid_entry_count = 0
    for entry in entries:
        key = normalized_entry_key(entry, task=task)
        if key is None:
            continue
        counter[key] += 1
        valid_entry_count += 1
    return counter, parse_ok, valid_entry_count


def project_key_counts(counter: Counter[tuple[Any, ...]], indices: tuple[int, ...]) -> Counter[tuple[Any, ...]]:
    projected: Counter[tuple[Any, ...]] = Counter()
    for key, count in counter.items():
        projected[tuple(key[idx] for idx in indices)] += count
    return projected


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
                if line:
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
                    f"Predictions missing {len(missing)} gold rows by {id_column}; "
                    f"first missing: {missing[:5]}"
                )
            return [pred_by_id[str(row[id_column])] for row in gold_rows]

    if len(prediction_rows) != len(gold_rows):
        raise ValueError(
            f"Prediction row count {len(prediction_rows)} does not match gold row count {len(gold_rows)}"
        )
    return [row[prediction_column] for row in prediction_rows]


def counter_matches(gold: Counter[tuple[Any, ...]], pred: Counter[tuple[Any, ...]]) -> int:
    return sum((gold & pred).values())


def evaluate(
    gold_rows: list[dict[str, Any]],
    predictions: list[Any],
    task: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_tp = full_pred_total = full_gold_total = 0
    pair_tp = pair_pred_total = pair_gold_total = 0
    context_tp = context_pred_total = context_gold_total = 0
    exact_match_count = 0
    parse_success_count = 0
    valid_pred_entry_count = 0
    per_row: list[dict[str, Any]] = []

    context_indices = (2, 3) if task == "table" else (2,)

    for row, prediction in zip(gold_rows, predictions):
        gold_keys, _, gold_valid_entries = key_counter(row["answer"], task=task)
        pred_keys, pred_parse_ok, pred_valid_entries = key_counter(prediction, task=task)

        full_matches = counter_matches(gold_keys, pred_keys)
        full_tp += full_matches
        full_pred_total += sum(pred_keys.values())
        full_gold_total += sum(gold_keys.values())

        gold_pairs = project_key_counts(gold_keys, (0, 1))
        pred_pairs = project_key_counts(pred_keys, (0, 1))
        pair_matches = counter_matches(gold_pairs, pred_pairs)
        pair_tp += pair_matches
        pair_pred_total += sum(pred_pairs.values())
        pair_gold_total += sum(gold_pairs.values())

        gold_contexts = project_key_counts(gold_keys, context_indices)
        pred_contexts = project_key_counts(pred_keys, context_indices)
        context_matches = counter_matches(gold_contexts, pred_contexts)
        context_tp += context_matches
        context_pred_total += sum(pred_contexts.values())
        context_gold_total += sum(gold_contexts.values())

        exact = gold_keys == pred_keys
        exact_match_count += int(exact)
        parse_success_count += int(pred_parse_ok)
        valid_pred_entry_count += pred_valid_entries

        per_row.append(
            {
                "source_sample_idx": row["source_sample_idx"],
                "context_id": row["context_id"],
                "gold_entry_count": gold_valid_entries,
                "pred_entry_count": pred_valid_entries,
                "full_matches": full_matches,
                "pair_matches": pair_matches,
                "context_matches": context_matches,
                "json_parse_ok": pred_parse_ok,
                "exact_match": exact,
            }
        )

    n = len(gold_rows)
    metrics = {
        "task": task,
        "sample_count": n,
        "json_parse_success_rate": round(parse_success_count / n, 6) if n else 0.0,
        "exact_row_match_rate": round(exact_match_count / n, 6) if n else 0.0,
        "gold_entry_count": full_gold_total,
        "pred_entry_count": full_pred_total,
        "valid_pred_entry_count": valid_pred_entry_count,
        "full_entry": {
            **prf(full_tp, full_pred_total, full_gold_total),
            "true_positive": full_tp,
        },
        "numeric_entity_datatype": {
            **prf(pair_tp, pair_pred_total, pair_gold_total),
            "true_positive": pair_tp,
        },
        "context_only": {
            **prf(context_tp, context_pred_total, context_gold_total),
            "true_positive": context_tp,
        },
    }
    return metrics, per_row


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.gold_dataset)[args.split]
    gold_rows = [dict(row) for row in dataset]

    if args.predictions is None:
        predictions = [row["answer"] for row in gold_rows]
        prediction_source = "gold_answers_sanity_check"
    else:
        prediction_rows = load_prediction_rows(Path(args.predictions))
        prediction_column = choose_prediction_column(prediction_rows, args.prediction_column)
        predictions = align_predictions(gold_rows, prediction_rows, prediction_column)
        prediction_source = args.predictions

    metrics, per_row = evaluate(gold_rows, predictions, task=args.task)
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
