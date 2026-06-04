from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd


Record = dict[str, Any]
FlagGetter = Callable[[Record], bool]


def load_jsonl_records(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_agentic_records(
    records: list[Record],
    score: bool,
    recall_k: tuple[int, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = _base_metrics(records, metadata)
    action_counts: dict[str, int] = {}
    for record in records:
        action = record["stm"]["final_action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    metrics["action_counts"] = action_counts
    metrics["flag_rate"] = _rate(action_counts.get("flag", 0), len(records))
    _add_scored_metrics(metrics, records, score, recall_k, section_key="stm")
    return metrics


def evaluate_single_llm_records(
    records: list[Record],
    score: bool,
    recall_k: tuple[int, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = _base_metrics(records, metadata)
    out_of_candidates = sum(
        1 for record in records if record["baseline"]["llm_selection_out_of_candidates"]
    )
    metrics["out_of_candidate_rate"] = _rate(out_of_candidates, len(records))
    _add_scored_metrics(metrics, records, score, recall_k, section_key="baseline")
    return metrics


def evaluate_fixed_memory_records(
    records: list[Record],
    recall_k: tuple[int, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = _base_metrics(records, metadata)
    flagged = sum(1 for record in records if record["stm"]["validation"]["status"] == "flagged")
    metrics["validation_flag_rate"] = _rate(flagged, len(records))
    _add_scored_metrics(metrics, records, True, recall_k, section_key="stm")
    return metrics


def write_agentic_breakdown(records: list[Record], output_dir: Path, score: bool) -> None:
    write_breakdown(
        records,
        output_dir,
        score,
        section_key="stm",
        flag_getter=lambda record: record["stm"]["final_action"] == "flag",
    )


def write_single_llm_breakdown(records: list[Record], output_dir: Path, score: bool) -> None:
    write_breakdown(records, output_dir, score, section_key="baseline")


def write_fixed_memory_breakdown(records: list[Record], output_dir: Path) -> None:
    write_breakdown(
        records,
        output_dir,
        True,
        section_key="stm",
        flag_getter=lambda record: record["stm"]["validation"]["status"] == "flagged",
    )


def write_breakdown(
    records: list[Record],
    output_dir: Path,
    score: bool,
    section_key: str,
    flag_getter: FlagGetter | None = None,
    recall_points: tuple[int, ...] = (20, 50, 100, 200),
) -> None:
    if not records or not score:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        section = record[section_key]
        candidate_tags = _candidate_tags(record, section_key)
        row = {
            "category": section["category"],
            "entity_type": record["gold"]["Type"],
            "correct": record["correct"],
        }
        for k in recall_points:
            row[f"recall{k}"] = record["gold"]["Tag"] in candidate_tags[:k]
        if flag_getter is not None:
            row["flagged"] = flag_getter(record)
        rows.append(row)

    df = pd.DataFrame(rows)
    breakdown = {
        "by_category": _group_breakdown(df, "category", recall_points, include_flag_rate=flag_getter is not None),
        "by_entity_type": _group_breakdown(
            df,
            "entity_type",
            recall_points,
            include_flag_rate=flag_getter is not None,
        ),
    }
    with (output_dir / "breakdown.json").open("w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2)


def _base_metrics(records: list[Record], metadata: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_examples": len(records)}
    metrics.update(metadata)
    return metrics


def _add_scored_metrics(
    metrics: dict[str, Any],
    records: list[Record],
    score: bool,
    recall_k: tuple[int, ...],
    section_key: str,
) -> None:
    if not score:
        return

    n = len(records)
    correct = sum(1 for record in records if record["correct"])
    recall_counts = {k: 0 for k in recall_k}
    for record in records:
        gold_tag = record["gold"]["Tag"]
        candidate_tags = _candidate_tags(record, section_key)
        for k in recall_k:
            recall_counts[k] += int(gold_tag in candidate_tags[:k])

    metrics["tag_accuracy"] = _rate(correct, n)
    metrics["recall_at_k"] = {str(k): _rate(recall_counts[k], n) for k in recall_k}


def _candidate_tags(record: Record, section_key: str) -> list[str]:
    return [candidate["tag"] for candidate in record[section_key]["top_k"]]


def _group_breakdown(
    df: pd.DataFrame,
    column: str,
    recall_points: tuple[int, ...],
    include_flag_rate: bool,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key, group in df.groupby(column):
        row = {
            "n": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }
        for k in recall_points:
            row[f"recall{k}"] = float(group[f"recall{k}"].mean())
        if include_flag_rate:
            row["flag_rate"] = float(group["flagged"].mean())
        result[str(key)] = row
    return result


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan
