#!/usr/bin/env python3
"""Offline arm-end metrics for Experiment B from persisted rounds.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FHS_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_reward_diagnostic" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"
ARMS = ("bandit", "random", "resample_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Experiment B arm-end rolling Recall@50 and reward summaries from rounds.jsonl."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--rounds-path", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_gold_tags(path: Path) -> dict[int, set[str]]:
    gold_by_fact: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            fact_id = int(record["fact_id"])
            gold_tags = {str(tag) for tag in record.get("ground_truth_concepts", []) if tag}
            gold_by_fact[fact_id] = gold_tags
    return gold_by_fact


def top_tags(candidates: list[dict[str, Any]], top_k: int) -> set[str]:
    ranked = sorted(candidates, key=lambda item: int(item.get("rank", 10**9)))
    return {str(item.get("tag")) for item in ranked[:top_k] if item.get("tag")}


def compute_recall_at_k(record: dict[str, Any], gold_tags: set[str], top_k: int) -> float:
    if not gold_tags:
        return 0.0
    candidates = record.get("accumulated_candidates_after")
    if not isinstance(candidates, list):
        raise ValueError(
            f"Missing accumulated_candidates_after for arm={record.get('arm')} fact_id={record.get('fact_id')}"
        )
    retrieved = top_tags(candidates, top_k)
    return len(gold_tags & retrieved) / len(gold_tags)


def collect_instance_rows(rounds_path: Path, gold_by_fact: dict[int, set[str]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    by_instance: dict[tuple[str, int], dict[str, Any]] = {}
    with rounds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            arm = str(record["arm"])
            fact_id = int(record["fact_id"])
            key = (arm, fact_id)
            state = by_instance.setdefault(
                key,
                {
                    "arm": arm,
                    "fact_id": fact_id,
                    "max_round_idx": -1,
                    "reward_sum": 0.0,
                    "final_record": None,
                },
            )
            state["reward_sum"] += float(record.get("reward_combined", 0.0))
            round_idx = int(record.get("round_idx", -1))
            if round_idx >= int(state["max_round_idx"]):
                state["max_round_idx"] = round_idx
                state["final_record"] = record

    rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in by_instance.values():
        fact_id = int(state["fact_id"])
        final_record = state["final_record"]
        if final_record is None:
            continue
        if fact_id not in gold_by_fact:
            raise ValueError(f"No gold tags found for fact_id={fact_id}")
        rows_by_arm[str(state["arm"])].append(
            {
                "fact_id": fact_id,
                "recall_at_50": compute_recall_at_k(final_record, gold_by_fact[fact_id], top_k),
                "reward_sum": float(state["reward_sum"]),
            }
        )

    for arm_rows in rows_by_arm.values():
        arm_rows.sort(key=lambda item: int(item["fact_id"]))
    return rows_by_arm


def rolling_mean(values: list[float], index: int, window: int) -> float:
    start = max(0, index - window + 1)
    window_values = values[start : index + 1]
    return sum(window_values) / len(window_values) if window_values else 0.0


def build_output_rows(rows_by_arm: dict[str, list[dict[str, Any]]], window: int) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = rows_by_arm.get(arm, [])
        recalls = [float(row["recall_at_50"]) for row in arm_rows]
        rewards = [float(row["reward_sum"]) for row in arm_rows]
        rolling_recalls: list[float] = []
        reward_total = 0.0
        for index, row in enumerate(arm_rows):
            reward_total += rewards[index]
            rolling_recall = rolling_mean(recalls, index, window)
            rolling_recalls.append(rolling_recall)
            output_rows.append(
                {
                    "arm": arm,
                    "instance_idx": index + 1,
                    "rolling_recall_at_50": round(rolling_recall, 8),
                    "cumulative_mean_reward": round(reward_total / (index + 1), 8),
                    "aulc": "",
                }
            )
        aulc = sum(rolling_recalls) / len(rolling_recalls) if rolling_recalls else 0.0
        output_rows.append(
            {
                "arm": arm,
                "instance_idx": "summary",
                "rolling_recall_at_50": "",
                "cumulative_mean_reward": "",
                "aulc": round(aulc, 8),
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    rounds_path = args.rounds_path or args.output_dir / "rounds.jsonl"
    out_csv = args.out_csv or args.output_dir / "arm_end_metrics.csv"
    gold_by_fact = load_gold_tags(args.sample_path)
    rows_by_arm = collect_instance_rows(rounds_path, gold_by_fact, args.top_k)
    rows = build_output_rows(rows_by_arm, args.window)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["arm", "instance_idx", "rolling_recall_at_50", "cumulative_mean_reward", "aulc"],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {arm: len(rows_by_arm.get(arm, [])) for arm in ARMS}
    print(json.dumps({"out_csv": str(out_csv), "instances_by_arm": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
