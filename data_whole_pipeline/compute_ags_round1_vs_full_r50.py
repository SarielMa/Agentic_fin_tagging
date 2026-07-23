#!/usr/bin/env python3
"""Compare Experiment B round-1-only and full-episode Recall@50 from rounds.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_reward_diagnostic" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
ARMS = ("bandit", "random", "resample_only")
MODALITIES = ("table", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute round-1-only vs full-episode R@50 for Experiment B from persisted candidates."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--rounds-path", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_gold_tags(path: Path) -> dict[int, set[str]]:
    gold_by_fact: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            gold_by_fact[int(record["fact_id"])] = {
                str(tag) for tag in record.get("ground_truth_concepts", []) if tag
            }
    return gold_by_fact


def top_tags(candidates: list[dict[str, Any]], top_k: int) -> set[str]:
    ranked = sorted(candidates, key=lambda item: int(item.get("rank", 10**9)))
    return {str(item.get("tag")) for item in ranked[:top_k] if item.get("tag")}


def recall_at_k(tags: set[str], gold_tags: set[str]) -> float:
    if not gold_tags:
        return 0.0
    return len(tags & gold_tags) / len(gold_tags)


def load_instances(rounds_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    instances: dict[tuple[str, int], dict[str, Any]] = {}
    with rounds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["arm"]), int(record["fact_id"]))
            state = instances.setdefault(
                key,
                {
                    "arm": str(record["arm"]),
                    "fact_id": int(record["fact_id"]),
                    "modality": str(record["modality"]),
                    "first_round_idx": 10**9,
                    "last_round_idx": -1,
                    "first_record": None,
                    "last_record": None,
                },
            )
            round_idx = int(record.get("round_idx", -1))
            if round_idx < int(state["first_round_idx"]):
                state["first_round_idx"] = round_idx
                state["first_record"] = record
            if round_idx >= int(state["last_round_idx"]):
                state["last_round_idx"] = round_idx
                state["last_record"] = record
    return instances


def summarize(
    instances: dict[tuple[str, int], dict[str, Any]],
    gold_by_fact: dict[int, set[str]],
    top_k: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for state in instances.values():
        fact_id = int(state["fact_id"])
        first_record = state["first_record"]
        last_record = state["last_record"]
        if first_record is None or last_record is None:
            continue
        if fact_id not in gold_by_fact:
            raise ValueError(f"No gold tags found for fact_id={fact_id}")
        round_1_candidates = first_record.get("candidates_before")
        full_candidates = last_record.get("accumulated_candidates_after")
        if not isinstance(round_1_candidates, list):
            raise ValueError(f"Missing candidates_before for arm={state['arm']} fact_id={fact_id}")
        if not isinstance(full_candidates, list):
            raise ValueError(f"Missing accumulated_candidates_after for arm={state['arm']} fact_id={fact_id}")

        round_1_top50 = top_tags(round_1_candidates, top_k)
        full_top50 = top_tags(full_candidates, top_k)
        gold_tags = gold_by_fact[fact_id]
        buckets[(str(state["arm"]), str(state["modality"]))].append(
            {
                "round_1_recall": recall_at_k(round_1_top50, gold_tags),
                "full_recall": recall_at_k(full_top50, gold_tags),
                "top50_diff": float(round_1_top50 != full_top50),
            }
        )

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for modality in MODALITIES:
            values = buckets.get((arm, modality), [])
            n = len(values)
            round_1_r50 = sum(item["round_1_recall"] for item in values) / n if n else 0.0
            full_r50 = sum(item["full_recall"] for item in values) / n if n else 0.0
            diff_fraction = sum(item["top50_diff"] for item in values) / n if n else 0.0
            rows.append(
                {
                    "arm": arm,
                    "modality": modality,
                    "n_facts": n,
                    "round_1_only_r50": round(round_1_r50, 8),
                    "full_episode_r50": round(full_r50, 8),
                    "difference": round(full_r50 - round_1_r50, 8),
                    "top50_membership_diff_fraction": round(diff_fraction, 8),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rounds_path = args.rounds_path or args.output_dir / "rounds.jsonl"
    out_csv = args.out_csv or args.output_dir / "round1_vs_full_r50.csv"
    gold_by_fact = load_gold_tags(args.sample_path)
    instances = load_instances(rounds_path)
    rows = summarize(instances, gold_by_fact, args.top_k)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "modality",
                "n_facts",
                "round_1_only_r50",
                "full_episode_r50",
                "difference",
                "top50_membership_diff_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_csv": str(out_csv), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
