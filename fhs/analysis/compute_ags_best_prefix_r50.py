#!/usr/bin/env python3
"""Oracle best-prefix Recall@50 diagnostic for Experiment B rounds.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FHS_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_reward_diagnostic" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"
ARMS = ("bandit", "random", "resample_only")
MODALITIES = ("table", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute round-1, full-episode, and oracle best-prefix R@50 from persisted Experiment B candidates."
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


def recall_at_k(candidates: list[dict[str, Any]], gold_tags: set[str], top_k: int) -> float:
    if not gold_tags:
        return 0.0
    retrieved = top_tags(candidates, top_k)
    return len(retrieved & gold_tags) / len(gold_tags)


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
                    "records": [],
                },
            )
            state["records"].append(record)
    for state in instances.values():
        state["records"].sort(key=lambda item: int(item.get("round_idx", -1)))
    return instances


def summarize(
    instances: dict[tuple[str, int], dict[str, Any]],
    gold_by_fact: dict[int, set[str]],
    top_k: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for state in instances.values():
        records = state["records"]
        if not records:
            continue
        fact_id = int(state["fact_id"])
        if fact_id not in gold_by_fact:
            raise ValueError(f"No gold tags found for fact_id={fact_id}")
        gold_tags = gold_by_fact[fact_id]

        first_candidates = records[0].get("candidates_before")
        if not isinstance(first_candidates, list):
            raise ValueError(f"Missing candidates_before for arm={state['arm']} fact_id={fact_id}")
        prefix_scores: list[tuple[int, float]] = [(1, recall_at_k(first_candidates, gold_tags, top_k))]
        for record in records:
            candidates = record.get("accumulated_candidates_after")
            if not isinstance(candidates, list):
                raise ValueError(
                    f"Missing accumulated_candidates_after for arm={state['arm']} fact_id={fact_id}"
                )
            prefix_scores.append((int(record.get("round_idx", -1)), recall_at_k(candidates, gold_tags, top_k)))

        round_1 = prefix_scores[0][1]
        full_episode = prefix_scores[-1][1]
        best_round, best_prefix = max(prefix_scores, key=lambda item: (item[1], -item[0]))
        buckets[(str(state["arm"]), str(state["modality"]))].append(
            {
                "round_1": round_1,
                "full_episode": full_episode,
                "best_prefix": best_prefix,
                "best_round": best_round,
                "later_beats_round_1": float(best_round > 1 and best_prefix > round_1),
                "full_beats_round_1": float(full_episode > round_1),
                "full_loses_to_round_1": float(full_episode < round_1),
            }
        )

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for modality in MODALITIES:
            values = buckets.get((arm, modality), [])
            n = len(values)
            best_rounds = Counter(int(item["best_round"]) for item in values)
            round_1_r50 = sum(float(item["round_1"]) for item in values) / n if n else 0.0
            full_r50 = sum(float(item["full_episode"]) for item in values) / n if n else 0.0
            best_r50 = sum(float(item["best_prefix"]) for item in values) / n if n else 0.0
            rows.append(
                {
                    "arm": arm,
                    "modality": modality,
                    "n_facts": n,
                    "round_1_only_r50": round(round_1_r50, 8),
                    "full_episode_r50": round(full_r50, 8),
                    "best_prefix_r50": round(best_r50, 8),
                    "best_minus_round_1": round(best_r50 - round_1_r50, 8),
                    "full_minus_round_1": round(full_r50 - round_1_r50, 8),
                    "later_prefix_beats_round_1_fraction": round(
                        sum(float(item["later_beats_round_1"]) for item in values) / n, 8
                    )
                    if n
                    else 0.0,
                    "full_beats_round_1_fraction": round(
                        sum(float(item["full_beats_round_1"]) for item in values) / n, 8
                    )
                    if n
                    else 0.0,
                    "full_loses_to_round_1_fraction": round(
                        sum(float(item["full_loses_to_round_1"]) for item in values) / n, 8
                    )
                    if n
                    else 0.0,
                    "best_prefix_round_counts": json.dumps(dict(sorted(best_rounds.items())), sort_keys=True),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rounds_path = args.rounds_path or args.output_dir / "rounds.jsonl"
    out_csv = args.out_csv or args.output_dir / "best_prefix_r50.csv"
    rows = summarize(load_instances(rounds_path), load_gold_tags(args.sample_path), args.top_k)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_csv": str(out_csv), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
