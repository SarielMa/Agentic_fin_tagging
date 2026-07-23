#!/usr/bin/env python3
"""Permutation null for Experiment B best-prefix R@50 from persisted rounds.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_reward_diagnostic" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
ARMS = ("bandit", "random", "resample_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace rounds 2-4 candidates with same-arm/different-fact candidates and compute best-prefix R@50."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--rounds-path", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--consensus-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260723)
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


def compact_candidates(candidates: list[dict[str, Any]]) -> list[tuple[str, int, float]]:
    compacted = []
    for idx, candidate in enumerate(candidates):
        tag = candidate.get("tag")
        if not tag:
            continue
        rank = int(candidate.get("rank", idx + 1) or idx + 1)
        consensus = float(candidate.get("consensus_agreement") or 0.0)
        compacted.append((str(tag), rank, consensus))
    return compacted


def load_instances(rounds_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    instances: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with rounds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            arm = str(record["arm"])
            fact_id = int(record["fact_id"])
            state = instances[arm].setdefault(
                fact_id,
                {
                    "arm": arm,
                    "fact_id": fact_id,
                    "modality": str(record["modality"]),
                    "round1": [],
                    "round_candidates": {},
                    "persisted_prefixes": [],
                },
            )
            round_idx = int(record["round_idx"])
            if not state["round1"]:
                state["round1"] = compact_candidates(record.get("candidates_before") or [])
            state["round_candidates"][round_idx] = compact_candidates(record.get("selected_candidates_after") or [])
            state["persisted_prefixes"].append(
                (round_idx, compact_candidates(record.get("accumulated_candidates_after") or []))
            )

    for arm_instances in instances.values():
        for state in arm_instances.values():
            state["persisted_prefixes"].sort(key=lambda item: item[0])
            state["round_indices"] = sorted(state["round_candidates"])
    return instances


def top_tags(candidates: list[tuple[str, int, float]], top_k: int) -> set[str]:
    return {tag for tag, _, _ in sorted(candidates, key=lambda item: item[1])[:top_k]}


def recall_from_tags(tags: set[str], gold_tags: set[str]) -> float:
    if not gold_tags:
        return 0.0
    return len(tags & gold_tags) / len(gold_tags)


def fuse_rounds(
    rounds: list[list[tuple[str, int, float]]],
    top_k: int,
    rrf_kappa: float,
    consensus_beta: float,
) -> set[str]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    best_consensus: dict[str, float] = {}
    first_round: dict[str, int] = {}
    for round_pos, candidates in enumerate(rounds, start=1):
        for tag, rank, consensus in candidates:
            if rank <= 0:
                continue
            scores[tag] += 1.0 / (rrf_kappa + rank)
            if tag not in best_rank or rank < best_rank[tag]:
                best_rank[tag] = rank
                best_consensus[tag] = consensus
            if tag not in first_round:
                first_round[tag] = round_pos
    ranked = sorted(
        scores,
        key=lambda tag: (
            -(scores[tag] + consensus_beta * best_consensus.get(tag, 0.0)),
            first_round.get(tag, 10**9),
            best_rank.get(tag, 10**9),
            tag,
        ),
    )
    return set(ranked[:top_k])


def best_minus_round1_from_prefix_tags(
    round1_tags: set[str],
    prefix_tags: list[set[str]],
    gold_tags: set[str],
) -> float:
    round1 = recall_from_tags(round1_tags, gold_tags)
    best = max([round1] + [recall_from_tags(tags, gold_tags) for tags in prefix_tags])
    return best - round1


def observed_persisted_by_arm(
    instances: dict[str, dict[int, dict[str, Any]]],
    gold_by_fact: dict[int, set[str]],
    top_k: int,
) -> dict[str, float]:
    observed = {}
    for arm, arm_instances in instances.items():
        deltas = []
        for fact_id, state in arm_instances.items():
            gold_tags = gold_by_fact[fact_id]
            round1_tags = top_tags(state["round1"], top_k)
            persisted_prefix_tags = [top_tags(candidates, top_k) for _, candidates in state["persisted_prefixes"]]
            deltas.append(best_minus_round1_from_prefix_tags(round1_tags, persisted_prefix_tags, gold_tags))
        observed[arm] = mean(deltas) if deltas else 0.0
    return observed


def observed_re_fused_by_arm(
    instances: dict[str, dict[int, dict[str, Any]]],
    gold_by_fact: dict[int, set[str]],
    top_k: int,
    rrf_kappa: float,
    consensus_beta: float,
) -> dict[str, float]:
    observed = {}
    for arm, arm_instances in instances.items():
        deltas = []
        for fact_id, state in arm_instances.items():
            rounds = [state["round1"]]
            round1_tags = top_tags(state["round1"], top_k)
            prefix_tags = []
            for round_idx in state["round_indices"]:
                rounds.append(state["round_candidates"][round_idx])
                prefix_tags.append(fuse_rounds(rounds, top_k, rrf_kappa, consensus_beta))
            deltas.append(best_minus_round1_from_prefix_tags(round1_tags, prefix_tags, gold_by_fact[fact_id]))
        observed[arm] = mean(deltas) if deltas else 0.0
    return observed


def permutation_rows(
    instances: dict[str, dict[int, dict[str, Any]]],
    gold_by_fact: dict[int, set[str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    donor_pools: dict[tuple[str, int], list[tuple[int, list[tuple[str, int, float]]]]] = defaultdict(list)
    for arm, arm_instances in instances.items():
        for fact_id, state in arm_instances.items():
            for round_idx, candidates in state["round_candidates"].items():
                donor_pools[(arm, round_idx)].append((fact_id, candidates))

    rows = []
    for permutation_idx in range(1, args.permutations + 1):
        for arm, arm_instances in instances.items():
            deltas = []
            for fact_id, state in arm_instances.items():
                rounds = [state["round1"]]
                round1_tags = top_tags(state["round1"], args.top_k)
                prefix_tags = []
                for round_idx in state["round_indices"]:
                    pool = [(donor_id, candidates) for donor_id, candidates in donor_pools[(arm, round_idx)] if donor_id != fact_id]
                    if not pool:
                        continue
                    _, donor_candidates = rng.choice(pool)
                    rounds.append(donor_candidates)
                    prefix_tags.append(fuse_rounds(rounds, args.top_k, args.rrf_kappa, args.consensus_beta))
                deltas.append(best_minus_round1_from_prefix_tags(round1_tags, prefix_tags, gold_by_fact[fact_id]))
            rows.append(
                {
                    "permutation_idx": permutation_idx,
                    "arm": arm,
                    "null_best_minus_round_1": round(mean(deltas), 8) if deltas else 0.0,
                }
            )
    return rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summary_rows(
    rows: list[dict[str, Any]],
    observed_persisted: dict[str, float],
    observed_re_fused: dict[str, float],
) -> list[dict[str, Any]]:
    by_arm: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(float(row["null_best_minus_round_1"]))
    summaries = []
    for arm in ARMS:
        values = by_arm.get(arm, [])
        observed = float(observed_persisted.get(arm, 0.0))
        ge_count = sum(value >= observed for value in values)
        summaries.append(
            {
                "arm": arm,
                "observed_persisted_best_minus_round_1": round(observed, 8),
                "observed_re_fused_best_minus_round_1": round(float(observed_re_fused.get(arm, 0.0)), 8),
                "null_mean": round(mean(values), 8) if values else 0.0,
                "null_sd": round((sum((value - mean(values)) ** 2 for value in values) / (len(values) - 1)) ** 0.5, 8)
                if len(values) > 1
                else 0.0,
                "null_q025": round(quantile(values, 0.025), 8),
                "null_q50": round(quantile(values, 0.50), 8),
                "null_q975": round(quantile(values, 0.975), 8),
                "p_ge_observed_persisted": round((ge_count + 1) / (len(values) + 1), 8) if values else 0.0,
                "permutations": len(values),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rounds_path = args.rounds_path or args.output_dir / "rounds.jsonl"
    out_csv = args.out_csv or args.output_dir / "null_search_permutation.csv"
    summary_csv = args.summary_csv or args.output_dir / "null_search_permutation_summary.csv"
    instances = load_instances(rounds_path)
    gold_by_fact = load_gold_tags(args.sample_path)
    rows = permutation_rows(instances, gold_by_fact, args)
    observed_persisted = observed_persisted_by_arm(instances, gold_by_fact, args.top_k)
    observed_re_fused = observed_re_fused_by_arm(
        instances,
        gold_by_fact,
        args.top_k,
        args.rrf_kappa,
        args.consensus_beta,
    )
    summaries = summary_rows(rows, observed_persisted, observed_re_fused)
    write_csv(out_csv, rows)
    write_csv(summary_csv, summaries)
    print(json.dumps({"out_csv": str(out_csv), "summary_csv": str(summary_csv), "rows": summaries}, indent=2))


if __name__ == "__main__":
    main()
