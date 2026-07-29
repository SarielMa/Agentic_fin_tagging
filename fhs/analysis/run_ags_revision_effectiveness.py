#!/usr/bin/env python3
"""Gate 2: when D- fires on a dimension, does the next revision move it toward gold?

Offline over the Experiment B rounds log. For every (round t, dimension d) where
the Feedback Agent fired D- on d, compares the hypothesis value for d at round t
against the value at round t+1, scoring each by graded closeness to the gold
concept's attribute on d.

Closeness is the graded form of the verdict being evaluated, using the same
branch logic as dimension_agreement_verdict:
  - if either side carries controlled categories: Jaccard of the category sets
  - otherwise: token overlap, |hypothesis tokens & gold tokens| / |hypothesis tokens|
A revision is improved / unchanged / worsened as closeness rises / holds / falls.

The null is replacing d with a value drawn at random from the observed value
distribution for that dimension. A refinement that does not beat that null is not
using the verdict, whatever its accuracy.
"""

from __future__ import annotations
# --- resolve local packages regardless of this file's depth in the tree ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        _sys.path.insert(0, str(_p / "src"))
        _sys.path.insert(0, str(_p / "analysis"))
        FHS_ROOT = _p
        break
# -------------------------------------------------------------------------

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    load_normalization_map,
    map_version,
    normalize_dimension_value,
    parse_candidate_symbolic_profile,
)
from run_ags_coverage_pilot import fact_context_key, load_jsonl, row_to_example
from run_ags_feedback_verdict_accuracy import (
    fired_dimensions,
    gold_candidate,
    hypothesis_dimensions_from_round,
)
from run_ags_reward_diagnostic import OPERATOR_DIMENSION
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    load_taxonomy,
    normalize_space,
    normalize_tag,
)


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_revision_effectiveness" / "qwen3_32b"
DEFAULT_ROUNDS_PATH = FHS_ROOT / "runs" / "runs_ags_reward_diagnostic" / "qwen3_32b" / "rounds.jsonl"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"

SOURCES = ("symbolic", "merged")
TARGET_GROUPS = ("all", "operator_targeted", "not_targeted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds-path", type=Path, default=DEFAULT_ROUNDS_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--gold-candidate-fields", choices=("compact", "full"), default="compact")
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def closeness(
    value: Any,
    dimension: str,
    gold_profile: dict[str, Any],
    normalization_map: dict[str, Any],
) -> tuple[float | None, str]:
    """Graded closeness of one dimension value to the gold concept, plus the branch used."""
    normalized = normalize_dimension_value(dimension.lower(), value, normalization_map)
    if not normalized["resolved"]:
        return None, "unresolved"
    gold_dimension = gold_profile["dimensions"].get(dimension.lower(), {})
    gold_categories = set(gold_dimension.get("categories", []))
    hypothesis_categories = set(normalized.get("categories", []))
    if hypothesis_categories or gold_categories:
        union = hypothesis_categories | gold_categories
        if not union:
            return None, "unresolved"
        return len(hypothesis_categories & gold_categories) / len(union), "category"
    hypothesis_tokens = set(normalized.get("tokens", []))
    gold_tokens = set(gold_profile.get("tokens", []))
    if not hypothesis_tokens or not gold_tokens:
        return None, "unresolved"
    return len(hypothesis_tokens & gold_tokens) / len(hypothesis_tokens), "token"


def classify(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "worsened"
    return "unchanged"


def load_rounds(path: Path) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            feedback = record.get("feedback") or {}
            rounds.append(
                {
                    "fact_id": int(record["fact_id"]),
                    "arm": str(record.get("arm", "")),
                    "round_idx": int(record.get("round_idx", -1)),
                    "modality": str(record.get("modality", "")),
                    "selected_operator": str(record.get("selected_operator", "")),
                    "hypothesis": hypothesis_dimensions_from_round(record),
                    "fired": {
                        "merged": fired_dimensions(feedback, "contradicted_dimensions"),
                        "symbolic": fired_dimensions(
                            feedback.get("symbolic_feedback") or {}, "contradicted_dimensions"
                        ),
                    },
                }
            )
    return rounds


def build_events(
    args: argparse.Namespace,
    rounds: list[dict[str, Any]],
    gold_profiles: dict[int, dict[str, Any]],
    context_by_fact: dict[int, str],
    value_pool: dict[str, list[str]],
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in rounds:
        episodes[(record["arm"], record["fact_id"])].append(record)
    for records in episodes.values():
        records.sort(key=lambda item: item["round_idx"])

    rng = random.Random(args.random_seed)
    events: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for (arm, fact_id), records in sorted(episodes.items()):
        gold_profile = gold_profiles.get(fact_id)
        if gold_profile is None:
            skipped["gold_profile_missing"] += 1
            continue
        for current, following in zip(records, records[1:]):
            target_dimension = OPERATOR_DIMENSION.get(current["selected_operator"])
            for source in SOURCES:
                for dimension in sorted(current["fired"][source]):
                    if dimension not in DIMENSIONS:
                        skipped["dimension_not_recognized"] += 1
                        continue
                    before_value = current["hypothesis"].get(dimension)
                    after_value = following["hypothesis"].get(dimension)
                    if before_value is None or after_value is None:
                        skipped["value_missing_in_one_round"] += 1
                        continue
                    before, before_branch = closeness(
                        before_value, dimension, gold_profile, normalization_map
                    )
                    after, after_branch = closeness(
                        after_value, dimension, gold_profile, normalization_map
                    )
                    if before is None or after is None:
                        skipped["closeness_unresolvable"] += 1
                        continue
                    if before_branch != after_branch:
                        skipped["branch_switched"] += 1
                        continue

                    pool = value_pool.get(dimension) or []
                    random_deltas: list[float] = []
                    for _ in range(args.random_draws):
                        if not pool:
                            break
                        drawn = rng.choice(pool)
                        drawn_closeness, drawn_branch = closeness(
                            drawn, dimension, gold_profile, normalization_map
                        )
                        if drawn_closeness is None or drawn_branch != before_branch:
                            continue
                        random_deltas.append(drawn_closeness - before)

                    events.append(
                        {
                            "arm": arm,
                            "fact_id": fact_id,
                            "context_key": context_by_fact[fact_id],
                            "modality": current["modality"],
                            "round_idx": current["round_idx"],
                            "dimension": dimension,
                            "feedback_source": source,
                            "selected_operator": current["selected_operator"],
                            "operator_targeted": bool(target_dimension == dimension),
                            "measure_branch": before_branch,
                            "value_before": normalize_space(before_value),
                            "value_after": normalize_space(after_value),
                            "value_changed": normalize_space(before_value) != normalize_space(after_value),
                            "closeness_before": round(before, 6),
                            "closeness_after": round(after, 6),
                            "delta": round(after - before, 6),
                            "outcome": classify(after - before),
                            "random_draws": len(random_deltas),
                            "random_mean_delta": round(sum(random_deltas) / len(random_deltas), 6)
                            if random_deltas
                            else None,
                            "random_improved_fraction": round(
                                sum(1 for value in random_deltas if value > 0) / len(random_deltas), 6
                            )
                            if random_deltas
                            else None,
                            "random_unchanged_fraction": round(
                                sum(1 for value in random_deltas if value == 0) / len(random_deltas), 6
                            )
                            if random_deltas
                            else None,
                            "random_worsened_fraction": round(
                                sum(1 for value in random_deltas if value < 0) / len(random_deltas), 6
                            )
                            if random_deltas
                            else None,
                        }
                    )

    coverage = {
        "episodes": len(episodes),
        "rounds": len(rounds),
        "events": len(events),
        "skipped": dict(skipped),
    }
    return events, coverage


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(events)
    if not n:
        return {"n": 0}
    with_random = [event for event in events if event["random_mean_delta"] is not None]
    return {
        "n": n,
        "improved": sum(1 for event in events if event["outcome"] == "improved"),
        "unchanged": sum(1 for event in events if event["outcome"] == "unchanged"),
        "worsened": sum(1 for event in events if event["outcome"] == "worsened"),
        "improved_fraction": round(sum(1 for event in events if event["outcome"] == "improved") / n, 6),
        "unchanged_fraction": round(sum(1 for event in events if event["outcome"] == "unchanged") / n, 6),
        "worsened_fraction": round(sum(1 for event in events if event["outcome"] == "worsened") / n, 6),
        "value_changed_fraction": round(sum(1 for event in events if event["value_changed"]) / n, 6),
        "mean_delta": round(sum(event["delta"] for event in events) / n, 6),
        "mean_closeness_before": round(sum(event["closeness_before"] for event in events) / n, 6),
        "mean_closeness_after": round(sum(event["closeness_after"] for event in events) / n, 6),
        "random_n": len(with_random),
        "random_improved_fraction": round(
            sum(event["random_improved_fraction"] for event in with_random) / len(with_random), 6
        )
        if with_random
        else 0.0,
        "random_unchanged_fraction": round(
            sum(event["random_unchanged_fraction"] for event in with_random) / len(with_random), 6
        )
        if with_random
        else 0.0,
        "random_worsened_fraction": round(
            sum(event["random_worsened_fraction"] for event in with_random) / len(with_random), 6
        )
        if with_random
        else 0.0,
        "random_mean_delta": round(
            sum(event["random_mean_delta"] for event in with_random) / len(with_random), 6
        )
        if with_random
        else 0.0,
    }


def bootstrap_delta_vs_random(
    events: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    usable = [event for event in events if event["random_mean_delta"] is not None]
    if not usable:
        return {"mean_delta_minus_random": (0.0, 0.0), "improved_fraction_minus_random": (0.0, 0.0)}
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in usable:
        by_context[event["context_key"]].append(event)
    context_keys = sorted(by_context)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        resampled: list[dict[str, Any]] = []
        for context_key in (rng.choice(context_keys) for _ in context_keys):
            resampled.extend(by_context[context_key])
        count = len(resampled)
        samples["mean_delta_minus_random"].append(
            sum(event["delta"] - event["random_mean_delta"] for event in resampled) / count
        )
        samples["improved_fraction_minus_random"].append(
            sum(
                float(event["outcome"] == "improved") - event["random_improved_fraction"]
                for event in resampled
            )
            / count
        )

    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        weight = pos - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {key: (percentile(values, 0.025), percentile(values, 0.975)) for key, values in samples.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = [row_to_example(row) for row in load_jsonl(args.sample_path)]
    context_by_fact = {example.example_idx: fact_context_key(example) for example in examples}
    concepts_by_tag = {concept.tag: concept for concept in load_taxonomy(args.taxonomy_jsonl)}
    normalization_map = load_normalization_map(args.normalization_map)

    gold_profiles: dict[int, dict[str, Any]] = {}
    for example in examples:
        if not example.gold_tags:
            continue
        concept = concepts_by_tag.get(normalize_tag(example.gold_tags[0]))
        if concept is None:
            continue
        gold_profiles[example.example_idx] = parse_candidate_symbolic_profile(
            gold_candidate(concept, args.gold_candidate_fields), normalization_map
        )

    rounds = load_rounds(args.rounds_path)
    value_pool: dict[str, list[str]] = defaultdict(list)
    for record in rounds:
        for dimension, value in record["hypothesis"].items():
            text = normalize_space(value)
            if text:
                value_pool[dimension].append(text)

    events, coverage = build_events(
        args, rounds, gold_profiles, context_by_fact, value_pool, normalization_map
    )
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True))
    if not events:
        raise RuntimeError("No D- revision events found.")

    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for source in SOURCES:
        for dimension in ("ALL", *DIMENSIONS):
            for target_group in TARGET_GROUPS:
                subset = [
                    event
                    for event in events
                    if event["feedback_source"] == source
                    and (dimension == "ALL" or event["dimension"] == dimension)
                    and (
                        target_group == "all"
                        or (target_group == "operator_targeted") == event["operator_targeted"]
                    )
                ]
                if not subset:
                    continue
                seed_offset += 1
                stats = summarize(subset)
                ci = bootstrap_delta_vs_random(subset, args.bootstrap_samples, args.bootstrap_seed + seed_offset)
                rows.append(
                    {
                        "feedback_source": source,
                        "dimension": dimension,
                        "target_group": target_group,
                        **stats,
                        "mean_delta_minus_random": round(stats["mean_delta"] - stats["random_mean_delta"], 6),
                        "mean_delta_minus_random_ci_low": round(ci["mean_delta_minus_random"][0], 6),
                        "mean_delta_minus_random_ci_high": round(ci["mean_delta_minus_random"][1], 6),
                        "improved_fraction_minus_random": round(
                            stats["improved_fraction"] - stats["random_improved_fraction"], 6
                        ),
                        "improved_minus_random_ci_low": round(ci["improved_fraction_minus_random"][0], 6),
                        "improved_minus_random_ci_high": round(ci["improved_fraction_minus_random"][1], 6),
                        "beats_random": bool(ci["mean_delta_minus_random"][0] > 0.0),
                    }
                )

    with (args.output_dir / "per_event.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    fieldnames = list(rows[0])
    with (args.output_dir / "revision_effectiveness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headline = next(
        row
        for row in rows
        if row["feedback_source"] == "symbolic" and row["dimension"] == "ALL" and row["target_group"] == "all"
    )
    targeted = next(
        (
            row
            for row in rows
            if row["feedback_source"] == "symbolic"
            and row["dimension"] == "ALL"
            and row["target_group"] == "operator_targeted"
        ),
        None,
    )
    metrics = {
        "experiment": "ags_revision_effectiveness",
        "question": (
            "Given D- fired on dimension d at round t, is the value for d at round t+1 closer to the "
            "gold concept's attribute on d than the previous value was?"
        ),
        "closeness_measure": (
            "graded form of dimension_agreement_verdict: category-set Jaccard when either side carries "
            "controlled categories, otherwise |hypothesis tokens & gold tokens| / |hypothesis tokens|; "
            "events whose branch changes between t and t+1 are excluded as non-comparable"
        ),
        "null_model": (
            f"replace d with a value drawn from the observed distribution of values for d across all "
            f"logged hypotheses, {args.random_draws} draws per event, same branch required"
        ),
        "coverage": coverage,
        "headline": headline,
        "headline_operator_targeted": targeted,
        "config": {
            "rounds_path": str(args.rounds_path),
            "gold_candidate_fields": args.gold_candidate_fields,
            "random_draws": args.random_draws,
            "random_seed": args.random_seed,
            "normalization_map_version": map_version(args.normalization_map),
            "bootstrap": {
                "iterations": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "unit": "context",
            },
        },
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "headline": headline}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
