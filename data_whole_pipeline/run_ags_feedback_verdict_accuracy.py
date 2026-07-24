#!/usr/bin/env python3
"""Gate: is the Feedback Agent's D- verdict better than chance?

Offline over the Experiment B rounds log. For every round and every dimension,
compares what the Feedback Agent said (D+ support / D- contradict / D? unresolved,
derived from the retrieved neighborhood) against ground truth derived from the
gold concept: does the hypothesis's value for that dimension actually disagree
with gold?

Reports precision and recall of D- per dimension, split by the agreement layer
that resolved the ground-truth verdict (exact vs lexical), for the merged
feedback and for its symbolic and LLM layers separately.

The chance reference is the base rate of true disagreement among resolvable
dimensions. A D- precision at the base rate means the verdict carries no
information, and iteration was never given usable input.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    agree,
    load_normalization_map,
    map_version,
)
from run_ags_component_validation import agreement_reason_layer
from run_ags_coverage_pilot import fact_context_key, load_jsonl, row_to_example
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    load_taxonomy,
    normalize_tag,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_feedback_verdict_accuracy" / "qwen3_32b"
DEFAULT_ROUNDS_PATH = SCRIPT_DIR / "runs_ags_reward_diagnostic" / "qwen3_32b" / "rounds.jsonl"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"

SOURCES = ("merged", "symbolic", "llm")
LAYERS = ("exact", "lexical")
ARMS = ("bandit", "random", "resample_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds-path", type=Path, default=DEFAULT_ROUNDS_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument(
        "--gold-candidate-fields",
        choices=("compact", "full"),
        default="compact",
        help="compact mirrors the tag/label/type view the feedback agent scored candidates under.",
    )
    parser.add_argument("--default-top-m", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def gold_candidate(concept: Any, fields: str) -> dict[str, Any]:
    candidate = {
        "tag": concept.tag,
        "type": concept.entity_type,
        "standard_label": concept.standard_label,
    }
    if fields == "full":
        candidate["documentation"] = concept.documentation
        candidate["retrieval_text"] = concept.retrieval_text
    return candidate


def hypothesis_dimensions_from_round(record: dict[str, Any]) -> dict[str, str]:
    """Recover the hypothesis's dimension values from the logged verdicts."""
    values: dict[str, str] = {}
    verdict_sources = [
        record.get("dimension_verdicts") or [],
        ((record.get("feedback") or {}).get("symbolic_feedback") or {}).get("dimension_verdicts") or [],
    ]
    for verdicts in verdict_sources:
        for verdict in verdicts:
            dimension = str(verdict.get("dimension", "")).upper()
            if not dimension or dimension in values:
                continue
            for candidate_verdict in verdict.get("candidate_verdicts") or []:
                raw = candidate_verdict.get("raw_hypothesis_value")
                if raw not in (None, ""):
                    values[dimension] = raw
                    break
    return values


def fired_dimensions(feedback: dict[str, Any], key: str) -> set[str]:
    return {str(dimension).upper() for dimension in (feedback.get(key) or [])}


def collect_observations(
    args: argparse.Namespace,
    gold_by_fact: dict[int, str],
    concepts_by_tag: dict[str, Any],
    context_by_fact: dict[int, str],
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    round_count = 0

    with args.rounds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            round_count += 1
            fact_id = int(record["fact_id"])
            gold_tag = gold_by_fact.get(fact_id)
            concept = concepts_by_tag.get(gold_tag) if gold_tag else None
            if concept is None:
                skipped["gold_concept_missing"] += 1
                continue

            hypothesis = hypothesis_dimensions_from_round(record)
            if not hypothesis:
                skipped["hypothesis_dimensions_unrecoverable"] += 1
                continue

            feedback = record.get("feedback") or {}
            fired = {
                "merged": fired_dimensions(feedback, "contradicted_dimensions"),
                "symbolic": fired_dimensions(feedback.get("symbolic_feedback") or {}, "contradicted_dimensions"),
                "llm": fired_dimensions(feedback.get("llm_feedback") or {}, "contradicted_dimensions"),
            }
            supported = {
                "merged": fired_dimensions(feedback, "supported_dimensions"),
                "symbolic": fired_dimensions(feedback.get("symbolic_feedback") or {}, "supported_dimensions"),
                "llm": fired_dimensions(feedback.get("llm_feedback") or {}, "supported_dimensions"),
            }

            symbolic = feedback.get("symbolic_feedback") or {}
            has_symbolic = bool(symbolic.get("dimension_verdicts"))
            top_m = int(symbolic.get("top_m") or args.default_top_m)
            gold_rank_before = record.get("gold_rank_before")
            # The symbolic layer scores the top-M retrieved candidates with the same agree()
            # machinery used for ground truth here, so when gold sits inside that window the
            # comparison shares an input and is circular. Track it and slice on it.
            gold_in_window = gold_rank_before is not None and int(gold_rank_before) <= top_m

            truth = agree(gold_candidate(concept, args.gold_candidate_fields), hypothesis, normalization_map)
            for verdict in truth.verdicts:
                dimension = str(verdict.get("dimension", "")).upper()
                matched = verdict.get("matched")
                if matched is None:
                    skipped["truth_unresolvable"] += 1
                    continue
                observations.append(
                    {
                        "fact_id": fact_id,
                        "context_key": context_by_fact[fact_id],
                        "arm": str(record.get("arm", "")),
                        "round_idx": int(record.get("round_idx", -1)),
                        "modality": str(record.get("modality", "")),
                        "dimension": dimension,
                        "truth_layer": agreement_reason_layer(verdict.get("reason")),
                        "truth_reason": verdict.get("reason"),
                        "truth_disagrees": bool(matched is False),
                        "gold_rank_before": gold_rank_before,
                        "feedback_top_m": top_m,
                        "gold_in_feedback_window": gold_in_window,
                        "has_symbolic_feedback": has_symbolic,
                        **{f"d_minus_{source}": bool(dimension in fired[source]) for source in SOURCES},
                        **{f"d_plus_{source}": bool(dimension in supported[source]) for source in SOURCES},
                    }
                )
            if args.log_every and round_count % args.log_every == 0:
                print(f"scanned {round_count} rounds, {len(observations)} observations", flush=True)

    coverage = {
        "rounds_scanned": round_count,
        "observations": len(observations),
        "skipped": dict(skipped),
    }
    return observations, coverage


def rates(rows: list[dict[str, Any]], source: str) -> dict[str, float]:
    fired_key = f"d_minus_{source}"
    true_positive = sum(1 for row in rows if row[fired_key] and row["truth_disagrees"])
    false_positive = sum(1 for row in rows if row[fired_key] and not row["truth_disagrees"])
    false_negative = sum(1 for row in rows if not row[fired_key] and row["truth_disagrees"])
    fired = true_positive + false_positive
    positives = true_positive + false_negative
    base_rate = positives / len(rows) if rows else 0.0
    precision = true_positive / fired if fired else 0.0
    recall = true_positive / positives if positives else 0.0
    return {
        "n": len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "d_minus_fired": fired,
        "true_disagreements": positives,
        "base_rate": base_rate,
        "precision": precision,
        "recall": recall,
        "precision_minus_base_rate": precision - base_rate,
        "lift": precision / base_rate if base_rate else 0.0,
    }


def bootstrap_context(
    rows: list[dict[str, Any]],
    source: str,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Resample contexts with replacement and recompute the rates from scratch."""
    if not rows:
        return {key: (0.0, 0.0) for key in ("precision", "recall", "precision_minus_base_rate")}
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_context[row["context_key"]].append(row)
    context_keys = sorted(by_context)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        resampled: list[dict[str, Any]] = []
        for context_key in (rng.choice(context_keys) for _ in context_keys):
            resampled.extend(by_context[context_key])
        stats = rates(resampled, source)
        for key in ("precision", "recall", "precision_minus_base_rate"):
            samples[key].append(stats[key])

    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        weight = pos - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {key: (percentile(values, 0.025), percentile(values, 0.975)) for key, values in samples.items()}


ARM_GROUPS = {
    "all": None,
    "feedback_arms": {"bandit", "random"},
    "bandit": {"bandit"},
    "random": {"random"},
    "resample_only": {"resample_only"},
}


def subset_for(
    observations: list[dict[str, Any]],
    dimension: str,
    layer: str,
    window: str,
    arm_group: str,
) -> list[dict[str, Any]]:
    arms = ARM_GROUPS[arm_group]
    return [
        row
        for row in observations
        if (dimension == "ALL" or row["dimension"] == dimension)
        and (layer == "all" or row["truth_layer"] == layer)
        and (
            window == "all"
            or (window == "gold_in_window") == bool(row["gold_in_feedback_window"])
        )
        and (arms is None or row["arm"] in arms)
    ]


def slice_specs() -> list[tuple[str, str, str, str]]:
    """Explicit slices, kept to the decision-relevant cuts rather than a full cross product."""
    specs: list[tuple[str, str, str, str]] = []
    for layer in ("all", *LAYERS):
        specs.append(("ALL", layer, "all", "all"))
        for window in ("gold_in_window", "gold_outside_window"):
            specs.append(("ALL", layer, window, "all"))
    for arm_group in ("feedback_arms", "bandit", "random", "resample_only"):
        specs.append(("ALL", "all", "all", arm_group))
    for window in ("gold_in_window", "gold_outside_window"):
        specs.append(("ALL", "all", window, "feedback_arms"))
    for dimension in DIMENSIONS:
        specs.append((dimension, "all", "all", "all"))
        for layer in LAYERS:
            specs.append((dimension, layer, "all", "all"))
        for window in ("gold_in_window", "gold_outside_window"):
            specs.append((dimension, "all", window, "feedback_arms"))
    return specs


def build_rows(args: argparse.Namespace, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for dimension, layer, window, arm_group in slice_specs():
        subset = subset_for(observations, dimension, layer, window, arm_group)
        for source in SOURCES:
            seed_offset += 1
            stats = rates(subset, source)
            ci = bootstrap_context(subset, source, args.bootstrap_samples, args.bootstrap_seed + seed_offset)
            rows.append(
                {
                    "dimension": dimension,
                    "truth_layer": layer,
                    "gold_window": window,
                    "arm_group": arm_group,
                    "feedback_source": source,
                    **{key: (round(value, 6) if isinstance(value, float) else value) for key, value in stats.items()},
                    "precision_ci_low": round(ci["precision"][0], 6),
                    "precision_ci_high": round(ci["precision"][1], 6),
                    "recall_ci_low": round(ci["recall"][0], 6),
                    "recall_ci_high": round(ci["recall"][1], 6),
                    "precision_minus_base_ci_low": round(ci["precision_minus_base_rate"][0], 6),
                    "precision_minus_base_ci_high": round(ci["precision_minus_base_rate"][1], 6),
                    "beats_chance": bool(ci["precision_minus_base_rate"][0] > 0.0),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = load_jsonl(args.sample_path)
    examples = [row_to_example(row) for row in sample_rows]
    gold_by_fact = {
        example.example_idx: normalize_tag(example.gold_tags[0])
        for example in examples
        if example.gold_tags
    }
    context_by_fact = {example.example_idx: fact_context_key(example) for example in examples}
    concepts_by_tag = {concept.tag: concept for concept in load_taxonomy(args.taxonomy_jsonl)}
    normalization_map = load_normalization_map(args.normalization_map)

    observations, coverage = collect_observations(
        args, gold_by_fact, concepts_by_tag, context_by_fact, normalization_map
    )
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True))
    if not observations:
        raise RuntimeError("No evaluable observations; cannot run the gate.")

    rows = build_rows(args, observations)

    with (args.output_dir / "per_verdict.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation, ensure_ascii=False) + "\n")

    fieldnames = list(rows[0])
    with (args.output_dir / "feedback_verdict_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def find(dimension, layer, window, arm_group, source):
        return next(
            row
            for row in rows
            if row["dimension"] == dimension
            and row["truth_layer"] == layer
            and row["gold_window"] == window
            and row["arm_group"] == arm_group
            and row["feedback_source"] == source
        )

    # Headline is the non-circular slice: gold outside the feedback window, and only the
    # arms that actually produce symbolic feedback.
    headline = find("ALL", "all", "gold_outside_window", "feedback_arms", "symbolic")
    circular = find("ALL", "all", "gold_in_window", "feedback_arms", "symbolic")
    llm_headline = find("ALL", "all", "gold_outside_window", "feedback_arms", "llm")
    any_beats_chance = [
        f"{row['dimension']}/{row['truth_layer']}/{row['feedback_source']}"
        for row in rows
        if row["beats_chance"] and row["n"] >= 30
    ]
    gate = {
        "headline_slice": "dimension=ALL, truth_layer=all, gold_outside_window, feedback_arms, source=symbolic",
        "headline_source": "symbolic",
        "n_observations": headline["n"],
        "base_rate": headline["base_rate"],
        "precision": headline["precision"],
        "recall": headline["recall"],
        "precision_minus_base_rate": headline["precision_minus_base_rate"],
        "precision_minus_base_ci": [
            headline["precision_minus_base_ci_low"],
            headline["precision_minus_base_ci_high"],
        ],
        "lift": headline["lift"],
        "pooled_beats_chance": headline["beats_chance"],
        "slices_beating_chance_n_ge_30": any_beats_chance,
        "circular_slice_precision": circular["precision"],
        "circular_slice_lift": circular["lift"],
        "circular_slice_n": circular["n"],
        "llm_layer_precision": llm_headline["precision"],
        "llm_layer_base_rate": llm_headline["base_rate"],
        "llm_layer_lift": llm_headline["lift"],
        "llm_layer_beats_chance": llm_headline["beats_chance"],
        "circularity_note": (
            "The symbolic layer scores the top-M retrieved candidates with the same agree() "
            "machinery used to derive ground truth from gold, so rounds with gold inside that "
            "window share an input between prediction and truth. The headline uses only rounds "
            "with gold outside the window. resample_only produces no symbolic feedback at all "
            "and is excluded from the headline."
        ),
        "verdict": (
            "D- carries signal above the base rate"
            if headline["beats_chance"]
            else "D- is at or near chance: iteration was never given usable input"
        ),
    }
    metrics = {
        "experiment": "ags_feedback_verdict_accuracy",
        "gate": gate,
        "coverage": coverage,
        "config": {
            "rounds_path": str(args.rounds_path),
            "sample_path": str(args.sample_path),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "gold_candidate_fields": args.gold_candidate_fields,
            "normalization_map_version": map_version(args.normalization_map),
            "bootstrap": {
                "iterations": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "unit": "context",
            },
            "ground_truth": (
                "agree(gold_concept, hypothesis_dimensions): matched is False means the hypothesis "
                "genuinely disagrees with gold on that dimension and D- should fire"
            ),
        },
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
