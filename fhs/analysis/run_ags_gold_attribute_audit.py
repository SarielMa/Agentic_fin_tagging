#!/usr/bin/env python3
"""Is the gold-side attribute recoverable for each dimension, or is closeness zero by construction?

Offline over the Experiment B revision logs plus the taxonomy. No GPU.

revision_effectiveness.csv reports 0 improvements out of 558 TEMPORAL events with
mean closeness 0.0018 unchanged before and after, despite values moving on 17% of them,
and SCOPE (0.0028) and QUALIFIER (0.0070) pinned similarly low, while FAMILY (0.318),
ROLE (0.158) and EVENT (0.108) are substantive. This audit tests whether the gold-side
attribute exists for those dimensions at all.

The mechanism under test is the category branch of `closeness`:

    if hypothesis_categories or gold_categories:
        return |h & g| / |h | g|

When the gold concept yields no categories on a dimension but the hypothesis does, the
union is the hypothesis's own set, the intersection is empty, and the score is exactly
0.0 for every value the hypothesis could ever take. Revision cannot move it. That is a
property of the measurement, not of the revision.

Gold categories are parsed from `gold_candidate(concept, fields)`, which under the
deployed `compact` setting is tag + entity_type + standard_label only -- documentation
and retrieval_text are dropped. The audit reports coverage under both settings, so the
question "is the attribute missing, or merely not looked for" is answered separately
from "is the pipeline configured to look for it".
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
from run_ags_coverage_pilot import load_jsonl, row_to_example
from run_ags_feedback_verdict_accuracy import gold_candidate
from run_ags_revision_effectiveness import (
    SOURCES,
    TARGET_GROUPS,
    bootstrap_delta_vs_random,
    closeness,
    summarize,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    load_taxonomy,
    normalize_tag,
)


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_gold_attribute_audit" / "qwen3_32b"
DEFAULT_EVENTS_PATH = FHS_ROOT / "runs" / "runs_ags_revision_effectiveness" / "qwen3_32b" / "per_event.jsonl"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"

EXAMPLES_PER_DIMENSION = 15
# A dimension is a measurement artifact when the gold side fails to yield the attribute on
# most events, so closeness is pinned at zero regardless of what the revision does.
GOLD_COVERAGE_FLOOR = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--gold-candidate-fields", choices=("compact", "full"), default="compact")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def build_gold_profiles(
    examples: list[Any],
    concepts_by_tag: dict[str, Any],
    normalization_map: dict[str, Any],
    fields: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    concepts: dict[int, Any] = {}
    for example in examples:
        if not example.gold_tags:
            continue
        concept = concepts_by_tag.get(normalize_tag(example.gold_tags[0]))
        if concept is None:
            continue
        concepts[example.example_idx] = concept
        profiles[example.example_idx] = parse_candidate_symbolic_profile(
            gold_candidate(concept, fields), normalization_map
        )
    return profiles, concepts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = [row_to_example(row) for row in load_jsonl(args.sample_path)]
    concepts_by_tag = {concept.tag: concept for concept in load_taxonomy(args.taxonomy_jsonl)}
    normalization_map = load_normalization_map(args.normalization_map)

    gold_profiles, gold_concepts = build_gold_profiles(
        examples, concepts_by_tag, normalization_map, args.gold_candidate_fields
    )
    # Diagnostic counterfactual: does the attribute appear if documentation and
    # retrieval_text are not dropped?
    other_fields = "full" if args.gold_candidate_fields == "compact" else "compact"
    gold_profiles_other, _ = build_gold_profiles(
        examples, concepts_by_tag, normalization_map, other_fields
    )

    vocab_sizes = {
        dimension: len(normalization_map.get("dimensions", {}).get(dimension.lower(), {}) or {})
        for dimension in DIMENSIONS
    }

    events = [json.loads(line) for line in args.events_path.open("r", encoding="utf-8") if line.strip()]
    symbolic = [event for event in events if event["feedback_source"] == "symbolic"]

    # --- per-dimension coverage --------------------------------------------------
    coverage_rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []
    gold_value_distribution: dict[str, list[dict[str, Any]]] = {}
    hypothesis_value_distribution: dict[str, list[dict[str, Any]]] = {}
    category_cooccurrence: dict[str, dict[str, Any]] = {}

    for dimension in DIMENSIONS:
        subset = [event for event in symbolic if event["dimension"] == dimension]
        key = dimension.lower()
        fact_ids = sorted({event["fact_id"] for event in subset})

        gold_categories_by_fact = {
            fact_id: tuple(
                gold_profiles.get(fact_id, {}).get("dimensions", {}).get(key, {}).get("categories", [])
            )
            for fact_id in fact_ids
        }
        gold_categories_other = {
            fact_id: tuple(
                gold_profiles_other.get(fact_id, {})
                .get("dimensions", {})
                .get(key, {})
                .get("categories", [])
            )
            for fact_id in fact_ids
        }
        facts_with_gold = sum(1 for value in gold_categories_by_fact.values() if value)
        events_with_gold = sum(1 for event in subset if gold_categories_by_fact.get(event["fact_id"]))
        events_with_gold_other = sum(
            1 for event in subset if gold_categories_other.get(event["fact_id"])
        )

        # A dimension with a controlled vocabulary defines its attribute as the category set,
        # so gold is recoverable only if gold yields categories. ROLE and EVENT define no
        # categories at all: their attribute is by design the concept's text tokens, which are
        # always present, so they are not exposed to the category-branch failure mode.
        has_vocab = vocab_sizes[dimension] > 0
        hypothesis_nonempty = 0
        hypothesis_categories_nonempty = 0
        recoverable_events = 0
        zero_cause: Counter[str] = Counter()
        for event in subset:
            normalized = normalize_dimension_value(key, event["value_before"], normalization_map)
            if normalized["resolved"] and normalized["raw"]:
                hypothesis_nonempty += 1
            gold_set = set(gold_categories_by_fact.get(event["fact_id"], ()))
            hypothesis_set = set(normalized["categories"])
            if hypothesis_set:
                hypothesis_categories_nonempty += 1
            if has_vocab:
                recoverable_events += int(bool(gold_set))
            else:
                recoverable_events += int(bool(gold_profiles.get(event["fact_id"], {}).get("tokens")))

            is_zero = event["closeness_before"] == 0.0
            if event["measure_branch"] == "token":
                zero_cause["token_zero" if is_zero else "token_nonzero"] += 1
            elif not gold_set and hypothesis_set:
                # Structural: union is the hypothesis's own set, intersection empty, always 0.
                zero_cause["category_structural_zero_gold_empty"] += 1
            elif gold_set and not hypothesis_set:
                zero_cause["category_zero_hypothesis_empty"] += 1
            elif is_zero:
                zero_cause["category_zero_disjoint"] += 1
            else:
                zero_cause["category_nonzero"] += 1

        zero_before = sum(1 for event in subset if event["closeness_before"] == 0.0)
        zero_both = sum(
            1 for event in subset if event["closeness_before"] == 0.0 and event["closeness_after"] == 0.0
        )
        max_closeness = max(
            (max(event["closeness_before"], event["closeness_after"]) for event in subset), default=0.0
        )
        branches = Counter(event["measure_branch"] for event in subset)
        # Non-zero is attainable on the category branch only if some gold concept in the
        # sample actually carries a category on this dimension.
        gold_categories_anywhere = sum(
            1
            for profile in gold_profiles.values()
            if profile.get("dimensions", {}).get(key, {}).get("categories")
        )

        n = len(subset) or 1
        coverage_rows.append(
            {
                "dimension": dimension,
                "n_events": len(subset),
                "n_facts": len(fact_ids),
                "vocab_categories_defined": vocab_sizes[dimension],
                "gold_attr_nonempty_facts": facts_with_gold,
                "gold_attr_nonempty_fact_fraction": round(facts_with_gold / (len(fact_ids) or 1), 6),
                "gold_attr_nonempty_event_fraction": round(events_with_gold / n, 6),
                f"gold_attr_nonempty_event_fraction_{other_fields}": round(events_with_gold_other / n, 6),
                "gold_profiles_with_category_in_sample": gold_categories_anywhere,
                "hypothesis_value_nonempty_fraction": round(hypothesis_nonempty / n, 6),
                "hypothesis_categories_nonempty_fraction": round(hypothesis_categories_nonempty / n, 6),
                "branch_category": branches.get("category", 0),
                "branch_token": branches.get("token", 0),
                "closeness_exactly_zero_before": zero_before,
                "closeness_exactly_zero_fraction": round(zero_before / n, 6),
                "closeness_zero_before_and_after": zero_both,
                "max_closeness_observed": round(max_closeness, 6),
                "nonzero_attainable": bool(max_closeness > 0.0),
                "nonzero_events": zero_cause["token_nonzero"] + zero_cause["category_nonzero"],
                "nonzero_event_fraction": round(
                    (zero_cause["token_nonzero"] + zero_cause["category_nonzero"]) / n, 6
                ),
                "gold_attribute_recoverable_fraction": round(recoverable_events / n, 6),
                "zero_structural_gold_empty": zero_cause["category_structural_zero_gold_empty"],
                "zero_hypothesis_empty": zero_cause["category_zero_hypothesis_empty"],
                "zero_categories_disjoint": zero_cause["category_zero_disjoint"],
                "zero_token_branch": zero_cause["token_zero"],
                "reportable": bool(recoverable_events / n >= GOLD_COVERAGE_FLOOR),
            }
        )

        distribution = Counter(
            ("|".join(gold_categories_by_fact.get(event["fact_id"], ())) or "<empty>")
            for event in subset
        )
        gold_value_distribution[dimension] = [
            {"gold_categories": value, "events": count, "fraction": round(count / n, 6)}
            for value, count in distribution.most_common(20)
        ]

        hypothesis_distribution: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for event in subset:
            hypothesis_set = normalize_dimension_value(
                key, event["value_before"], normalization_map
            )["categories"]
            gold_set = sorted(gold_categories_by_fact.get(event["fact_id"], ()))
            hypothesis_distribution["|".join(hypothesis_set) or "<empty>"] += 1
            if gold_set and hypothesis_set:
                pair_counts[("|".join(gold_set), "|".join(hypothesis_set))] += 1
        hypothesis_value_distribution[dimension] = [
            {"hypothesis_categories": value, "events": count, "fraction": round(count / n, 6)}
            for value, count in hypothesis_distribution.most_common(20)
        ]
        both_present = sum(pair_counts.values())
        disjoint_pairs = sum(
            count
            for (gold_value, hypothesis_value), count in pair_counts.items()
            if not (set(gold_value.split("|")) & set(hypothesis_value.split("|")))
        )
        category_cooccurrence[dimension] = {
            "events_with_both_sides_non_empty": both_present,
            "disjoint_events": disjoint_pairs,
            "disjoint_fraction": round(disjoint_pairs / both_present, 6) if both_present else None,
            "top_pairs": [
                {"gold": gold_value, "hypothesis": hypothesis_value, "events": count}
                for (gold_value, hypothesis_value), count in pair_counts.most_common(10)
            ],
        }

        # Hand-inspectable examples, spread across distinct facts where possible.
        seen_facts: set[int] = set()
        chosen: list[dict[str, Any]] = []
        for event in sorted(subset, key=lambda item: (item["fact_id"], item["round_idx"])):
            if len(chosen) >= EXAMPLES_PER_DIMENSION:
                break
            if event["fact_id"] in seen_facts:
                continue
            seen_facts.add(event["fact_id"])
            chosen.append(event)
        for event in subset:
            if len(chosen) >= EXAMPLES_PER_DIMENSION:
                break
            if event not in chosen:
                chosen.append(event)

        for event in chosen:
            fact_id = event["fact_id"]
            concept = gold_concepts.get(fact_id)
            profile = gold_profiles.get(fact_id, {})
            before_norm = normalize_dimension_value(key, event["value_before"], normalization_map)
            after_norm = normalize_dimension_value(key, event["value_after"], normalization_map)
            example_records.append(
                {
                    "dimension": dimension,
                    "fact_id": fact_id,
                    "round_idx": event["round_idx"],
                    "gold_tag": profile.get("tag", ""),
                    "gold_standard_label": getattr(concept, "standard_label", ""),
                    "gold_text_parsed": gold_candidate(concept, args.gold_candidate_fields)
                    if concept is not None
                    else {},
                    "gold_attribute_categories": list(
                        profile.get("dimensions", {}).get(key, {}).get("categories", [])
                    ),
                    f"gold_attribute_categories_{other_fields}": list(
                        gold_profiles_other.get(fact_id, {})
                        .get("dimensions", {})
                        .get(key, {})
                        .get("categories", [])
                    ),
                    "hypothesis_before": event["value_before"],
                    "hypothesis_before_categories": before_norm["categories"],
                    "hypothesis_after": event["value_after"],
                    "hypothesis_after_categories": after_norm["categories"],
                    "value_changed": event["value_changed"],
                    "measure_branch": event["measure_branch"],
                    "closeness_before": event["closeness_before"],
                    "closeness_after": event["closeness_after"],
                    "delta": event["delta"],
                    "outcome": event["outcome"],
                }
            )

    reportable = [row["dimension"] for row in coverage_rows if row["reportable"]]
    artifact = [row["dimension"] for row in coverage_rows if not row["reportable"]]

    # --- recomputed effectiveness over reportable dimensions only ----------------
    filtered_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for source in SOURCES:
        for dimension in ("ALL", *DIMENSIONS):
            for target_group in TARGET_GROUPS:
                if dimension == "ALL":
                    pool = [event for event in events if event["dimension"] in reportable]
                elif dimension in reportable:
                    pool = [event for event in events if event["dimension"] == dimension]
                else:
                    continue
                subset = [
                    event
                    for event in pool
                    if event["feedback_source"] == source
                    and (
                        target_group == "all"
                        or (target_group == "operator_targeted") == event["operator_targeted"]
                    )
                ]
                if not subset:
                    continue
                seed_offset += 1
                stats = summarize(subset)
                ci = bootstrap_delta_vs_random(
                    subset, args.bootstrap_samples, args.bootstrap_seed + seed_offset
                )
                filtered_rows.append(
                    {
                        "feedback_source": source,
                        "dimension": dimension,
                        "target_group": target_group,
                        **stats,
                        "mean_delta_minus_random": round(
                            stats["mean_delta"] - stats["random_mean_delta"], 6
                        ),
                        "mean_delta_minus_random_ci_low": round(ci["mean_delta_minus_random"][0], 6),
                        "mean_delta_minus_random_ci_high": round(ci["mean_delta_minus_random"][1], 6),
                        "improved_fraction_minus_random": round(
                            stats["improved_fraction"] - stats["random_improved_fraction"], 6
                        ),
                        "improved_minus_random_ci_low": round(
                            ci["improved_fraction_minus_random"][0], 6
                        ),
                        "improved_minus_random_ci_high": round(
                            ci["improved_fraction_minus_random"][1], 6
                        ),
                        "beats_random": bool(ci["mean_delta_minus_random"][0] > 0.0),
                    }
                )

    with (args.output_dir / "gold_attribute_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    with (args.output_dir / "gold_attribute_examples.jsonl").open("w", encoding="utf-8") as handle:
        for record in example_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (args.output_dir / "revision_effectiveness_filtered.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered_rows[0]))
        writer.writeheader()
        writer.writerows(filtered_rows)

    def all_row(rows: list[dict[str, Any]], source: str, group: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in rows
                if row["feedback_source"] == source
                and row["dimension"] == "ALL"
                and row["target_group"] == group
            ),
            None,
        )

    original_rows = list(
        csv.DictReader(
            (args.events_path.parent / "revision_effectiveness.csv").open("r", encoding="utf-8")
        )
    )
    original_all = all_row(original_rows, "symbolic", "all")
    filtered_all = all_row(filtered_rows, "symbolic", "all")

    metrics = {
        "experiment": "ags_gold_attribute_audit",
        "question": (
            "For each dimension, is the gold-side attribute extracted at all, or is closeness "
            "zero by construction because the gold concept yields no categories while the "
            "hypothesis does?"
        ),
        "mechanism": (
            "closeness takes the category branch whenever either side carries categories. If gold "
            "carries none and the hypothesis carries some, the union is the hypothesis's own set, "
            "the intersection is empty, and the score is exactly 0.0 for every value the hypothesis "
            "could take, so revision cannot move it."
        ),
        "gold_candidate_fields": {
            "deployed": args.gold_candidate_fields,
            "compact_includes": ["tag", "entity_type", "standard_label"],
            "full_adds": ["documentation", "retrieval_text"],
            "note": (
                "gold categories are matched against this text only; the compact setting drops "
                "documentation and retrieval_text, which is where period/scope wording lives"
            ),
        },
        "normalization_map": {
            "version": map_version(args.normalization_map),
            "categories_defined_per_dimension": vocab_sizes,
            "note": (
                "ROLE and EVENT have no controlled categories defined, so they always take the "
                "token branch and are unaffected by the category-branch failure mode"
            ),
        },
        "reportability_rule": (
            "A dimension is reportable when the gold attribute is recoverable on at least "
            f"{GOLD_COVERAGE_FLOOR:.0%} of its events. Recoverable means gold yields a non-empty "
            "category set for dimensions that define a controlled vocabulary, and non-empty gold "
            "tokens for dimensions that define none (ROLE, EVENT), whose attribute is the "
            "concept's text by design."
        ),
        "gold_coverage_floor": GOLD_COVERAGE_FLOOR,
        "reportable_dimensions": reportable,
        "artifact_dimensions": artifact,
        "coverage": coverage_rows,
        "gold_value_distribution_top20": gold_value_distribution,
        "hypothesis_value_distribution_top20": hypothesis_value_distribution,
        "category_cooccurrence": category_cooccurrence,
        "temporal_vocabulary_conflates_two_axes": (
            "The TEMPORAL vocabulary mixes period type (instant, duration) with maturity "
            "classification (current, noncurrent, prior, future, year_1..year_5). Hypotheses only "
            "ever emit period type; gold labels almost only yield maturity, because the literal "
            "word 'Current' in a label such as NotesAndLoansReceivableGrossCurrent means the "
            "balance-sheet classification, not a time reference. The two sets cannot intersect, so "
            "Jaccard is 0 no matter how good the hypothesis is -- 'as of December 31, 2024' against "
            "gold 'current' scores 0. This is a false-friend match, which is worse than an absent "
            "gold attribute: it produces a confident zero rather than an unresolved skip."
        ),
        "all_rows_shift": {
            "original_symbolic_all": original_all,
            "filtered_symbolic_all": filtered_all,
            "excluded_dimensions": artifact,
        },
        "config": {
            "events_path": str(args.events_path),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "bootstrap": {
                "iterations": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "unit": "context",
            },
        },
        "rows": filtered_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "reportable": reportable,
                "artifact": artifact,
                "coverage": [
                    {
                        "dimension": row["dimension"],
                        "n_events": row["n_events"],
                        "gold_attr_nonempty_event_fraction": row["gold_attr_nonempty_event_fraction"],
                        f"gold_{other_fields}": row[f"gold_attr_nonempty_event_fraction_{other_fields}"],
                        "zero_fraction": row["closeness_exactly_zero_fraction"],
                        "max_closeness": row["max_closeness_observed"],
                    }
                    for row in coverage_rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
