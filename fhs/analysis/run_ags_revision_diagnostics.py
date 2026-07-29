#!/usr/bin/env python3
"""Table 14: revision-stage diagnostics, on the FROZEN TEST SPLIT.

Replaces the development placeholders that `run_ags_revision_effectiveness.py`
(Panel B) and `run_ags_gold_attribute_audit.py` (Panel A) produced over the
250-fact tabular coverage-pilot sample. Same two panels, same estimators, same
closeness measure and the same null -- different data, and one source change
described below.

Panel A -- gold-attribute recoverability
    Revision can only be scored on a dimension whose gold-side attribute is
    recoverable from the taxonomy representation. Where it is not, `closeness`
    takes the category branch with an empty gold set, the union collapses to
    the hypothesis's own categories, and the score is exactly 0.0 for every
    value the hypothesis could take. That is a property of the measurement,
    not of the revision, so those dimensions are excluded from Panel B before
    anything is concluded from them.

Panel B -- revision effectiveness
    For every dimension on which D- fired at round t, compare the value's
    graded closeness to gold before and after that round's revision, against a
    null that replaces the value with one drawn from the dimension's observed
    value distribution.

Where the test-split revision events come from
----------------------------------------------
The AGS-Seq test runs -- runs_fintagging_grounding_baseline/qwen3_32b_ags_seq
and _ags_seq_random, both over all 2,509 test facts at B=4 rounds. Their
`ags_seq_rounds` log carries everything the dev rounds log did: the verdicts
(`dimension_verdicts`, whose `candidate_verdicts[*].raw_hypothesis_value`
recovers the pre-revision value), which dimensions D- fired on (`D_minus`),
the operator and its targeted dimension (`directive.target_dimension`), and --
new relative to dev -- the post-revision hypothesis itself
(`revised_hypothesis`).

That last field is why this is not a line-for-line port. The dev script had to
recover the after-value by pairing round t with round t+1, which silently
drops the final round of every episode. Here the revision is logged directly,
so the after-value is read rather than inferred and the last round is kept. I
verified the two agree before relying on it: across 3,379 paired comparisons,
`revised_hypothesis[t]` equals the recovered pre-revision hypothesis at t+1
with zero mismatches, so this is strictly more coverage of the same quantity.

One layer, not two. The dev tables reported a `symbolic` and a `merged`
feedback source. The test AGS-Seq runs set `llm_feedback_enabled=False`
(ags_sequential_arms.episode_feedback refuses to run otherwise), so on test
there is exactly one verifier layer and a "merged" column would be the
deterministic column relabelled. It is reported once, as `deterministic`.

Outputs
-------
  per_event.jsonl              one row per scored (arm, fact, round, dimension)
  panel_a_recoverability.csv   Panel A
  panel_b_effectiveness.csv    Panel B, the paper's row set
  revision_effectiveness.csv   the full dimension x target-group grid
  metrics.json                 the above plus prose statistics and coverage

CPU only. No generation, no retrieval, no GPU.
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
from typing import Any, Iterator

from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    load_normalization_map,
    map_version,
    normalize_dimension_value,
    parse_candidate_symbolic_profile,
)
from run_ags_feedback_verdict_accuracy import gold_candidate
from run_ags_revision_effectiveness import (
    bootstrap_delta_vs_random,
    classify,
    closeness,
    summarize,
)
from run_ags_reward_diagnostic import OPERATOR_DIMENSION
from run_ags_verification_quality import hypothesis_dimensions_from_round, stream_jsonl
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    load_taxonomy,
    normalize_space,
    normalize_tag,
)


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_revision_diagnostics" / "qwen3_32b"
DEFAULT_SEQ_TRACES = {
    "learned": FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline" / "qwen3_32b_ags_seq" / "grounding_traces.jsonl",
    "random": SCRIPT_DIR
    / "runs_fintagging_grounding_baseline"
    / "qwen3_32b_ags_seq_random"
    / "grounding_traces.jsonl",
}

TARGET_GROUPS = ("all", "operator_targeted", "not_targeted")
# Same floor the dev audit used: below this the gold attribute is missing often
# enough that closeness is pinned near zero by construction.
GOLD_COVERAGE_FLOOR = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-trace-learned", type=Path, default=DEFAULT_SEQ_TRACES["learned"])
    parser.add_argument("--seq-trace-random", type=Path, default=DEFAULT_SEQ_TRACES["random"])
    parser.add_argument(
        "--arms",
        default="learned,random",
        help="Comma-separated subset of learned,random. Both are pooled for the paper rows; "
        "per-arm rows are always emitted alongside.",
    )
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--gold-candidate-fields", choices=("compact", "full"), default="compact")
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: stop after N facts per arm.")
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


# ------------------------------------------------------------------ round pass


def iter_rounds(
    trace_path: Path,
    arm: str,
    limit: int | None,
    log_every: int,
) -> Iterator[dict[str, Any]]:
    """Flatten one arm's trace into the round records Panel B consumes.

    `value_after` comes from the round's own logged `revised_hypothesis`, so
    every round including the last contributes; see the module docstring for
    why that is equivalent to the dev script's consecutive-round pairing.
    """
    facts = 0
    for record in stream_jsonl(trace_path):
        if limit is not None and facts >= limit:
            break
        facts += 1
        gold_tags = [normalize_tag(tag) for tag in record.get("gold_tags", [])]
        for round_record in record.get("ags_seq_rounds") or []:
            directive = round_record.get("directive") or {}
            target_dimension = normalize_space(directive.get("target_dimension", "")).upper() or None
            if target_dimension is None:
                # PERTURB and CHANGE_STRATEGY target no single dimension; fall back to the
                # operator name map so the dev definition of "targeted" still applies.
                target_dimension = OPERATOR_DIMENSION.get(round_record.get("selected_operator", ""))
            yield {
                "arm": arm,
                "fact_id": int(record["example_idx"]),
                "context_key": str(record.get("context_id")),
                "modality": str(record.get("input_type", "")),
                "gold_tags": gold_tags,
                "round_idx": int(round_record.get("round_idx", -1)),
                "selected_operator": str(round_record.get("selected_operator", "")),
                "target_dimension": (target_dimension or "").upper() or None,
                "hypothesis_before": hypothesis_dimensions_from_round(round_record),
                "hypothesis_after": (round_record.get("revised_hypothesis") or {}).get("dimensions") or {},
                "d_minus": {str(dim).upper() for dim in (round_record.get("D_minus") or [])},
            }
        if log_every and facts % log_every == 0:
            print(f"[{arm}] scanned {facts} facts", flush=True)


def build_events(
    args: argparse.Namespace,
    rounds: list[dict[str, Any]],
    gold_profiles: dict[int, dict[str, Any]],
    value_pool: dict[str, list[str]],
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.random_seed)
    events: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for record in rounds:
        gold_profile = gold_profiles.get(record["fact_id"])
        if gold_profile is None:
            skipped["gold_profile_missing"] += 1
            continue
        for dimension in sorted(record["d_minus"]):
            if dimension not in DIMENSIONS:
                skipped["dimension_not_recognized"] += 1
                continue
            before_value = record["hypothesis_before"].get(dimension)
            after_value = record["hypothesis_after"].get(dimension)
            if before_value is None or after_value is None:
                skipped["value_missing"] += 1
                continue
            before, before_branch = closeness(before_value, dimension, gold_profile, normalization_map)
            after, after_branch = closeness(after_value, dimension, gold_profile, normalization_map)
            if before is None or after is None:
                skipped["closeness_unresolvable"] += 1
                continue
            if before_branch != after_branch:
                # A value that switches measurement branch is not comparable before to after.
                skipped["branch_switched"] += 1
                continue

            pool = value_pool.get(dimension) or []
            random_deltas: list[float] = []
            for _ in range(args.random_draws):
                if not pool:
                    break
                drawn = rng.choice(pool)
                drawn_closeness, drawn_branch = closeness(drawn, dimension, gold_profile, normalization_map)
                if drawn_closeness is None or drawn_branch != before_branch:
                    continue
                random_deltas.append(drawn_closeness - before)

            events.append(
                {
                    "arm": record["arm"],
                    "fact_id": record["fact_id"],
                    "context_key": record["context_key"],
                    "modality": record["modality"],
                    "round_idx": record["round_idx"],
                    "dimension": dimension,
                    "feedback_source": "deterministic",
                    "selected_operator": record["selected_operator"],
                    "operator_targeted": bool(record["target_dimension"] == dimension),
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
        "rounds": len(rounds),
        "events": len(events),
        "skipped": dict(skipped),
    }
    return events, coverage


# ----------------------------------------------------------------------- panels


def _recoverable(profile: dict[str, Any], key: str, has_vocab: bool) -> bool:
    if has_vocab:
        return bool(profile.get("dimensions", {}).get(key, {}).get("categories"))
    # ROLE and EVENT define no controlled vocabulary; their attribute is the concept's
    # own text, so the category branch never applies to them.
    return bool(profile.get("tokens"))


def panel_a(
    events: list[dict[str, Any]],
    gold_profiles: dict[int, dict[str, Any]],
    normalization_map: dict[str, Any],
    gold_profiles_other: dict[int, dict[str, Any]] | None = None,
    other_fields: str = "full",
) -> list[dict[str, Any]]:
    """Gold-attribute recoverability per dimension.

    Recoverable means: for a dimension defining a controlled vocabulary, gold
    yields a non-empty category set; for a dimension defining none (ROLE,
    EVENT), gold yields non-empty tokens -- their attribute is the concept's
    text by design, so they are recoverable by construction and are not exposed
    to the empty-gold-set failure mode at all.

    `gold_profiles_other` is the same measurement under the other
    gold-candidate field setting. It separates two very different diagnoses
    that produce the identical `recoverable` number: the attribute genuinely
    does not exist for this dimension, versus it exists but the deployed
    `compact` view (tag + type + standard_label) does not look at the text
    carrying it. On test these differ sharply for FAMILY, so reporting only
    the deployed column would be misleading.

    Note that this is event-weighted, not concept-weighted, and the two can
    diverge a lot: D- fires on a dimension precisely when the hypothesis
    disagrees with gold, which is correlated with gold being unrecoverable
    there. The concept-weighted figure is reported alongside for contrast.
    """
    vocab_sizes = {
        dimension: len(normalization_map.get("dimensions", {}).get(dimension.lower(), {}) or {})
        for dimension in DIMENSIONS
    }
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        subset = [event for event in events if event["dimension"] == dimension]
        key = dimension.lower()
        has_vocab = vocab_sizes[dimension] > 0
        recoverable = 0
        recoverable_other = 0
        for event in subset:
            if _recoverable(gold_profiles.get(event["fact_id"], {}), key, has_vocab):
                recoverable += 1
            if gold_profiles_other is not None and _recoverable(
                gold_profiles_other.get(event["fact_id"], {}), key, has_vocab
            ):
                recoverable_other += 1
        n = len(subset)
        fact_ids = {event["fact_id"] for event in subset}
        recoverable_facts = sum(
            1 for fact_id in fact_ids if _recoverable(gold_profiles.get(fact_id, {}), key, has_vocab)
        )
        max_closeness = max(
            (max(event["closeness_before"], event["closeness_after"]) for event in subset), default=0.0
        )
        fraction = recoverable / n if n else 0.0
        row = {
            "dimension": dimension,
            "n": n,
            "recoverable": round(fraction, 6),
            "max_closeness": round(max_closeness, 6),
            "included": bool(n and fraction >= GOLD_COVERAGE_FLOOR),
            "vocab_categories_defined": vocab_sizes[dimension],
            "label_derived": not has_vocab,
            "n_facts": len(fact_ids),
            "recoverable_concept_weighted": round(recoverable_facts / len(fact_ids), 6) if fact_ids else 0.0,
            "zero_closeness_before_fraction": round(
                sum(1 for event in subset if event["closeness_before"] == 0.0) / n, 6
            )
            if n
            else 0.0,
        }
        if gold_profiles_other is not None:
            row[f"recoverable_{other_fields}"] = round(recoverable_other / n, 6) if n else 0.0
            # Large gap => the attribute exists but the deployed field view misses it.
            row["attribute_missed_by_deployed_view"] = bool(
                n and (recoverable_other - recoverable) / n >= 0.25
            )
        rows.append(row)
    return rows


PANEL_B_SPEC = [
    ("All events", "ALL", "all"),
    ("Targeted", "ALL", "operator_targeted"),
    ("ROLE, targ.", "ROLE", "operator_targeted"),
    ("EVENT, targ.", "EVENT", "operator_targeted"),
    ("FAMILY, targ.", "FAMILY", "operator_targeted"),
    ("QUALIFIER", "QUALIFIER", "all"),
]


def score_subset(
    args: argparse.Namespace, subset: list[dict[str, Any]], seed_offset: int
) -> dict[str, Any] | None:
    if not subset:
        return None
    stats = summarize(subset)
    ci = bootstrap_delta_vs_random(subset, args.bootstrap_samples, args.bootstrap_seed + seed_offset)
    return {
        **stats,
        "mean_delta_minus_random": round(stats["mean_delta"] - stats["random_mean_delta"], 6),
        "mean_delta_minus_random_ci_low": round(ci["mean_delta_minus_random"][0], 6),
        "mean_delta_minus_random_ci_high": round(ci["mean_delta_minus_random"][1], 6),
        "improved_fraction_minus_random": round(
            stats["improved_fraction"] - stats["random_improved_fraction"], 6
        ),
        "improved_minus_random_ci_low": round(ci["improved_fraction_minus_random"][0], 6),
        "improved_minus_random_ci_high": round(ci["improved_fraction_minus_random"][1], 6),
        "beats_random": bool(ci["improved_fraction_minus_random"][0] > 0.0),
    }


def select(events: list[dict[str, Any]], dimension: str, group: str, included: set[str]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        # "ALL" means the included dimensions only -- excluding the artifact ones is
        # the whole point of Panel A, and TEMPORAL alone is ~36% of events.
        if (event["dimension"] in included if dimension == "ALL" else event["dimension"] == dimension)
        and (group == "all" or (group == "operator_targeted") == event["operator_targeted"])
    ]


def build_grid(
    args: argparse.Namespace, events: list[dict[str, Any]], included: set[str], arms: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for arm in ("pooled", *arms):
        pool = events if arm == "pooled" else [event for event in events if event["arm"] == arm]
        for dimension in ("ALL", *DIMENSIONS):
            for group in TARGET_GROUPS:
                subset = select(pool, dimension, group, included)
                seed_offset += 1
                stats = score_subset(args, subset, seed_offset)
                if stats is None:
                    continue
                rows.append({"arm": arm, "dimension": dimension, "target_group": group, **stats})
    return rows


def panel_b(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, dimension, group in PANEL_B_SPEC:
        row = next(
            (
                candidate
                for candidate in rows
                if candidate["arm"] == "pooled"
                and candidate["dimension"] == dimension
                and candidate["target_group"] == group
            ),
            None,
        )
        if row is None:
            out.append({"group": label, "n": 0, "improved": None, "random": None, "ci_low": None, "ci_high": None})
            continue
        out.append(
            {
                "group": label,
                "n": row["n"],
                "improved": round(row["improved_fraction"], 3),
                "random": round(row["random_improved_fraction"], 3),
                "ci_low": round(row["improved_minus_random_ci_low"], 3),
                "ci_high": round(row["improved_minus_random_ci_high"], 3),
                "ci_excludes_zero_favoring_revision": bool(row["improved_minus_random_ci_low"] > 0.0),
            }
        )
    return out


def prose_statistics(
    events: list[dict[str, Any]], rounds: list[dict[str, Any]], included: set[str]
) -> dict[str, Any]:
    """The sentences Appendix G states in text rather than in the table."""
    n = len(events) or 1
    targeted = [event for event in events if event["operator_targeted"]]
    by_dimension = Counter(event["dimension"] for event in events)
    rounds_changing_any = sum(
        1
        for record in rounds
        if any(
            normalize_space(record["hypothesis_before"].get(dimension, ""))
            != normalize_space(record["hypothesis_after"].get(dimension, ""))
            for dimension in DIMENSIONS
        )
    )
    return {
        "value_unchanged_fraction_all_events": round(
            sum(1 for event in events if not event["value_changed"]) / n, 6
        ),
        "rounds_changing_any_value_fraction": round(rounds_changing_any / (len(rounds) or 1), 6),
        "value_changed_fraction_targeted": round(
            sum(1 for event in targeted if event["value_changed"]) / (len(targeted) or 1), 6
        ),
        "targeted_events": len(targeted),
        "events_by_dimension": dict(by_dimension),
        "event_share_by_dimension": {
            dimension: round(count / n, 6) for dimension, count in by_dimension.items()
        },
        "excluded_dimension_event_share": round(
            sum(count for dimension, count in by_dimension.items() if dimension not in included) / n, 6
        ),
    }


# ------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    trace_by_arm = {"learned": args.seq_trace_learned, "random": args.seq_trace_random}
    for arm in arms:
        if arm not in trace_by_arm:
            raise SystemExit(f"Unknown arm '{arm}'; expected learned and/or random.")
        if not Path(trace_by_arm[arm]).exists():
            raise SystemExit(f"Missing trace for arm '{arm}': {trace_by_arm[arm]}")

    print(f"Loading taxonomy from {args.taxonomy_jsonl}", flush=True)
    concepts_by_tag = {concept.tag: concept for concept in load_taxonomy(args.taxonomy_jsonl)}
    normalization_map = load_normalization_map(args.normalization_map)

    rounds: list[dict[str, Any]] = []
    for arm in arms:
        print(f"Reading arm '{arm}' from {trace_by_arm[arm]}", flush=True)
        rounds.extend(iter_rounds(Path(trace_by_arm[arm]), arm, args.limit, args.log_every))
    if not rounds:
        raise SystemExit("No rounds found in the sequential traces.")

    # Gold profiles, one parse per distinct fact rather than per event. The second set is
    # the diagnostic counterfactual: is the attribute absent, or merely not looked for?
    other_fields = "full" if args.gold_candidate_fields == "compact" else "compact"
    gold_profiles: dict[int, dict[str, Any]] = {}
    gold_profiles_other: dict[int, dict[str, Any]] = {}
    for record in rounds:
        fact_id = record["fact_id"]
        if fact_id in gold_profiles or not record["gold_tags"]:
            continue
        concept = concepts_by_tag.get(record["gold_tags"][0])
        if concept is None:
            continue
        gold_profiles[fact_id] = parse_candidate_symbolic_profile(
            gold_candidate(concept, args.gold_candidate_fields), normalization_map
        )
        gold_profiles_other[fact_id] = parse_candidate_symbolic_profile(
            gold_candidate(concept, other_fields), normalization_map
        )

    # The null draws from the observed distribution of values for that dimension,
    # pooled over every logged hypothesis on both sides of every revision.
    value_pool: dict[str, list[str]] = defaultdict(list)
    for record in rounds:
        for source in (record["hypothesis_before"], record["hypothesis_after"]):
            for dimension, value in source.items():
                text = normalize_space(value)
                if text and dimension in DIMENSIONS:
                    value_pool[dimension].append(text)

    events, coverage = build_events(args, rounds, gold_profiles, value_pool, normalization_map)
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True), flush=True)
    if not events:
        raise SystemExit("No D- revision events found; nothing to score.")

    rows_a = panel_a(events, gold_profiles, normalization_map, gold_profiles_other, other_fields)
    included = {row["dimension"] for row in rows_a if row["included"]}
    print(f"Included dimensions: {sorted(included)}", flush=True)
    missed = [row["dimension"] for row in rows_a if row.get("attribute_missed_by_deployed_view")]
    if missed:
        print(
            f"NOTE: on {missed} the gold attribute is far more recoverable under "
            f"'{other_fields}' gold-candidate fields than under the deployed "
            f"'{args.gold_candidate_fields}'. Those dimensions are excluded because the "
            f"deployed view cannot see the attribute, not because it is absent.",
            flush=True,
        )

    grid = build_grid(args, events, included, arms)
    rows_b = panel_b(grid)
    prose = prose_statistics(events, rounds, included)

    # --------------------------------------------------------------- write out
    with (args.output_dir / "per_event.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_csv(name: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("panel_a_recoverability.csv", rows_a)
    write_csv("panel_b_effectiveness.csv", rows_b)
    write_csv("revision_effectiveness.csv", grid)

    metrics = {
        "experiment": "ags_revision_diagnostics",
        "split": "test",
        "table": "Table 14 (Appendix G)",
        "panel_a": rows_a,
        "panel_b": rows_b,
        "prose_statistics": prose,
        "included_dimensions": sorted(included),
        "excluded_dimensions": sorted(set(DIMENSIONS) - included),
        "coverage": coverage,
        "config": {
            "arms": arms,
            "traces": {arm: str(trace_by_arm[arm]) for arm in arms},
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "gold_candidate_fields": {
                "deployed": args.gold_candidate_fields,
                "counterfactual_column": other_fields,
                "compact_includes": ["tag", "entity_type", "standard_label"],
                "full_adds": ["documentation", "retrieval_text"],
                "note": (
                    "gold categories are matched against this text only; a dimension whose "
                    f"recoverable_{other_fields} greatly exceeds recoverable is excluded because "
                    "the deployed view cannot see the attribute, not because it is absent"
                ),
            },
            "gold_coverage_floor": GOLD_COVERAGE_FLOOR,
            "random_draws": args.random_draws,
            "random_seed": args.random_seed,
            "normalization_map_version": map_version(args.normalization_map),
            "bootstrap": {
                "iterations": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "unit": "context",
            },
            "limit": args.limit,
            "feedback_layers": (
                "deterministic only: the test AGS-Seq runs set llm_feedback_enabled=False, so a "
                "'merged' column would be the deterministic column relabelled"
            ),
            "after_value_source": (
                "the round's own logged revised_hypothesis, verified equal to the recovered "
                "pre-revision hypothesis of round t+1 (3,379 comparisons, 0 mismatches), which "
                "keeps the final round the dev consecutive-pairing approach dropped"
            ),
        },
        "grid": grid,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n=== Table 14 Panel A: gold-attribute recoverability (test) ===", flush=True)
    print(
        f"  {'Dimension':<12} {'n':>7} {'Recov.':>8} {'Max':>6}  Included   "
        f"[{other_fields}: {'Recov.':>7}]",
        flush=True,
    )
    for row in rows_a:
        print(
            f"  {row['dimension']:<12} {row['n']:>7} {row['recoverable']:>8.3f} "
            f"{row['max_closeness']:>6.2f}  {'yes' if row['included'] else 'no':<9}  "
            f"{row.get(f'recoverable_{other_fields}', 0.0):>7.3f}",
            flush=True,
        )

    print("\n=== Table 14 Panel B: revision effectiveness (test) ===", flush=True)
    print(f"  {'Group':<16} {'n':>7} {'Impr.':>7} {'Rand.':>7}  95% CI", flush=True)
    for row in rows_b:
        if row["improved"] is None:
            print(f"  {row['group']:<16} {'--':>7}  (no events)", flush=True)
            continue
        mark = " †" if row["ci_excludes_zero_favoring_revision"] else ""
        print(
            f"  {row['group']:<16} {row['n']:>7} {row['improved']:>7.3f} {row['random']:>7.3f}  "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]{mark}",
            flush=True,
        )

    print(f"\nWrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
