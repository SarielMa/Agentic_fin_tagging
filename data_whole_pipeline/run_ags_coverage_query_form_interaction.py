#!/usr/bin/env python3
"""Label-coverage x query-form interaction on the frozen AGS 661-fact sample.

Fully offline: direct retrieval needs no LLM, the free-text one-pass generations are
reused from the coverage-pilot Arm A records, and the structured hypotheses are reused
from the component-validation log. Only BM25 retrieval is re-run, twice per cell, with
label_coverage_weight off (0.0) and on (1.0).

The 2 x 5 grid:

  label_coverage_weight in {0.0, 1.0}
  query_form in
     raw_context      build_direct_query, no LLM              (== direct_retrieval)
     freetext         reused one-pass grounding query          (== one_pass_grounding)
     structured_def   primary structured hypothesis, query_def rendering
     structured_lab   primary structured hypothesis, query_lab rendering
     structured_dual  RRF fusion of the def and lab retrievals of the primary hypothesis

All five forms are a single retrieval per fact, so coverage off/on is the identical
toggle everywhere and the def/lab/dual cells differ from each other only in rendering --
the multi-hypothesis J=3 fusion of the deployed pipeline is deliberately held out, so
this isolates rendering x coverage rather than confounding it with fusion. raw_context
and freetext reproduce runs_ags_label_coverage_ablation exactly (verified: identical
retriever config, type_filter on, pool over the full type-filtered index).

The claim under test is super-additivity: that coverage helps more under label-form
rendering than under raw context or free text. The primary read is the difference of
per-form coverage gains,

    interaction = gain(structured_lab) - gain(raw_context)
                = gain(structured_lab) - gain(freetext)

bootstrapped directly at the context level as a paired difference of differences, not
compared CI-to-CI. The sharper secondary read stratifies each form's coverage gain by
gold label token count and asks whether structured_lab's profile is flatter across
label lengths than the short-label-concentrated profiles of raw_context and freetext.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    fact_context_key,
    load_jsonl,
    row_to_example,
)
from run_ags_label_coverage_ablation import (
    LABEL_TOKEN_STRATA,
    check_one_pass_reuse,
    label_token_count,
    label_token_stratum,
    load_one_pass_generations,
    temporary_retriever_weight,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    SCRIPT_DIR,
    Example,
    TaxonomyRetriever,
    build_direct_query,
    first_gold_rank,
    fuse_round_candidates,
    load_taxonomy,
    normalize_space,
    normalize_tag,
    retrieval_query_from_grounding,
    retrieve_candidates,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_coverage_query_form_interaction" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
DEFAULT_ONE_PASS_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "arm_A_hypothesis_candidates.jsonl"
DEFAULT_COMPONENT_DIR = SCRIPT_DIR / "runs_ags_component_validation" / "qwen3_32b"

QUERY_FORMS = ("raw_context", "freetext", "structured_def", "structured_lab", "structured_dual")
STRUCTURED_FORMS = ("structured_def", "structured_lab", "structured_dual")
WEIGHT_LABELS = ("off", "on")
MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
REPORT_METRICS = ("recall_at_10", "recall_at_50", "mrr")
# The primary hypothesis; def/lab/dual all render this one, so they differ only in rendering.
PRIMARY_HYPOTHESIS_IDX = 0
# stratum -> ordinal position, for the length-profile slope.
STRATUM_X = {"1": 0, "2": 1, "3-4": 2, "5+": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--one-pass-path", type=Path, default=DEFAULT_ONE_PASS_PATH)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--label-coverage-weight-off", type=float, default=0.0)
    parser.add_argument("--label-coverage-weight-on", type=float, default=1.0)
    parser.add_argument("--label-coverage-pool-multiplier", type=int, default=0)
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_primary_hypotheses(component_dir: Path) -> dict[int, dict[str, Any]]:
    """The primary structured hypothesis per fact, carrying its rendered def/lab queries."""
    primary: dict[int, dict[str, Any]] = {}
    for record in load_jsonl(component_dir / "hypotheses.jsonl"):
        if int(record.get("hypothesis_idx", 0)) != PRIMARY_HYPOTHESIS_IDX:
            continue
        fact_id = int(record["fact_id"])
        if fact_id in primary:
            raise ValueError(f"Duplicate primary hypothesis for fact_id={fact_id}")
        primary[fact_id] = record
    return primary


def structured_queries(
    example: Example,
    hypothesis: dict[str, Any],
) -> dict[str, str]:
    return {
        "def": retrieval_query_from_grounding(example, normalize_space(hypothesis.get("query_def", ""))),
        "lab": retrieval_query_from_grounding(example, normalize_space(hypothesis.get("query_lab", ""))),
    }


def retrieve_form(
    retriever: TaxonomyRetriever,
    example: Example,
    form: str,
    raw_query: str,
    freetext_query: str,
    structured: dict[str, str],
    top_k: int,
    rrf_kappa: float,
) -> int | None:
    if form == "raw_context":
        candidates = retrieve_candidates(retriever, raw_query, example.entity_type, top_k)
    elif form == "freetext":
        candidates = retrieve_candidates(retriever, freetext_query, example.entity_type, top_k)
    elif form == "structured_def":
        candidates = retrieve_candidates(retriever, structured["def"], example.entity_type, top_k)
    elif form == "structured_lab":
        candidates = retrieve_candidates(retriever, structured["lab"], example.entity_type, top_k)
    elif form == "structured_dual":
        def_candidates = retrieve_candidates(retriever, structured["def"], example.entity_type, top_k)
        lab_candidates = retrieve_candidates(retriever, structured["lab"], example.entity_type, top_k)
        candidates = fuse_round_candidates(
            [{"round": 1, "candidates": def_candidates}, {"round": 2, "candidates": lab_candidates}],
            top_k,
            rrf_kappa,
        )
    else:
        raise ValueError(f"Unsupported query_form={form}")
    return first_gold_rank([candidate["tag"] for candidate in candidates], example.gold_tags)


def metric_values(ranks: dict[int, int | None]) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {metric: {} for metric in METRICS}
    for fact_id, rank in ranks.items():
        for depth in DEPTHS:
            values[f"recall_at_{depth}"][fact_id] = float(rank is not None and rank <= depth)
        values["mrr"][fact_id] = 0.0 if rank is None else 1.0 / rank
    return values


def fact_ids_for_modality(examples: list[Example], modality: str) -> list[int]:
    return [
        example.example_idx
        for example in examples
        if modality == "pooled" or example.input_type == modality
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ols_slope(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of y on x; None when x has no spread."""
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = [row_to_example(row) for row in load_jsonl(args.sample_path)]
    examples_by_id = {example.example_idx: example for example in examples}
    context_keys = {fact_context_key(example) for example in examples}

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    taxonomy_by_tag = {concept.tag: concept for concept in taxonomy}

    stratum_by_fact: dict[int, str] = {}
    token_count_by_fact: dict[int, int] = {}
    for example in examples:
        if len(example.gold_tags) != 1:
            raise ValueError(f"fact_id={example.example_idx} has {len(example.gold_tags)} gold tags")
        concept = taxonomy_by_tag.get(normalize_tag(example.gold_tags[0]))
        if concept is None:
            raise ValueError(f"gold concept absent from taxonomy for fact_id={example.example_idx}")
        count = label_token_count(concept.standard_label or concept.raw_tag)
        token_count_by_fact[example.example_idx] = count
        stratum_by_fact[example.example_idx] = label_token_stratum(count)

    generations, provenance = load_one_pass_generations(args.one_pass_path)
    reuse_check = check_one_pass_reuse(examples, generations, provenance)
    if not reuse_check["passed"]:
        raise RuntimeError(f"one-pass reuse check failed: {reuse_check['failures']}")

    primary_hypotheses = load_primary_hypotheses(args.component_dir)
    missing_hypotheses = sorted({example.example_idx for example in examples} - set(primary_hypotheses))
    if missing_hypotheses:
        raise RuntimeError(f"{len(missing_hypotheses)} facts lack a primary structured hypothesis")

    raw_queries = {example.example_idx: build_direct_query(example) for example in examples}
    freetext_queries = {
        example.example_idx: retrieval_query_from_grounding(
            example, generations[example.example_idx]["query_text"]
        )
        for example in examples
    }
    structured_by_fact = {
        example.example_idx: structured_queries(example, primary_hypotheses[example.example_idx])
        for example in examples
    }

    retriever = TaxonomyRetriever(taxonomy, type_filter=args.type_filter)
    weights = {"off": args.label_coverage_weight_off, "on": args.label_coverage_weight_on}

    ranks: dict[tuple[str, str], dict[int, int | None]] = {}
    for form in QUERY_FORMS:
        for weight_label, weight in weights.items():
            with temporary_retriever_weight(retriever, weight, args.label_coverage_pool_multiplier):
                fact_ranks: dict[int, int | None] = {}
                for position, example in enumerate(examples, start=1):
                    fact_ranks[example.example_idx] = retrieve_form(
                        retriever,
                        example,
                        form,
                        raw_queries[example.example_idx],
                        freetext_queries[example.example_idx],
                        structured_by_fact[example.example_idx],
                        args.top_k,
                        args.rrf_kappa,
                    )
                    if args.log_every and position % args.log_every == 0:
                        print(f"{form}/{weight_label}: {position}/{len(examples)}", flush=True)
                ranks[(form, weight_label)] = fact_ranks

    values = {key: metric_values(fact_ranks) for key, fact_ranks in ranks.items()}

    per_fact_rows = [
        {
            "fact_id": example.example_idx,
            "context_key": fact_context_key(example),
            "modality": example.input_type,
            "entity_type": example.entity_type,
            "gold_tag": normalize_tag(example.gold_tags[0]),
            "gold_label_token_count": token_count_by_fact[example.example_idx],
            "gold_label_tokens": stratum_by_fact[example.example_idx],
            **{
                f"rank_{form}_{weight_label}": ranks[(form, weight_label)][example.example_idx]
                for form in QUERY_FORMS
                for weight_label in WEIGHT_LABELS
            },
        }
        for example in examples
    ]
    write_jsonl(args.output_dir / "per_fact_ranks.jsonl", per_fact_rows)

    # --- per-form coverage gain --------------------------------------------------
    def gain_series(form: str, metric: str, fact_ids: list[int]) -> dict[int, float]:
        off_values = values[(form, "off")][metric]
        on_values = values[(form, "on")][metric]
        return {fact_id: on_values[fact_id] - off_values[fact_id] for fact_id in fact_ids}

    gain_rows: list[dict[str, Any]] = []
    seed_offset = 0
    gain_ci_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for form in QUERY_FORMS:
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                paired = gain_series(form, metric, fact_ids)
                ci = bootstrap_context_ci(
                    paired, examples_by_id, fact_ids, args.bootstrap_samples, args.bootstrap_seed + seed_offset
                )
                gain_ci_cache[(form, modality, metric)] = ci
                gain_rows.append(
                    {
                        "query_form": form,
                        "modality": modality,
                        "metric": metric,
                        "n_facts": ci.get("fact_count", len(fact_ids)),
                        "n_contexts": ci.get("context_count", 0),
                        "off": round(mean([values[(form, "off")][metric][f] for f in fact_ids]), 6),
                        "on": round(mean([values[(form, "on")][metric][f] for f in fact_ids]), 6),
                        "gain": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                    }
                )

    # --- interaction: paired difference of differences ---------------------------
    interaction_rows: list[dict[str, Any]] = []
    seed_offset = 100_000
    contrasts = (("structured_lab", "raw_context"), ("structured_lab", "freetext"))
    for target, reference in contrasts:
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                target_gain = gain_series(target, metric, fact_ids)
                reference_gain = gain_series(reference, metric, fact_ids)
                dod = {fact_id: target_gain[fact_id] - reference_gain[fact_id] for fact_id in fact_ids}
                ci = bootstrap_context_ci(
                    dod, examples_by_id, fact_ids, args.bootstrap_samples, args.bootstrap_seed + seed_offset
                )
                target_ci = gain_ci_cache[(target, modality, metric)]
                reference_ci = gain_ci_cache[(reference, modality, metric)]
                interaction_rows.append(
                    {
                        "target_form": target,
                        "reference_form": reference,
                        "modality": modality,
                        "metric": metric,
                        "n_facts": ci.get("fact_count", len(fact_ids)),
                        "n_contexts": ci.get("context_count", 0),
                        "target_gain": target_ci["mean"],
                        "reference_gain": reference_ci["mean"],
                        "interaction": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                        "interaction_positive_significant": bool(ci["ci_low"] > 0.0),
                    }
                )

    # --- length-stratified coverage gain, pooled ---------------------------------
    stratum_rows: list[dict[str, Any]] = []
    seed_offset = 200_000
    slope_by_form: dict[str, dict[str, float | None]] = {}
    for form in QUERY_FORMS:
        slope_by_form[form] = {}
        for metric in METRICS:
            stratum_points: list[tuple[float, float]] = []
            for stratum in LABEL_TOKEN_STRATA:
                seed_offset += 1
                fact_ids = [
                    example.example_idx
                    for example in examples
                    if stratum_by_fact[example.example_idx] == stratum
                ]
                paired = gain_series(form, metric, fact_ids)
                ci = bootstrap_context_ci(
                    paired, examples_by_id, fact_ids, args.bootstrap_samples, args.bootstrap_seed + seed_offset
                )
                stratum_points.append((float(STRATUM_X[stratum]), ci["mean"]))
                stratum_rows.append(
                    {
                        "query_form": form,
                        "gold_label_tokens": stratum,
                        "metric": metric,
                        "n_facts": len(fact_ids),
                        "n_contexts": ci.get("context_count", 0),
                        "off": round(mean([values[(form, "off")][metric][f] for f in fact_ids]), 6),
                        "on": round(mean([values[(form, "on")][metric][f] for f in fact_ids]), 6),
                        "gain": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                    }
                )
            short = next(y for x, y in stratum_points if x == STRATUM_X["1"])
            long = next(y for x, y in stratum_points if x == STRATUM_X["5+"])
            slope_by_form[form][metric] = {
                "slope": ols_slope(stratum_points),
                "short_1tok_gain": round(short, 6),
                "long_5plus_gain": round(long, 6),
                "decay_short_minus_long": round(short - long, 6),
            }

    write_csv(
        args.output_dir / "coverage_gain_by_form.csv",
        gain_rows,
        [
            "query_form",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "off",
            "on",
            "gain",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
        ],
    )
    write_csv(
        args.output_dir / "coverage_interaction.csv",
        interaction_rows,
        [
            "target_form",
            "reference_form",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "target_gain",
            "reference_gain",
            "interaction",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "interaction_positive_significant",
        ],
    )
    write_csv(
        args.output_dir / "coverage_gain_by_length.csv",
        stratum_rows,
        [
            "query_form",
            "gold_label_tokens",
            "metric",
            "n_facts",
            "n_contexts",
            "off",
            "on",
            "gain",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
        ],
    )

    # --- pre-registered decision -------------------------------------------------
    def interaction_for(target: str, reference: str, metric: str, modality: str = "pooled") -> dict[str, Any]:
        return next(
            row
            for row in interaction_rows
            if row["target_form"] == target
            and row["reference_form"] == reference
            and row["metric"] == metric
            and row["modality"] == modality
        )

    primary_reads = {
        f"structured_lab_minus_{reference}": {
            metric: interaction_for("structured_lab", reference, metric)
            for metric in REPORT_METRICS
        }
        for reference in ("raw_context", "freetext")
    }

    # Interaction is significant only if it is positive with CI excluding zero against BOTH
    # references, on the primary R@10 read (pooled).
    lab_vs_raw_r10 = interaction_for("structured_lab", "raw_context", "recall_at_10")
    lab_vs_free_r10 = interaction_for("structured_lab", "freetext", "recall_at_10")
    interaction_significant = bool(
        lab_vs_raw_r10["interaction_positive_significant"]
        and lab_vs_free_r10["interaction_positive_significant"]
    )

    # Flatter profile: structured_lab decays less across label length than both baselines,
    # on R@10. Decay = short(1-tok) gain minus long(5+) gain; smaller is flatter.
    lab_decay = slope_by_form["structured_lab"]["recall_at_10"]["decay_short_minus_long"]
    raw_decay = slope_by_form["raw_context"]["recall_at_10"]["decay_short_minus_long"]
    free_decay = slope_by_form["freetext"]["recall_at_10"]["decay_short_minus_long"]
    flatter_profile = bool(lab_decay < raw_decay and lab_decay < free_decay)

    claim_combination = bool(interaction_significant and flatter_profile)
    if claim_combination:
        decision = "claim_combination"
        interpretation = (
            "Label-form rendering makes coverage exploitable: the coverage gain under "
            "structured_lab exceeds the gain under both raw_context and free text with CIs "
            "excluding zero, and its length profile is flatter. Report as a method contribution. "
            "Coverage stays enabled for all methods in the main results table."
        )
    else:
        decision = "shared_index_improvement"
        interpretation = (
            "The coverage x rendering interaction is not established: it is not positive with a "
            "CI excluding zero against both references"
            + ("" if interaction_significant else " (interaction not significant)")
            + (", and the structured_lab length profile is not flatter than both baselines" if not flatter_profile else "")
            + ". Coverage is a shared index improvement, not an AGS-specific effect. Note it in one "
            "sentence in the setup section, make no AGS-specific claim, and keep it enabled for all "
            "methods including baselines -- disabling it for baselines is not justified."
        )

    stratum_counts = Counter(stratum_by_fact.values())
    metrics = {
        "experiment": "ags_coverage_query_form_interaction",
        "question": (
            "Is label coverage super-additive with label-form rendering? Only a positive "
            "coverage x rendering interaction would justify an AGS-specific coverage claim."
        ),
        "design": {
            "query_forms": list(QUERY_FORMS),
            "structured_forms_use": (
                "the primary structured hypothesis (idx 0), rendered def/lab; dual is the RRF "
                "fusion of that hypothesis's def and lab retrievals. J=3 multi-hypothesis fusion "
                "is held out so def/lab/dual differ only in rendering."
            ),
            "reused_generations": {
                "freetext": str(args.one_pass_path),
                "structured": str(args.component_dir / "hypotheses.jsonl"),
            },
            "raw_and_freetext_reproduce": "runs_ags_label_coverage_ablation (verified identical retriever config)",
            "retriever": {
                "type_filter": args.type_filter,
                "label_coverage_weight_off": args.label_coverage_weight_off,
                "label_coverage_weight_on": args.label_coverage_weight_on,
                "label_coverage_pool_multiplier": args.label_coverage_pool_multiplier,
                "top_k": args.top_k,
                "rrf_kappa": args.rrf_kappa,
            },
        },
        "sample": {
            "fact_count": len(examples),
            "context_count": len(context_keys),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
            "label_token_strata_counts": {s: stratum_counts.get(s, 0) for s in LABEL_TOKEN_STRATA},
        },
        "coverage_gain_rows": gain_rows,
        "interaction_rows": interaction_rows,
        "gain_by_length_rows": stratum_rows,
        "primary_read_pooled": primary_reads,
        "length_profile": slope_by_form,
        "decision_inputs": {
            "interaction_significant_both_references_r10_pooled": interaction_significant,
            "lab_vs_raw_r10": {
                "interaction": lab_vs_raw_r10["interaction"],
                "ci": [lab_vs_raw_r10["ci_low"], lab_vs_raw_r10["ci_high"]],
            },
            "lab_vs_freetext_r10": {
                "interaction": lab_vs_free_r10["interaction"],
                "ci": [lab_vs_free_r10["ci_low"], lab_vs_free_r10["ci_high"]],
            },
            "flatter_profile_r10": flatter_profile,
            "decay_short_minus_long_r10": {
                "structured_lab": lab_decay,
                "raw_context": raw_decay,
                "freetext": free_decay,
            },
        },
        "decision": decision,
        "claim_combination": claim_combination,
        "coverage_enabled_for_all_methods": True,
        "interpretation": interpretation,
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "context",
            "difference_of_differences": "paired per fact, bootstrapped directly (not CI-to-CI)",
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "decision": decision,
                "claim_combination": claim_combination,
                "lab_vs_raw_r10": metrics["decision_inputs"]["lab_vs_raw_r10"],
                "lab_vs_freetext_r10": metrics["decision_inputs"]["lab_vs_freetext_r10"],
                "flatter_profile_r10": flatter_profile,
                "decay_r10": metrics["decision_inputs"]["decay_short_minus_long_r10"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
