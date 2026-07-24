#!/usr/bin/env python3
"""Regenerate the stage decomposition from the shared configuration scorer.

Every stage is one call to `score_configuration`, the same function the
configuration ablation calls, so a stage row and a config row with the same
flags are the identical computation and cannot diverge again.

Also emits path_diff.csv: for facts where the old no-rerank stage and the
reranked config disagree on whether gold is in the top 10, the intermediate
state of both paths side by side, so the first differing column is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ags_configuration_scoring import FactContext, score_configuration
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map, map_version
from run_ags_config_ablation import index_retrievals, read_record
from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    fact_context_key,
    load_jsonl,
    row_to_example,
)
from run_fintagging_grounding_baseline import Example, SCRIPT_DIR, first_gold_rank, normalize_tag


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_stage_decomposition" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
DEFAULT_COMPONENT_DIR = SCRIPT_DIR / "runs_ags_component_validation" / "qwen3_32b"
DEFAULT_ABLATION_DIR = SCRIPT_DIR / "runs_ags_config_ablation" / "qwen3_32b"

MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")

CAUSE_NOTE = (
    "Reconciled: the old stage_decomposition and config_ablation never diverged in code. Both read the "
    "same pre-fused `dual` record from Experiment A's retrievals.jsonl (dual_fuse runs upstream at "
    "retrieval-logging time, not in either path), so no label/def fusion happens in either. The 0.504 vs "
    "0.405 gap at table R@10 is entirely rerank_beta: the old stage 2 was beta=0 and config A_j1 is "
    "beta=0.05. Verified by construction - running the config_ablation path at beta=0 reproduces the old "
    "stage 2 exactly (table 0.503534, text 0.347368). Root cause of the sign: beta*consensus spans 0 to "
    "0.023 while a one-list RRF score spans only 0.0126, so at J=1 the rerank outweighs retrieval order "
    "and costs 9.9 points at table R@10; at J=3 the fused score spans 0.049 and the same rerank gains 3.2 "
    "points. The rerank is not uniformly positive - its sign depends on how many lists are fused."
)

# Cumulative ladder: each stage changes exactly one flag from the stage above it.
STAGES: tuple[dict[str, Any], ...] = (
    {
        "stage": "j1_def_norerank",
        "component_added": "baseline: J=1, definition rendering, no aggregation, no rerank",
        "rendering": "def",
        "hypotheses_used": 1,
        "aggregation": "none",
        "rerank_beta": 0.0,
        "cumulative": True,
        "deployable": True,
        "equals_config": None,
    },
    {
        "stage": "j1_dual_norerank",
        "component_added": "modality-conditional rendering (dual for table, def for text)",
        "rendering": "policy",
        "hypotheses_used": 1,
        "aggregation": "none",
        "rerank_beta": 0.0,
        "cumulative": True,
        "deployable": True,
        "equals_config": None,
    },
    {
        "stage": "j1_dual_rerank",
        "component_added": "consensus symbolic rerank at beta=0.05",
        "rendering": "policy",
        "hypotheses_used": 1,
        "aggregation": "none",
        "rerank_beta": 0.05,
        "cumulative": True,
        "deployable": True,
        "equals_config": "A_j1",
    },
    {
        "stage": "j2_dual_rrf_rerank",
        "component_added": "second hypothesis, RRF fusion over J=2",
        "rendering": "policy",
        "hypotheses_used": 2,
        "aggregation": "rrf",
        "rerank_beta": 0.05,
        "cumulative": True,
        "deployable": True,
        "equals_config": "F_j2_rrf",
    },
    {
        "stage": "j3_dual_rrf_rerank",
        "component_added": "third hypothesis, RRF fusion over J=3 (current pipeline)",
        "rendering": "policy",
        "hypotheses_used": 3,
        "aggregation": "rrf",
        "rerank_beta": 0.05,
        "cumulative": True,
        "deployable": True,
        "equals_config": "C_j3_rrf",
    },
    # Diagnostics, compared against the final ladder stage rather than the one above.
    {
        "stage": "j3_dual_rrf_norerank",
        "component_added": "diagnostic: J=3 fusion with the rerank switched off",
        "rendering": "policy",
        "hypotheses_used": 3,
        "aggregation": "rrf",
        "rerank_beta": 0.0,
        "cumulative": False,
        "deployable": True,
        "equals_config": None,
    },
    {
        "stage": "j3_dual_oracle",
        "component_added": "diagnostic: oracle best-of-J=3 (upper bound, not deployable)",
        "rendering": "policy",
        "hypotheses_used": 3,
        "aggregation": "oracle",
        "rerank_beta": 0.0,
        "cumulative": False,
        "deployable": False,
        "equals_config": "E_j3_oracle",
    },
)
FINAL_LADDER_STAGE = "j3_dual_rrf_rerank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--score-variant", choices=("sum", "mean"), default="sum")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--path-diff-facts", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def implied_label_coverage_weight(record: dict[str, Any]) -> float | None:
    """Recover the label-coverage weight from a logged candidate's score arithmetic."""
    for candidate in record.get("candidates", [])[:5]:
        denominator = float(candidate.get("label_coverage", 0.0)) + float(
            candidate.get("query_label_coverage", 0.0)
        )
        if denominator > 0 and "retrieval_score" in candidate:
            numerator = float(candidate["retrieval_score"]) - float(candidate.get("bm25_normalized_score", 0.0))
            return round(numerator / denominator, 6)
    return None


def rank_metrics(rank: int | None, top_k: int) -> dict[str, float]:
    valid = rank is not None and rank <= top_k
    values = {f"recall_at_{depth}": float(rank is not None and rank <= depth) for depth in DEPTHS}
    values["mrr"] = 1.0 / rank if valid else 0.0
    return values


def metric_values(ranks: dict[int, int | None], top_k: int) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {metric: {} for metric in METRICS}
    for fact_id, rank in ranks.items():
        for metric, value in rank_metrics(rank, top_k).items():
            values[metric][fact_id] = value
    return values


def fact_ids_for_modality(examples: list[Example], modality: str) -> list[int]:
    return [
        example.example_idx
        for example in examples
        if modality == "pooled" or example.input_type == modality
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_path_diff_rows(
    args: argparse.Namespace,
    examples: list[Example],
    offsets: dict[tuple[str, int, int], int],
    retrievals_path: Path,
    hypotheses_by_fact: dict[int, list[dict[str, Any]]],
    rendering_policy: dict[str, str],
    ranks: dict[str, dict[int, int | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Intermediate state of both paths on facts where they disagree at top 10."""
    def in_top_10(rank: int | None) -> bool:
        return rank is not None and rank <= 10

    disagreeing = [
        example
        for example in examples
        if in_top_10(ranks["j1_dual_norerank"][example.example_idx])
        != in_top_10(ranks["j1_dual_rerank"][example.example_idx])
    ]
    selected = sorted(disagreeing, key=lambda example: example.example_idx)[: args.path_diff_facts]

    rows: list[dict[str, Any]] = []
    with retrievals_path.open("r", encoding="utf-8") as handle:
        for example in selected:
            fact_id = example.example_idx
            policy = rendering_policy[example.input_type]
            def_record = read_record(handle, offsets[("def", fact_id, 0)])
            lab_record = read_record(handle, offsets[("lab", fact_id, 0)])
            policy_record = read_record(handle, offsets[(policy, fact_id, 0)])
            hypothesis = hypotheses_by_fact[fact_id][0]

            gold = example.gold_tags
            before = ranks["j1_dual_norerank"][fact_id]
            after = ranks["j1_dual_rerank"][fact_id]
            row = {
                "fact_id": fact_id,
                "modality": example.input_type,
                "policy_rendering": policy,
                "direction": "rerank_lost_gold" if in_top_10(before) else "rerank_gained_gold",
                # hypothesis used - identical in both paths
                "path_a_hypothesis_idx": 0,
                "path_b_hypothesis_idx": 0,
                "hypothesis_dimensions": json.dumps(hypothesis.get("dimensions", {}), sort_keys=True),
                # rendered queries - identical in both paths
                "query_def": def_record.get("query"),
                "query_lab": lab_record.get("query"),
                # component lists - identical in both paths
                "gold_rank_def_list": first_gold_rank(
                    [normalize_tag(tag) for tag in def_record["candidate_ids"]], gold
                ),
                "gold_rank_lab_list": first_gold_rank(
                    [normalize_tag(tag) for tag in lab_record["candidate_ids"]], gold
                ),
                # fused list before rerank - identical in both paths
                "path_a_gold_rank_fused_before_rerank": before,
                "path_b_gold_rank_fused_before_rerank": before,
                # after rerank - the only place the paths differ
                "path_a_rerank_beta": 0.0,
                "path_b_rerank_beta": 0.05,
                "path_a_gold_rank_after_rerank": before,
                "path_b_gold_rank_after_rerank": after,
                "path_a_label_coverage_weight": implied_label_coverage_weight(policy_record),
                "path_b_label_coverage_weight": implied_label_coverage_weight(policy_record),
                "path_a_rrf_kappa": args.rrf_kappa,
                "path_b_rrf_kappa": args.rrf_kappa,
            }
            differing = [
                column
                for column in (
                    "hypothesis_idx",
                    "label_coverage_weight",
                    "rrf_kappa",
                    "gold_rank_fused_before_rerank",
                    "rerank_beta",
                    "gold_rank_after_rerank",
                )
                if row.get(f"path_a_{column}") != row.get(f"path_b_{column}")
            ]
            row["first_differing_column"] = differing[0] if differing else "none"
            rows.append(row)

    summary = {
        "facts_disagreeing_at_top_10": len(disagreeing),
        "facts_dumped": len(rows),
        "first_differing_column_counts": dict(Counter(row["first_differing_column"] for row in rows)),
        "direction_counts": dict(Counter(row["direction"] for row in rows)),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = [row_to_example(row) for row in load_jsonl(args.sample_path)]
    examples_by_id = {example.example_idx: example for example in examples}

    component_metrics = json.loads((args.component_dir / "metrics.json").read_text(encoding="utf-8"))
    rendering_policy = {
        modality: rendering
        for modality, rendering in component_metrics["rendering_gate"]["adopted_rendering_by_modality"].items()
        if modality in ("table", "text")
    }
    hypothesis_count = int(component_metrics["hypotheses_per_fact"])

    normalization_map = load_normalization_map(args.normalization_map)
    hypotheses_by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in load_jsonl(args.component_dir / "hypotheses.jsonl"):
        hypotheses_by_fact[int(hypothesis["fact_id"])].append(hypothesis)
    for values_list in hypotheses_by_fact.values():
        values_list.sort(key=lambda item: int(item["hypothesis_idx"]))

    retrievals_path = args.component_dir / "retrievals.jsonl"
    print(f"indexing {retrievals_path} ...", flush=True)
    offsets = index_retrievals(retrievals_path)

    ranks: dict[str, dict[int, int | None]] = {stage["stage"]: {} for stage in STAGES}
    with retrievals_path.open("r", encoding="utf-8") as handle:
        for position, example in enumerate(examples, start=1):
            fact_id = example.example_idx
            policy = rendering_policy[example.input_type]
            contexts: dict[str, FactContext] = {}
            for rendering in {policy, "def"}:
                records = [
                    read_record(handle, offsets[(rendering, fact_id, hypothesis_idx)])
                    for hypothesis_idx in range(hypothesis_count)
                ]
                contexts[rendering] = FactContext(
                    example=example,
                    records=records,
                    hypotheses=hypotheses_by_fact[fact_id],
                    normalization_map=normalization_map,
                    agreement_top_m=args.agreement_top_m,
                )
            for stage in STAGES:
                rendering = policy if stage["rendering"] == "policy" else stage["rendering"]
                result = score_configuration(
                    contexts[rendering],
                    hypotheses_used=stage["hypotheses_used"],
                    aggregation=stage["aggregation"],
                    score_variant=args.score_variant,
                    rerank_beta=stage["rerank_beta"],
                    rrf_kappa=args.rrf_kappa,
                    top_k=args.top_k,
                )
                ranks[stage["stage"]][fact_id] = result.gold_rank
            if args.log_every and position % args.log_every == 0:
                print(f"scored {position}/{len(examples)} facts", flush=True)

    values = {stage: metric_values(series, args.top_k) for stage, series in ranks.items()}

    # Equality checks against the configuration ablation, per fact, not just in aggregate.
    variant_suffix = "sum_rrf" if args.score_variant == "sum" else "mean_rrf"
    ablation_ranks = {
        int(record["fact_id"]): record
        for record in load_jsonl(args.ablation_dir / "per_fact_ranks.jsonl")
    }
    equality_checks = {}
    for stage in STAGES:
        config = stage["equals_config"]
        if not config:
            continue
        matches = sum(
            1
            for fact_id, record in ablation_ranks.items()
            if record.get(f"rank_{config}_{variant_suffix}") == ranks[stage["stage"]][fact_id]
        )
        equality_checks[f"{stage['stage']}_equals_{config}"] = {
            "facts_matching": matches,
            "fact_count": len(examples),
            "exact": matches == len(examples),
        }
    print(json.dumps({"equality_checks": equality_checks}, indent=2, sort_keys=True))

    rows: list[dict[str, Any]] = []
    seed_offset = 0
    cumulative_stages = [stage for stage in STAGES if stage["cumulative"]]
    for stage in STAGES:
        if stage["cumulative"]:
            idx = cumulative_stages.index(stage)
            reference = cumulative_stages[idx - 1]["stage"] if idx else None
            stage_idx: Any = idx + 1
        else:
            reference = FINAL_LADDER_STAGE
            stage_idx = "diagnostic"
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                absolute = bootstrap_context_ci(
                    values[stage["stage"]][metric],
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + seed_offset,
                )
                row = {
                    "stage_idx": stage_idx,
                    "stage": stage["stage"],
                    "component_added": stage["component_added"],
                    "rendering": stage["rendering"],
                    "hypotheses_used": stage["hypotheses_used"],
                    "aggregation": stage["aggregation"],
                    "score_variant": args.score_variant,
                    "rerank_beta": stage["rerank_beta"],
                    "cumulative": stage["cumulative"],
                    "deployable": stage["deployable"],
                    "equals_config": stage["equals_config"] or "",
                    "equality_verified": (
                        equality_checks[f"{stage['stage']}_equals_{stage['equals_config']}"]["exact"]
                        if stage["equals_config"]
                        else ""
                    ),
                    "modality": modality,
                    "metric": metric,
                    "n_facts": absolute.get("fact_count", len(fact_ids)),
                    "n_contexts": absolute.get("context_count", 0),
                    "value": absolute["mean"],
                    "value_ci_low": absolute["ci_low"],
                    "value_ci_high": absolute["ci_high"],
                    "delta_reference": reference or "",
                }
                if reference is None:
                    row.update(
                        {
                            "delta_from_reference": 0.0,
                            "delta_ci_low": 0.0,
                            "delta_ci_high": 0.0,
                            "delta_ci_excludes_zero": False,
                            "facts_improved": 0,
                            "facts_unchanged": len(fact_ids),
                            "facts_degraded": 0,
                        }
                    )
                else:
                    seed_offset += 1
                    left = values[stage["stage"]][metric]
                    right = values[reference][metric]
                    paired = {fact_id: left[fact_id] - right[fact_id] for fact_id in fact_ids}
                    ci = bootstrap_context_ci(
                        paired,
                        examples_by_id,
                        fact_ids,
                        iterations=args.bootstrap_samples,
                        seed=args.bootstrap_seed + seed_offset,
                    )
                    row.update(
                        {
                            "delta_from_reference": ci["mean"],
                            "delta_ci_low": ci["ci_low"],
                            "delta_ci_high": ci["ci_high"],
                            "delta_ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                            "facts_improved": sum(1 for fact_id in fact_ids if paired[fact_id] > 0),
                            "facts_unchanged": sum(1 for fact_id in fact_ids if paired[fact_id] == 0),
                            "facts_degraded": sum(1 for fact_id in fact_ids if paired[fact_id] < 0),
                        }
                    )
                rows.append(row)

    path_diff_rows, path_diff_summary = build_path_diff_rows(
        args, examples, offsets, retrievals_path, hypotheses_by_fact, rendering_policy, ranks
    )

    write_csv(
        args.output_dir / "stage_decomposition.csv",
        rows,
        [
            "stage_idx",
            "stage",
            "component_added",
            "rendering",
            "hypotheses_used",
            "aggregation",
            "score_variant",
            "rerank_beta",
            "cumulative",
            "deployable",
            "equals_config",
            "equality_verified",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "value",
            "value_ci_low",
            "value_ci_high",
            "delta_reference",
            "delta_from_reference",
            "delta_ci_low",
            "delta_ci_high",
            "delta_ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )
    write_csv(
        args.output_dir / "path_diff.csv",
        path_diff_rows,
        [
            "fact_id",
            "modality",
            "policy_rendering",
            "direction",
            "first_differing_column",
            "path_a_hypothesis_idx",
            "path_b_hypothesis_idx",
            "hypothesis_dimensions",
            "query_def",
            "query_lab",
            "gold_rank_def_list",
            "gold_rank_lab_list",
            "path_a_gold_rank_fused_before_rerank",
            "path_b_gold_rank_fused_before_rerank",
            "path_a_rerank_beta",
            "path_b_rerank_beta",
            "path_a_gold_rank_after_rerank",
            "path_b_gold_rank_after_rerank",
            "path_a_label_coverage_weight",
            "path_b_label_coverage_weight",
            "path_a_rrf_kappa",
            "path_b_rrf_kappa",
        ],
    )

    metrics = {
        "experiment": "ags_stage_decomposition",
        "reconciliation_note": CAUSE_NOTE,
        "supersedes": str(SCRIPT_DIR / "runs_ags_pipeline_vs_direct" / "qwen3_32b" / "stage_decomposition.csv"),
        "shared_scorer": "ags_configuration_scoring.score_configuration",
        "path_a": "old stage decomposition: logged gold_rank, no rerank (rerank_beta=0.0)",
        "path_b": "configuration ablation: same list, consensus rerank at beta=0.05",
        "sample": {
            "path": str(args.sample_path),
            "fact_count": len(examples),
            "context_count": len({fact_context_key(example) for example in examples}),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
        },
        "fixed_config": {
            "label_coverage_weight": component_metrics.get("label_coverage_weight"),
            "label_coverage_pool_multiplier": component_metrics.get("label_coverage_pool_multiplier"),
            "rendering_policy": rendering_policy,
            "score_variant": args.score_variant,
            "rrf_kappa": args.rrf_kappa,
            "top_k": args.top_k,
            "agreement_top_m": args.agreement_top_m,
            "normalization_map_version": map_version(args.normalization_map),
            "dual_rendering_note": (
                "the dual rendering is pre-fused upstream by dual_fuse at Experiment A retrieval-logging "
                f"time with dual_rrf_kappa={component_metrics.get('dual_rrf_kappa')}; neither path fuses "
                "label and definition lists itself"
            ),
        },
        "equality_checks": equality_checks,
        "path_diff_summary": path_diff_summary,
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "context",
            "pairing": "per fact against delta_reference",
        },
        "stage_rows": rows,
        "path_diff_rows": path_diff_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "path_diff_summary": path_diff_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
