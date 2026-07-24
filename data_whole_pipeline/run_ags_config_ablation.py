#!/usr/bin/env python3
"""Pipeline configuration ablation: how many hypotheses, and how to aggregate them.

Entirely offline over the Experiment A logs (hypotheses.jsonl, retrievals.jsonl).
No generation, no retrieval, no GPU.

Fixed for every configuration: label_coverage_weight = 1.0, modality-conditional
rendering (dual for table, def for text), consensus symbolic rerank at beta = 0.05.

  A. j1                  J=1, no aggregation, + rerank
  B. j3_select           J=3, symbolic selection of one hypothesis, + rerank
  C. j3_rrf              J=3, RRF fusion over all J, + rerank      [current pipeline]
  D. j3_select_union     J=3, symbolic selection for the head, union of all J as
                         the candidate pool, + rerank
  E. j3_oracle           J=3, oracle best-of-J                     [upper bound]
  F. j2_rrf              J=2, RRF fusion, + rerank

Symbolic selection reuses the rule already established in Experiment A
(j3_symbolic_select): argmax over hypotheses of the mean agreement between a
hypothesis and its own top-M retrieved candidates.

The consensus rerank adds beta * consensus_agreement to an RRF score whose scale
grows with the number of fused lists, so at fixed beta the rerank is far stronger
in a one-list configuration than in a three-list one. Every configuration is
therefore scored twice: `sum_rrf` (as the pipeline implements it today) and
`mean_rrf` (RRF divided by the number of lists, putting every configuration on
one retrieval-score scale so beta means the same thing everywhere).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ags_configuration_scoring import FactContext, score_configuration
from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    load_normalization_map,
    map_version,
)
from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    fact_context_key,
    load_jsonl,
    row_to_example,
)
from run_fintagging_grounding_baseline import Example, SCRIPT_DIR


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_config_ablation" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
DEFAULT_COMPONENT_DIR = SCRIPT_DIR / "runs_ags_component_validation" / "qwen3_32b"
DEFAULT_PIPELINE_DIR = SCRIPT_DIR / "runs_ags_pipeline_vs_direct" / "qwen3_32b"

MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
PRIMARY_METRICS = ("recall_at_10", "mrr")
SCORE_VARIANTS = ("sum_rrf", "mean_rrf")
VARIANT_FLAG = {"sum_rrf": "sum", "mean_rrf": "mean"}

CONFIGS = {
    "A_j1": {
        "label": "J=1, no aggregation, + rerank",
        "hypotheses_used": 1,
        "generation_calls_per_fact": 1,
        "aggregation": "none",
        "aggregation_flag": "none",
        "deployable": True,
    },
    "B_j3_select": {
        "label": "J=3, symbolic selection, + rerank",
        "hypotheses_used": 3,
        "generation_calls_per_fact": 3,
        "aggregation": "symbolic_selection",
        "aggregation_flag": "selection",
        "deployable": True,
    },
    "C_j3_rrf": {
        "label": "J=3, RRF fusion, + rerank (current pipeline)",
        "hypotheses_used": 3,
        "generation_calls_per_fact": 3,
        "aggregation": "rrf_fusion",
        "aggregation_flag": "rrf",
        "deployable": True,
    },
    "D_j3_select_union": {
        "label": "J=3, symbolic selection head + union pool, + rerank",
        "hypotheses_used": 3,
        "generation_calls_per_fact": 3,
        "aggregation": "symbolic_selection_over_union",
        "aggregation_flag": "selection_union",
        "deployable": True,
    },
    "E_j3_oracle": {
        "label": "J=3, oracle best-of-J (upper bound)",
        "hypotheses_used": 3,
        "generation_calls_per_fact": 3,
        "aggregation": "oracle",
        "aggregation_flag": "oracle",
        "deployable": False,
    },
    "F_j2_rrf": {
        "label": "J=2, RRF fusion, + rerank",
        "hypotheses_used": 2,
        "generation_calls_per_fact": 2,
        "aggregation": "rrf_fusion",
        "aggregation_flag": "rrf",
        "deployable": True,
    },
}
CONFIG_ORDER = tuple(CONFIGS)
BASELINE_CONFIG = "A_j1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--consensus-beta", type=float, default=0.05)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--expected-label-coverage-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def index_retrievals(path: Path) -> dict[tuple[str, int, int], int]:
    """Byte offsets of every retrieval record, so facts can be loaded one at a time."""
    offsets: dict[tuple[str, int, int], int] = {}
    with path.open("rb") as handle:
        offset = handle.tell()
        for line in handle:
            if line.strip():
                record = json.loads(line)
                key = (
                    str(record["rendering"]),
                    int(record["fact_id"]),
                    int(record["hypothesis_idx"]),
                )
                offsets[key] = offset
            offset += len(line)
    return offsets


def read_record(handle: Any, offset: int) -> dict[str, Any]:
    handle.seek(offset)
    return json.loads(handle.readline())


def build_fact_rankings(
    args: argparse.Namespace,
    example: Example,
    records: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
) -> tuple[dict[tuple[str, str], int | None], int]:
    """Gold rank per (config, score variant) for one fact, plus the selected hypothesis.

    Every configuration is one call to the shared `score_configuration`, so a config
    row here and a stage row in the decomposition with the same flags are the same
    computation.
    """
    context = FactContext(
        example=example,
        records=records,
        hypotheses=hypotheses,
        normalization_map=normalization_map,
        agreement_top_m=args.agreement_top_m,
    )
    ranks: dict[tuple[str, str], int | None] = {}
    for variant in SCORE_VARIANTS:
        for config in CONFIG_ORDER:
            spec = CONFIGS[config]
            result = score_configuration(
                context,
                hypotheses_used=spec["hypotheses_used"],
                aggregation=spec["aggregation_flag"],
                score_variant=VARIANT_FLAG[variant],
                rerank_beta=0.0 if spec["aggregation_flag"] == "oracle" else args.consensus_beta,
                rrf_kappa=args.rrf_kappa,
                top_k=args.top_k,
            )
            ranks[(config, variant)] = result.gold_rank
    return ranks, context.selected_idx_by_j[len(records)]


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


def paired_stats(
    args: argparse.Namespace,
    examples_by_id: dict[int, Example],
    fact_ids: list[int],
    left: dict[int, float],
    right: dict[int, float],
    seed: int,
) -> dict[str, Any]:
    paired = {fact_id: left[fact_id] - right[fact_id] for fact_id in fact_ids}
    ci = bootstrap_context_ci(paired, examples_by_id, fact_ids, iterations=args.bootstrap_samples, seed=seed)
    return {
        "difference": ci["mean"],
        "diff_ci_low": ci["ci_low"],
        "diff_ci_high": ci["ci_high"],
        "diff_ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
        "facts_improved": sum(1 for fact_id in fact_ids if paired[fact_id] > 0),
        "facts_unchanged": sum(1 for fact_id in fact_ids if paired[fact_id] == 0),
        "facts_degraded": sum(1 for fact_id in fact_ids if paired[fact_id] < 0),
    }


def build_config_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[tuple[str, str], dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for variant in SCORE_VARIANTS:
        for config in CONFIG_ORDER:
            spec = CONFIGS[config]
            for modality in MODALITIES:
                fact_ids = fact_ids_for_modality(examples, modality)
                for metric in METRICS:
                    seed_offset += 1
                    absolute = bootstrap_context_ci(
                        values[(config, variant)][metric],
                        examples_by_id,
                        fact_ids,
                        iterations=args.bootstrap_samples,
                        seed=args.bootstrap_seed + seed_offset,
                    )
                    row = {
                        "config": config,
                        "description": spec["label"],
                        "score_variant": variant,
                        "aggregation": spec["aggregation"],
                        "hypotheses_used": spec["hypotheses_used"],
                        "generation_calls_per_fact": spec["generation_calls_per_fact"],
                        "deployable": spec["deployable"],
                        "modality": modality,
                        "metric": metric,
                        "n_facts": absolute.get("fact_count", len(fact_ids)),
                        "n_contexts": absolute.get("context_count", 0),
                        "value": absolute["mean"],
                        "ci_low": absolute["ci_low"],
                        "ci_high": absolute["ci_high"],
                    }
                    if config == BASELINE_CONFIG:
                        row.update(
                            {
                                "difference": 0.0,
                                "diff_ci_low": 0.0,
                                "diff_ci_high": 0.0,
                                "diff_ci_excludes_zero": False,
                                "facts_improved": 0,
                                "facts_unchanged": len(fact_ids),
                                "facts_degraded": 0,
                                "read": "baseline",
                            }
                        )
                    else:
                        seed_offset += 1
                        row.update(
                            paired_stats(
                                args,
                                examples_by_id,
                                fact_ids,
                                values[(config, variant)][metric],
                                values[(BASELINE_CONFIG, variant)][metric],
                                args.bootstrap_seed + seed_offset,
                            )
                        )
                        row["read"] = (
                            "primary"
                            if (
                                modality == "table"
                                and metric in PRIMARY_METRICS
                                and config in ("C_j3_rrf", "B_j3_select")
                            )
                            else "secondary"
                        )
                    rows.append(row)
    return rows


def build_modality_depth_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[tuple[str, str], dict[str, dict[int, float]]],
    reference_values: dict[str, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    """Text subset: shallow J=1 def-rendering arm against the current pipeline."""
    fact_ids = fact_ids_for_modality(examples, "text")
    rows: list[dict[str, Any]] = []
    seed_offset = 40_000
    arms: dict[str, dict[str, dict[int, float]]] = {}
    for variant in SCORE_VARIANTS:
        arms[f"text_shallow_j1_{variant}"] = values[("A_j1", variant)]
        arms[f"pipeline_j3_rrf_{variant}"] = values[("C_j3_rrf", variant)]
    arms["reference_one_pass_freetext"] = reference_values["one_pass_freetext"]

    baseline_key = "pipeline_j3_rrf_sum_rrf"
    for arm, arm_values in arms.items():
        for metric in METRICS:
            seed_offset += 1
            absolute = bootstrap_context_ci(
                arm_values[metric],
                examples_by_id,
                fact_ids,
                iterations=args.bootstrap_samples,
                seed=args.bootstrap_seed + seed_offset,
            )
            row = {
                "arm": arm,
                "modality": "text",
                "metric": metric,
                "n_facts": absolute.get("fact_count", len(fact_ids)),
                "n_contexts": absolute.get("context_count", 0),
                "value": absolute["mean"],
                "ci_low": absolute["ci_low"],
                "ci_high": absolute["ci_high"],
            }
            if arm == baseline_key:
                row.update(
                    {
                        "difference_vs_pipeline": 0.0,
                        "diff_ci_low": 0.0,
                        "diff_ci_high": 0.0,
                        "diff_ci_excludes_zero": False,
                        "facts_improved": 0,
                        "facts_unchanged": len(fact_ids),
                        "facts_degraded": 0,
                    }
                )
            else:
                seed_offset += 1
                stats = paired_stats(
                    args,
                    examples_by_id,
                    fact_ids,
                    arm_values[metric],
                    arms[baseline_key][metric],
                    args.bootstrap_seed + seed_offset,
                )
                row.update(
                    {
                        "difference_vs_pipeline": stats["difference"],
                        "diff_ci_low": stats["diff_ci_low"],
                        "diff_ci_high": stats["diff_ci_high"],
                        "diff_ci_excludes_zero": stats["diff_ci_excludes_zero"],
                        "facts_improved": stats["facts_improved"],
                        "facts_unchanged": stats["facts_unchanged"],
                        "facts_degraded": stats["facts_degraded"],
                    }
                )
            rows.append(row)
    return rows


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

    component_metrics = json.loads((args.component_dir / "metrics.json").read_text(encoding="utf-8"))
    if component_metrics.get("label_coverage_weight") != args.expected_label_coverage_weight:
        raise RuntimeError(
            f"Experiment A ran at label_coverage_weight={component_metrics.get('label_coverage_weight')}, "
            f"expected {args.expected_label_coverage_weight}"
        )
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

    ranks: dict[tuple[str, str], dict[int, int | None]] = {
        (config, variant): {} for config in CONFIG_ORDER for variant in SCORE_VARIANTS
    }
    selected_hypothesis: dict[int, int] = {}
    with retrievals_path.open("r", encoding="utf-8") as handle:
        for position, example in enumerate(examples, start=1):
            fact_id = example.example_idx
            rendering = rendering_policy[example.input_type]
            records = [
                read_record(handle, offsets[(rendering, fact_id, hypothesis_idx)])
                for hypothesis_idx in range(hypothesis_count)
            ]
            fact_ranks, selected_idx = build_fact_rankings(
                args,
                example,
                records,
                hypotheses_by_fact[fact_id],
                normalization_map,
            )
            selected_hypothesis[fact_id] = selected_idx
            for key, rank in fact_ranks.items():
                ranks[key][fact_id] = rank
            if args.log_every and position % args.log_every == 0:
                print(f"consolidated {position}/{len(examples)} facts", flush=True)

    values = {key: metric_values(series, args.top_k) for key, series in ranks.items()}

    pipeline_ranks = {
        int(record["fact_id"]): record
        for record in load_jsonl(args.pipeline_dir / "per_fact_ranks.jsonl")
    }
    reference_values = {
        "one_pass_freetext": metric_values(
            {
                fact_id: record.get("rank_one_pass_freetext")
                for fact_id, record in pipeline_ranks.items()
            },
            args.top_k,
        )
    }

    # C under sum_rrf reproduces the deployed pipeline, so it is checkable against it.
    pipeline_agreement = sum(
        1
        for fact_id, record in pipeline_ranks.items()
        if record.get("rank_pipeline") == ranks[("C_j3_rrf", "sum_rrf")][fact_id]
    )
    reproduction_check = {
        "compared_against": str(args.pipeline_dir / "per_fact_ranks.jsonl"),
        "facts_matching_deployed_pipeline": pipeline_agreement,
        "fact_count": len(examples),
        "exact": pipeline_agreement == len(examples),
    }
    print(json.dumps({"c_reproduces_pipeline": reproduction_check}, indent=2, sort_keys=True))

    config_rows = build_config_rows(args, examples, examples_by_id, values)
    modality_rows = build_modality_depth_rows(args, examples, examples_by_id, values, reference_values)

    per_fact_rows = [
        {
            "fact_id": example.example_idx,
            "context_key": fact_context_key(example),
            "modality": example.input_type,
            "rendering": rendering_policy[example.input_type],
            "selected_hypothesis_idx": selected_hypothesis[example.example_idx],
            **{
                f"rank_{config}_{variant}": ranks[(config, variant)][example.example_idx]
                for config in CONFIG_ORDER
                for variant in SCORE_VARIANTS
            },
        }
        for example in examples
    ]
    with (args.output_dir / "per_fact_ranks.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_fact_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        args.output_dir / "config_ablation.csv",
        config_rows,
        [
            "config",
            "description",
            "score_variant",
            "aggregation",
            "hypotheses_used",
            "generation_calls_per_fact",
            "deployable",
            "modality",
            "metric",
            "read",
            "n_facts",
            "n_contexts",
            "value",
            "ci_low",
            "ci_high",
            "difference",
            "diff_ci_low",
            "diff_ci_high",
            "diff_ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )
    write_csv(
        args.output_dir / "modality_depth.csv",
        modality_rows,
        [
            "arm",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "value",
            "ci_low",
            "ci_high",
            "difference_vs_pipeline",
            "diff_ci_low",
            "diff_ci_high",
            "diff_ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )

    metrics = {
        "experiment": "ags_config_ablation",
        "sample": {
            "path": str(args.sample_path),
            "fact_count": len(examples),
            "context_count": len({fact_context_key(example) for example in examples}),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
        },
        "fixed_config": {
            "label_coverage_weight": args.expected_label_coverage_weight,
            "label_coverage_pool_multiplier": component_metrics.get("label_coverage_pool_multiplier"),
            "type_filter": True,
            "rendering_policy": rendering_policy,
            "consensus_beta": args.consensus_beta,
            "rrf_kappa": args.rrf_kappa,
            "top_k": args.top_k,
            "agreement_top_m": args.agreement_top_m,
            "normalization_map_version": map_version(args.normalization_map),
            "source_logs": {
                "hypotheses": str(args.component_dir / "hypotheses.jsonl"),
                "retrievals": str(retrievals_path),
            },
        },
        "configs": {config: CONFIGS[config] for config in CONFIG_ORDER},
        "score_variants": {
            "sum_rrf": "RRF scores summed across fused lists, exactly as the pipeline implements it; "
            "the retrieval-score scale grows with the number of lists, so at fixed beta the rerank "
            "is relatively stronger in configurations that fuse fewer lists",
            "mean_rrf": "RRF divided by the number of fused lists, putting every configuration on one "
            "retrieval-score scale so beta carries the same weight everywhere",
        },
        "symbolic_selection_rule": (
            "argmax over hypotheses of mean_agreement(own top-M candidates, own dimensions), "
            f"top_m={args.agreement_top_m}, ties resolved toward the earlier hypothesis; this is the "
            "j3_symbolic_select rule from Experiment A"
        ),
        "selected_hypothesis_distribution": dict(Counter(selected_hypothesis.values())),
        "c_reproduces_pipeline": reproduction_check,
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "context",
            "pairing": f"per fact against {BASELINE_CONFIG}",
        },
        "config_rows": config_rows,
        "modality_depth_rows": modality_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    primary = [row for row in config_rows if row.get("read") == "primary"]
    print(json.dumps({"output_dir": str(args.output_dir), "primary_reads": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
