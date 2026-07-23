#!/usr/bin/env python3
"""Pipeline vs. direct retrieval under label coverage, on the frozen 661-fact sample.

With label_coverage_weight = 1.0 for every arm, compares:
  1. direct_retrieval  raw serialized fact + context as query
  2. one_pass_def      single hypothesis, definition rendering
  3. pipeline          J=3 structured hypotheses, modality-conditional dual
                       rendering, RRF fusion over the J retrievals, consensus
                       symbolic rerank at beta = 0.05

Arm 1 is read from the label-coverage ablation; arms 2 and 3 are offline
re-consolidation of the Experiment A (component validation) hypotheses and
retrievals. No generation and no retrieval is performed here: every arm's
per-fact gold rank comes from artifacts that were already produced with label
coverage enabled, which this script verifies before computing anything.

Also decomposes the pipeline gain into cumulative stages so each component's
contribution is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    fact_context_key,
    load_jsonl,
    row_to_example,
)
from run_fintagging_grounding_baseline import Example, SCRIPT_DIR


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_pipeline_vs_direct" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
DEFAULT_ABLATION_DIR = SCRIPT_DIR / "runs_ags_label_coverage_ablation" / "qwen3_32b"
DEFAULT_COMPONENT_DIR = SCRIPT_DIR / "runs_ags_component_validation" / "qwen3_32b"

MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
PRIMARY_METRICS = ("recall_at_10", "mrr")

ARMS = ("direct_retrieval", "one_pass_def", "pipeline")
CONTRASTS = (
    ("pipeline", "direct_retrieval"),
    ("pipeline", "one_pass_def"),
    ("pipeline", "one_pass_freetext"),
    ("one_pass_def", "direct_retrieval"),
)
STAGES = (
    "one_pass_def",
    "plus_dual_rendering",
    "plus_j3_union_oracle",
    "plus_rrf_fusion",
    "plus_consensus_rerank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--consensus-beta", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--expected-label-coverage-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def rank_metrics(rank: int | None, top_k: int) -> dict[str, float]:
    valid = rank is not None and rank <= top_k
    values = {f"recall_at_{depth}": float(rank is not None and rank <= depth) for depth in DEPTHS}
    values["mrr"] = 1.0 / rank if valid else 0.0
    return values


def as_rank(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_ablation_ranks(path: Path) -> dict[str, dict[int, int | None]]:
    """Direct-retrieval and free-text one-pass ranks from the label-coverage run."""
    direct: dict[int, int | None] = {}
    freetext: dict[int, int | None] = {}
    for record in load_jsonl(path):
        fact_id = int(record["fact_id"])
        direct[fact_id] = as_rank(record.get("rank_direct_retrieval_on"))
        freetext[fact_id] = as_rank(record.get("rank_one_pass_grounding_on"))
    return {"direct_retrieval": direct, "one_pass_freetext": freetext}


def load_a1_ranks(path: Path) -> tuple[dict[tuple[str, int, int], int | None], dict[str, Any]]:
    """Per (rendering, fact_id, hypothesis_idx) gold rank from the A1 retrievals."""
    ranks: dict[tuple[str, int, int], int | None] = {}
    renderings: Counter[str] = Counter()
    hypothesis_indices: Counter[int] = Counter()
    coverage_fields_present = 0
    coverage_weight_observed: set[float] = set()
    record_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_count += 1
            rendering = str(record["rendering"])
            fact_id = int(record["fact_id"])
            hypothesis_idx = int(record["hypothesis_idx"])
            renderings[rendering] += 1
            hypothesis_indices[hypothesis_idx] += 1
            ranks[(rendering, fact_id, hypothesis_idx)] = as_rank(record.get("gold_rank"))

            top = (record.get("candidates") or [None])[0]
            if isinstance(top, dict) and "label_coverage" in top and "retrieval_score" in top:
                coverage_fields_present += 1
                normalized = float(top.get("bm25_normalized_score", 0.0))
                coverage = float(top.get("label_coverage", 0.0))
                query_coverage = float(top.get("query_label_coverage", 0.0))
                score = float(top.get("retrieval_score", 0.0))
                denominator = coverage + query_coverage
                if denominator > 0:
                    coverage_weight_observed.add(round((score - normalized) / denominator, 6))

    provenance = {
        "path": str(path),
        "record_count": record_count,
        "renderings": dict(renderings),
        "hypotheses_per_fact": dict(hypothesis_indices),
        "records_carrying_label_coverage_fields": coverage_fields_present,
        "implied_label_coverage_weight": sorted(coverage_weight_observed),
    }
    return ranks, provenance


def load_consolidation_ranks(path: Path, variants: set[str]) -> tuple[dict[str, dict[int, int | None]], dict[int, str]]:
    ranks: dict[str, dict[int, int | None]] = {variant: {} for variant in variants}
    rendering_by_fact: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            variant = str(record["variant"])
            if variant not in variants:
                continue
            fact_id = int(record["fact_id"])
            ranks[variant][fact_id] = as_rank(record.get("gold_rank"))
            rendering_by_fact[fact_id] = str(record["rendering"])
    return ranks, rendering_by_fact


def check_label_coverage(
    component_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    a1_provenance: dict[str, Any],
    expected_weight: float,
) -> dict[str, Any]:
    component_weight = component_metrics.get("label_coverage_weight")
    component_multiplier = component_metrics.get("label_coverage_pool_multiplier")
    ablation_config = ablation_metrics.get("config", {})
    ablation_weight = ablation_config.get("label_coverage_weight_on")
    ablation_multiplier = ablation_config.get("label_coverage_pool_multiplier")

    failures = []
    if component_weight != expected_weight:
        failures.append(
            f"component validation ran at label_coverage_weight={component_weight}, expected {expected_weight}; "
            "A1's logged queries must be re-scored with label coverage enabled before this comparison is valid"
        )
    if ablation_weight != expected_weight:
        failures.append(f"ablation on-arm ran at label_coverage_weight={ablation_weight}, expected {expected_weight}")
    if component_multiplier != ablation_multiplier:
        failures.append(
            f"pool multiplier differs: component={component_multiplier} ablation={ablation_multiplier}"
        )
    implied = a1_provenance["implied_label_coverage_weight"]
    if implied and any(abs(value - expected_weight) > 1e-6 for value in implied):
        failures.append(f"A1 candidate scores imply label_coverage_weight in {implied}, expected {expected_weight}")
    if not a1_provenance["records_carrying_label_coverage_fields"]:
        failures.append("A1 candidates carry no label-coverage fields, so coverage was off when they were retrieved")

    return {
        "passed": not failures,
        "failures": failures,
        "component_label_coverage_weight": component_weight,
        "component_label_coverage_pool_multiplier": component_multiplier,
        "ablation_label_coverage_weight": ablation_weight,
        "ablation_label_coverage_pool_multiplier": ablation_multiplier,
        "a1_implied_label_coverage_weight": implied,
    }


def build_arm_ranks(
    args: argparse.Namespace,
    examples: list[Example],
    ablation_ranks: dict[str, dict[int, int | None]],
    a1_ranks: dict[tuple[str, int, int], int | None],
    consolidation_ranks: dict[str, dict[int, int | None]],
    rendering_policy: dict[str, str],
    hypothesis_count: int,
) -> dict[str, dict[int, int | None]]:
    consensus_variant = f"rrf_consensus_beta_{args.consensus_beta:g}"
    ranks: dict[str, dict[int, int | None]] = {
        "direct_retrieval": ablation_ranks["direct_retrieval"],
        "one_pass_freetext": ablation_ranks["one_pass_freetext"],
        "one_pass_def": {},
        "plus_dual_rendering": {},
        "plus_j3_union_oracle": {},
        "plus_rrf_fusion": consolidation_ranks["plain_rrf"],
        "plus_consensus_rerank": consolidation_ranks[consensus_variant],
    }
    for example in examples:
        fact_id = example.example_idx
        rendering = rendering_policy[example.input_type]
        ranks["one_pass_def"][fact_id] = a1_ranks[("def", fact_id, 0)]
        ranks["plus_dual_rendering"][fact_id] = a1_ranks[(rendering, fact_id, 0)]
        hypothesis_ranks = [
            a1_ranks[(rendering, fact_id, hypothesis_idx)]
            for hypothesis_idx in range(hypothesis_count)
        ]
        observed = [rank for rank in hypothesis_ranks if rank is not None]
        ranks["plus_j3_union_oracle"][fact_id] = min(observed) if observed else None
    ranks["pipeline"] = ranks["plus_consensus_rerank"]
    return ranks


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


def paired_row(
    args: argparse.Namespace,
    examples_by_id: dict[int, Example],
    fact_ids: list[int],
    left: dict[int, float],
    right: dict[int, float],
    seed: int,
) -> dict[str, Any]:
    paired = {fact_id: left[fact_id] - right[fact_id] for fact_id in fact_ids}
    ci = bootstrap_context_ci(
        paired,
        examples_by_id,
        fact_ids,
        iterations=args.bootstrap_samples,
        seed=seed,
    )
    return {
        "n_facts": len(fact_ids),
        "n_contexts": ci.get("context_count", 0),
        "value_left": round(mean([left[fact_id] for fact_id in fact_ids]), 6),
        "value_right": round(mean([right[fact_id] for fact_id in fact_ids]), 6),
        "difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
        "facts_improved": sum(1 for fact_id in fact_ids if paired[fact_id] > 0),
        "facts_unchanged": sum(1 for fact_id in fact_ids if paired[fact_id] == 0),
        "facts_degraded": sum(1 for fact_id in fact_ids if paired[fact_id] < 0),
    }


def build_contrast_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[str, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for left_arm, right_arm in CONTRASTS:
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                row = paired_row(
                    args,
                    examples_by_id,
                    fact_ids,
                    values[left_arm][metric],
                    values[right_arm][metric],
                    args.bootstrap_seed + seed_offset,
                )
                rows.append(
                    {
                        "contrast": f"{left_arm}_minus_{right_arm}",
                        "arm_left": left_arm,
                        "arm_right": right_arm,
                        "modality": modality,
                        "metric": metric,
                        "read": (
                            "primary"
                            if (
                                modality == "table"
                                and metric in PRIMARY_METRICS
                                and left_arm == "pipeline"
                                and right_arm in ("direct_retrieval", "one_pass_def")
                            )
                            else "secondary"
                        ),
                        **row,
                    }
                )
    return rows


def build_arm_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[str, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 5_000
    for arm in (*ARMS, "one_pass_freetext"):
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                ci = bootstrap_context_ci(
                    values[arm][metric],
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + seed_offset,
                )
                rows.append(
                    {
                        "arm": arm,
                        "modality": modality,
                        "metric": metric,
                        "n_facts": ci.get("fact_count", len(fact_ids)),
                        "n_contexts": ci.get("context_count", 0),
                        "value": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                    }
                )
    return rows


def build_stage_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[str, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 10_000
    for stage_idx, stage in enumerate(STAGES):
        previous = STAGES[stage_idx - 1] if stage_idx else None
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                absolute = bootstrap_context_ci(
                    values[stage][metric],
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + seed_offset,
                )
                row = {
                    "stage_idx": stage_idx + 1,
                    "stage": stage,
                    "component_added": {
                        "one_pass_def": "baseline: J=1, definition rendering",
                        "plus_dual_rendering": "modality-conditional rendering (dual for table, def for text)",
                        "plus_j3_union_oracle": "J=3 hypotheses, oracle union over their rankings",
                        "plus_rrf_fusion": "RRF fusion of the J retrievals into one ranking (unweighted)",
                        "plus_consensus_rerank": f"consensus symbolic rerank at beta={args.consensus_beta:g}",
                    }[stage],
                    "deployable": stage != "plus_j3_union_oracle",
                    "modality": modality,
                    "metric": metric,
                    "n_facts": absolute.get("fact_count", len(fact_ids)),
                    "n_contexts": absolute.get("context_count", 0),
                    "value": absolute["mean"],
                    "value_ci_low": absolute["ci_low"],
                    "value_ci_high": absolute["ci_high"],
                }
                if previous is None:
                    row.update(
                        {
                            "delta_from_previous": 0.0,
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
                    delta = paired_row(
                        args,
                        examples_by_id,
                        fact_ids,
                        values[stage][metric],
                        values[previous][metric],
                        args.bootstrap_seed + seed_offset,
                    )
                    row.update(
                        {
                            "delta_from_previous": delta["difference"],
                            "delta_ci_low": delta["ci_low"],
                            "delta_ci_high": delta["ci_high"],
                            "delta_ci_excludes_zero": delta["ci_excludes_zero"],
                            "facts_improved": delta["facts_improved"],
                            "facts_unchanged": delta["facts_unchanged"],
                            "facts_degraded": delta["facts_degraded"],
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
    ablation_metrics = json.loads((args.ablation_dir / "metrics.json").read_text(encoding="utf-8"))

    a1_ranks, a1_provenance = load_a1_ranks(args.component_dir / "retrievals.jsonl")
    coverage_check = check_label_coverage(
        component_metrics,
        ablation_metrics,
        a1_provenance,
        args.expected_label_coverage_weight,
    )
    print(json.dumps({"label_coverage_check": coverage_check}, ensure_ascii=False, indent=2, sort_keys=True))
    if not coverage_check["passed"]:
        raise RuntimeError(
            "Aborting: arms were not all retrieved under label coverage. "
            f"Failures: {coverage_check['failures']}"
        )

    consensus_variant = f"rrf_consensus_beta_{args.consensus_beta:g}"
    consolidation_ranks, rendering_by_fact = load_consolidation_ranks(
        args.component_dir / "consolidation_rankings.jsonl",
        {"plain_rrf", consensus_variant},
    )
    rendering_policy = component_metrics["rendering_gate"]["adopted_rendering_by_modality"]
    rendering_policy = {
        modality: rendering
        for modality, rendering in rendering_policy.items()
        if modality in ("table", "text")
    }
    policy_mismatches = [
        fact_id
        for fact_id, rendering in rendering_by_fact.items()
        if rendering != rendering_policy[examples_by_id[fact_id].input_type]
    ]
    if policy_mismatches:
        raise ValueError(
            f"{len(policy_mismatches)} consolidated facts used a rendering other than the adopted policy "
            f"{rendering_policy}: {policy_mismatches[:10]}"
        )

    hypothesis_count = int(component_metrics["hypotheses_per_fact"])
    ablation_ranks = load_ablation_ranks(args.ablation_dir / "per_fact_ranks.jsonl")
    ranks = build_arm_ranks(
        args,
        examples,
        ablation_ranks,
        a1_ranks,
        consolidation_ranks,
        rendering_policy,
        hypothesis_count,
    )

    missing = {
        name: sorted(set(examples_by_id) - set(series))[:10]
        for name, series in ranks.items()
        if set(series) != set(examples_by_id)
    }
    if missing:
        raise ValueError(f"Arms are missing facts from the frozen sample: {missing}")

    values = {name: metric_values(series, args.top_k) for name, series in ranks.items()}

    per_fact_rows = [
        {
            "fact_id": example.example_idx,
            "context_key": fact_context_key(example),
            "modality": example.input_type,
            "rendering": rendering_policy[example.input_type],
            **{f"rank_{name}": ranks[name][example.example_idx] for name in ranks},
        }
        for example in examples
    ]
    with (args.output_dir / "per_fact_ranks.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_fact_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    contrast_rows = build_contrast_rows(args, examples, examples_by_id, values)
    arm_rows = build_arm_rows(args, examples, examples_by_id, values)
    stage_rows = build_stage_rows(args, examples, examples_by_id, values)

    write_csv(
        args.output_dir / "pipeline_vs_direct.csv",
        contrast_rows,
        [
            "contrast",
            "arm_left",
            "arm_right",
            "modality",
            "metric",
            "read",
            "n_facts",
            "n_contexts",
            "value_left",
            "value_right",
            "difference",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )
    write_csv(
        args.output_dir / "arm_metrics.csv",
        arm_rows,
        ["arm", "modality", "metric", "n_facts", "n_contexts", "value", "ci_low", "ci_high"],
    )
    write_csv(
        args.output_dir / "stage_decomposition.csv",
        stage_rows,
        [
            "stage_idx",
            "stage",
            "component_added",
            "deployable",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "value",
            "value_ci_low",
            "value_ci_high",
            "delta_from_previous",
            "delta_ci_low",
            "delta_ci_high",
            "delta_ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )

    retriever_config = {
        "index": "BM25 over us_gaap_2024_enriched_retrieval.jsonl, type-filtered",
        "label_coverage_weight": args.expected_label_coverage_weight,
        "label_coverage_pool_multiplier": component_metrics.get("label_coverage_pool_multiplier"),
        "type_filter": True,
        "top_k": args.top_k,
        "rrf_kappa": component_metrics.get("rrf_kappa"),
        "dual_rrf_kappa": component_metrics.get("dual_rrf_kappa"),
    }
    metrics = {
        "experiment": "ags_pipeline_vs_direct",
        "sample": {
            "path": str(args.sample_path),
            "fact_count": len(examples),
            "context_count": len({fact_context_key(example) for example in examples}),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
        },
        "retriever_config_by_arm": {
            "direct_retrieval": {
                **retriever_config,
                "source": str(args.ablation_dir / "per_fact_ranks.jsonl"),
                "query": "build_direct_query: entity + datatype + visible context text",
                "generation": "none",
            },
            "one_pass_def": {
                **retriever_config,
                "source": str(args.component_dir / "retrievals.jsonl"),
                "query": "structured hypothesis 0, definition rendering",
                "generation": "Experiment A hypotheses.jsonl (Qwen3-32B, vllm, temperature 0.8, J=3)",
            },
            "one_pass_freetext": {
                **retriever_config,
                "source": str(args.ablation_dir / "per_fact_ranks.jsonl"),
                "query": "free-text one-pass query description",
                "generation": "coverage-pilot Arm A (Qwen3-32B, vllm, temperature 0.0, J=1)",
            },
            "pipeline": {
                **retriever_config,
                "source": str(args.component_dir / "consolidation_rankings.jsonl"),
                "query": f"J={hypothesis_count} structured hypotheses, rendering policy {rendering_policy}",
                "consolidation": f"RRF fusion (unweighted, kappa={component_metrics.get('rrf_kappa')}) "
                f"then consensus symbolic rerank at beta={args.consensus_beta:g}",
                "generation": "Experiment A hypotheses.jsonl (Qwen3-32B, vllm, temperature 0.8, J=3)",
            },
        },
        "rendering_policy": rendering_policy,
        "consensus_beta": args.consensus_beta,
        "hypotheses_per_fact": hypothesis_count,
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "context",
            "pairing": "per fact",
        },
        "label_coverage_check": coverage_check,
        "a1_provenance": a1_provenance,
        "rendering_gate_status": component_metrics["rendering_gate"]["status"],
        "rendering_gate_claim_allowed": component_metrics["rendering_gate"]["claim_allowed"],
        "arm_rows": arm_rows,
        "contrast_rows": contrast_rows,
        "stage_rows": stage_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    primary = [row for row in contrast_rows if row["read"] == "primary"]
    print(json.dumps({"output_dir": str(args.output_dir), "primary_reads": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
