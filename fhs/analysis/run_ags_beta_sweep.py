#!/usr/bin/env python3
"""Beta sweep under range-normalized rerank scoring, at J = 1, 2, 3.

Offline over the Experiment A logs. No generation, no retrieval, no GPU.

The deployed rerank adds beta * agree(c) to an unnormalized RRF score whose scale
grows with the number of fused lists, so beta's effective strength depends on J:
a one-list RRF score spans about 0.013 while beta * agree spans up to 0.023, and a
three-list score spans about 0.049. This sweep re-scores with

    score = minmax(S_wRRF) + beta * agree(c)

which is rank-equivalent to S_wRRF + beta * s_range * agree(c), s_range being the
observed range of S_wRRF over that fact's candidate pool. Both scalings are swept
so the change is visible, at beta in {0, 0.05, 0.1, 0.2, 0.4} and J in {1, 2, 3}.

Note that range normalization also collapses the sum/mean RRF distinction: dividing
every score by the number of lists is a positive rescaling, which min-max removes.
The run asserts this rather than assuming it.
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

from ags_configuration_scoring import FactContext, score_configuration
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map, map_version
from run_ags_config_ablation import index_retrievals, read_record
from run_ags_coverage_pilot import fact_context_key, load_jsonl, row_to_example
from run_fintagging_grounding_baseline import Example, SCRIPT_DIR


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_beta_sweep" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"
DEFAULT_COMPONENT_DIR = FHS_ROOT / "runs" / "runs_ags_component_validation" / "qwen3_32b"

MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
BETAS = (0.0, 0.05, 0.1, 0.2, 0.4)
J_VALUES = (1, 2, 3)
SCALINGS = ("raw", "range_normalized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--score-variant", choices=("sum", "mean"), default="sum")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def aggregation_for(j: int) -> str:
    return "none" if j == 1 else "rrf"


def config_key(j: int, beta: float, scaling: str) -> str:
    # At beta=0 the rerank never runs, so both scalings are the same computation.
    return f"J{j}_beta{beta:g}_{'shared' if beta == 0.0 else scaling}"


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


class ContextBootstrap:
    """Context-level bootstrap that pre-aggregates per context, so each draw is O(contexts)."""

    def __init__(self, examples: list[Example], iterations: int, seed: int) -> None:
        self.iterations = iterations
        self.by_modality: dict[str, list[int]] = {
            modality: [
                example.example_idx
                for example in examples
                if modality == "pooled" or example.input_type == modality
            ]
            for modality in MODALITIES
        }
        self.contexts: dict[str, list[list[int]]] = {}
        self.draws: dict[str, list[list[int]]] = {}
        for modality, fact_ids in self.by_modality.items():
            grouped: dict[str, list[int]] = defaultdict(list)
            index = {example.example_idx: example for example in examples}
            for fact_id in fact_ids:
                grouped[fact_context_key(index[fact_id])].append(fact_id)
            keys = sorted(grouped)
            self.contexts[modality] = [grouped[key] for key in keys]
            rng = random.Random(seed + len(modality))
            self.draws[modality] = [
                [rng.randrange(len(keys)) for _ in keys] for _ in range(iterations)
            ]

    def interval(self, values: dict[int, float], modality: str) -> tuple[float, float, float]:
        groups = self.contexts[modality]
        totals = [sum(values[fact_id] for fact_id in group) for group in groups]
        counts = [len(group) for group in groups]
        observed = sum(totals) / sum(counts) if counts else 0.0
        samples: list[float] = []
        for draw in self.draws[modality]:
            total = 0.0
            count = 0
            for index in draw:
                total += totals[index]
                count += counts[index]
            samples.append(total / count if count else 0.0)
        samples.sort()

        def percentile(q: float) -> float:
            pos = (len(samples) - 1) * q
            lower = int(pos)
            upper = min(lower + 1, len(samples) - 1)
            weight = pos - lower
            return samples[lower] * (1.0 - weight) + samples[upper] * weight

        return observed, percentile(0.025), percentile(0.975)

    def paired(
        self,
        left: dict[int, float],
        right: dict[int, float],
        modality: str,
    ) -> tuple[float, float, float]:
        difference = {fact_id: left[fact_id] - right[fact_id] for fact_id in self.by_modality[modality]}
        return self.interval(difference, modality)


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

    configs: list[tuple[str, int, float, str]] = []
    for j in J_VALUES:
        for beta in BETAS:
            for scaling in SCALINGS:
                key = config_key(j, beta, scaling)
                if any(existing[0] == key for existing in configs):
                    continue
                configs.append((key, j, beta, scaling))

    ranks: dict[str, dict[int, int | None]] = {key: {} for key, _, _, _ in configs}
    variant_equivalence = {"checked": 0, "mismatched": 0}

    with retrievals_path.open("r", encoding="utf-8") as handle:
        for position, example in enumerate(examples, start=1):
            fact_id = example.example_idx
            rendering = rendering_policy[example.input_type]
            records = [
                read_record(handle, offsets[(rendering, fact_id, hypothesis_idx)])
                for hypothesis_idx in range(hypothesis_count)
            ]
            context = FactContext(
                example=example,
                records=records,
                hypotheses=hypotheses_by_fact[fact_id],
                normalization_map=normalization_map,
                agreement_top_m=args.agreement_top_m,
            )
            for key, j, beta, scaling in configs:
                result = score_configuration(
                    context,
                    hypotheses_used=j,
                    aggregation=aggregation_for(j),
                    score_variant=args.score_variant,
                    rerank_beta=beta,
                    rrf_kappa=args.rrf_kappa,
                    top_k=args.top_k,
                    score_scaling=scaling,
                )
                ranks[key][fact_id] = result.gold_rank

            # Range normalization should make sum and mean RRF identical; verify, don't assume.
            for j in (2, 3):
                for beta in (0.05, 0.4):
                    alternate = score_configuration(
                        context,
                        hypotheses_used=j,
                        aggregation="rrf",
                        score_variant="mean",
                        rerank_beta=beta,
                        rrf_kappa=args.rrf_kappa,
                        top_k=args.top_k,
                        score_scaling="range_normalized",
                    )
                    variant_equivalence["checked"] += 1
                    if alternate.gold_rank != ranks[config_key(j, beta, "range_normalized")][fact_id]:
                        variant_equivalence["mismatched"] += 1
            if args.log_every and position % args.log_every == 0:
                print(f"scored {position}/{len(examples)} facts", flush=True)

    values = {key: metric_values(series, args.top_k) for key, series in ranks.items()}
    bootstrap = ContextBootstrap(examples, args.bootstrap_samples, args.bootstrap_seed)

    rows: list[dict[str, Any]] = []
    for key, j, beta, scaling in configs:
        no_rerank = config_key(j, 0.0, scaling)
        same_beta_j1 = config_key(1, beta, scaling)
        for modality in MODALITIES:
            for metric in METRICS:
                observed, low, high = bootstrap.interval(values[key][metric], modality)
                row = {
                    "config": key,
                    "hypotheses_used": j,
                    "rerank_beta": beta,
                    "score_scaling": scaling if beta > 0 else "n/a (rerank off)",
                    "score_variant": args.score_variant,
                    "modality": modality,
                    "metric": metric,
                    "n_facts": len(bootstrap.by_modality[modality]),
                    "n_contexts": len(bootstrap.contexts[modality]),
                    "value": round(observed, 6),
                    "ci_low": round(low, 6),
                    "ci_high": round(high, 6),
                }
                delta, delta_low, delta_high = bootstrap.paired(
                    values[key][metric], values[no_rerank][metric], modality
                )
                row.update(
                    {
                        "delta_vs_beta0": round(delta, 6),
                        "delta_vs_beta0_ci_low": round(delta_low, 6),
                        "delta_vs_beta0_ci_high": round(delta_high, 6),
                        "delta_vs_beta0_excludes_zero": bool(delta_low > 0.0 or delta_high < 0.0),
                    }
                )
                delta, delta_low, delta_high = bootstrap.paired(
                    values[key][metric], values[same_beta_j1][metric], modality
                )
                row.update(
                    {
                        "delta_vs_j1_same_beta": round(delta, 6),
                        "delta_vs_j1_ci_low": round(delta_low, 6),
                        "delta_vs_j1_ci_high": round(delta_high, 6),
                        "delta_vs_j1_excludes_zero": bool(delta_low > 0.0 or delta_high < 0.0),
                    }
                )
                rows.append(row)

    per_fact = [
        {
            "fact_id": example.example_idx,
            "context_key": fact_context_key(example),
            "modality": example.input_type,
            **{f"rank_{key}": ranks[key][example.example_idx] for key, _, _, _ in configs},
        }
        for example in examples
    ]
    with (args.output_dir / "per_fact_ranks.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_fact:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        args.output_dir / "beta_sweep.csv",
        rows,
        [
            "config",
            "hypotheses_used",
            "rerank_beta",
            "score_scaling",
            "score_variant",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "value",
            "ci_low",
            "ci_high",
            "delta_vs_beta0",
            "delta_vs_beta0_ci_low",
            "delta_vs_beta0_ci_high",
            "delta_vs_beta0_excludes_zero",
            "delta_vs_j1_same_beta",
            "delta_vs_j1_ci_low",
            "delta_vs_j1_ci_high",
            "delta_vs_j1_excludes_zero",
        ],
    )

    def best_for(scaling: str, modality: str, metric: str) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if row["modality"] == modality
            and row["metric"] == metric
            and (row["score_scaling"] == scaling or row["rerank_beta"] == 0.0)
        ]
        return max(candidates, key=lambda row: (row["value"], -row["rerank_beta"]))

    metrics = {
        "experiment": "ags_beta_sweep",
        "change": (
            "score = minmax(S_wRRF over the fact's candidate pool) + beta * agree(c), rank-equivalent "
            "to S_wRRF + beta * s_range * agree(c); swept against the deployed raw scaling"
        ),
        "sample": {
            "path": str(args.sample_path),
            "fact_count": len(examples),
            "context_count": len({fact_context_key(example) for example in examples}),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
        },
        "grid": {"betas": list(BETAS), "hypotheses": list(J_VALUES), "scalings": list(SCALINGS)},
        "fixed_config": {
            "rendering_policy": rendering_policy,
            "score_variant": args.score_variant,
            "rrf_kappa": args.rrf_kappa,
            "top_k": args.top_k,
            "agreement_top_m": args.agreement_top_m,
            "label_coverage_weight": component_metrics.get("label_coverage_weight"),
            "normalization_map_version": map_version(args.normalization_map),
        },
        "sum_mean_equivalence_under_range_normalization": {
            **variant_equivalence,
            "equivalent": variant_equivalence["mismatched"] == 0,
        },
        "best_by_scaling": {
            scaling: {
                f"{modality}_{metric}": best_for(scaling, modality, metric)
                for modality in ("table", "text")
                for metric in ("recall_at_10", "mrr")
            }
            for scaling in SCALINGS
        },
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "context",
            "pairing": "per fact",
        },
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sum_mean_equivalence": metrics["sum_mean_equivalence_under_range_normalization"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
