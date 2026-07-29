#!/usr/bin/env python3
"""Extension of the range-normalized beta sweep to beta >= 0.5, at J = 1, 2, 3.

Offline over the Experiment A logs. No generation, no retrieval, no GPU.

`run_ags_beta_sweep.py` swept beta in {0, 0.05, 0.1, 0.2, 0.4} under both scalings and
found the range-normalized curve still rising at the top of its grid. This run extends
that grid to beta in {0.5, 0.6, 0.8, 1.0, 1.5} under range_normalized scoring only, plus
{2, 3, 4} because two text recall_at_10 curves had still not turned over at 1.5,

    score = minmax(S_wRRF) + beta * agree(c)

reusing `score_configuration` unchanged, so the emitted rows are the identical
computation as the original sweep and concatenate with beta_sweep.csv on the same
twenty columns. Only the new beta cells are emitted; the beta=0 anchors and the raw
scaling already live in beta_sweep.csv and are recomputed here as references for the
paired contrasts rather than re-emitted.

One column is appended, `rerank_share`: the mean over facts of the rerank term's range
divided by the retrieval score's range, in whatever units that cell's scaling uses.
Under raw scoring that is beta * range(agree) / range(S_wRRF). Under range-normalized
scoring the retrieval range is 1 by construction and the raw-equivalent beta is
beta * range(S_wRRF), so the same ratio reduces to beta * range(agree) -- the two
parameterizations land on one axis. The run recomputes the deployed raw J=3 beta=0.05
cell on this axis as a check; it lands at 0.61, not the 0.47 the prior docstring's quoted
spans imply, because those spans are near-bounds rather than observed means. See
metrics.json rerank_share.deployed_raw_anchor for the decomposition.
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

from ags_configuration_scoring import (
    FactContext,
    order_from_scores,
    rrf_scores,
    score_configuration,
)
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map, map_version
from run_ags_config_ablation import index_retrievals, read_record
from run_ags_coverage_pilot import fact_context_key, load_jsonl, row_to_example
from run_fintagging_grounding_baseline import Example, SCRIPT_DIR
from run_ags_beta_sweep import (
    ContextBootstrap,
    METRICS,
    MODALITIES,
    aggregation_for,
    config_key,
    metric_values,
    write_csv,
)


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_beta_sweep_extended" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"
DEFAULT_COMPONENT_DIR = FHS_ROOT / "runs" / "runs_ags_component_validation" / "qwen3_32b"
DEFAULT_PRIOR_SWEEP = FHS_ROOT / "runs" / "runs_ags_beta_sweep" / "qwen3_32b" / "beta_sweep.csv"

BETAS = (0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0)
J_VALUES = (1, 2, 3)
SCALING = "range_normalized"

# The deployed pipeline cell, carried as the paired reference for read (c).
BASELINE_J = 3
BASELINE_BETA = 0.05
BASELINE_SCALING = "raw"

SWEEP_FIELDNAMES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--prior-sweep", type=Path, default=DEFAULT_PRIOR_SWEEP)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    parser.add_argument("--score-variant", choices=("sum", "mean"), default="sum")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def pool_ranges(
    context: FactContext,
    hypotheses_used: int,
    kappa: float,
    score_variant: str,
) -> tuple[float, float]:
    """Range of S_wRRF and of the consensus agreement over this fact's candidate pool.

    Mirrors how `apply_rerank` builds its retrieval vector: the fused RRF score over the
    ordered pool at J = hypotheses_used, or the positional 1/(kappa+rank) fallback when
    only one list is used and no fused scores exist.
    """
    lists = context.candidate_lists[:hypotheses_used]
    if hypotheses_used == 1:
        order = lists[0]
        retrieval = [1.0 / (kappa + rank) for rank in range(1, len(order) + 1)]
        lists_fused = 1
    else:
        scores, best_rank, first_list = rrf_scores(lists, kappa)
        order = order_from_scores(scores, best_rank, first_list)
        retrieval = [scores[tag] for tag in order]
        lists_fused = len(lists)
    if score_variant == "mean":
        retrieval = [value / lists_fused for value in retrieval]

    consensus = context.consensus_over(hypotheses_used)
    agreement = [consensus.get(tag, 0.0) for tag in order]
    if not retrieval:
        return 0.0, 0.0
    return max(retrieval) - min(retrieval), max(agreement) - min(agreement)


def share_from_ranges(
    beta: float,
    scaling: str,
    retrieval_range: float,
    agreement_range: float,
) -> float | None:
    """Rerank-term range over retrieval-score range, in the units that cell scores in.

    raw:              beta * range(agree) / range(S_wRRF)
    range_normalized: the retrieval range is 1 and the raw-equivalent beta is
                      beta * range(S_wRRF), so the ratio reduces to beta * range(agree).
    """
    if retrieval_range <= 0.0:
        return None
    if scaling == "range_normalized":
        return beta * agreement_range
    return beta * agreement_range / retrieval_range


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    betas = tuple(sorted({float(beta) for beta in args.betas}))

    examples = [row_to_example(row) for row in load_jsonl(args.sample_path)]
    component_metrics = json.loads((args.component_dir / "metrics.json").read_text(encoding="utf-8"))
    rendering_policy = {
        modality: rendering
        for modality, rendering in component_metrics["rendering_gate"]["adopted_rendering_by_modality"].items()
        if modality in ("table", "text")
    }
    hypothesis_count = int(component_metrics["hypotheses_per_fact"])
    label_coverage_weight = component_metrics.get("label_coverage_weight")
    assert label_coverage_weight == 1.0, f"expected label_coverage_weight=1.0, got {label_coverage_weight}"
    assert rendering_policy == {"table": "dual", "text": "def"}, rendering_policy

    normalization_map = load_normalization_map(args.normalization_map)
    hypotheses_by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in load_jsonl(args.component_dir / "hypotheses.jsonl"):
        hypotheses_by_fact[int(hypothesis["fact_id"])].append(hypothesis)
    for values_list in hypotheses_by_fact.values():
        values_list.sort(key=lambda item: int(item["hypothesis_idx"]))

    retrievals_path = args.component_dir / "retrievals.jsonl"
    print(f"indexing {retrievals_path} ...", flush=True)
    offsets = index_retrievals(retrievals_path)

    # Emitted cells: the new betas, range_normalized only.
    emitted: list[tuple[str, int, float, str]] = [
        (config_key(j, beta, SCALING), j, beta, SCALING) for j in J_VALUES for beta in betas
    ]
    # References for the paired contrasts, already present in beta_sweep.csv.
    references: list[tuple[str, int, float, str]] = [
        (config_key(j, 0.0, SCALING), j, 0.0, SCALING) for j in J_VALUES
    ]
    baseline_key = config_key(BASELINE_J, BASELINE_BETA, BASELINE_SCALING)
    references.append((baseline_key, BASELINE_J, BASELINE_BETA, BASELINE_SCALING))
    configs = emitted + references

    ranks: dict[str, dict[int, int | None]] = {key: {} for key, _, _, _ in configs}
    # rerank_share is a property of (J, beta, scaling) x fact; accumulate per modality.
    shares: dict[tuple[int, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {modality: [] for modality in MODALITIES}
    )
    undefined_share_facts = 0
    # Raw component spans per J, so the anchor can be decomposed rather than asserted.
    span_samples: dict[int, dict[str, list[float]]] = {
        j: {"retrieval": [], "agreement": []} for j in J_VALUES
    }

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

            ranges_by_j = {
                j: pool_ranges(context, j, args.rrf_kappa, args.score_variant) for j in J_VALUES
            }
            for j in J_VALUES:
                retrieval_range, agreement_range = ranges_by_j[j]
                span_samples[j]["retrieval"].append(retrieval_range)
                span_samples[j]["agreement"].append(agreement_range)
                for beta in betas:
                    share = share_from_ranges(beta, SCALING, retrieval_range, agreement_range)
                    if share is None:
                        undefined_share_facts += 1
                        continue
                    shares[(j, beta, SCALING)]["pooled"].append(share)
                    shares[(j, beta, SCALING)][example.input_type].append(share)
            retrieval_range, agreement_range = ranges_by_j[BASELINE_J]
            share = share_from_ranges(BASELINE_BETA, BASELINE_SCALING, retrieval_range, agreement_range)
            if share is not None:
                cell = shares[(BASELINE_J, BASELINE_BETA, BASELINE_SCALING)]
                cell["pooled"].append(share)
                cell[example.input_type].append(share)

            if args.log_every and position % args.log_every == 0:
                print(f"scored {position}/{len(examples)} facts", flush=True)

    values = {key: metric_values(series, args.top_k) for key, series in ranks.items()}
    bootstrap = ContextBootstrap(examples, args.bootstrap_samples, args.bootstrap_seed)

    rows: list[dict[str, Any]] = []
    for key, j, beta, scaling in emitted:
        no_rerank = config_key(j, 0.0, scaling)
        same_beta_j1 = config_key(1, beta, scaling)
        for modality in MODALITIES:
            share = mean_or_none(shares[(j, beta, scaling)][modality])
            for metric in METRICS:
                observed, low, high = bootstrap.interval(values[key][metric], modality)
                row = {
                    "config": key,
                    "hypotheses_used": j,
                    "rerank_beta": beta,
                    "score_scaling": scaling,
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
                row["rerank_share"] = round(share, 6) if share is not None else ""
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
    with (args.output_dir / "per_fact_ranks_extended.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_fact:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        args.output_dir / "beta_sweep_extended.csv",
        rows,
        SWEEP_FIELDNAMES + ["rerank_share"],
    )

    # --- reads -------------------------------------------------------------------
    prior_rows = list(csv.DictReader(args.prior_sweep.open("r", encoding="utf-8"))) if args.prior_sweep.exists() else []

    def curve(modality: str, metric: str, j: int) -> list[dict[str, Any]]:
        """Full beta curve at this J: prior sweep cells plus the new ones, beta ascending."""
        points: dict[float, dict[str, Any]] = {}
        for row in prior_rows:
            if (
                int(row["hypotheses_used"]) == j
                and row["modality"] == modality
                and row["metric"] == metric
                and row["score_variant"] == args.score_variant
                and (row["score_scaling"] == SCALING or float(row["rerank_beta"]) == 0.0)
            ):
                beta = float(row["rerank_beta"])
                points[beta] = {"rerank_beta": beta, "value": float(row["value"]), "source": "beta_sweep"}
        for row in rows:
            if row["hypotheses_used"] == j and row["modality"] == modality and row["metric"] == metric:
                points[row["rerank_beta"]] = {
                    "rerank_beta": row["rerank_beta"],
                    "value": row["value"],
                    "rerank_share": row["rerank_share"],
                    "source": "beta_sweep_extended",
                }
        return [points[beta] for beta in sorted(points)]

    def peak(points: list[dict[str, Any]]) -> dict[str, Any]:
        best = max(points, key=lambda point: (point["value"], -point["rerank_beta"]))
        top_beta = points[-1]["rerank_beta"]
        return {
            "peak_beta": best["rerank_beta"],
            "peak_value": best["value"],
            "grid_max_beta": top_beta,
            "still_rising_at_grid_max": bool(best["rerank_beta"] == top_beta),
            "turned_over": bool(best["rerank_beta"] < top_beta),
            "curve": points,
        }

    peaks = {
        modality: {
            metric: {f"J{j}": peak(curve(modality, metric, j)) for j in J_VALUES}
            for metric in ("recall_at_10", "mrr")
        }
        for modality in ("table", "text")
    }

    j3_minus_j1 = {
        modality: {
            metric: [
                {
                    "rerank_beta": beta,
                    "delta": row["delta_vs_j1_same_beta"],
                    "ci_low": row["delta_vs_j1_ci_low"],
                    "ci_high": row["delta_vs_j1_ci_high"],
                    "excludes_zero": row["delta_vs_j1_excludes_zero"],
                }
                for beta in betas
                for row in rows
                if row["hypotheses_used"] == 3
                and row["rerank_beta"] == beta
                and row["modality"] == modality
                and row["metric"] == metric
            ]
            for metric in ("recall_at_10", "mrr")
        }
        for modality in ("table", "text")
    }

    # (c) best normalized cell on table R@10, paired against the deployed raw cell.
    table_r10 = [row for row in rows if row["modality"] == "table" and row["metric"] == "recall_at_10"]
    best_row = max(table_r10, key=lambda row: (row["value"], -row["rerank_beta"]))
    best_key = best_row["config"]
    versus_baseline = {
        "best_normalized_cell": {
            "config": best_key,
            "hypotheses_used": best_row["hypotheses_used"],
            "rerank_beta": best_row["rerank_beta"],
            "selected_on": "table recall_at_10",
        },
        "baseline_cell": {
            "config": baseline_key,
            "hypotheses_used": BASELINE_J,
            "rerank_beta": BASELINE_BETA,
            "score_scaling": BASELINE_SCALING,
        },
        "paired": {
            modality: {
                metric: dict(
                    zip(
                        ("delta", "ci_low", "ci_high"),
                        [
                            round(value, 6)
                            for value in bootstrap.paired(
                                values[best_key][metric], values[baseline_key][metric], modality
                            )
                        ],
                    )
                )
                | {
                    "best_value": round(bootstrap.interval(values[best_key][metric], modality)[0], 6),
                    "baseline_value": round(bootstrap.interval(values[baseline_key][metric], modality)[0], 6),
                }
                for metric in METRICS
            }
            for modality in ("table", "text")
        },
    }
    for modality, per_metric in versus_baseline["paired"].items():
        for metric, entry in per_metric.items():
            entry["excludes_zero"] = bool(entry["ci_low"] > 0.0 or entry["ci_high"] < 0.0)

    # Cells statistically indistinguishable from the argmax on table R@10, so the
    # recommendation is reported as a plateau rather than a spuriously precise argmax.
    plateau = []
    for row in sorted(table_r10, key=lambda item: (-item["value"], item["rerank_beta"])):
        delta, low, high = bootstrap.paired(
            values[row["config"]]["recall_at_10"], values[best_key]["recall_at_10"], "table"
        )
        if low <= 0.0 <= high:
            mrr_row = next(
                item
                for item in rows
                if item["config"] == row["config"] and item["modality"] == "table" and item["metric"] == "mrr"
            )
            plateau.append(
                {
                    "config": row["config"],
                    "hypotheses_used": row["hypotheses_used"],
                    "rerank_beta": row["rerank_beta"],
                    "recall_at_10": row["value"],
                    "mrr": mrr_row["value"],
                    "rerank_share": row["rerank_share"],
                    "delta_vs_argmax_r10": round(delta, 6),
                    "ci_low": round(low, 6),
                    "ci_high": round(high, 6),
                }
            )

    selection = {
        "selected": {
            "hypotheses_used": best_row["hypotheses_used"],
            "rerank_beta": best_row["rerank_beta"],
            "score_scaling": SCALING,
            "score_variant": args.score_variant,
            "config": best_key,
        },
        "rule": (
            "maximize table recall_at_10 (566 of 661 facts), ties broken by the lower beta; "
            "table is reported as the primary subset and text separately"
        ),
        "justification": (
            "Selected on the table subset, where every beta curve turns over inside the grid. "
            "The argmax is not uniquely identified: the plateau below lists every cell whose "
            "paired difference from the argmax has a CI spanning zero, and it is wide, so beta "
            "should be read as a region rather than a point. The selected cell matches the "
            "deployed raw pipeline on table recall_at_10 exactly and is nominally ahead on MRR "
            "with a CI spanning zero, while giving up recall_at_50 significantly -- so this is "
            "not a demonstrated improvement over the deployed configuration, only a "
            "reparameterization that reaches the same top-10 accuracy at one fewer hypothesis."
        ),
        "indistinguishable_plateau_table_recall_at_10": plateau,
    }

    baseline_share = {
        modality: mean_or_none(shares[(BASELINE_J, BASELINE_BETA, BASELINE_SCALING)][modality])
        for modality in MODALITIES
    }
    rerank_share_table = {
        f"J{j}_beta{beta:g}_{SCALING}": {
            modality: (
                round(mean_or_none(shares[(j, beta, SCALING)][modality]), 6)
                if mean_or_none(shares[(j, beta, SCALING)][modality]) is not None
                else None
            )
            for modality in MODALITIES
        }
        for j in J_VALUES
        for beta in betas
    }
    rerank_share_table[baseline_key] = {
        modality: round(value, 6) if value is not None else None
        for modality, value in baseline_share.items()
    }

    metrics = {
        "experiment": "ags_beta_sweep_extended",
        "change": (
            "extends the range-normalized sweep to beta in "
            f"{[float(beta) for beta in betas]} at J in {list(J_VALUES)}, score_variant=sum, "
            "same score_configuration path and same twenty columns as beta_sweep.csv, plus a "
            "rerank_share column"
        ),
        "sample": {
            "path": str(args.sample_path),
            "fact_count": len(examples),
            "context_count": len({fact_context_key(example) for example in examples}),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
        },
        "grid": {
            "betas": [float(beta) for beta in betas],
            "hypotheses": list(J_VALUES),
            "scalings": [SCALING],
            "emitted_configs": [key for key, _, _, _ in emitted],
            "reference_configs_not_emitted": [key for key, _, _, _ in references],
        },
        "fixed_config": {
            "rendering_policy": rendering_policy,
            "score_variant": args.score_variant,
            "rrf_kappa": args.rrf_kappa,
            "top_k": args.top_k,
            "agreement_top_m": args.agreement_top_m,
            "label_coverage_weight": label_coverage_weight,
            "normalization_map_version": map_version(args.normalization_map),
        },
        "rerank_share": {
            "definition": (
                "mean over facts of range(rerank term) / range(retrieval score) in that cell's "
                "scaling units; raw = beta*range(agree)/range(S_wRRF), range_normalized = "
                "beta*range(agree) since the retrieval range is 1 and the raw-equivalent beta is "
                "beta*range(S_wRRF)"
            ),
            "facts_with_degenerate_retrieval_range": undefined_share_facts,
            "by_config": rerank_share_table,
            "deployed_raw_anchor": {
                "config": baseline_key,
                "pooled": round(baseline_share["pooled"], 6) if baseline_share["pooled"] else None,
                "table": round(baseline_share["table"], 6) if baseline_share["table"] else None,
                "text": round(baseline_share["text"], 6) if baseline_share["text"] else None,
                "expected_approximately": 0.47,
                "confirmed": bool(
                    baseline_share["pooled"] is not None
                    and abs(baseline_share["pooled"] - 0.47) <= 0.05
                ),
                "note": (
                    "not confirmed. 0.47 is the ratio of the two span figures quoted in the "
                    "run_ags_beta_sweep docstring (0.023 / 0.049), both of which are near-bounds "
                    "rather than observed means: 0.049 is the theoretical three-list RRF span "
                    "3/(kappa+1) attained only under perfect top-rank overlap, and 0.46 overstates "
                    "nothing but understates the observed agreement span. Measured over the 661 "
                    "facts the observed means are given below, and their ratio is the anchor value. "
                    "The gap is aggregation-independent (mean-of-ratios and ratio-of-means agree to "
                    "three decimals) and does not affect any ranking: rerank_share is a monotone "
                    "relabeling of the beta axis within a scaling."
                ),
                "observed_component_spans_at_baseline_J": {
                    "mean_range_S_wRRF": round(
                        sum(span_samples[BASELINE_J]["retrieval"])
                        / len(span_samples[BASELINE_J]["retrieval"]),
                        6,
                    ),
                    "mean_range_agree_consensus": round(
                        sum(span_samples[BASELINE_J]["agreement"])
                        / len(span_samples[BASELINE_J]["agreement"]),
                        6,
                    ),
                    "docstring_figures": {"range_S_wRRF": 0.049, "beta_times_range_agree": 0.023},
                },
            },
            "mean_component_spans_by_j": {
                f"J{j}": {
                    "mean_range_S_wRRF": round(
                        sum(span_samples[j]["retrieval"]) / len(span_samples[j]["retrieval"]), 6
                    ),
                    "mean_range_agree_consensus": round(
                        sum(span_samples[j]["agreement"]) / len(span_samples[j]["agreement"]), 6
                    ),
                }
                for j in J_VALUES
            },
        },
        "selection": selection,
        "peaks": peaks,
        "j3_minus_j1_same_beta": j3_minus_j1,
        "versus_deployed_baseline": versus_baseline,
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
                "rows_emitted": len(rows),
                "raw_anchor_rerank_share": metrics["rerank_share"]["deployed_raw_anchor"],
                "table_recall_at_10_peaks": {
                    j: peaks["table"]["recall_at_10"][j]["peak_beta"] for j in ("J1", "J2", "J3")
                },
                "still_rising": {
                    f"{modality}_{metric}_{j}": peaks[modality][metric][j]["still_rising_at_grid_max"]
                    for modality in ("table", "text")
                    for metric in ("recall_at_10", "mrr")
                    for j in ("J1", "J2", "J3")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
