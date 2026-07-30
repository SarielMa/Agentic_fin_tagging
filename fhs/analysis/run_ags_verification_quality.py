#!/usr/bin/env python3
"""Table 3 / Table 13: verification quality, on the FROZEN TEST SPLIT.

Replaces the development-set placeholders that
an earlier development-set run produced (250 tabular facts from the
coverage-pilot sample, n=5,765 dimension verdicts). Same question, same
estimator, same ground-truth definition -- different data.

What is measured
----------------
For every dimension verdict issued during grounding we ask whether the
hypothesis genuinely disagreed with the gold concept on that dimension, and
score the precision and recall of the D- (contradict) verdict against that
truth. The chance reference is the base rate of true disagreement among the
resolvable dimensions; a precision at the base rate means the verdict carries
no information.

Ground truth is `agree(gold_concept, hypothesis_dimensions)`: `matched is
False` means the hypothesis really does disagree there and D- should fire.
The verifiers never see the gold concept -- it is revealed only here.

Where the test-split verdicts come from
---------------------------------------
Both layers are scored on ONE shared unit -- (fact, hypothesis, dimension) over
the J=2 frozen-AGS hypotheses on all 2,509 test facts -- and on ONE shared
assessed window, the top-M=10 cluster representatives of AGS's own fused
ranking. That is what makes "deterministic beats LLM on identical dimensions"
a paired claim rather than two unrelated measurements.

  deterministic  recomputed here by `symbolic_feedback_from_candidates`, the
                 production verifier, over the representatives read from
                 runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/
                 bm25_candidates.jsonl. Pure and deterministic given the
                 candidates, so replaying it reproduces generation time
                 exactly; the test AGS-Seq run logged its verdicts but only
                 from round 2 onward, which is a different unit.
  llm            read verbatim from runs_ags_table5_ablation/qwen3_32b/
                 llm_verifier_calls.jsonl -- the full-test rerun (5,018 calls =
                 2,509 facts x 2 hypotheses, parse_rate 1.0) over the SAME
                 representatives, judging FAMILY/ROLE/EVENT only. Its
                 per-candidate judgements are folded up to a dimension verdict
                 with the identical >=0.6 / <=0.25 support-fraction thresholds
                 the deterministic layer uses, so the two differ in who judges,
                 not in how votes are counted.
  merged         `merge_feedback_layers(deterministic, llm)`, the production
                 merge.

The LLM layer abstains on QUALIFIER/SCOPE/TEMPORAL by construction. Those rows
stay in the denominator, exactly as on dev: an abstention is a missed
detection, and suppressing them would flatter the LLM layer.

Table 13's closing note -- verification quality under learned versus random
operator selection -- is a separate pass over the two sequential arms' test
traces (--arm-comparison), which do log their own per-round verdicts.

Outputs
-------
  per_verdict.jsonl          one row per scored (fact, hypothesis, dimension)
  verification_quality.csv   every slice x layer with bootstrap CIs
  table3.csv, table13.csv    the two paper tables, ready to typeset
  metrics.json               everything above plus coverage and config

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

from ags_sequential_arms import cluster_representatives
from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    VERDICT_CONTRADICT,
    VERDICT_SUPPORT,
    VERDICT_UNRESOLVED,
    agree,
    canonical_hypothesis_dimensions,
    is_unresolved,
    load_normalization_map,
    map_version,
    merge_feedback_layers,
    symbolic_feedback_from_candidates,
)
from run_ags_component_validation import agreement_reason_layer
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    load_taxonomy,
    normalize_tag,
)


DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_verification_quality" / "qwen3_32b"
DEFAULT_TEST_TRACE = (
    FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline" / "qwen3_32b_frozen_ags" / "bm25_candidates.jsonl"
)
DEFAULT_LLM_CALLS = FHS_ROOT / "runs" / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_calls.jsonl"
DEFAULT_SEQ_TRACES = {
    "learned": FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline" / "qwen3_32b_ags_seq" / "grounding_traces.jsonl",
    "random": SCRIPT_DIR
    / "runs_fintagging_grounding_baseline"
    / "qwen3_32b_ags_seq_random"
    / "grounding_traces.jsonl",
}

# The layer names the paper uses. "deterministic" is the symbolic verifier.
SOURCES = ("deterministic", "llm", "merged")
LLM_DIMENSIONS = ("FAMILY", "ROLE", "EVENT")
LAYERS = ("exact", "lexical")

# Same thresholds symbolic_feedback_from_candidates applies, so the two layers
# are folded up from per-candidate votes identically.
SUPPORT_THRESHOLD = 0.6
CONTRADICT_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-trace",
        type=Path,
        default=DEFAULT_TEST_TRACE,
        help="Frozen-AGS test-split trace. This is the split apply_server_fintagging_frozen_ags.sh runs.",
    )
    parser.add_argument("--llm-calls", type=Path, default=DEFAULT_LLM_CALLS)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument(
        "--gold-candidate-fields",
        choices=("compact", "full"),
        default="compact",
        help="compact mirrors the tag/label/type view the verifier scored candidates under.",
    )
    parser.add_argument("--top-m", type=int, default=10, help="Must match the LLM verifier run's --top-m.")
    parser.add_argument("--cluster-scan-depth", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument(
        "--arm-comparison",
        dest="arm_comparison",
        action="store_true",
        default=True,
        help="Also score the deterministic layer on the AGS-Seq learned/random test traces "
        "(Table 13's closing note). Streams two ~4.8GB files.",
    )
    parser.add_argument("--no-arm-comparison", dest="arm_comparison", action="store_false")
    parser.add_argument("--seq-trace-learned", type=Path, default=DEFAULT_SEQ_TRACES["learned"])
    parser.add_argument("--seq-trace-random", type=Path, default=DEFAULT_SEQ_TRACES["random"])
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: stop after N facts.")
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


# --------------------------------------------------------------------------- io


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_llm_verdicts(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """(fact_id, hypothesis_idx) -> the verifier's per-candidate judgements.

    Only rows that actually parsed are kept; an unparseable call is a missing
    measurement, not an abstention, and is reported separately in coverage.
    """
    verdicts: dict[tuple[int, int], dict[str, Any]] = {}
    unusable = 0
    for row in stream_jsonl(path):
        key = (int(row["fact_id"]), int(row["hypothesis_idx"]))
        if row.get("parse_ok") and row.get("verdicts_by_tag"):
            verdicts[key] = {
                "candidate_tags": [normalize_tag(tag) for tag in row.get("candidate_tags", [])],
                "verdicts_by_tag": {
                    normalize_tag(tag): value for tag, value in (row.get("verdicts_by_tag") or {}).items()
                },
                "parse_mode": row.get("parse_mode"),
            }
        else:
            unusable += 1
            verdicts.pop(key, None)
    verdicts["__unusable__"] = unusable  # type: ignore[index]
    return verdicts


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


# ------------------------------------------------------------------ llm folding


def llm_feedback_from_verdicts(
    hypothesis_dimensions: dict[str, Any],
    representative_tags: list[str],
    verdicts_by_tag: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fold the LLM's per-candidate judgements into one verdict per dimension.

    Mirrors symbolic_feedback_from_candidates exactly: the same assessed window,
    the same comparable-subset rule (a `null` judgement is an abstention and is
    dropped from the denominator, not counted as disagreement), and the same
    >=0.6 support / <=0.25 contradict thresholds. The only difference between
    the two layers is who issued the per-candidate judgement.

    Dimensions the LLM was never asked about (QUALIFIER/SCOPE/TEMPORAL) come
    back unresolved -- an abstention, which is how it will be scored.
    """
    canonical = canonical_hypothesis_dimensions(hypothesis_dimensions)
    verdict_records: list[dict[str, Any]] = []
    supported: list[str] = []
    contradicted: list[str] = []
    unresolved: list[str] = []

    for dimension, value in canonical.items():
        upper = dimension.upper()
        if is_unresolved(value) or upper not in LLM_DIMENSIONS:
            unresolved.append(upper)
            verdict_records.append(
                {
                    "dimension": upper,
                    "verdict": VERDICT_UNRESOLVED,
                    "source_layer": "llm",
                    "confidence": 0.0,
                    "candidate_support_fraction": 0.0,
                    "reason": "hypothesis_unresolved" if is_unresolved(value) else "dimension_not_judged_by_llm",
                }
            )
            continue

        comparable = [
            verdicts_by_tag.get(tag, {}).get(upper)
            for tag in representative_tags
            if verdicts_by_tag.get(tag, {}).get(upper) is not None
        ]
        if not comparable:
            unresolved.append(upper)
            support_fraction = 0.0
            confidence = 0.0
            verdict_name = VERDICT_UNRESOLVED
        else:
            support_fraction = sum(1 for matched in comparable if matched is True) / len(comparable)
            if support_fraction >= SUPPORT_THRESHOLD:
                supported.append(upper)
                verdict_name = VERDICT_SUPPORT
            elif support_fraction <= CONTRADICT_THRESHOLD:
                contradicted.append(upper)
                verdict_name = VERDICT_CONTRADICT
            else:
                unresolved.append(upper)
                verdict_name = VERDICT_UNRESOLVED
            confidence = abs(support_fraction - 0.5) * 2.0

        verdict_records.append(
            {
                "dimension": upper,
                "verdict": verdict_name,
                "source_layer": "llm",
                "confidence": round(confidence, 6),
                "candidate_support_fraction": round(support_fraction, 6),
                "comparable_candidates": len(comparable),
            }
        )

    structural = {"FAMILY", "ROLE", "EVENT"}
    structural_contradictions = [dim for dim in contradicted if dim in structural]
    return {
        "supported_dimensions": supported,
        "contradicted_dimensions": contradicted,
        "unresolved_dimensions": unresolved,
        "dimension_verdicts": verdict_records,
        "structural_mismatch": {
            "is_mismatch": bool(len(structural_contradictions) >= 2),
            "reason": "llm_contradictions=" + ",".join(structural_contradictions)
            if len(structural_contradictions) >= 2
            else "",
        },
        "source_layer": "llm",
    }


def fired(feedback: dict[str, Any], key: str) -> set[str]:
    return {str(dimension).upper() for dimension in (feedback.get(key) or [])}


# ------------------------------------------------------------------ observation


def collect_observations(
    args: argparse.Namespace,
    llm_verdicts: dict[tuple[int, int], dict[str, Any]],
    concepts_by_tag: dict[str, Any],
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    facts_scanned = 0
    hypotheses_scanned = 0
    llm_missing_calls = 0

    for record in stream_jsonl(args.test_trace):
        if args.limit is not None and facts_scanned >= args.limit:
            break
        facts_scanned += 1
        fact_id = int(record["example_idx"])

        gold_tags = [normalize_tag(tag) for tag in record.get("gold_tags", [])]
        concept = concepts_by_tag.get(gold_tags[0]) if gold_tags else None
        if concept is None:
            skipped["gold_concept_missing"] += 1
            continue

        # The fused AGS ranking the verifier actually inspected, and the same
        # cluster-representative window the LLM verifier was handed.
        ranking = record.get("final_candidates") or record.get("candidates") or []
        representatives = cluster_representatives(
            ranking, normalization_map, args.top_m, args.cluster_scan_depth
        )
        representative_tags = [normalize_tag(candidate.get("tag", "")) for candidate in representatives]
        if not representatives:
            skipped["no_candidates"] += 1
            continue

        gold_rank = next(
            (int(candidate["rank"]) for candidate in ranking if normalize_tag(candidate.get("tag", "")) in set(gold_tags)),
            None,
        )
        gold_in_window = bool(set(gold_tags) & set(representative_tags))

        for hypothesis in record.get("frozen_ags_hypotheses", []):
            hyp_idx = int(hypothesis["hypothesis_idx"])
            hypothesis_dimensions = hypothesis.get("dimensions", {})
            hypotheses_scanned += 1

            deterministic_fb = symbolic_feedback_from_candidates(
                hypothesis_dimensions,
                representatives,
                top_m=len(representatives),
                normalization_map=normalization_map,
            )

            call = llm_verdicts.get((fact_id, hyp_idx))
            if call is None:
                llm_missing_calls += 1
                # No usable LLM call: it issued nothing, which is exactly how an
                # abstention is scored. Recorded so the gap stays visible.
                llm_fb = llm_feedback_from_verdicts(hypothesis_dimensions, [], {})
            else:
                llm_fb = llm_feedback_from_verdicts(
                    hypothesis_dimensions, call["candidate_tags"], call["verdicts_by_tag"]
                )
            merged_fb = merge_feedback_layers(deterministic_fb, llm_fb)

            d_minus = {
                "deterministic": fired(deterministic_fb, "contradicted_dimensions"),
                "llm": fired(llm_fb, "contradicted_dimensions"),
                "merged": fired(merged_fb, "contradicted_dimensions"),
            }
            d_plus = {
                "deterministic": fired(deterministic_fb, "supported_dimensions"),
                "llm": fired(llm_fb, "supported_dimensions"),
                "merged": fired(merged_fb, "supported_dimensions"),
            }

            truth = agree(gold_candidate(concept, args.gold_candidate_fields), hypothesis_dimensions, normalization_map)
            for verdict in truth.verdicts:
                dimension = str(verdict.get("dimension", "")).upper()
                matched = verdict.get("matched")
                if matched is None:
                    skipped["truth_unresolvable"] += 1
                    continue
                observations.append(
                    {
                        "fact_id": fact_id,
                        "context_key": str(record.get("context_id")),
                        "hypothesis_idx": hyp_idx,
                        "modality": str(record.get("input_type", "")),
                        "datatype": str(record.get("type", "")),
                        "dimension": dimension,
                        "truth_layer": agreement_reason_layer(verdict.get("reason")),
                        "truth_reason": verdict.get("reason"),
                        "truth_disagrees": bool(matched is False),
                        "gold_rank": gold_rank,
                        "gold_in_window": gold_in_window,
                        "assessed_window_size": len(representatives),
                        "llm_call_present": call is not None,
                        **{f"d_minus_{source}": bool(dimension in d_minus[source]) for source in SOURCES},
                        **{f"d_plus_{source}": bool(dimension in d_plus[source]) for source in SOURCES},
                    }
                )

        if args.log_every and facts_scanned % args.log_every == 0:
            print(f"scanned {facts_scanned} facts, {len(observations)} observations", flush=True)

    coverage = {
        "facts_scanned": facts_scanned,
        "hypotheses_scanned": hypotheses_scanned,
        "observations": len(observations),
        "llm_calls_loaded": len(llm_verdicts),
        "llm_calls_unusable": llm_verdicts.get("__unusable__", 0),
        "hypotheses_without_llm_call": llm_missing_calls,
        "skipped": dict(skipped),
    }
    return observations, coverage


# ---------------------------------------------------------------------- scoring


def rates(counts: dict[str, int]) -> dict[str, Any]:
    true_positive = counts["tp"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    n = counts["n"]
    issued = counts.get("issued", 0)
    d_minus_fired = true_positive + false_positive
    positives = true_positive + false_negative
    base_rate = positives / n if n else 0.0
    precision = true_positive / d_minus_fired if d_minus_fired else 0.0
    recall = true_positive / positives if positives else 0.0
    # F1 is reported because precision and recall move in opposite directions across the
    # layers: the LLM abstains far more, which flatters precision-only comparisons.
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "d_minus_fired": d_minus_fired,
        "true_disagreements": positives,
        "verdicts_issued": issued,
        # Non-abstention: the layer said something (D- or D+) rather than staying silent.
        # An abstention is a missed detection, so it stays in every denominator above.
        "coverage": issued / n if n else 0.0,
        "base_rate": base_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_minus_base_rate": precision - base_rate,
        "lift": precision / base_rate if base_rate else 0.0,
    }


def count_rows(rows: list[dict[str, Any]], source: str) -> dict[str, int]:
    fired_key = f"d_minus_{source}"
    supported_key = f"d_plus_{source}"
    counts = {"n": len(rows), "tp": 0, "fp": 0, "fn": 0, "issued": 0}
    for row in rows:
        if row[fired_key] or row[supported_key]:
            counts["issued"] += 1
        if row[fired_key]:
            counts["tp" if row["truth_disagrees"] else "fp"] += 1
        elif row["truth_disagrees"]:
            counts["fn"] += 1
    return counts


CONTRASTS = (
    ("deterministic_minus_llm_f1", "deterministic", "llm", "f1"),
    ("deterministic_minus_llm_recall", "deterministic", "llm", "recall"),
    ("merged_minus_deterministic_f1", "merged", "deterministic", "f1"),
)


def paired_layer_bootstrap(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """CI on a between-layer difference, resampling contexts jointly.

    Both layers are scored on the identical observation set, so a context must enter
    or leave both replicates together. Resampling them independently would discard
    the pairing and widen the interval for no reason.
    """
    if not rows:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_excludes_zero": False, "contexts": 0}
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_context[row["context_key"]].append(row)
    keys = sorted(by_context)
    per_context = {key: (count_rows(by_context[key], left), count_rows(by_context[key], right)) for key in keys}

    rng = random.Random(seed)
    samples: list[float] = []
    size = len(keys)
    fields = ("n", "tp", "fp", "fn", "issued")
    for _ in range(iterations):
        total_left = dict.fromkeys(fields, 0)
        total_right = dict.fromkeys(fields, 0)
        for _ in range(size):
            left_counts, right_counts = per_context[keys[rng.randrange(size)]]
            for field in fields:
                total_left[field] += left_counts[field]
                total_right[field] += right_counts[field]
        samples.append(rates(total_left)[metric] - rates(total_right)[metric])

    ordered = sorted(samples)

    def percentile(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (1.0 - (pos - lower)) + ordered[upper] * (pos - lower)

    low, high = percentile(0.025), percentile(0.975)
    return {
        "mean": round(sum(samples) / len(samples), 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "contexts": size,
        "iterations": iterations,
    }


def bootstrap_context(
    rows: list[dict[str, Any]],
    source: str,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Resample source contexts with replacement, recomputing the rates each time.

    Contexts, not facts: this benchmark puts ~21 facts in one table, so
    fact-level resampling would treat 21 correlated draws as independent. The
    per-context sufficient statistics (n, tp, fp, fn) are precomputed once and
    summed per replicate, which is the same estimator as rebuilding the row
    lists and vastly cheaper.
    """
    keys = ("precision", "recall", "precision_minus_base_rate")
    if not rows:
        return {key: (0.0, 0.0) for key in keys}

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_context[row["context_key"]].append(row)
    context_keys = sorted(by_context)
    per_context = [count_rows(by_context[key], source) for key in context_keys]

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {key: [] for key in keys}
    size = len(per_context)
    for _ in range(iterations):
        total = {"n": 0, "tp": 0, "fp": 0, "fn": 0}
        for _ in range(size):
            counts = per_context[rng.randrange(size)]
            total["n"] += counts["n"]
            total["tp"] += counts["tp"]
            total["fp"] += counts["fp"]
            total["fn"] += counts["fn"]
        stats = rates(total)
        for key in keys:
            samples[key].append(stats[key])

    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        weight = pos - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {key: (percentile(values, 0.025), percentile(values, 0.975)) for key, values in samples.items()}


def subset_for(
    observations: list[dict[str, Any]],
    dimension: str,
    layer: str,
    window: str,
    modality: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in observations
        if (dimension == "ALL" or row["dimension"] == dimension)
        and (layer == "all" or row["truth_layer"] == layer)
        and (window == "all" or (window == "gold_in_window") == bool(row["gold_in_window"]))
        and (modality == "all" or row["modality"] == modality)
    ]


def slice_specs() -> list[tuple[str, str, str, str]]:
    """The decision-relevant cuts, not a full cross product.

    Every row the two tables print, plus the diagnostic cuts that make them
    readable: the exact/lexical split (only the exact layer is verification in
    the strict sense -- see Limitations), and the modality split.
    """
    specs: list[tuple[str, str, str, str]] = [("ALL", "all", "all", "all")]
    for window in ("gold_in_window", "gold_outside_window"):
        specs.append(("ALL", "all", window, "all"))
    for layer in LAYERS:
        specs.append(("ALL", layer, "all", "all"))
    for modality in ("table", "text"):
        specs.append(("ALL", "all", "all", modality))
    for dimension in DIMENSIONS:
        specs.append((dimension, "all", "all", "all"))
    for window in ("gold_in_window", "gold_outside_window"):
        specs.append(("FAMILY", "all", window, "all"))
    return specs


def build_rows(args: argparse.Namespace, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for dimension, layer, window, modality in slice_specs():
        subset = subset_for(observations, dimension, layer, window, modality)
        for source in SOURCES:
            seed_offset += 1
            stats = rates(count_rows(subset, source))
            ci = bootstrap_context(subset, source, args.bootstrap_samples, args.bootstrap_seed + seed_offset)
            rows.append(
                {
                    "dimension": dimension,
                    "truth_layer": layer,
                    "gold_window": window,
                    "modality": modality,
                    "verifier": source,
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


# ------------------------------------------------------------- the paper tables


def find_row(rows: list[dict[str, Any]], dimension: str, layer: str, window: str, modality: str, verifier: str):
    return next(
        row
        for row in rows
        if row["dimension"] == dimension
        and row["truth_layer"] == layer
        and row["gold_window"] == window
        and row["modality"] == modality
        and row["verifier"] == verifier
    )


TABLE3_SPEC = [
    ("Deterministic", "ALL", "all", "all", "all", "deterministic"),
    ("LLM", "ALL", "all", "all", "all", "llm"),
    ("Merged", "ALL", "all", "all", "all", "merged"),
]

TABLE13_SPEC = [
    ("All", "deterministic", "ALL", "all", "all", "all", "deterministic", True),
    ("All", "LLM", "ALL", "all", "all", "all", "llm", True),
    ("All", "merged", "ALL", "all", "all", "all", "merged", True),
    ("Gold in window", "merged", "ALL", "all", "gold_in_window", "all", "merged", False),
    ("Gold outside", "merged", "ALL", "all", "gold_outside_window", "all", "merged", False),
    ("FAMILY", "merged", "FAMILY", "all", "all", "all", "merged", True),
    ("FAMILY", "LLM", "FAMILY", "all", "all", "all", "llm", False),
]


def table3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label, dimension, layer, window, modality, verifier in TABLE3_SPEC:
        row = find_row(rows, dimension, layer, window, modality, verifier)
        out.append(
            {
                "verifier": label,
                "precision": round(row["precision"], 3),
                "recall": round(row["recall"], 3),
                "f1": round(row["f1"], 3),
                "coverage": round(row["coverage"], 3),
                "precision_minus_base": round(row["precision_minus_base_rate"], 3),
                "ci_excludes_zero": row["beats_chance"],
                "n": row["n"],
                "base_rate": round(row["base_rate"], 3),
            }
        )
    return out


def table13(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for condition, layer_label, dimension, layer, window, modality, verifier, show_delta in TABLE13_SPEC:
        row = find_row(rows, dimension, layer, window, modality, verifier)
        out.append(
            {
                "condition": condition,
                "layer": layer_label,
                "precision": round(row["precision"], 3),
                "recall": round(row["recall"], 3),
                "precision_minus_base": round(row["precision_minus_base_rate"], 3) if show_delta else None,
                "ci_excludes_zero": row["beats_chance"] if show_delta else None,
                "n": row["n"],
                "base_rate": round(row["base_rate"], 3),
                "d_minus_fired": row["d_minus_fired"],
            }
        )
    return out


def llm_firing_by_dimension(observations: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """"Issued a verdict on N of M opportunities", per dimension.

    An opportunity is a scored dimension observation; a firing is any non-
    abstention (D- or D+). This is the sentence Table 13's caption carries.
    """
    summary: dict[str, dict[str, int]] = {}
    for dimension in DIMENSIONS:
        rows = [row for row in observations if row["dimension"] == dimension]
        summary[dimension] = {
            "opportunities": len(rows),
            "llm_issued_any_verdict": sum(1 for row in rows if row["d_minus_llm"] or row["d_plus_llm"]),
            "llm_d_minus": sum(1 for row in rows if row["d_minus_llm"]),
            "llm_true_positives": sum(1 for row in rows if row["d_minus_llm"] and row["truth_disagrees"]),
            "deterministic_d_minus": sum(1 for row in rows if row["d_minus_deterministic"]),
            "deterministic_true_positives": sum(
                1 for row in rows if row["d_minus_deterministic"] and row["truth_disagrees"]
            ),
        }
    return summary


# ------------------------------------------------- learned vs random arm compare


def hypothesis_dimensions_from_round(record: dict[str, Any]) -> dict[str, str]:
    """Recover the pre-revision hypothesis values from a logged round.

    The round logs the verdicts, not the hypothesis they were issued against;
    each candidate verdict carries `raw_hypothesis_value`, which is that value
    verbatim. Dimensions the hypothesis left UNRESOLVED have no candidate
    verdicts and are therefore absent -- which is correct, since agree() would
    skip them anyway.
    """
    values: dict[str, str] = {}
    for verdict in record.get("dimension_verdicts") or []:
        dimension = str(verdict.get("dimension", "")).upper()
        if not dimension or dimension in values:
            continue
        for candidate_verdict in verdict.get("candidate_verdicts") or []:
            raw = candidate_verdict.get("raw_hypothesis_value")
            if raw not in (None, ""):
                values[dimension] = raw
                break
    return values


def score_seq_arm(
    trace_path: Path,
    concepts_by_tag: dict[str, Any],
    normalization_map: dict[str, Any],
    gold_candidate_fields: str,
    limit: int | None,
    log_every: int,
    label: str,
) -> dict[str, Any]:
    """Deterministic-layer precision/recall over one sequential arm's test trace.

    Different unit from the main table -- (fact, round, dimension) over rounds
    2..B rather than (fact, hypothesis, dimension) at round one -- so these
    numbers are comparable to each other, not to Table 3.
    """
    counts = {"n": 0, "tp": 0, "fp": 0, "fn": 0}
    per_context: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "tp": 0, "fp": 0, "fn": 0})
    facts = 0
    rounds = 0
    skipped: Counter[str] = Counter()

    for record in stream_jsonl(trace_path):
        if limit is not None and facts >= limit:
            break
        facts += 1
        gold_tags = [normalize_tag(tag) for tag in record.get("gold_tags", [])]
        concept = concepts_by_tag.get(gold_tags[0]) if gold_tags else None
        if concept is None:
            skipped["gold_concept_missing"] += 1
            continue
        context_key = str(record.get("context_id"))
        gold = gold_candidate(concept, gold_candidate_fields)

        for round_record in record.get("ags_seq_rounds") or []:
            hypothesis = hypothesis_dimensions_from_round(round_record)
            if not hypothesis:
                skipped["hypothesis_dimensions_unrecoverable"] += 1
                continue
            rounds += 1
            contradicted = {str(dim).upper() for dim in (round_record.get("D_minus") or [])}
            truth = agree(gold, hypothesis, normalization_map)
            for verdict in truth.verdicts:
                dimension = str(verdict.get("dimension", "")).upper()
                matched = verdict.get("matched")
                if matched is None:
                    skipped["truth_unresolvable"] += 1
                    continue
                disagrees = matched is False
                bucket = "tp" if (dimension in contradicted and disagrees) else (
                    "fp" if dimension in contradicted else ("fn" if disagrees else None)
                )
                counts["n"] += 1
                per_context[context_key]["n"] += 1
                if bucket:
                    counts[bucket] += 1
                    per_context[context_key][bucket] += 1

        if log_every and facts % log_every == 0:
            print(f"[{label}] scanned {facts} facts, {counts['n']} verdicts", flush=True)

    stats = rates(counts)
    return {
        "arm": label,
        "trace": str(trace_path),
        "facts": facts,
        "rounds": rounds,
        "skipped": dict(skipped),
        **{key: (round(value, 6) if isinstance(value, float) else value) for key, value in stats.items()},
        # Popped by the caller for the paired bootstrap, not serialized.
        "_per_context": {key: dict(value) for key, value in per_context.items()},
    }


def paired_arm_bootstrap(
    per_context_a: dict[str, dict[str, int]],
    per_context_b: dict[str, dict[str, int]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """CI on the learned-minus-random precision difference, resampling contexts jointly.

    The arms run the same instances in the same order, so a context must be drawn into or
    out of both replicates together; resampling them independently would inflate the interval
    by discarding the pairing. "Statistically identical" is a claim about this interval
    containing zero, not about the two point estimates rounding to the same digits.
    """
    keys = sorted(set(per_context_a) | set(per_context_b))
    if not keys:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_excludes_zero": False, "contexts": 0}
    empty = {"n": 0, "tp": 0, "fp": 0, "fn": 0}
    rng = random.Random(seed)
    deltas: list[float] = []
    size = len(keys)
    for _ in range(iterations):
        total_a = {"n": 0, "tp": 0, "fp": 0, "fn": 0}
        total_b = {"n": 0, "tp": 0, "fp": 0, "fn": 0}
        for _ in range(size):
            key = keys[rng.randrange(size)]
            for total, source in ((total_a, per_context_a), (total_b, per_context_b)):
                counts = source.get(key, empty)
                for field in ("n", "tp", "fp", "fn"):
                    total[field] += counts[field]
        deltas.append(rates(total_a)["precision"] - rates(total_b)["precision"])

    ordered = sorted(deltas)

    def percentile(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (1.0 - (pos - lower)) + ordered[upper] * (pos - lower)

    low, high = percentile(0.025), percentile(0.975)
    return {
        "mean_delta": round(sum(deltas) / len(deltas), 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "contexts": size,
        "iterations": iterations,
    }


# ------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.test_trace, args.llm_calls, args.taxonomy_jsonl):
        if not Path(path).exists():
            raise SystemExit(f"Missing required input: {path}")

    print(f"Loading taxonomy from {args.taxonomy_jsonl}", flush=True)
    concepts_by_tag = {concept.tag: concept for concept in load_taxonomy(args.taxonomy_jsonl)}
    normalization_map = load_normalization_map(args.normalization_map)

    print(f"Loading LLM verifier verdicts from {args.llm_calls}", flush=True)
    llm_verdicts = load_llm_verdicts(args.llm_calls)
    print(f"  {len(llm_verdicts) - 1} usable calls, {llm_verdicts['__unusable__']} unusable", flush=True)

    observations, coverage = collect_observations(args, llm_verdicts, concepts_by_tag, normalization_map)
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True), flush=True)
    if not observations:
        raise SystemExit("No evaluable observations; nothing to score.")

    rows = build_rows(args, observations)
    paper_table3 = table3(rows)
    paper_table13 = table13(rows)
    firing = llm_firing_by_dimension(observations)

    # Task 1: paired between-layer contrasts on the pooled slice, context as the unit.
    layer_contrasts = {
        name: {
            "left": left,
            "right": right,
            "metric": metric,
            **paired_layer_bootstrap(
                observations, left, right, metric, args.bootstrap_samples, args.bootstrap_seed + offset
            ),
        }
        for offset, (name, left, right, metric) in enumerate(CONTRASTS, start=9000)
    }

    arm_comparison: dict[str, Any] | None = None
    if args.arm_comparison:
        arms = []
        for label, path in (("learned", args.seq_trace_learned), ("random", args.seq_trace_random)):
            if not Path(path).exists():
                print(f"Skipping arm comparison for {label}: missing {path}", flush=True)
                continue
            print(f"Scoring sequential arm '{label}' from {path}", flush=True)
            arms.append(
                score_seq_arm(
                    Path(path),
                    concepts_by_tag,
                    normalization_map,
                    args.gold_candidate_fields,
                    args.limit,
                    args.log_every,
                    label,
                )
            )
        per_context_by_arm = {arm["arm"]: arm.pop("_per_context") for arm in arms}
        if len(arms) == 2:
            paired = paired_arm_bootstrap(
                per_context_by_arm["learned"],
                per_context_by_arm["random"],
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            arm_comparison = {
                "arms": arms,
                "precision_delta": round(arms[0]["precision"] - arms[1]["precision"], 6),
                "precision_delta_paired_bootstrap": paired,
                "statistically_identical": not paired["ci_excludes_zero"],
                "note": (
                    "Deterministic-layer verification quality on the sequential arms' own logged "
                    "round 2..B verdicts. Unit is (fact, round, dimension), not the round-one "
                    "(fact, hypothesis, dimension) unit of Table 3, so these are comparable to "
                    "each other only. The CI is a paired context-level bootstrap: the arms run "
                    "the same instances in the same order, so contexts are resampled jointly."
                ),
            }
        elif arms:
            arm_comparison = {"arms": arms, "note": "Only one arm trace was available."}

    # --------------------------------------------------------------- write out
    with (args.output_dir / "per_verdict.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation, ensure_ascii=False) + "\n")

    def write_csv(name: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("verification_quality.csv", rows)
    write_csv("table3.csv", paper_table3)
    write_csv("table13.csv", paper_table13)

    metrics = {
        "experiment": "ags_verification_quality",
        "split": "test",
        "unit": "(fact, hypothesis, dimension) over J=2 frozen-AGS hypotheses, top-M cluster representatives",
        "table3": paper_table3,
        "table13": paper_table13,
        "layer_contrasts": layer_contrasts,
        "llm_firing_by_dimension": firing,
        "scope": (
            "Table 3 evaluates whether a verifier can determine that the HYPOTHESIS is wrong on "
            "a dimension -- the signal D- revision feedback is built from. It does not evaluate "
            "candidate reranking ability; see runs_ags_verifier_bridge for that comparison."
        ),
        "arm_comparison": arm_comparison,
        "coverage": coverage,
        "config": {
            "test_trace": str(args.test_trace),
            "llm_calls": str(args.llm_calls),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "gold_candidate_fields": args.gold_candidate_fields,
            "top_m": args.top_m,
            "cluster_scan_depth": args.cluster_scan_depth,
            "support_threshold": SUPPORT_THRESHOLD,
            "contradict_threshold": CONTRADICT_THRESHOLD,
            "normalization_map_version": map_version(args.normalization_map),
            "bootstrap": {"iterations": args.bootstrap_samples, "seed": args.bootstrap_seed, "unit": "context"},
            "limit": args.limit,
            "ground_truth": (
                "agree(gold_concept, hypothesis_dimensions): matched is False means the hypothesis "
                "genuinely disagrees with gold on that dimension and D- should fire"
            ),
        },
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n=== Table 3 (test): hypothesis-level absolute disagreement detection ===", flush=True)
    print(f"  {'Verifier':<16} {'Prec.':>6} {'Rec.':>6} {'F1':>6} {'Cov.':>6} {'P-base':>8}", flush=True)
    for row in paper_table3:
        mark = "†" if row["ci_excludes_zero"] else " "
        print(
            f"  {row['verifier']:<16} {row['precision']:>6.3f} {row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} {row['coverage']:>6.3f} {row['precision_minus_base']:>+7.3f}{mark}",
            flush=True,
        )
    print(f"  n={paper_table3[0]['n']}  base rate={paper_table3[0]['base_rate']:.3f}", flush=True)
    print("\n  Paired context-level contrasts (2.5-97.5%):", flush=True)
    for name, contrast in layer_contrasts.items():
        mark = " *" if contrast["ci_excludes_zero"] else ""
        print(
            f"    {name:<34} {contrast['mean']:+.4f} "
            f"[{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]{mark}",
            flush=True,
        )

    print("\n=== Table 13 (test) ===", flush=True)
    for row in paper_table13:
        delta = "—" if row["precision_minus_base"] is None else f"{row['precision_minus_base']:+.3f}"
        mark = "†" if row["ci_excludes_zero"] else ""
        print(
            f"  {row['condition']:<16} {row['layer']:<14} {row['precision']:.3f}  {row['recall']:.3f}  "
            f"{delta}{mark}",
            flush=True,
        )
    if arm_comparison and len(arm_comparison.get("arms", [])) == 2:
        learned, rand = arm_comparison["arms"]
        paired = arm_comparison["precision_delta_paired_bootstrap"]
        verdict = "statistically identical" if arm_comparison["statistically_identical"] else "DIFFERENT"
        print(
            f"\n  learned vs random deterministic precision: "
            f"{learned['precision']:.4f} versus {rand['precision']:.4f}\n"
            f"  paired delta {paired['mean_delta']:+.4f} "
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] -> {verdict}",
            flush=True,
        )
    print(f"\nWrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
