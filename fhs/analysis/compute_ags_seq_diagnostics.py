#!/usr/bin/env python3
"""Sequential-search diagnostics for the finished AGS-Seq arms, consolidated into two tables.

Replaces the five separate diagnostic tables (T22-T26) with:

    TABLE A  "Does the search change the outcome?"      (was T22, T23)
             round-1 vs full R@50, top-50 membership churn, oracle best-prefix
             advantage, and the permutation null for that advantage.
    TABLE B  "Search dynamics and learning"             (was T24, T25, T26)
             stream AULC, final-third R@50, reward density, episode behaviour.

Offline only: reads the two finished traces, runs no generation and no retrieval.

    python compute_ags_seq_diagnostics.py                     # full run
    python compute_ags_seq_diagnostics.py --limit 100         # smoke test

TWO ARMS, NOT THREE -- READ THIS BEFORE USING THE TABLES
---------------------------------------------------------
The brief asked for three arms sharing one instance order: learned selection, random
selection, and perturbation-only. The finished run has only the first two. There is no
perturbation-only arm to read:

  * `ags_sequential_arms.py` implements exactly two query modes, `ags_seq` (Thompson
    sampling over per-operator posteriors) and `ags_seq_random` (uniform draw from the
    same admissible slate); its module docstring calls directive selection "the single
    difference between them". PERTURB is an *operator* inside the slate
    (PERTURB_OPERATOR = "O_perturb", :90), never a separate arm.
  * A perturbation-only arm therefore needs a third generation run, which the brief rules
    out ("no new jobs, no generation, no retrieval").
  * It cannot be reconstructed from the logged counterfactual either. Each round stores
    `replay_directive` / `delta_replay`, but that replay is the *runner-up* directive
    frozen at decision time (ags_sequential_arms.py:464, :833-852) and applies to one
    round only. Chaining it into a full episode is impossible, because round r+1's
    feedback state depends on the candidate set round r actually produced.

A three-arm version of exactly these diagnostics does exist, on a different run:
`runs_ags_reward_diagnostic/qwen3_32b/rounds.jsonl` has arms bandit / random /
resample_only, where `resample_only` IS the perturbation-only arm. It is not a substitute
here -- it is 250 facts drawn from 14 source contexts and is tabular-only (zero text
rounds), so it cannot support the modality split this brief asks for, and a context-level
bootstrap over 14 clusters is far too coarse to put a usable interval on anything. Use it
only if the paper wants the third arm more than it wants the sample size, and if so, say
in the caption that the row comes from the 250-fact pilot.

FOUR MEASUREMENT DECISIONS THE CAPTION HAS TO CARRY
---------------------------------------------------
1. ROUND-LEVEL R@50 IS A RETRIEVAL-STAGE QUANTITY. The listwise reranker runs once, over
   the final pool, so a "reranked round one" does not exist and no prefix of the episode
   can be scored at rerank stage. Every number here is the retrieval stage. Sanity check:
   the full-episode R@50 computed below reproduces each run's metrics.json
   bm25_retrieval.recall_at_50 (asserted, see `checks` in metrics.json).

2. THE NOVELTY GATE WAS OFF (`ags_seq_config.novelty_gate = false`). `gate_would_reject`
   is therefore counterfactual: it records what the gate *would* have rejected had it been
   enabled, and no round was actually suppressed. Table B's gate-reject column must be
   labelled as the counterfactual rate, not as realized behaviour.

3. THE PERMUTATION NULL DESTROYS CONSOLIDATION, DELIBERATELY. Substituting another fact's
   round-r candidate list does not re-run sum-RRF; it asks the narrower question the brief
   specifies -- how much oracle advantage does *any* extra list buy, versus the list this
   fact's own search actually produced. Because gold rarely appears in a foreign fact's
   list, the null sits near the round-1 baseline by construction. That is the point: it
   calibrates the observed advantage, it is not a competing method.

4. AULC HERE IS OVER THE INSTANCE STREAM, NOT OVER ROUNDS. It is the area under a
   window-50 rolling mean of per-fact final R@50, in the order the arm processed facts,
   normalized by stream length. This measures whether the bandit improves as it accumulates
   posterior mass. It is a different quantity from the within-episode AULC in
   compute_ags_seq_arm_metrics.py (mean R@50 across rounds 1..B of one episode); the two
   are not comparable and should not appear in the same table under one name.
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
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from run_fintagging_grounding_baseline import SCRIPT_DIR, first_gold_rank, normalize_tag

DEFAULT_RUNS_ROOT = FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline"
DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_seq_diag" / "qwen3_32b"

ARMS = {
    "AGS-Seq (learned)": "qwen3_32b_ags_seq",
    "AGS-Seq-random": "qwen3_32b_ags_seq_random",
}
MODALITIES = ("table", "text")
DEPTH = 50
# Kept for the reward-density row: "no measurable movement" needs a tolerance as well as an
# exact-zero count, because utility is a float difference and exact zeros understate the mass.
NEAR_ZERO = 0.01
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--window", type=int, default=50, help="Rolling window for the stream AULC curve.")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--limit", type=int, default=None, help="Facts per arm, for smoke tests.")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def stream_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def top_tags(tags: list[Any], depth: int) -> list[str]:
    return [normalize_tag(tag) for tag in tags[:depth]]


def read_fact(record: dict[str, Any], stream_index: int) -> dict[str, Any]:
    """One compact per-fact row. Only the top-`DEPTH` tags are retained per round: every
    quantity in both tables is an R@50 or a rank, so the tail of the 200-candidate list is
    never consulted and holding it would cost ~2 GB per arm for nothing."""
    gold = [normalize_tag(tag) for tag in record.get("gold_tags", [])]
    round1_full = record.get("round1_candidates") or []
    final_full = record.get("candidate_union_tags") or []

    # Stored as frozensets: the permutation null does ~75M membership tests per arm, and
    # only membership at depth 50 is ever asked of these lists, never order.
    round_lists: list[frozenset[str]] = []
    round_rows: list[dict[str, Any]] = []
    for round_record in record.get("ags_seq_rounds", []):
        round_lists.append(frozenset(top_tags(round_record.get("candidate_list") or [], DEPTH)))
        round_rows.append(
            {
                "round_idx": round_record.get("round_idx"),
                "selected_operator": round_record.get("selected_operator"),
                "selected_mode": round_record.get("selected_mode"),
                "runner_up_operator": round_record.get("runner_up_operator"),
                "delta_y": float(round_record.get("delta_y") or 0.0),
                "delta_replay": float(round_record.get("delta_replay") or 0.0),
                "reward": float(round_record.get("reward") or 0.0),
                "rank_before": round_record.get("rank_before"),
                "rank_after": round_record.get("rank_after"),
                "utility_before": round_record.get("utility_before"),
                "utility_after": round_record.get("utility_after"),
                "gold_in_union_before": bool(round_record.get("gold_in_union_before")),
                "gold_in_union_after": bool(round_record.get("gold_in_union_after")),
                "gate_would_reject": bool(round_record.get("gate_would_reject")),
                "gate_rejections_this_round": int(round_record.get("gate_rejections_this_round") or 0),
                "neighborhood_novelty_n": float(round_record.get("neighborhood_novelty_n") or 0.0),
                "D_plus_count": int(round_record.get("D_plus_count") or 0),
                "D_minus_count": int(round_record.get("D_minus_count") or 0),
                "D_question_count": int(round_record.get("D_question_count") or 0),
                "psi_values": [float(value) for value in (round_record.get("psi_values") or [])],
            }
        )

    round1_tags = top_tags(round1_full, DEPTH)
    final_tags = top_tags(final_full, DEPTH)
    return {
        "fact_id": int(record.get("example_idx")),
        "context_id": record.get("context_id"),
        "modality": record.get("input_type"),
        "stream_index": stream_index,
        "gold": gold,
        "gold_set": frozenset(gold),
        "round1_top": round1_tags,
        "final_top": final_tags,
        "round1_r50": 1.0 if first_gold_rank(round1_full, gold) and first_gold_rank(round1_full, gold) <= DEPTH else 0.0,
        "final_r50": 1.0 if first_gold_rank(final_full, gold) and first_gold_rank(final_full, gold) <= DEPTH else 0.0,
        "round1_rank_gold": record.get("round1_rank_gold"),
        "final_rank_gold": record.get("final_rank_gold"),
        "gold_ever_in_union": first_gold_rank(final_full, gold) is not None,
        "membership_changed": 1.0 if set(round1_tags) != set(final_tags) else 0.0,
        "realized_rounds": int(record.get("realized_rounds") or 1),
        "stop_reason": record.get("stop_reason") or "unknown",
        "round_lists": round_lists,
        "rounds": round_rows,
        "parity_ok": bool((record.get("ags_seq_round1_parity") or {}).get("round1_parity_ok", True)),
    }


def gold_hit(tags: frozenset[str], gold_set: frozenset[str]) -> bool:
    return not gold_set.isdisjoint(tags)


# --------------------------------------------------------------------------------------
# 1. Round-1 vs full, and the oracle best prefix
# --------------------------------------------------------------------------------------


def oracle_prefix_r50(fact: dict[str, Any], round_lists: list[list[str]] | None = None) -> float:
    """Best R@50 attainable by stopping the episode after some round, chosen with gold.

    The prefix value is the R@50 of that round's consolidated list, which is what the trace
    stores per round; no re-fusion is performed or needed.
    """
    lists = fact["round_lists"] if round_lists is None else round_lists
    best = fact["round1_r50"]
    if best >= 1.0:
        return best
    gold_set = fact["gold_set"]
    for tags in lists:
        if gold_hit(tags, gold_set):
            return 1.0
    return best


def permutation_null(
    facts: list[dict[str, Any]],
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Null: each fact keeps its own round 1 but borrows every later round's candidate list
    from another fact processed at the same round index in the same arm."""
    observed_adv = [oracle_prefix_r50(fact) - fact["round1_r50"] for fact in facts]
    observed = float(np.mean(observed_adv))

    # Donor pools, one per round index: only facts that actually reached that round.
    by_round: dict[int, list[list[str]]] = defaultdict(list)
    for fact in facts:
        for offset, tags in enumerate(fact["round_lists"]):
            by_round[offset].append(tags)

    # Only facts that missed at round 1 can register any advantage, real or null; the rest
    # contribute a structural zero to both numerator and denominator.
    eligible = [fact for fact in facts if fact["round1_r50"] < 1.0 and fact["round_lists"]]
    draws = np.empty(permutations, dtype=float)
    for iteration in range(permutations):
        hits = 0
        for fact in eligible:
            gold_set = fact["gold_set"]
            for offset in range(len(fact["round_lists"])):
                pool = by_round[offset]
                if not gold_set.isdisjoint(pool[int(rng.integers(0, len(pool)))]):
                    hits += 1
                    break
        draws[iteration] = hits / len(facts)

    return {
        "observed_advantage": round(observed, 6),
        "null_mean": round(float(np.mean(draws)), 6),
        "null_sd": round(float(np.std(draws, ddof=1)), 6) if permutations > 1 else None,
        "null_q0975": round(float(np.quantile(draws, 0.975)), 6),
        "p_value": round(float(np.mean(draws >= observed)), 6),
        "permutations": permutations,
        "facts": len(facts),
    }


# --------------------------------------------------------------------------------------
# 2. Stream AULC
# --------------------------------------------------------------------------------------


def stream_aulc(facts: list[dict[str, Any]], window: int) -> dict[str, Any]:
    ordered = sorted(facts, key=lambda fact: fact["stream_index"])
    series = np.asarray([fact["final_r50"] for fact in ordered], dtype=float)
    if series.size == 0:
        return {}
    effective = min(window, series.size)
    kernel = np.ones(effective) / effective
    rolling = np.convolve(series, kernel, mode="valid")
    thirds = np.array_split(series, 3)
    return {
        "aulc": round(float(np.mean(rolling)), 6),
        "window": effective,
        "curve_points": int(rolling.size),
        "stream_third_1_r50": round(float(np.mean(thirds[0])), 6),
        "stream_third_2_r50": round(float(np.mean(thirds[1])), 6),
        "stream_third_3_r50": round(float(np.mean(thirds[2])), 6),
        "stream_third_3_minus_1": round(float(np.mean(thirds[2]) - np.mean(thirds[0])), 6),
    }


# --------------------------------------------------------------------------------------
# 3. Reward density and behaviour
# --------------------------------------------------------------------------------------


def reward_density(facts: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [round_row["delta_y"] for fact in facts for round_row in fact["rounds"]]
    replays = [round_row["delta_replay"] for fact in facts for round_row in fact["rounds"]]
    rewards = [round_row["reward"] for fact in facts for round_row in fact["rounds"]]
    if not deltas:
        return {}
    delta_array = np.asarray(deltas)
    replay_array = np.asarray(replays)
    return {
        "rounds": len(deltas),
        "delta_y_zero_frac": round(float(np.mean(delta_array == 0.0)), 6),
        "delta_y_near_zero_frac": round(float(np.mean(np.abs(delta_array) < NEAR_ZERO)), 6),
        "delta_y_positive_frac": round(float(np.mean(delta_array > 0.0)), 6),
        "no_gold_episode_frac": round(
            float(np.mean([0.0 if fact["gold_ever_in_union"] else 1.0 for fact in facts])), 6
        ),
        "delta_y_quantiles": {
            str(q): round(float(np.quantile(delta_array, q)), 8) for q in QUANTILES
        },
        "delta_replay_quantiles": {
            str(q): round(float(np.quantile(replay_array, q)), 8) for q in QUANTILES
        },
        "cumulative_mean_reward": round(float(np.mean(rewards)), 8),
        "mean_delta_y": round(float(np.mean(delta_array)), 8),
        "mean_delta_replay": round(float(np.mean(replay_array)), 8),
    }


def behavior(facts: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = [round_row for fact in facts for round_row in fact["rounds"]]
    stop_reasons = Counter(fact["stop_reason"] for fact in facts)
    operators = Counter(round_row["selected_operator"] for round_row in rounds)
    # "Informative" = the round moved utility at all. With a dense reward this would be
    # nearly every round; it is reported per operator so a dead operator is visible.
    informative = Counter(
        round_row["selected_operator"] for round_row in rounds if round_row["delta_y"] != 0.0
    )
    return {
        "mean_realized_rounds": round(float(np.mean([fact["realized_rounds"] for fact in facts])), 6),
        "max_realized_rounds": int(max(fact["realized_rounds"] for fact in facts)),
        "gate_would_reject_rate": round(
            float(np.mean([1.0 if round_row["gate_would_reject"] else 0.0 for round_row in rounds])), 6
        )
        if rounds
        else None,
        "gate_rejections_total": int(sum(round_row["gate_rejections_this_round"] for round_row in rounds)),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "operator_counts": dict(sorted(operators.items())),
        "operator_informative_counts": dict(sorted(informative.items())),
        "operator_informative_rate": {
            operator: round(informative.get(operator, 0) / count, 6)
            for operator, count in sorted(operators.items())
        },
        "mean_neighborhood_novelty": round(
            float(np.mean([round_row["neighborhood_novelty_n"] for round_row in rounds])), 6
        )
        if rounds
        else None,
    }


def posterior_condition_numbers(
    facts: list[dict[str, Any]],
    ridge: float,
    forgetting: float,
    stride: int = 250,
) -> dict[str, Any]:
    """Reconstruct cond(A_o) over the stream.

    Only the END-OF-STREAM condition number per operator is persisted
    (ags_seq_posteriors.json); the trajectory is not. It is recoverable, but the decay is
    easy to get wrong: `_moments` (ags_sequential_arms.py:699-712) weights each stored psi by
    zeta^(S - step_i) where `step` is a GLOBAL counter incremented on EVERY operator's update
    (:684), not a per-operator one. So

        A_o(s) = lambda*I + sum_{step_i < s, op_i = o} zeta^(s - step_i) psi_i psi_i^T

    and the correct recursion decays every operator's matrix at every global step, adding the
    outer product only for the operator actually selected:

        M_o(s+1) = zeta * (M_o(s) + [op_s = o] psi_s psi_s^T)

    Decaying only on an operator's own updates -- the natural-looking recursion -- leaves an
    operator with 344 records carrying 344 steps of decay instead of 7,527, and overstates the
    condition number by two orders of magnitude. `final_matches_persisted` in metrics.json is
    the guard: it compares this reconstruction against ags_seq_posteriors.json.
    """
    ordered = sorted(facts, key=lambda fact: fact["stream_index"])
    events = [
        (round_row["selected_operator"], np.asarray(round_row["psi_values"], dtype=float))
        for fact in ordered
        for round_row in fact["rounds"]
        if round_row["psi_values"]
    ]
    if not events:
        return {}
    dim = events[0][1].size
    operators = sorted({operator for operator, _ in events})
    matrices = {operator: np.zeros((dim, dim)) for operator in operators}
    trajectory: dict[str, list[dict[str, float]]] = defaultdict(list)

    for step, (operator, psi) in enumerate(events):
        matrices[operator] += np.outer(psi, psi)
        for name in operators:
            matrices[name] *= forgetting
        if (step + 1) % stride == 0 or step == len(events) - 1:
            for name in operators:
                trajectory[name].append(
                    {
                        "step": step + 1,
                        "condition_number": round(
                            float(np.linalg.cond(matrices[name] + ridge * np.eye(dim))), 4
                        ),
                    }
                )
    return {
        "steps": len(events),
        "trajectory_stride": stride,
        "final": {
            operator: round(float(np.linalg.cond(matrices[operator] + ridge * np.eye(dim))), 6)
            for operator in operators
        },
        "trajectory": {operator: points for operator, points in sorted(trajectory.items())},
    }


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


def context_bootstrap(
    facts: list[dict[str, Any]],
    value: Any,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Context-level bootstrap CI for a per-fact statistic.

    Resampling is over source contexts, not facts: this benchmark puts ~21 facts under one
    table, and a fact-level bootstrap would treat them as independent and understate the
    interval.
    """
    if not facts:
        return {}
    by_context: dict[Any, list[float]] = defaultdict(list)
    for fact in facts:
        by_context[fact["context_id"]].append(float(value(fact)))
    contexts = list(by_context)
    means = np.asarray([float(np.mean(by_context[context])) for context in contexts])
    sizes = np.asarray([len(by_context[context]) for context in contexts], dtype=float)
    observed = float(np.sum(means * sizes) / np.sum(sizes))

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, len(contexts), size=len(contexts))
        weights = sizes[picks]
        draws[index] = float(np.sum(means[picks] * weights) / np.sum(weights))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": round(observed, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "contexts": len(contexts),
        "facts": len(facts),
    }


def paired_context_bootstrap(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    value: Any,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    if not shared:
        return {}
    paired = [
        {
            "context_id": left[fact_id]["context_id"],
            "diff": float(value(left[fact_id])) - float(value(right[fact_id])),
        }
        for fact_id in shared
    ]
    result = context_bootstrap(paired, lambda row: row["diff"], iterations, seed)
    result["ci_excludes_zero"] = bool(result["ci_low"] > 0 or result["ci_high"] < 0)
    return result


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def subset(facts: list[dict[str, Any]], modality: str) -> list[dict[str, Any]]:
    if modality == "pooled":
        return facts
    return [fact for fact in facts if fact["modality"] == modality]


def table_a_row(arm: str, modality: str, facts: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + len(facts))
    null = permutation_null(facts, args.permutations, rng)
    round1 = context_bootstrap(facts, lambda fact: fact["round1_r50"], args.bootstrap_samples, args.seed)
    full = context_bootstrap(facts, lambda fact: fact["final_r50"], args.bootstrap_samples, args.seed)
    diff = context_bootstrap(
        facts, lambda fact: fact["final_r50"] - fact["round1_r50"], args.bootstrap_samples, args.seed
    )
    churn = context_bootstrap(facts, lambda fact: fact["membership_changed"], args.bootstrap_samples, args.seed)
    return {
        "arm": arm,
        "modality": modality,
        "facts": len(facts),
        "contexts": round1.get("contexts"),
        "round1_r50": round1.get("mean"),
        "full_r50": full.get("mean"),
        "full_minus_round1": diff.get("mean"),
        "diff_ci_low": diff.get("ci_low"),
        "diff_ci_high": diff.get("ci_high"),
        "diff_ci_excludes_zero": bool(diff.get("ci_low", 0) > 0 or diff.get("ci_high", 0) < 0),
        "membership_changed_pct": round(100.0 * churn.get("mean", 0.0), 4),
        "oracle_prefix_advantage": null["observed_advantage"],
        "null_mean": null["null_mean"],
        "null_q0975": null["null_q0975"],
        "p_value": null["p_value"],
        "permutations": null["permutations"],
    }


def table_b_row(arm: str, modality: str, facts: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    curve = stream_aulc(facts, args.window)
    density = reward_density(facts)
    dynamics = behavior(facts)
    return {
        "arm": arm,
        "modality": modality,
        "facts": len(facts),
        "aulc": curve.get("aulc"),
        "aulc_window": curve.get("window"),
        "stream_third_1_r50": curve.get("stream_third_1_r50"),
        "stream_third_2_r50": curve.get("stream_third_2_r50"),
        "stream_third_3_r50": curve.get("stream_third_3_r50"),
        "stream_third_3_minus_1": curve.get("stream_third_3_minus_1"),
        "delta_y_zero_frac": density.get("delta_y_zero_frac"),
        "delta_y_near_zero_frac": density.get("delta_y_near_zero_frac"),
        "no_gold_frac": density.get("no_gold_episode_frac"),
        "mean_rounds": dynamics.get("mean_realized_rounds"),
        "gate_reject_rate_counterfactual": dynamics.get("gate_would_reject_rate"),
        "cumulative_mean_reward": density.get("cumulative_mean_reward"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    reported: dict[str, float] = {}
    for arm, run_dir in ARMS.items():
        trace = args.runs_root / run_dir / "bm25_candidates.jsonl"
        if not trace.exists():
            raise SystemExit(f"Missing trace for {arm}: {trace}")
        facts = []
        for stream_index, record in enumerate(stream_jsonl(trace, args.limit)):
            if not configs.get(arm):
                configs[arm] = record.get("ags_seq_config") or {}
            facts.append(read_fact(record, stream_index))
        arms[arm] = facts
        metrics_path = args.runs_root / run_dir / "metrics.json"
        if metrics_path.exists():
            reported[arm] = json.load(metrics_path.open())["bm25_retrieval"]["recall_at_50"]
        print(f"{arm}: {len(facts)} facts from {trace}", flush=True)

    # Pairing: the two arms must cover the same facts, or nothing below is paired.
    fact_sets = {arm: {fact["fact_id"] for fact in facts} for arm, facts in arms.items()}
    shared = set.intersection(*fact_sets.values())
    pairing = {
        "shared_facts": len(shared),
        "unmatched": {arm: sorted(ids - shared)[:5] for arm, ids in fact_sets.items() if ids - shared},
        "same_stream_order": all(
            [fact["fact_id"] for fact in sorted(facts, key=lambda row: row["stream_index"])]
            == [fact["fact_id"] for fact in sorted(next(iter(arms.values())), key=lambda row: row["stream_index"])]
            for facts in arms.values()
        ),
    }

    table_a: list[dict[str, Any]] = []
    table_b: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for arm, facts in arms.items():
        details[arm] = {"by_modality": {}}
        for modality in ("pooled",) + MODALITIES:
            rows = subset(facts, modality)
            if not rows:
                continue
            table_a.append(table_a_row(arm, modality, rows, args))
            table_b.append(table_b_row(arm, modality, rows, args))
            details[arm]["by_modality"][modality] = {
                "stream_aulc": stream_aulc(rows, args.window),
                "reward_density": reward_density(rows),
                "behavior": behavior(rows),
            }
        config = configs.get(arm, {})
        details[arm]["posterior_condition_numbers"] = posterior_condition_numbers(
            facts,
            ridge=float(config.get("posterior_ridge", 1.0)),
            forgetting=float(config.get("posterior_forgetting", 1.0)),
        )
        details[arm]["config"] = config
        details[arm]["parity_failures"] = sum(1 for fact in facts if not fact["parity_ok"])

        # Guard: the reconstructed end-of-stream cond(A_o) must match the run's own snapshot.
        snapshot_path = next(
            (path for path in (args.runs_root / ARMS[arm]).glob("*posteriors.json")), None
        )
        persisted = {}
        if snapshot_path is not None:
            snapshot = json.load(snapshot_path.open())
            persisted = {
                operator: values.get("condition_number")
                for operator, values in (snapshot.get("operators") or {}).items()
            }
        reconstructed = details[arm]["posterior_condition_numbers"].get("final", {})
        details[arm]["posterior_condition_numbers"]["persisted"] = persisted
        details[arm]["posterior_condition_numbers"]["final_matches_persisted"] = {
            operator: bool(
                operator in persisted
                and persisted[operator] is not None
                and args.limit is None
                and math.isclose(value, persisted[operator], rel_tol=1e-04)
            )
            for operator, value in reconstructed.items()
        }

    # Paired arm contrast on the headline quantity.
    keyed = {arm: {fact["fact_id"]: fact for fact in facts} for arm, facts in arms.items()}
    left, right = list(ARMS)
    contrasts = {
        f"{left} - {right}": {
            "full_r50": paired_context_bootstrap(
                keyed[left], keyed[right], lambda fact: fact["final_r50"], args.bootstrap_samples, args.seed
            ),
            "full_minus_round1": paired_context_bootstrap(
                keyed[left],
                keyed[right],
                lambda fact: fact["final_r50"] - fact["round1_r50"],
                args.bootstrap_samples,
                args.seed,
            ),
        }
    }

    # The retrieval-stage full-episode R@50 must reproduce each run's own metrics.json.
    checks = {}
    for arm, facts in arms.items():
        computed = float(np.mean([fact["final_r50"] for fact in facts]))
        expected = reported.get(arm)
        checks[arm] = {
            "computed_full_r50": round(computed, 6),
            "metrics_json_recall_at_50": expected,
            "matches": bool(expected is not None and args.limit is None and math.isclose(computed, expected, abs_tol=1e-06)),
        }

    write_csv(args.output_dir / "seq_diag_A.csv", table_a)
    write_csv(args.output_dir / "seq_diag_B.csv", table_b)
    metrics = {
        "arms_present": list(ARMS),
        "perturbation_only_arm": (
            "NOT AVAILABLE. ags_sequential_arms.py implements two arms only (ags_seq, "
            "ags_seq_random); PERTURB is an operator in the slate, not an arm. The logged "
            "counterfactual replay is the runner-up directive for a single round and cannot be "
            "chained into an episode. A three-arm version (bandit/random/resample_only) exists "
            "only in runs_ags_reward_diagnostic/qwen3_32b, which is 250 tabular-only facts over "
            "14 source contexts."
        ),
        "pairing": pairing,
        "checks": checks,
        "contrasts": contrasts,
        "per_arm": details,
        "table_a": table_a,
        "table_b": table_b,
        "settings": {
            "depth": DEPTH,
            "near_zero_tolerance": NEAR_ZERO,
            "permutations": args.permutations,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "source context (context_id)",
            "window": args.window,
            "seed": args.seed,
            "limit": args.limit,
            "stage": "retrieval (bm25_retrieval); the listwise reranker runs once over the final pool and has no per-round analogue",
            "novelty_gate": "disabled in the run; gate_reject_rate is counterfactual",
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    for row in table_a:
        if row["modality"] == "pooled":
            print(
                f"  A {row['arm']:22s} rd1={row['round1_r50']:.4f} full={row['full_r50']:.4f} "
                f"diff={row['full_minus_round1']:+.4f} churn={row['membership_changed_pct']:.1f}% "
                f"oracle={row['oracle_prefix_advantage']:.4f} null={row['null_mean']:.4f} p={row['p_value']:.3f}",
                flush=True,
            )
    for row in table_b:
        if row["modality"] == "pooled":
            print(
                f"  B {row['arm']:22s} aulc={row['aulc']:.4f} T3={row['stream_third_3_r50']:.4f} "
                f"dy0={row['delta_y_zero_frac']:.4f} nogold={row['no_gold_frac']:.4f} "
                f"rounds={row['mean_rounds']:.2f} gate={row['gate_reject_rate_counterfactual']:.4f}",
                flush=True,
            )
    print(json.dumps(checks, indent=2), flush=True)
    print(f"\nWrote {args.output_dir}/seq_diag_A.csv, seq_diag_B.csv, metrics.json", flush=True)


if __name__ == "__main__":
    main()
