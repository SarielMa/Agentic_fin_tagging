#!/usr/bin/env python3
"""T7: the efficiency table (comparing_methods/ags_t7_t28_spec.md).

Offline instrumentation of runs that already exist -- no generation, no retrieval, no GPU.
Reads each method's trace, writes efficiency.csv and metrics.json.

The claim this table supports (paper section 4.6): AGS spends J generation calls and no model
call after that, so its LLM-call count is low and flat while the sequential and iterative
methods scale with rounds. metrics.json therefore asserts AGS's measured LLM calls fall in
[J, J + tolerance] and flags any post-generation model call, per the spec's Output block.

See efficiency/efficiency.py's module docstring for the four logging complications this has to
work around (two trace schemas, AGS-Seq's replay-conflated totals, the uncounted hypothesis
retry, and the shared listwise-rerank call that lives outside every method's counter).

    python efficiency/run_t7_efficiency.py                    # full table
    python efficiency/run_t7_efficiency.py --limit 50         # smoke test
    python efficiency/run_t7_efficiency.py --report-wallclock # opt in to the column
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from efficiency.efficiency import (  # noqa: E402
    DEFAULT_RUNS_DIR,
    HYPOTHESIS_CALL_KIND,
    METHODS,
    diagnostics,
    load_tokenizer,
    read_method,
    summarize,
)

AGS_LABEL = "AGS"
AGS_J = 2  # frozen config: FrozenAgsConfig.hypotheses (ags_frozen_grounding.py:75)
# One extra invocation per hypothesis is the most the uncounted retry can add
# (sample_hypotheses retries exactly once), so J=2 can legitimately measure up to 4.
AGS_CALL_TOLERANCE = 0.25

DEFAULT_HARDWARE_NOTE = "single B200 GPU, vLLM backend, Qwen3-32B"
DEFAULT_BATCH_NOTE = (
    "per-fact single-prompt generation except one-pass grounding, which batches 32 prompts "
    "per vLLM call; wall-clock therefore not comparable across rows"
)

CSV_COLUMNS = (
    "method",
    "n_facts",
    "llm_calls_per_fact",
    "retrieval_calls_per_fact",
    "completion_tokens_per_fact",
    "wallclock_sec_per_fact",
    "rerank_llm_calls_per_fact",
    "replay_llm_calls_per_fact",
    "replay_completion_tokens_per_fact",
    "llm_calls_per_fact_lower_bound",
    "llm_calls_per_fact_upper_bound",
    "hardware_note",
    "batch_note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_t7_t28" / "qwen3_32b")
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-32B",
        help="Generation-time tokenizer, used to recover completion tokens for the two legacy "
        "traces whose logs stored raw_output but no token count.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer loading; one-pass grounding's completion tokens are then unavailable.",
    )
    parser.add_argument(
        "--report-wallclock",
        action="store_true",
        help="Include the wall-clock column. Off by default: the six methods were not timed "
        "under identical conditions, which is exactly the case where the spec says to drop it.",
    )
    parser.add_argument("--hardware-note", default=DEFAULT_HARDWARE_NOTE)
    parser.add_argument("--batch-note", default=DEFAULT_BATCH_NOTE)
    parser.add_argument("--limit", type=int, default=None, help="Facts per method, for smoke tests.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ags_assertions(summary: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    """The spec's Output block: assert AGS is within [J, J + small tolerance], and flag any
    post-generation model call."""
    measured = summary.get("llm_calls_per_fact")
    upper = summary.get("llm_calls_per_fact_upper_bound")
    within = measured is not None and AGS_J <= measured <= AGS_J + AGS_CALL_TOLERANCE

    # Anything other than hypothesis sampling in AGS's own call log would be a model call after
    # generation, contradicting section 4.6.
    unexpected = sorted(kind for kind in diag.get("call_kinds", {}) if kind != HYPOTHESIS_CALL_KIND)
    rerank_calls = float(summary.get("rerank_llm_calls_per_fact") or 0.0)

    return {
        "J": AGS_J,
        "tolerance": AGS_CALL_TOLERANCE,
        "measured_llm_calls_per_fact": measured,
        "within_J_plus_tolerance": bool(within),
        "retry_adjusted_upper_bound": upper,
        "unexpected_post_generation_call_kinds": unexpected,
        "post_generation_model_call_detected": bool(unexpected) or rerank_calls > 0,
        "shared_rerank_calls_per_fact": rerank_calls,
        "shared_rerank_note": (
            "All six runs set RUN_RERANK=1, so each fact also passes through one shared listwise "
            "tag-selection LLM call logged in qwen_rerank_predictions.jsonl. That call is applied "
            "identically to every row and is NOT part of any method's total_llm_calls, so it does "
            "not affect the cross-method comparison -- but it IS a post-generation model call. "
            "Section 4.6's 'no model call after generation' claim describes AGS's own ranking "
            "(the metrics.json 'bm25_retrieval' block); if the paper's headline AGS number is the "
            "'qwen_reranked' block instead, the claim needs this +1 shared call stated explicitly."
        ),
    }


def main() -> None:
    args = parse_args()
    tokenizer = None if args.no_tokenizer else load_tokenizer(args.tokenizer_model)

    summaries: list[dict[str, Any]] = []
    all_diagnostics: dict[str, Any] = {}
    incomplete: list[str] = []

    for spec in METHODS:
        print(f"Reading {spec.label} ({spec.run_dir}) ...", flush=True)
        costs = read_method(spec, runs_dir=args.runs_dir, tokenizer=tokenizer, limit=args.limit)
        if not costs.rows:
            print(f"  SKIPPED: no facts read ({'; '.join(costs.notes) or 'no trace'})", flush=True)
            all_diagnostics[spec.label] = diagnostics(costs)
            continue
        summary = summarize(costs, args.report_wallclock, args.hardware_note, args.batch_note)
        summaries.append(summary)
        all_diagnostics[spec.label] = diagnostics(costs)
        if not costs.run_complete:
            incomplete.append(spec.label)
        print(
            f"  n={summary['n_facts']} llm={summary['llm_calls_per_fact']} "
            f"retrieval={summary['retrieval_calls_per_fact']} "
            f"tokens={summary['completion_tokens_per_fact']}",
            flush=True,
        )

    ags_summary = next((row for row in summaries if row["method"] == AGS_LABEL), None)
    assertions: dict[str, Any] = {}
    if ags_summary is not None:
        assertions["ags"] = ags_assertions(ags_summary, all_diagnostics.get(AGS_LABEL, {}))
        if not assertions["ags"]["within_J_plus_tolerance"]:
            print(
                f"\nWARNING: AGS measured {assertions['ags']['measured_llm_calls_per_fact']} LLM "
                f"calls/fact, outside [J={AGS_J}, J+{AGS_CALL_TOLERANCE}]. The spec says to "
                "investigate before reporting: something after generation may be calling the model.",
                flush=True,
            )

    metrics = {
        "assertions": assertions,
        "methods": all_diagnostics,
        "wallclock_column_reported": bool(args.report_wallclock),
        "wallclock_policy": (
            "Reported on request." if args.report_wallclock else
            "Dropped (spec option b). Only four of six methods logged a per-fact timer at all, "
            "and one-pass grounding's generation is batched 32-at-a-time with no per-fact timing, "
            "so the six rows were not timed under identical conditions. Measured values for the "
            "methods that do have them are kept under methods.*.measured_wallclock_sec_per_fact "
            "and must not be read as a cross-method comparison."
        ),
        "token_column_basis": (
            "Completion tokens only, excluding prompt tokens, per the spec ('prompt tokens are "
            "dominated by the context and would swamp the signal'). Counted with the generation "
            "model's own tokenizer, not estimated from characters."
        ),
        "incomplete_runs": incomplete,
        "limit": args.limit,
        "artifact_paths": {
            "efficiency_csv": str(args.output_dir / "efficiency.csv"),
            "runs_dir": str(args.runs_dir),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "efficiency.csv", summaries)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if incomplete:
        print(
            f"\nWARNING: still-running or partial runs included: {', '.join(incomplete)}. "
            "Re-run this script once they finish.",
            flush=True,
        )
    print(json.dumps(assertions, indent=2, sort_keys=True), flush=True)
    print(f"\nWrote {args.output_dir / 'efficiency.csv'} and metrics.json", flush=True)


if __name__ == "__main__":
    main()
