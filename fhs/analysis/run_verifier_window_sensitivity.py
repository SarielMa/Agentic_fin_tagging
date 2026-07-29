#!/usr/bin/env python3
"""tab:llm_window_sensitivity: the K_v rows that come from real generation runs.

run_ags_verifier_ablation.py already has a --sensitivity-top-m path, but that one TRUNCATES
the logged K_v=10 call windows, and a verifier shown five candidates writes a different prompt
than one shown ten and cut to five. It names its rows "Hybrid AGS (M=k)" for exactly that
reason; build_verifier_ablation_table.py only ever reads "Hybrid AGS (K_v=k)", so the
truncated diagnostic cannot reach the paper table even by accident. This script produces the
K_v= rows, and it produces them only from a verifier that was actually run at that window.

Each window is its own generation run on GPU, writing
    <run-dir>/verdicts_m<K>/llm_verifier_verdicts.json
    <run-dir>/verdicts_m<K>/llm_verifier_calls.jsonl
    <run-dir>/verdicts_m<K>/llm_verifier_summary.json
A window with no summary.json has not finished generating and is SKIPPED, not estimated -- the
calls file is streamed as the job runs, so its presence alone proves nothing.

The K_v=10 row is not computed here. It is the deployed configuration and already sits in
verifier_ablation.csv as "Hybrid AGS (full)"; the table builder reads it from there. What this
script needs K_v=10 for is the paired-bootstrap baseline, and it rehydrates that from
<run-dir>/per_fact/hybrid_ags_full.jsonl rather than re-scoring 2,509 facts for 13 minutes to
land on numbers that are already on disk. Every metric is a function of the gold rank
(core.py:463-473: rank is None means a miss), so the rehydration is exact, not an
approximation -- it reproduces the recorded pooled row to all six printed digits.

Existing rows in verifier_window_sensitivity.csv for windows NOT regenerated in this run are
carried through, so this can be re-run as each generation job lands.

CPU only.
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
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import TOP_KS, AblationConfig, aggregate, evaluate, reset_consensus_cache  # noqa: E402
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts  # noqa: E402
from run_ags_verifier_ablation import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_OUTPUT_DIR,
    METRICS,
    bootstrap_rows,
    load_call_windows,
    write_csv,
)

# The windows the paper reports either side of the deployed K_v=10.
DEFAULT_WINDOWS = (5, 20)


def variant_name(top_m: int) -> str:
    return f"Hybrid AGS (K_v={top_m})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        metavar="K[=DIR]",
        help="Window to evaluate, e.g. --window 5 or --window 5=/path/to/verdicts_m5. "
        f"Repeatable. Defaults to {' and '.join(str(k) for k in DEFAULT_WINDOWS)}, each read "
        "from <run-dir>/verdicts_m<K>/.",
    )
    parser.add_argument(
        "--baseline-per-fact",
        type=Path,
        default=None,
        help="Per-fact ranks of the K_v=10 hybrid arm, used only as the paired-bootstrap "
        "baseline. Defaults to <run-dir>/per_fact/hybrid_ags_full.jsonl.",
    )
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-parse-rate",
        type=float,
        default=0.95,
        help="Refuse a generation run whose responses did not parse this cleanly. A low parse "
        "rate looks like verifier abstention in the metrics but is a truncation bug; see the "
        "note llm_verifier_summary.json carries.",
    )
    return parser.parse_args()


def resolve_windows(args: argparse.Namespace) -> list[tuple[int, Path]]:
    specs = args.window if args.window else [str(k) for k in DEFAULT_WINDOWS]
    out: list[tuple[int, Path]] = []
    for spec in specs:
        key, _, path = spec.partition("=")
        try:
            top_m = int(key)
        except ValueError:
            raise SystemExit(f"--window expects K or K=DIR, got {spec!r}")
        out.append((top_m, Path(path) if path else args.run_dir / f"verdicts_m{top_m}"))
    return sorted(out)


def rehydrate_baseline(path: Path) -> list[dict[str, Any]]:
    """Per-fact ranks -> the metric-bearing rows bootstrap_rows and aggregate expect.

    Mirrors core.py's evaluate() exactly: a None rank is a miss, so it scores 0 on every
    recall depth and contributes 0.0 to MRR rather than being dropped from the mean.
    """
    if not path.exists():
        raise SystemExit(
            f"baseline per-fact ranks not found: {path}\n"
            "Run run_ags_verifier_ablation.py first -- it writes per_fact/ for every arm."
        )
    rows: list[dict[str, Any]] = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        rank = record["rank"]
        row: dict[str, Any] = {
            "fact_id": record["fact_id"],
            "context_id": record["context_id"],
            "modality": record["modality"],
            "rank": rank,
            "mrr": 0.0 if rank is None else 1.0 / rank,
            "top1_accuracy": bool(rank == 1),
        }
        for k in TOP_KS:
            row[f"recall_at_{k}"] = bool(rank is not None and rank <= k)
        rows.append(row)
    if not rows:
        raise SystemExit(f"baseline per-fact file is empty: {path}")
    return rows


def load_generation_run(top_m: int, directory: Path, min_parse_rate: float) -> dict[str, Any] | None:
    """Verdicts + calls for one window, or None if that run has not finished."""
    summary_path = directory / "llm_verifier_summary.json"
    verdicts_path = directory / "llm_verifier_verdicts.json"
    calls_path = directory / "llm_verifier_calls.jsonl"

    if not summary_path.exists():
        state = "directory absent" if not directory.exists() else "still generating (no summary yet)"
        print(f"  K_v={top_m:<3} SKIPPED: {state} -- {directory}", flush=True)
        return None
    if not verdicts_path.exists():
        print(f"  K_v={top_m:<3} SKIPPED: summary present but no verdicts.json -- {directory}", flush=True)
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Guard against a directory named for one window holding another window's run: every
    # number downstream would be attributed to the wrong K_v and nothing would look wrong.
    generated = summary.get("top_m")
    if generated != top_m:
        raise SystemExit(f"{summary_path} was generated at top_m={generated}, not {top_m}")
    parse_rate = float(summary.get("parse_rate", 0.0))
    if parse_rate < min_parse_rate:
        raise SystemExit(
            f"K_v={top_m}: parse_rate {parse_rate:.3f} < {min_parse_rate}. Truncated responses "
            f"read as verifier abstention and would understate this window. See {summary_path} "
            "-- raise --query-max-new-tokens and regenerate."
        )

    verdicts = load_llm_verifier_verdicts(verdicts_path)
    if not verdicts:
        raise SystemExit(f"no verdicts loaded from {verdicts_path}")
    _, call_cost = load_call_windows(calls_path)
    return {"verdicts": verdicts, "call_cost": call_cost, "summary": summary, "dir": directory}


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    windows = resolve_windows(args)
    baseline_path = args.baseline_per_fact or args.run_dir / "per_fact" / "hybrid_ags_full.jsonl"

    normalization_map = load_normalization_map(args.normalization_map)
    baseline_rows = rehydrate_baseline(baseline_path)
    baseline_agg = aggregate(baseline_rows)
    print(f"baseline K_v=10 from {baseline_path.name}: {len(baseline_rows)} facts", flush=True)
    print(f"  recall@10 {baseline_agg['recall_at_10']:.6f}  mrr {baseline_agg['mrr']:.6f}  "
          f"top1 {baseline_agg['top1_accuracy']:.6f}", flush=True)

    facts = list(load_test_facts(args.test_trace).values())
    if args.limit:
        facts = facts[: args.limit]
    print(f"test facts: {len(facts)}", flush=True)
    if args.limit is None and len(facts) != len(baseline_rows):
        raise SystemExit(
            f"baseline has {len(baseline_rows)} facts but the trace has {len(facts)}. The "
            "paired bootstrap needs both arms over the same fact set; regenerate the baseline."
        )

    print("\n=== candidate-window sensitivity (real generation runs) ===", flush=True)
    fresh: list[dict[str, Any]] = []
    done: list[int] = []
    pending: list[int] = []
    for top_m, directory in windows:
        run = load_generation_run(top_m, directory, args.min_parse_rate)
        if run is None:
            pending.append(top_m)
            continue

        config = AblationConfig(
            name=variant_name(top_m),
            beta=0.6,
            verifier_mode="hybrid",
            # Same deployed settings as the arms in run_ags_verifier_ablation.py, so the only
            # thing that differs between this row and the K_v=10 row is the window.
            truncate_pool_to_top_k=True,
            llm_unjudged_fill="mean",
            llm_verifier_top_m=top_m,
            llm_verifier_verdicts=run["verdicts"],
        )
        reset_consensus_cache()
        started = time.time()
        rows = [evaluate(fact, config, normalization_map) for fact in facts]
        elapsed = time.time() - started
        print(f"  K_v={top_m:<3} {elapsed:7.1f}s  verdicts {len(run['verdicts'])}  "
              f"calls {run['call_cost']['calls']}", flush=True)

        fresh.extend(
            bootstrap_rows(
                variant_name(top_m),
                rows,
                baseline_rows,
                args.bootstrap_samples,
                args.seed,
                extra={
                    "top_m": top_m,
                    "source": "generation_run",
                    "verdicts_dir": str(directory),
                    "parse_rate": run["summary"].get("parse_rate"),
                    "verifier_llm_calls": run["call_cost"]["calls"],
                },
            )
        )
        done.append(top_m)

    # Carry through windows this invocation did not regenerate, so landing K_v=20 later does
    # not drop the K_v=5 row that is already correct.
    out_path = args.run_dir / "verifier_window_sensitivity.csv"
    kept: list[dict[str, Any]] = []
    if out_path.exists() and out_path.stat().st_size > 0:
        import csv as _csv

        regenerated = {variant_name(k) for k in done}
        with out_path.open(encoding="utf-8", newline="") as handle:
            for row in _csv.DictReader(handle):
                if row.get("variant") not in regenerated:
                    kept.append(row)
        if kept:
            carried = sorted({row["variant"] for row in kept})
            print(f"\ncarrying through {len(kept)} existing rows: {', '.join(carried)}", flush=True)

    write_csv(out_path, kept + fresh)
    print(f"\nwrote {out_path} ({len(kept) + len(fresh)} rows)", flush=True)

    if fresh:
        print("\n=== pooled, retrieval stage ===", flush=True)
        for row in fresh:
            if row["modality"] == "pooled" and row["metric"] in METRICS:
                flag = " *" if str(row.get("ci_excludes_zero", "")).lower() == "true" else ""
                print(f"  {row['variant']:<22} {row['metric']:<15} {row['value']:.6f}"
                      f"  delta {row['delta_vs_full']:+.6f}{flag}", flush=True)

    if pending:
        print(f"\nstill pending (no finished generation run): "
              f"{', '.join(f'K_v={k}' for k in pending)}", flush=True)
    print("\nRebuild the table with: python build_verifier_ablation_table.py "
          f"--run-dir {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
