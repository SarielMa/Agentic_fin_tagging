#!/usr/bin/env python3
"""Orchestrator for everything in Table 5 that needs no GPU (ags_table5_ablation_spec.md
section 5, steps 1-5): the dev beta sweep, every offline test row (including the freetext
equality assertion), and the w_cov=0 index ablation panel. Run this before spending any GPU
time on the LLM verifier (step 6, run_llm_verifier.py) -- a logging or replay problem here
should surface first, cheaply.

    1. verify section 0 logging is complete    (already done for this build; see
                                                 data_prep.dev_replay_sanity_check, run
                                                 inside run_beta_sweep.py)
    2. dev beta sweeps                          run_beta_sweep.py
    3. all OFFLINE rows                         run_test_rows.py
    4. assert 3.3 == Table 2 parallel sampling   inside run_test_rows.py
    5. 3.10 label coverage off, AGS + 2 baselines run_index_ablation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--skip-beta-sweep", action="store_true", help="Reuse an existing selected_betas.json.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if not args.skip_beta_sweep:
        run([python, str(HERE / "run_beta_sweep.py"), "--output-dir", str(args.output_dir)])
    else:
        print("Skipping beta sweep; reusing existing selected_betas.json", flush=True)

    test_rows_cmd = [
        python,
        str(HERE / "run_test_rows.py"),
        "--output-dir",
        str(args.output_dir),
        "--selected-betas",
        str(args.output_dir / "selected_betas.json"),
    ]
    if args.limit is not None:
        test_rows_cmd += ["--limit", str(args.limit)]
    verdicts_path = args.output_dir / "llm_verifier_verdicts.json"
    if verdicts_path.exists():
        test_rows_cmd += ["--llm-verifier-verdicts", str(verdicts_path)]
    run(test_rows_cmd)

    index_cmd = [python, str(HERE / "run_index_ablation.py"), "--output-dir", str(args.output_dir)]
    if args.limit is not None:
        index_cmd += ["--limit", str(args.limit)]
    run(index_cmd)

    print(f"\nAll offline rows done. Outputs in {args.output_dir}", flush=True)
    print("Remaining: run_llm_verifier.py (GPU, ~5k calls), then re-run run_test_rows.py to fold row 3.9 in.", flush=True)


if __name__ == "__main__":
    main()
