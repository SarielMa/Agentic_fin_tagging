#!/usr/bin/env python3
"""Fill tab:end_to_end's "AGS (hybrid verification)" Full-pipeline cells.

Same scope and same estimator as the nine rows already in that column: accuracy is
correct / n_gold_entities (a gold entity the extractor never produced counts as wrong), and
std is a context-level bootstrap, 2,000 resamples, seed 20260724.

The point estimate is recomputed from the predictions and checked against the run's own
fulltagging_metrics.json before anything is written -- if the two disagree the scope has
drifted and this refuses rather than reporting an interval around a number it did not compute.

CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402

TEST = REPO / "FinTagging_800_200_grounding_test_JSON" / "data" / "test.jsonl"
# Row label history, because this script fails SILENTLY when it drifts -- a non-matching label
# just prints "nothing to fill (row already populated)" and exits 0:
#   "AGS (hybrid verification)"  while the paper carried the hybrid framing
#   "AGS (full)"                 after the rewrite around the LLM-only result
#   "FHS (full)"                 after 2026-07-29, when the method was renamed to Factorized
#                                Hypothesis Search to match the paper's title (AGS appeared 64
#                                times and was never expanded anywhere).
# It is tab:ablation's own bold baseline row: the llm_drop / fused-window arm. The measurement
# that fills it is apply_server_fulltagging_llmonly_rerank.sh.
LABEL = r"\textbf{FHS (full)}"
SEED, SAMPLES = 20260724, 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--paper", type=Path, required=True)
    p.add_argument("--test-jsonl", type=Path, default=TEST)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads((args.run_dir / "fulltagging_metrics.json").read_text(encoding="utf-8"))
    published = metrics["rerank_gold_entity_scope"]["accuracy"]
    n_gold_total = metrics["n_gold_entities"]
    r50 = metrics["rerank_gold_entity_scope"].get("recall_at_50")

    gold: dict[int, int] = defaultdict(int)
    for line in args.test_jsonl.open(encoding="utf-8"):
        rec = json.loads(line)
        gold[int(rec["context_id"])] += int(rec["ground_truth_count"])

    correct: dict[int, int] = defaultdict(int)
    with (args.run_dir / "qwen_rerank_predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            tags = {normalize_tag(t) for t in (rec.get("gold_tags") or [])}
            sel = rec.get("selected_tag")
            if sel and normalize_tag(sel) in tags:
                correct[int(rec["context_id"])] += 1

    acc = sum(correct.values()) / n_gold_total
    if abs(acc - published) > 1e-6:
        raise SystemExit(f"recomputed {acc:.6f} != published {published:.6f}; scope drifted, refusing to write")

    ctx = sorted(gold)
    num = np.array([correct.get(c, 0) for c in ctx], dtype=float)
    den = np.array([gold[c] for c in ctx], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ctx), size=(SAMPLES, len(ctx)))
    std = (num[idx].sum(axis=1) / den[idx].sum(axis=1)).std(ddof=1)

    text = args.paper.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Scope to tab:end_to_end. The same row label also appears in tab:main_results, whose
    # last column is an em-dash "---" -- a substring test for "--" matches it and silently
    # rewrites that table instead. Bounding the search by the label is the only safe way.
    end = next((i for i, l in enumerate(lines) if r"\label{tab:end_to_end}" in l), None)
    if end is None:
        raise SystemExit(f"{args.paper}: no \\label{{tab:end_to_end}} found")
    start = next(i for i in range(end, -1, -1) if lines[i].lstrip().startswith(r"\begin{table"))

    target = [
        i
        for i in range(start, end)
        if lines[i].strip().startswith(LABEL)
        and lines[i].rstrip().endswith(r"\\")
        and any(c.strip() == "--" for c in lines[i].rstrip().removesuffix(r"\\").split("&"))
    ]
    if not target:
        print("nothing to fill (row already populated)")
        return
    i = target[0]
    parts = [p.strip() for p in lines[i].rstrip().removesuffix(r"\\").split("&")]
    parts[-3] = f"\\textbf{{{r50:.3f}}}" if r50 is not None else parts[-3]
    parts[-2] = f"\\textbf{{{acc:.3f}}}"
    parts[-1] = f"\\textbf{{{std:.3f}}}"
    lines[i] = " & ".join(parts) + r" \\"
    args.paper.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    print(f"filled: acc={acc:.6f} (matches published) std={std:.4f}")
    print(lines[i])


if __name__ == "__main__":
    main()
