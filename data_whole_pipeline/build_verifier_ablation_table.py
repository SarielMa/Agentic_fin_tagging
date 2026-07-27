#!/usr/bin/env python3
"""Assemble the verifier LaTeX row bodies from whatever has finished.

Emits row bodies matching the three tables that are currently blank in acl_latex.tex, with
their exact row labels and column orders, so each fragment is a paste between \\midrule and
\\bottomrule:

    tab:ablation (first block)              R@1 R@10 R@50 R@200 MRR Acc(reranked)
    tab:verifier_reranker_interaction       MRR Final-Acc
    tab:llm_window_sensitivity              R@10 R@50 R@200 MRR R@1  (retrieval stage only)

Cells read -- when the number does not exist yet (a GPU rerank job that has not landed, or a
window whose generation run has not finished). Rerun as jobs complete; it is idempotent.

WHY THE INTERACTION TABLE HAS NO RECALL COLUMN
    The listwise reranker reorders the existing top 20 and leaves everything below it in
    place, so recall at every depth is fixed for a given verifier setting. Reporting it would
    show two identical pairs and invite the reading that the reranker was measured on recall.

CPU only, seconds to run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs_ags_verifier_ablation" / "qwen3_32b"

# (latex label, csv variant, rerank-dir suffix, is-baseline)
ABLATION_ROWS: tuple[tuple[str, str, str, bool], ...] = (
    (r"\textbf{AGS (hybrid verification; full)}", "Hybrid AGS (full)", "hybrid_full", True),
    (r"$-$ candidate-level LLM verifier (deterministic core)", "- LLM verifier", "no_llm", False),
    (r"$-$ deterministic consensus verifier", "- deterministic verifier", "no_determ", False),
    (r"LLM verifier only", "LLM verifier only", "llm_only", False),
    (r"No verifier (fused retrieval score only)", "- both verifiers", "no_verifier", False),
)
# R@1 leftmost, then depth, then MRR; final reranked accuracy is appended as the last column
# so the row reads left-to-right from the arm's own ranking to what the deployed pipeline
# ultimately selects -- the same quantity Table 2 reports.
ABLATION_METRICS = ("top1_accuracy", "recall_at_10", "recall_at_50", "recall_at_200", "mrr")
WINDOW_METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")
BASELINE = "Hybrid AGS (full)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--modality", default="pooled")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: Any, digits: int = 3, dagger: bool = False, bold: bool = False) -> str:
    if value is None or value == "":
        return "--"
    text = f"{float(value):.{digits}f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    return text + (r"$^{\dagger}$" if dagger else "")


def reranked(run_dir: Path, arm: str | None) -> dict[str, Any]:
    """The qwen_reranked block for an arm, or {} if its GPU job has not landed."""
    if arm is None:
        return {}
    path = run_dir / f"rerank_{arm}" / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("qwen_reranked", {})


def indexed(rows: list[dict[str, str]], modality: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["variant"], row["metric"]): row for row in rows if row["modality"] == modality}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    ablation = indexed(read_csv(run_dir / "verifier_ablation.csv"), args.modality)
    window = indexed(read_csv(run_dir / "verifier_window_sensitivity.csv"), args.modality)
    cost = {row["variant"]: row for row in read_csv(run_dir / "verifier_ablation_cost.csv")}
    fragments: dict[str, list[str]] = {}

    # ---- tab:ablation, first block ----------------------------------------------------------
    lines = []
    for label, variant, arm, is_baseline in ABLATION_ROWS:
        cells = []
        for metric in ABLATION_METRICS:
            row = ablation.get((variant, metric))
            if row is None:
                cells.append("--")
                continue
            flag = (not is_baseline) and row.get("ci_excludes_zero", "").lower() == "true"
            cells.append(num(row["value"], dagger=flag, bold=is_baseline))
        cells.append(num(reranked(run_dir, arm).get("accuracy"), bold=is_baseline))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    fragments["table_ablation_block.tex"] = lines

    # ---- tab:verifier_reranker_interaction --------------------------------------------------
    # Recall is deliberately absent: the reranker reorders the existing top 20 and leaves the
    # rest in place, so recall at every depth is fixed for a given verifier setting and only
    # MRR and top-1 can move.
    lines = []
    for llm_state, variant, arm in (("Off", "- LLM verifier", "no_llm"), ("On ", BASELINE, "hybrid_full")):
        retrieval = {metric: ablation.get((variant, metric)) for metric in ("mrr", "top1_accuracy")}
        for rerank_state in ("Off", "On "):
            if rerank_state.strip() == "Off":
                mrr = retrieval["mrr"]["value"] if retrieval["mrr"] else None
                acc = retrieval["top1_accuracy"]["value"] if retrieval["top1_accuracy"] else None
            else:
                block = reranked(run_dir, arm)
                mrr = block.get("mrr")
                acc = block.get("accuracy")
            lines.append(f"{llm_state} & {rerank_state} & {num(mrr)} & {num(acc)} " + r"\\")
    # Emit in the table's own row order: Off/Off, On/Off, Off/On, On/On.
    order = [0, 2, 1, 3]
    fragments["table_interaction.tex"] = [lines[i] for i in order]

    # ---- tab:llm_window_sensitivity ----------------------------------------------------------
    # Retrieval stage only, and no cost columns: each window is its own generation run, so a
    # row exists only once that run has produced verdicts. Nothing here is derived by
    # truncating a larger window.
    lines = []
    for top_m in (5, 10, 20):
        if top_m == 10:
            source, variant = ablation, BASELINE
        else:
            source, variant = window, f"Hybrid AGS (K_v={top_m})"
        cells = []
        for metric in WINDOW_METRICS:
            row = source.get((variant, metric))
            if row is None:
                cells.append("--")
                continue
            flag = top_m != 10 and row.get("ci_excludes_zero", "").lower() == "true"
            cells.append(num(row["value"], dagger=flag))
        lines.append(f"{top_m:<2} & " + " & ".join(cells) + r" \\")
    fragments["table_window_sensitivity.tex"] = lines

    for name, lines in fragments.items():
        (run_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n=== {name} ===")
        print("\n".join(lines))

    pending = [arm for _, _, arm, _ in ABLATION_ROWS if not reranked(run_dir, arm)]
    if pending:
        print(f"\nreranked accuracy still pending: {', '.join(pending)}")


if __name__ == "__main__":
    main()
