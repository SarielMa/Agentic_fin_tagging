#!/usr/bin/env python3
"""Assemble the verifier LaTeX row bodies from whatever has finished.

Emits row bodies matching the four tables in appendix_verifier_ablation.tex, with their exact
row labels and column orders, so each fragment is a paste between \\midrule and \\bottomrule:

    tab:ablation (first block)              R@1 R@10 MRR | Acc(reranked) std
    tab:verifier_reranker_interaction       MRR Final-Acc
    tab:llm_window_sensitivity              R@10 R@50 R@200 MRR R@1  (retrieval stage only)
    tab:verifiercost                        LLM calls, completion tokens, scoring seconds

Cells read -- when the number does not exist yet (a GPU rerank job that has not landed, or a
window whose generation run has not finished). Rerun as jobs complete; it is idempotent.

--appendix ALSO WRITES THE ROWS INTO THE PAPER FILE
    Without it this script only refreshes fragments under the run directory, and the appendix
    keeps whatever rows were last pasted into it by hand -- which is how it sat with four
    unfilled `%% >>> ...` markers while every fragment on disk was current. With it, the lines
    between each marker and the following \\bottomrule are replaced by that fragment. The
    marker line stays, so this is idempotent and can be re-run as each job lands.

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

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs_ags_verifier_ablation" / "qwen3_32b"

# (latex label, csv variant, rerank-dir suffix, is-baseline)
# "LLM verifier only" is deliberately NOT a row here. It and "- deterministic consensus
# verifier" are the same ablation: core.py:403-411 sends both to llm_only_consensus_scores with
# identical pool, hypotheses, verdicts, dimensions and unjudged_fill, differing in exactly one
# argument -- abstention="drop" vs "negative". Neither computes a deterministic term. Listing
# both under "Verification architecture" implied a component difference that does not exist,
# and since "negative" scores higher, it invited reading an abstention prior as a component
# effect. The architecture row is "drop", the neutral removal: hybrid falls back to the
# symbolic verdict on an abstention (core.py:167), and with that term deleted there is nothing
# to fall back to, so averaging over the dimensions actually ruled on is what removal means.
# "negative" adds a new assumption -- silence is evidence against -- exactly the way
# llm_unjudged_fill="zero" smuggled in a top-K_v prior. It is reported as sensitivity instead.
#
# THESE LABELS NO LONGER MATCH THE PAPER, DELIBERATELY. The rewrite around the LLM-only result
# renamed the block: the bold baseline is now "AGS (full)" and it is the no_determ arm, not
# hybrid_full; "$-$ LLM reranker (fused retrieval score only)" is no_verifier; "Program-driven
# score instead of LLM reranker" is no_llm; and no hybrid_full row survives. Injection matches on
# the label text, so the mismatch is inert -- inject_paper reports "labels drifted" and writes
# nothing (verified against a copy). tab:ablation is already filled and was checked cell by cell
# against the fused reranks, so nothing is lost by that.
#
# Do NOT just swap in the new strings. The daggers come from verifier_ablation.csv's
# ci_excludes_zero, whose deltas are all computed against "Hybrid AGS (full)". Relabelling
# no_determ as the baseline while its CIs still reference hybrid_full would print significance
# marks against a baseline the table no longer shows. Re-enabling this path means recomputing the
# paired bootstraps with no_determ as the reference first.
ABLATION_ROWS: tuple[tuple[str, str, str, bool], ...] = (
    (r"\textbf{AGS (hybrid verification; full)}", "Hybrid AGS (full)", "hybrid_full", True),
    (r"$-$ candidate-level LLM verifier (deterministic core)", "- LLM verifier", "no_llm", False),
    (r"$-$ deterministic consensus verifier", "- deterministic verifier", "no_determ", False),
    (r"No verifier (fused retrieval score only)", "- both verifiers", "no_verifier", False),
)
# The abstention-rule sensitivity, reported separately from the architecture block.
ABSTENTION_ROWS: tuple[tuple[str, str, str], ...] = (
    (r"Abstention dropped (averaged over dimensions ruled on)", "- deterministic verifier", "no_determ"),
    (r"Abstention counted as non-support", "LLM verifier only", "llm_only"),
)
# R@1 leftmost, then depth, then MRR; final reranked accuracy is appended as the last column
# so the row reads left-to-right from the arm's own ranking to what the deployed pipeline
# ultimately selects -- the same quantity Table 2 reports.
# Paper-wide convention: recall and MRR are pre-final-rerank, Acc. is post. The
# ablation table shows R@1, R@10, MRR only -- deeper recall cannot move (see caption).
ABLATION_METRICS = ("top1_accuracy", "recall_at_10", "mrr")
WINDOW_METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")
BASELINE = "Hybrid AGS (full)"

# The appendix lays its ablation table out differently from the main paper's tab:ablation --
# full recall depths, then MRR, then the two accuracies -- and its caption is written against
# that order ("Acc. (retr.) is top-1 on the arm's own ranking"). Emitting the main-paper
# fragment there put R@1 under an R@10 heading and MRR under R@200 with no LaTeX error to show
# for it, so the two layouts get two fragments rather than one shared one.
APPENDIX_ABLATION_METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")

DEFAULT_APPENDIX = SCRIPT_DIR.parent / "appendix_verifier_ablation.tex"
# fragment filename -> the `%% >>> NAME` marker it fills. The two differ because the appendix
# names its markers after the tables and the fragments are named after what generates them;
# mapping them explicitly beats renaming either and breaking the other's references.
MARKER_FOR_FRAGMENT: dict[str, str] = {
    "table_verifier_ablation_appendix.tex": "table_verifier_ablation.tex",
    "table_verifier_interaction_appendix.tex": "table_verifier_interaction.tex",
    "table_window_sensitivity.tex": "table_verifier_window.tex",
    "table_verifier_cost.tex": "table_verifier_cost.tex",
}
# Columns each marker's tabular declares, so a fragment can never be pasted under headings it
# does not match. table_ablation_block.tex and table_interaction.tex are NOT injected: they
# belong to the main paper's tables, which have their own column orders.
# acl_latex.tex was the paper until the rewrite around the LLM-only result; it now lives on as
# acl_latex_old_July28.tex and the live paper is acl_latex_llmonly.tex. Pointing here at the old
# name did not fail loudly -- it just found no file, so an injection run reported nothing and
# changed nothing.
DEFAULT_PAPER = SCRIPT_DIR.parent / "comparing_methods" / "acl_latex_llmonly.tex"

# tab:ablation's deterministic-core diagnostic rows: paper row label -> rerank dir(s).
# Only the last two cells (Final Acc., std) are written for these. Their retrieval-stage cells
# come from the older table-5 run and are not recomputed here, so replacing the whole row would
# overwrite measurements this script does not own.
#
# -ensemble is two dirs because the row is defined as the mean of the idx=0 and idx=1 replicas
# (run_test_rows.py:229-238); a single rerank of either index would be a different row.
DIAGNOSTIC_ROWS: dict[str, tuple[str, ...]] = {
    r"$-$ ensemble ($J{=}1$, single hypothesis)": ("rerank_ensemble_idx0", "rerank_ensemble_idx1"),
    r"$-$ label-form (definition-form only)": ("rerank_label_form",),
    r"$-$ definition-form (label-form only)": ("rerank_definition_form",),
    r"$-$ summed fusion (mean RRF)": ("rerank_mean_fusion",),
    r"$-$ score normalization (raw fused scores)": ("rerank_raw_scaling",),
    r"$-$ label coverage ($w_{\mathrm{cov}}{=}0$)": ("rerank_wcov0",),
    r"Oracle best single hypothesis (deterministic signal)": ("rerank_oracle_single",),
}
# tab label in acl_latex.tex -> the fragment whose rows update it. These are the MAIN-PAPER
# fragments, whose column orders were written for exactly these two tables.
PAPER_TARGETS: dict[str, str] = {
    "tab:ablation": "table_ablation_block.tex",
    "tab:llm_window_sensitivity": "table_window_sensitivity.tex",
}

MARKER_COLUMNS: dict[str, int] = {
    "table_verifier_ablation.tex": 7,
    "table_verifier_interaction.tex": 3,
    "table_verifier_window.tex": 6,
    "table_verifier_cost.tex": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--modality", default="pooled")
    parser.add_argument(
        "--appendix",
        type=Path,
        nargs="?",
        const=DEFAULT_APPENDIX,
        default=None,
        help=f"Inject the fragments into this paper file (default {DEFAULT_APPENDIX.name} "
        "beside the repo). Omit to only refresh the fragments under --run-dir.",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        nargs="?",
        const=DEFAULT_PAPER,
        default=None,
        help=f"Update the matching rows of the submitted paper (default {DEFAULT_PAPER}). "
        "Rows are matched by label, so section headings and unrelated diagnostic rows are "
        "left alone.",
    )
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


def arm_dir(run_dir: Path, arm: str) -> Path:
    """The directory holding an arm's rerank, preferring the clean fused-window rerun.

    stage_fused_reranks.sh writes the reranks built from the fused-window verdicts to
    rerank_<arm>_k10fused/, leaving the original rerank_<arm>/ in place. Those originals are the
    PROVISIONAL ones whose candidate window was ordered by the deterministic score -- exactly what
    the fused rerun exists to replace. Reading rerank_<arm>/ unconditionally silently published
    the stale numbers with no error to show for it, so the fused directory wins whenever it
    exists. Arms with no LLM in them (no_llm, no_verifier) have no fused variant and fall back,
    correctly: there is no window for the deterministic score to have ordered.
    """
    fused = run_dir / f"rerank_{arm}_k10fused"
    return fused if (fused / "metrics.json").exists() else run_dir / f"rerank_{arm}"


def reranked(run_dir: Path, arm: str | None) -> dict[str, Any]:
    """The qwen_reranked block for an arm, or {} if its GPU job has not landed."""
    if arm is None:
        return {}
    path = arm_dir(run_dir, arm) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("qwen_reranked", {})


def reranked_accuracy_se(run_dir: Path, arm: str | None, iterations: int = 2000, seed: int = 20260724) -> float | None:
    """Bootstrap standard error of an arm's final accuracy, resampling source contexts.

    Same resampling unit and seed as the paired CIs, but one-sample: this is the sampling
    uncertainty of that cell's own value, not of a difference. It is NOT a seed-to-seed
    standard deviation -- there is one generation run and every arm replays its hypotheses.
    """
    if arm is None:
        return None
    # Gate on metrics.json, not on the predictions file. The GPU job streams predictions as it
    # goes and only writes metrics.json at the end, so a running job has a partial predictions
    # file on disk -- bootstrapping it yields a confident-looking std over whatever fraction has
    # landed, printed next to a "--" accuracy. The std and the accuracy it qualifies must appear
    # together or not at all, so both use the same completion signal.
    directory = arm_dir(run_dir, arm)
    if not (directory / "metrics.json").exists():
        return None
    path = directory / "qwen_rerank_predictions.jsonl"
    if not path.exists():
        return None
    by_context: dict[Any, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            gold = {tag for tag in record.get("gold_tags", [])}
            ranking = record.get("final_ranking") or []
            hit = float(bool(ranking) and ranking[0] in gold)
            by_context.setdefault(record.get("context_id"), []).append(hit)
    if not by_context:
        return None
    contexts = list(by_context)
    means = np.asarray([float(np.mean(by_context[c])) for c in contexts])
    sizes = np.asarray([len(by_context[c]) for c in contexts], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, len(contexts), size=len(contexts))
        weights = sizes[picks]
        draws[index] = float(np.sum(means[picks] * weights) / np.sum(weights))
    return float(np.std(draws, ddof=1))


def indexed(rows: list[dict[str, str]], modality: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["variant"], row["metric"]): row for row in rows if row["modality"] == modality}


def diagnostic_cells(run_dir: Path, arms: tuple[str, ...]) -> tuple[str, str] | None:
    """(Final Acc., std) for a diagnostic row, or None until every arm it needs has landed.

    A multi-arm row is the mean over its arms, which is only meaningful once all of them
    exist -- reporting the mean of one of two replicas would silently be a different row.
    """
    accs: list[float] = []
    ses: list[float] = []
    for arm in arms:
        path = run_dir / arm / "metrics.json"
        if not path.exists():
            return None
        block = json.loads(path.read_text(encoding="utf-8")).get("qwen_reranked") or {}
        acc = block.get("accuracy")
        if acc is None:
            return None
        accs.append(float(acc))
        n = block.get("n") or block.get("count")
        ses.append(float(np.sqrt(acc * (1.0 - acc) / n)) if n else float("nan"))
    mean_acc = sum(accs) / len(accs)
    # Replicas are averaged, so their standard errors combine as independent means would.
    mean_se = float(np.sqrt(sum(se**2 for se in ses)) / len(ses)) if not any(np.isnan(ses)) else None
    return num(mean_acc), num(mean_se)


def inject_diagnostics(paper: Path, run_dir: Path) -> int:
    """Write Final Acc./std into tab:ablation's diagnostic rows, leaving their other cells alone."""
    text = paper.read_text(encoding="utf-8")
    lines = text.splitlines()
    end = next(i for i, l in enumerate(lines) if r"\label{tab:ablation}" in l)
    start = next(i for i in range(end, -1, -1) if lines[i].lstrip().startswith(r"\begin{table"))

    filled = 0
    for i in range(start, end):
        line = lines[i].rstrip()
        if not line.endswith(r"\\"):
            continue
        arms = DIAGNOSTIC_ROWS.get(row_label(line))
        if arms is None:
            continue
        cells = diagnostic_cells(run_dir, arms)
        if cells is None:
            continue
        body = line.removesuffix(r"\\").rstrip()
        parts = [p.strip() for p in body.split("&")]
        if len(parts) != 7:
            raise SystemExit(f"{paper}:{i + 1}: expected 7 fields in a tab:ablation row, got {len(parts)}")
        parts[-2], parts[-1] = cells
        new = " & ".join(parts) + r" \\"
        if new != line:
            lines[i] = new
            filled += 1

    if filled:
        paper.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return filled


def row_label(row: str) -> str:
    """The text before the first & -- what identifies a row inside its table."""
    return row.split("&", 1)[0].strip()


def inject_paper(paper: Path, fragments: dict[str, list[str]]) -> None:
    """Update rows in acl_latex.tex, which has no markers and cannot take a block paste.

    tab:ablation interleaves \\midrule and \\multicolumn section headings between the rows this
    script generates, and it carries eight further diagnostic rows whose Final columns are
    blank for a different reason (no GPU rerank was ever run for them). Replacing a contiguous
    span would destroy the headings and silently claim the diagnostics too, so rows are matched
    by their label -- the text before the first & -- and only exact matches are rewritten.

    PAPER_TARGETS deliberately omits tab:verifier_reranker_interaction: that table carries a
    std column this script has no generator for, and it is already populated, so overwriting it
    would mean inventing one of its five columns.
    """
    if not paper.exists():
        raise SystemExit(f"paper not found: {paper}")
    original = paper.read_text(encoding="utf-8")
    lines = original.splitlines()

    for label, fragment_name in PAPER_TARGETS.items():
        marker = rf"\label{{{label}}}"
        end = next((i for i, line in enumerate(lines) if marker in line), None)
        if end is None:
            raise SystemExit(f"{paper}: no \\label{{{label}}} found")
        start = next(
            (i for i in range(end, -1, -1) if lines[i].lstrip().startswith(r"\begin{table")), None
        )
        if start is None:
            raise SystemExit(f"{paper}: \\label{{{label}}} is not inside a table environment")

        wanted = {row_label(row): row for row in fragments[fragment_name]}
        seen: set[str] = set()
        for i in range(start, end):
            key = row_label(lines[i])
            if key not in wanted or not lines[i].rstrip().endswith(r"\\"):
                continue
            # Guard the same way the marker path does: a row whose field count no longer
            # matches the one it is replacing means the paper's columns moved under us.
            before = lines[i].rstrip().removesuffix(r"\\").count("&")
            after = wanted[key].rstrip().removesuffix(r"\\").count("&")
            if before != after:
                raise SystemExit(
                    f"{paper}:{i + 1}: {label} row {key!r} has {before + 1} fields but the "
                    f"generated row has {after + 1} -- refusing to inject.\n"
                    f"  paper:     {lines[i].strip()}\n  generated: {wanted[key]}"
                )
            if lines[i] != wanted[key]:
                lines[i] = wanted[key]
            seen.add(key)

        missing = sorted(set(wanted) - seen)
        if missing:
            raise SystemExit(f"{paper}: {label} has no row matching {missing} -- labels drifted")
        print(f"  {label}: {len(seen)} rows matched")

    # acl_latex.tex has no trailing newline; adding one would show up in every future diff.
    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if updated == original:
        print(f"\n{paper}: already current, unchanged")
    else:
        paper.write_text(updated, encoding="utf-8")
        print(f"\nupdated {paper}")


def inject(appendix: Path, fragments: dict[str, list[str]]) -> None:
    """Replace each `%% >>> NAME` marker's row block in `appendix` with its fragment.

    A block runs from the line after the marker to the next \\bottomrule. Rewriting only that
    span leaves captions, labels and the surrounding tabular preamble untouched, and keeping
    the marker line makes re-running safe -- the second run overwrites the first run's rows
    rather than stacking another copy underneath them.
    """
    if not appendix.exists():
        raise SystemExit(f"appendix not found: {appendix}")
    original = appendix.read_text(encoding="utf-8")
    lines = original.splitlines()

    marker_to_fragment = {marker: name for name, marker in MARKER_FOR_FRAGMENT.items()}
    out: list[str] = []
    filled: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        stripped = line.strip()
        if not stripped.startswith("%% >>> "):
            index += 1
            continue

        marker = stripped[len("%% >>> ") :].strip()
        name = marker_to_fragment.get(marker)
        if name is None:
            raise SystemExit(f"{appendix}:{index + 1}: no fragment is generated for marker {marker!r}")

        end = index + 1
        while end < len(lines) and lines[end].strip() != r"\bottomrule":
            end += 1
        if end >= len(lines):
            raise SystemExit(f"{appendix}:{index + 1}: marker {marker!r} has no \\bottomrule after it")

        # A row with the wrong field count either fails to compile (too many) or silently
        # prints every number one column left of its heading (too few). Neither is worth
        # discovering in a PDF, so refuse before writing.
        expected = MARKER_COLUMNS[marker]
        for row in fragments[name]:
            fields = row.rstrip().removesuffix(r"\\").count("&") + 1
            if fields != expected:
                raise SystemExit(
                    f"{appendix}: {marker} declares {expected} columns but a generated row has "
                    f"{fields} fields -- refusing to inject.\n  {row}"
                )

        out.extend(fragments[name])
        filled.append(f"{marker} ({len(fragments[name])} rows)")
        index = end  # \bottomrule itself is re-emitted by the next loop iteration

    missing = sorted(set(MARKER_FOR_FRAGMENT.values()) - {entry.split(" (")[0] for entry in filled})
    updated = "\n".join(out) + "\n"
    if updated == original:
        print(f"\n{appendix}: already current, unchanged")
    else:
        appendix.write_text(updated, encoding="utf-8")
        print(f"\ninjected into {appendix}:")
        for entry in filled:
            print(f"  {entry}")
    if missing:
        print(f"  WARNING: no marker found for {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    ablation = indexed(read_csv(run_dir / "verifier_ablation.csv"), args.modality)
    # The two LLM-only arms are only correct when the CSV was produced with the neutral
    # unjudged fill. An older CSV silently regresses those rows by ~0.028 Recall@10, so
    # refuse to emit rather than overwrite a corrected table with stale values.
    summary_path = run_dir / "verifier_ablation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("llm_unjudged_fill") != "mean":
            raise SystemExit(
                "verifier_ablation.csv predates the neutral-fill fix (llm_unjudged_fill != "
                "'mean'). Re-run run_ags_verifier_ablation.py before rebuilding, or the "
                "'- deterministic verifier' and 'LLM verifier only' rows will regress."
            )
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
        # Empty spacer column separates the retrieval-stage block from the reranked one,
        # matching the \cmidrule grouping in the table header.
        reranked_cell = num(reranked(run_dir, arm).get("accuracy"), bold=is_baseline)
        se_cell = num(reranked_accuracy_se(run_dir, arm))
        lines.append(f"{label} & " + " & ".join(cells) + f" & & {reranked_cell} & {se_cell}" + r" \\")
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

    # ---- tab:verifierablation (appendix layout) ----------------------------------------------
    # Same numbers as table_ablation_block.tex, ordered for the appendix's own header:
    # R@10 R@50 R@200 MRR | Acc.(retr.) Acc.(rerank). Acc.(retr.) is top1_accuracy -- the arm's
    # own top-1 before the listwise stage -- which is what that caption says it is.
    lines = []
    for label, variant, arm, is_baseline in ABLATION_ROWS:
        cells = []
        for metric in APPENDIX_ABLATION_METRICS:
            row = ablation.get((variant, metric))
            if row is None:
                cells.append("--")
                continue
            flag = (not is_baseline) and row.get("ci_excludes_zero", "").lower() == "true"
            cells.append(num(row["value"], dagger=flag, bold=is_baseline))
        cells.append(num(reranked(run_dir, arm).get("accuracy"), bold=is_baseline))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    fragments["table_verifier_ablation_appendix.tex"] = lines

    # ---- tab:verifierinteraction (appendix layout) -------------------------------------------
    # Accuracy only. The appendix tabular is llc and its caption discusses top-1 alone; the
    # main paper's version of this table carries MRR as well.
    lines = []
    interaction_cells: list[tuple[str, str, Any]] = []
    for llm_state, variant, arm in (("Off", "- LLM verifier", "no_llm"), ("On ", BASELINE, "hybrid_full")):
        retrieval_row = ablation.get((variant, "top1_accuracy"))
        interaction_cells.append(
            (llm_state, "Off", retrieval_row["value"] if retrieval_row else None)
        )
        interaction_cells.append((llm_state, "On ", reranked(run_dir, arm).get("accuracy")))
    # Table's own row order: Off/Off, On/Off, Off/On, On/On.
    for index in (0, 2, 1, 3):
        llm_state, rerank_state, value = interaction_cells[index]
        lines.append(f"{llm_state} & {rerank_state} & {num(value)} " + r"\\")
    fragments["table_verifier_interaction_appendix.tex"] = lines

    # ---- abstention-rule sensitivity ---------------------------------------------------------
    # Same ablation twice, under the two readings of an abstention. Reported apart from the
    # architecture block so the gap between them is read as what it is -- a policy choice on a
    # verifier that abstains on 44% of dimension opportunities -- and not as a component.
    lines = []
    for label, variant, arm in ABSTENTION_ROWS:
        cells = []
        for metric in APPENDIX_ABLATION_METRICS:
            row = ablation.get((variant, metric))
            cells.append("--" if row is None else num(row["value"]))
        cells.append(num(reranked(run_dir, arm).get("accuracy")))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    fragments["table_verifier_abstention.tex"] = lines

    # ---- tab:verifiercost --------------------------------------------------------------------
    # Per fact, straight from verifier_ablation_cost.csv. The deterministic arms issue no calls,
    # so their LLM/token cells are a true zero rather than a missing measurement -- printed as 0
    # and not as --, which here would wrongly read "job still running".
    lines = []
    for label, variant, _, is_baseline in ABLATION_ROWS:
        row = cost.get(variant)
        if row is None:
            lines.append(f"{label} & -- & -- & --" + r" \\")
            continue
        calls = float(row["verifier_llm_calls_per_fact"])
        tokens = float(row["verifier_completion_tokens_per_fact"])
        seconds = float(row["scoring_cpu_sec_per_fact"])
        cells = [
            f"{calls:.1f}",
            f"{round(tokens):,}".replace(",", "{,}"),
            f"{seconds:.3f}",
        ]
        if is_baseline:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    fragments["table_verifier_cost.tex"] = lines

    for name, lines in fragments.items():
        (run_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n=== {name} ===")
        print("\n".join(lines))

    if args.appendix:
        inject(args.appendix, fragments)
    if args.paper:
        print(f"\n=== {args.paper} ===")
        inject_paper(args.paper, fragments)
        filled = inject_diagnostics(args.paper, run_dir)
        print(f"  tab:ablation diagnostics: {filled} rows written")
        waiting = [
            label for label, arms in DIAGNOSTIC_ROWS.items() if diagnostic_cells(run_dir, arms) is None
        ]
        if waiting:
            print(f"  still waiting on {len(waiting)} diagnostic rows: {', '.join(sorted(waiting))}")

    pending = [arm for _, _, arm, _ in ABLATION_ROWS if not reranked(run_dir, arm)]
    if pending:
        print(f"\nreranked accuracy still pending: {', '.join(pending)}")


if __name__ == "__main__":
    main()
