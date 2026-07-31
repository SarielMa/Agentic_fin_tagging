#!/usr/bin/env python3
"""Generate RESULTS_INDEX.md: which directory holds THIS version's result for each paper table.

WHY
    runs_fintagging_grounding_baseline/ has 28 subdirectories and
    runs_ags_verifier_ablation/qwen3_32b/ has 29, accumulated over weeks: smoke runs, diagnostics,
    hint/judge6 probes, quarantined output, ask-3 artifacts, and the current batch, all side by
    side with similar names. Reading a stale directory is the single most expensive mistake
    available here, so the index is generated from disk + the job ledger rather than written by
    hand, and it can be re-run at any time.

USAGE
    python3 make_results_index.py            # writes RESULTS_INDEX.md
    python3 make_results_index.py --print    # also dumps it to stdout
CPU only, reads metadata only, seconds.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "runs_fintagging_grounding_baseline"
ABL = ROOT / "runs_ags_verifier_ablation" / "qwen3_32b"
LEDGER = ROOT / "ask6_batch_20260730_jobids.txt"

# ---------------------------------------------------------------- the authoritative map
# paper location -> (path, what the table takes from it)
# Only this batch's outputs. Anything not named here is NOT a source for the current version.
CURRENT: list[tuple[str, Path, str]] = [
    ("tab:main_results / Direct retrieval",      BASE / "qwen3_32b_direct_retrieval_wcov1",              "metrics.json: bm25_retrieval + qwen_reranked"),
    ("tab:main_results / One-pass free-text",    BASE / "qwen3_32b_one_pass_grounding_wcov1",            "metrics.json (queries seeded from the published run, w_cov the only variable)"),
    ("tab:main_results / One-pass structured",   BASE / "qwen3_32b_one_pass_structured",                 "metrics.json -- NOT rerun: its config pins w_cov=1 already"),
    ("tab:main_results / Parallel stochastic",   BASE / "qwen3_32b_parallel_sampling_wcov1",             "metrics.json (i.i.d. arm: PARALLEL_PROMPT_STYLE=plain, T=0.8)"),
    ("tab:main_results / Parallel diversity",    BASE / "qwen3_32b_parallel_sampling_diversity_wcov1",   "metrics.json"),
    ("tab:main_results / Decomposed",            BASE / "qwen3_32b_decomposed_retrieval_wcov1",          "metrics.json"),
    ("tab:main_results / Intrinsic refine.",     BASE / "qwen3_32b_intrinsic_self_refinement_wcov1",     "metrics.json"),
    ("tab:main_results / Feedback refine.",      BASE / "qwen3_32b_retrieval_feedback_refinement_wcov1", "metrics.json"),
    ("tab:main_results / FHS (full)",            ABL / "rerank_arm6_full",                               "metrics.json: bm25_retrieval gives R@10/R@50/MRR, qwen_reranked gives Acc"),
    ("tab:main_results / FHS-Seq",               BASE / "qwen3_32b_seq_verifier_s0",                     "4 shards s0..s3, merge before aggregating"),
    ("tab:ablation / FHS (full)",                ABL / "rerank_arm6_full",                               "same run as the main-table FHS row"),
    ("tab:ablation / - verifier",                ABL / "rerank_no_verifier",                             "beta=0, no verdicts consumed, so ask-6 does not apply"),
    ("tab:ablation / Program-driven score",      ABL / "rerank_no_llm",                                  "verifier_mode=deterministic by definition of the row"),
    ("tab:ablation / - label-form",              ABL / "rerank_arm6_llmonly_label_form",                 "beta=0.8 per selected_betas.json (ranking count halves)"),
    ("tab:ablation / - definition-form",         ABL / "rerank_arm6_llmonly_definition_form",            "beta=0.2 per selected_betas.json"),
    ("tab:ablation / - ensemble (J=1)",          ABL / "rerank_arm6_llmonly_ensemble_idx0",              "arithmetic mean with ...ensemble_idx1"),
    ("tab:ablation / - ensemble (J=1)",          ABL / "rerank_arm6_llmonly_ensemble_idx1",              "the other half of that mean"),
    ("tab:ablation / - summed fusion",           ABL / "rerank_arm6_llmonly_mean_fusion",                "metrics.json"),
    ("tab:ablation / - score norm.",             ABL / "rerank_arm6_llmonly_raw_scaling",                "reuses verdicts_arm6_full: range-norm is monotone so the window is identical"),
    ("tab:ablation / - label coverage",          ABL / "rerank_wcov0",                                   "Acc only; retrieval columns from runs_ags_table5_ablation/qwen3_32b_rerun/index_ablation.csv"),
    ("tab:ablation / Oracle best single",        ABL / "rerank_oracle_single",                           "stays program-driven: the oracle has one window PER HYPOTHESIS, which --window-tags cannot express"),
    ("tab:llm_window_sensitivity / K_v=5",       ABL / "verdicts_arm6_k5",                               "CPU rescoring input"),
    ("tab:llm_window_sensitivity / K_v=10",      ABL / "verdicts_arm6_full",                             "must reproduce tab:ablation's FHS row to every digit"),
    ("tab:llm_window_sensitivity / K_v=20",      ABL / "verdicts_arm6_k20",                              "CPU rescoring input"),
    ("tab:verifierfull, tab:verifierbridge",     ABL / "verdicts_arm6_full",                             "pass --llm-calls <this>/llm_verifier_calls.jsonl explicitly; both scripts default to the OLD det-window log"),
]

# 这一批读取的输入。它们不是"旧结果",误删/误判会让在跑的 job 失败或静默走错窗口。
INPUTS: list[tuple[str, Path, str]] = [
    ("6 份 per-arm 窗口",          ABL / "arm_windows",                                  "verdicts_arm6_* 的 --window-tags 来源;window_full 已验证等于部署窗口(25,090/25,090)"),
    ("每-fact 基线",               ABL / "per_fact",                                     "K_v 敏感性脚本的 --baseline-per-fact"),
    ("frozen trace(所有消融的池)", BASE / "qwen3_32b_frozen_ags",                        "bm25_candidates.jsonl:十二个臂全部对它打分,谁都不许换"),
    ("已发表的 free-text query",   BASE / "qwen3_32b_one_pass_grounding",                "query_descriptions.jsonl 被播种进 _wcov1 目录;也是表 5 freetext 行的口径"),
]

VERDICT_DIRS = ["verdicts_arm6_full", "verdicts_arm6_def_only", "verdicts_arm6_lab_only",
                "verdicts_arm6_ensemble_idx0", "verdicts_arm6_ensemble_idx1",
                "verdicts_arm6_mean_fusion", "verdicts_arm6_k5", "verdicts_arm6_k20"]

# Why a directory that looks relevant must not be read. Matched as substrings, first hit wins.
STALE_REASONS = [
    ("_quarantine",      "quarantined: produced under the window for/else bug"),
    ("_superseded",      "archived this batch; see _superseded_20260730/WHY.md"),
    ("smoke",            "smoke test, partial by design"),
    ("CFGCHECK",         "config gate, not a result"),
    ("_DEV",             "development-sample run"),
    ("DIAGNOSTIC",       "diagnostic only, explicitly not for any table"),
    ("_rate",            "throughput probe, 40 facts"),
    ("_hint",            "symbolic-hint probe, a null result the paper does not print"),
    ("judge6",           "ask-6 probe under the OLD prompt scope (hypothesis_scope=all)"),
    ("k10fused",         "ask-3 era: superseded by the arm6 runs"),
    ("k5fused",          "ask-3 era"),
    ("k20fused",         "ask-3 era"),
    ("verdicts_k10_",    "ask-3 verdicts"),
    ("verdicts_k5_",     "ask-3 verdicts"),
    ("verdicts_k20_",    "ask-3 verdicts"),
    ("verdicts_m5",      "pre-fix: window cut from the deterministically reranked order"),
    ("verdicts_m20",     "pre-fix window"),
    ("fulltagging",      "extractor-driven pipeline, not a reported table"),
    ("hybrid_full",      "hybrid verifier: the paper is LLM-only"),
    ("llm_only",         "llm_strict arm, kept for the abstention contrast only"),
    ("_j2",              "interim J=2 probe"),
    ("varweight",        "dimension-weighting study, not in the paper"),
    ("neutral",          "abstention-scoring study, not in the paper"),
    ("_wcov0",           "w_cov=0, kept only for the coverage row"),
]


def job_states() -> dict[str, str]:
    """name -> state, from the ledger + sacct."""
    out: dict[str, str] = {}
    if not LEDGER.exists():
        return out
    for line in LEDGER.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        name, jid = parts
        try:
            r = subprocess.run(["sacct", "-n", "-j", jid, "--format=State%14", "-X"],
                               capture_output=True, text=True, timeout=30)
            out[name] = (r.stdout.strip().splitlines() or ["?"])[0].strip()
        except Exception:
            out[name] = "?"
    return out


def status_of(path: Path) -> str:
    if not path.exists():
        return "not started"
    if (path / "metrics.json").exists():
        return "READY (metrics.json)"
    if (path / "llm_verifier_verdicts.json").exists():
        return "READY (verdicts)"
    files = [p.name for p in path.iterdir()]
    if any(f.endswith(".jsonl") for f in files):
        return "in progress"
    return "empty"


def config_of(path: Path) -> str:
    for name in ("ranking_summary.json", "llm_verifier_summary.json"):
        p = path / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        bits = []
        for k in ("verifier_mode", "beta", "top_m", "llm_unjudged_fill", "window_source", "parse_rate"):
            if d.get(k) is not None:
                bits.append(f"{k}={d[k]}")
        dims = d.get("judge_dimensions") or d.get("llm_verifier_dimensions")
        if dims:
            bits.append(f"dims={len(dims)}")
        wt = d.get("window_tags_path")
        if wt:
            bits.append(f"window={Path(wt).name}")
        return ", ".join(bits)
    return ""


def main() -> None:
    states = job_states()
    lines: list[str] = []
    A = lines.append
    A("# 这一版结果在哪儿 — RESULTS_INDEX.md")
    A("")
    A(f"**自动生成**,由 `make_results_index.py` 读磁盘 + `ask6_batch_20260730_jobids.txt` + 每个 run 自己")
    A(f"记录的配置产出。刷新:`python3 make_results_index.py`。生成时间 {datetime.now():%Y-%m-%d %H:%M}。")
    A("")
    A("这一版 = 2026-07-30 03:11 一次性提交的 26 个 job(问6算6,代码冻结,指纹见 `FREEZE_MANIFEST.txt`)。")
    A("**只有下面点名的目录属于这一版。同一棵树下其余目录都不是**,原因见最后一节。")
    A("")
    A("## 1. 批次进度")
    A("")
    A("| job | 状态 |")
    A("|---|---|")
    for name, st in states.items():
        A(f"| `{name}` | {st} |")
    A("")
    A("## 2. 论文每张表从哪里取数")
    A("")
    A("| 论文位置 | 目录 | 状态 | 取什么 / 注意 |")
    A("|---|---|---|---|")
    for label, path, note in CURRENT:
        rel = path.relative_to(ROOT)
        A(f"| {label} | `{rel}` | {status_of(path)} | {note} |")
    A("")
    A("## 3. 八份问6 verdict 的自述配置")
    A("")
    A("每个 run 自己记录的配置,直接读盘;`dims` 必须是 6,`window_source` 必须是 fused。")
    A("")
    A("| 目录 | 状态 | 自述配置 |")
    A("|---|---|---|")
    for d in VERDICT_DIRS:
        p = ABL / d
        A(f"| `{d}` | {status_of(p)} | {config_of(p) or '(还没写 summary)'} |")
    A("")
    A("## 3b. 这一批读取的输入(不是旧结果,别动)")
    A("")
    A("| 是什么 | 路径 | 为什么关键 |")
    A("|---|---|---|")
    for label, path, note in INPUTS:
        A(f"| {label} | `{path.relative_to(ROOT)}` | {note} |")
    A("")
    A("## 4. 派生分析(等 verdict 落地后本地 CPU 跑,不占 GPU)")
    A("")
    A("| 产物 | 目录 | 命令要点 |")
    A("|---|---|---|")
    A("| K_v 敏感性 | `runs_ags_verifier_ablation/qwen3_32b/verifier_window_sensitivity.csv` | `run_verifier_window_sensitivity.py --verifier-mode llm_drop` |")
    A("| `tab:verifierfull` | `runs_ags_verification_quality/qwen3_32b_arm6/` | **必须显式** `--llm-calls .../verdicts_arm6_full/llm_verifier_calls.jsonl` |")
    A("| `tab:verifierbridge` | `runs_ags_verifier_bridge/qwen3_32b_arm6/` | 同上,默认路径是旧的 det 窗口 calls |")
    A("| 消融汇总 | `runs_ags_verifier_ablation/qwen3_32b/verifier_ablation.csv` + `table_*.tex` | 汇总 `rerank_arm6_*` |")
    A("")
    A("## 5. 不要读的目录(同一棵树下,名字很像)")
    A("")
    A("| 目录 | 为什么不能用 |")
    A("|---|---|")
    keep = {p.name for _, p, _ in CURRENT} | set(VERDICT_DIRS) | {p.name for _, p, _ in INPUTS} | {'.done'}
    for tree in (BASE, ABL):
        for p in sorted(tree.iterdir()):
            if not p.is_dir() or p.name in keep:
                continue
            reason = next((r for k, r in STALE_REASONS if k.lower() in p.name.lower()), "")
            if not reason:
                reason = "上一版或历史 run:不在这一批的 ledger 里"
            A(f"| `{p.relative_to(ROOT)}` | {reason} |")
    A("")
    A("## 6. 一条判据")
    A("")
    A("拿不准某个目录是不是这一版的,不要看名字或时间,跑:")
    A("")
    A("```")
    A("python3 verify_single_code_path.py      # 每个 run 自述的配置 vs 钉住的配置")
    A("```")
    A("")
    A("它读的是 run 自己写下的 `ranking_summary.json` / `llm_verifier_summary.json`,漂移会直接报出来。")
    A("")

    text = "\n".join(lines)
    (ROOT / "RESULTS_INDEX.md").write_text(text, encoding="utf-8")
    print(f"wrote {ROOT / 'RESULTS_INDEX.md'} ({len(lines)} lines)")
    if "--print" in sys.argv:
        print(text)


if __name__ == "__main__":
    main()
