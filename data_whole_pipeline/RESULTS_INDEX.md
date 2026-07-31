# 这一版结果在哪儿 — RESULTS_INDEX.md

**自动生成**,由 `make_results_index.py` 读磁盘 + `ask6_batch_20260730_jobids.txt` + 每个 run 自己
记录的配置产出。刷新:`python3 make_results_index.py`。生成时间 2026-07-30 16:23。

这一版 = 2026-07-30 03:11 一次性提交的 26 个 job(问6算6,代码冻结,指纹见 `FREEZE_MANIFEST.txt`)。
**只有下面点名的目录属于这一版。同一棵树下其余目录都不是**,原因见最后一节。

## 1. 批次进度

| job | 状态 |
|---|---|
| `bl_direct_retrieval` | COMPLETED |
| `bl_one_pass_grounding` | COMPLETED |
| `bl_parallel_sampling_iid` | COMPLETED |
| `bl_parallel_diversity` | RUNNING |
| `bl_decomposed` | COMPLETED |
| `bl_intrinsic` | COMPLETED |
| `bl_feedback` | COMPLETED |
| `vf_full` | COMPLETED |
| `vf_def_only` | COMPLETED |
| `vf_lab_only` | COMPLETED |
| `vf_ensemble_idx0` | COMPLETED |
| `vf_ensemble_idx1` | COMPLETED |
| `vf_mean_fusion` | COMPLETED |
| `vf_k5` | COMPLETED |
| `vf_k20` | COMPLETED |
| `rr_no_determ` | COMPLETED |
| `rr_llmonly_raw_scaling` | COMPLETED |
| `rr_llmonly_label_form` | COMPLETED |
| `rr_llmonly_definition_form` | COMPLETED |
| `rr_llmonly_ensemble_idx0` | COMPLETED |
| `rr_llmonly_ensemble_idx1` | COMPLETED |
| `rr_llmonly_mean_fusion` | COMPLETED |
| `seqvf_s0` | RUNNING |
| `seqvf_s1` | RUNNING |
| `seqvf_s2` | RUNNING |
| `seqvf_s3` | RUNNING |

## 2. 论文每张表从哪里取数

| 论文位置 | 目录 | 状态 | 取什么 / 注意 |
|---|---|---|---|
| tab:main_results / Direct retrieval | `runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval_wcov1` | READY (metrics.json) | metrics.json: bm25_retrieval + qwen_reranked |
| tab:main_results / One-pass free-text | `runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding_wcov1` | READY (metrics.json) | metrics.json (queries seeded from the published run, w_cov the only variable) |
| tab:main_results / One-pass structured | `runs_fintagging_grounding_baseline/qwen3_32b_one_pass_structured` | READY (metrics.json) | metrics.json -- NOT rerun: its config pins w_cov=1 already |
| tab:main_results / Parallel stochastic | `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_wcov1` | READY (metrics.json) | metrics.json (i.i.d. arm: PARALLEL_PROMPT_STYLE=plain, T=0.8) |
| tab:main_results / Parallel diversity | `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_diversity_wcov1` | in progress | metrics.json |
| tab:main_results / Decomposed | `runs_fintagging_grounding_baseline/qwen3_32b_decomposed_retrieval_wcov1` | READY (metrics.json) | metrics.json |
| tab:main_results / Intrinsic refine. | `runs_fintagging_grounding_baseline/qwen3_32b_intrinsic_self_refinement_wcov1` | READY (metrics.json) | metrics.json |
| tab:main_results / Feedback refine. | `runs_fintagging_grounding_baseline/qwen3_32b_retrieval_feedback_refinement_wcov1` | READY (metrics.json) | metrics.json |
| tab:main_results / FHS (full) | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_full` | READY (metrics.json) | metrics.json: bm25_retrieval gives R@10/R@50/MRR, qwen_reranked gives Acc |
| tab:main_results / FHS-Seq | `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s0` | in progress | 4 shards s0..s3, merge before aggregating |
| tab:ablation / FHS (full) | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_full` | READY (metrics.json) | same run as the main-table FHS row |
| tab:ablation / - verifier | `runs_ags_verifier_ablation/qwen3_32b/rerank_no_verifier` | READY (metrics.json) | beta=0, no verdicts consumed, so ask-6 does not apply |
| tab:ablation / Program-driven score | `runs_ags_verifier_ablation/qwen3_32b/rerank_no_llm` | READY (metrics.json) | verifier_mode=deterministic by definition of the row |
| tab:ablation / - label-form | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_label_form` | READY (metrics.json) | beta=0.8 per selected_betas.json (ranking count halves) |
| tab:ablation / - definition-form | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_definition_form` | READY (metrics.json) | beta=0.2 per selected_betas.json |
| tab:ablation / - ensemble (J=1) | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_ensemble_idx0` | READY (metrics.json) | arithmetic mean with ...ensemble_idx1 |
| tab:ablation / - ensemble (J=1) | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_ensemble_idx1` | READY (metrics.json) | the other half of that mean |
| tab:ablation / - summed fusion | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_mean_fusion` | READY (metrics.json) | metrics.json |
| tab:ablation / - score norm. | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_raw_scaling` | READY (metrics.json) | reuses verdicts_arm6_full: range-norm is monotone so the window is identical |
| tab:ablation / - label coverage | `runs_ags_verifier_ablation/qwen3_32b/rerank_wcov0` | READY (metrics.json) | Acc only; retrieval columns from runs_ags_table5_ablation/qwen3_32b_rerun/index_ablation.csv |
| tab:ablation / Oracle best single | `runs_ags_verifier_ablation/qwen3_32b/rerank_oracle_single` | READY (metrics.json) | stays program-driven: the oracle has one window PER HYPOTHESIS, which --window-tags cannot express |
| tab:llm_window_sensitivity / K_v=5 | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_k5` | READY (verdicts) | CPU rescoring input |
| tab:llm_window_sensitivity / K_v=10 | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_full` | READY (verdicts) | must reproduce tab:ablation's FHS row to every digit |
| tab:llm_window_sensitivity / K_v=20 | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_k20` | READY (verdicts) | CPU rescoring input |
| tab:verifierfull, tab:verifierbridge | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_full` | READY (verdicts) | pass --llm-calls <this>/llm_verifier_calls.jsonl explicitly; both scripts default to the OLD det-window log |

## 3. 八份问6 verdict 的自述配置

每个 run 自己记录的配置,直接读盘;`dims` 必须是 6,`window_source` 必须是 fused。

| 目录 | 状态 | 自述配置 |
|---|---|---|
| `verdicts_arm6_full` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_full.jsonl |
| `verdicts_arm6_def_only` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_def_only.jsonl |
| `verdicts_arm6_lab_only` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_lab_only.jsonl |
| `verdicts_arm6_ensemble_idx0` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_ensemble_idx0.jsonl |
| `verdicts_arm6_ensemble_idx1` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_ensemble_idx1.jsonl |
| `verdicts_arm6_mean_fusion` | READY (verdicts) | top_m=10, window_source=fused, parse_rate=1.0, dims=6, window=window_mean_fusion.jsonl |
| `verdicts_arm6_k5` | READY (verdicts) | top_m=5, window_source=fused, parse_rate=1.0, dims=6 |
| `verdicts_arm6_k20` | READY (verdicts) | top_m=20, window_source=fused, parse_rate=1.0, dims=6 |

## 3b. 这一批读取的输入(不是旧结果,别动)

| 是什么 | 路径 | 为什么关键 |
|---|---|---|
| 6 份 per-arm 窗口 | `runs_ags_verifier_ablation/qwen3_32b/arm_windows` | verdicts_arm6_* 的 --window-tags 来源;window_full 已验证等于部署窗口(25,090/25,090) |
| 每-fact 基线 | `runs_ags_verifier_ablation/qwen3_32b/per_fact` | K_v 敏感性脚本的 --baseline-per-fact |
| frozen trace(所有消融的池) | `runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags` | bm25_candidates.jsonl:十二个臂全部对它打分,谁都不许换 |
| 已发表的 free-text query | `runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding` | query_descriptions.jsonl 被播种进 _wcov1 目录;也是表 5 freetext 行的口径 |

## 4. 派生分析(等 verdict 落地后本地 CPU 跑,不占 GPU)

| 产物 | 目录 | 命令要点 |
|---|---|---|
| K_v 敏感性 | `runs_ags_verifier_ablation/qwen3_32b/verifier_window_sensitivity.csv` | `run_verifier_window_sensitivity.py --verifier-mode llm_drop` |
| `tab:verifierfull` | `runs_ags_verification_quality/qwen3_32b_arm6/` | **必须显式** `--llm-calls .../verdicts_arm6_full/llm_verifier_calls.jsonl` |
| `tab:verifierbridge` | `runs_ags_verifier_bridge/qwen3_32b_arm6/` | 同上,默认路径是旧的 det 窗口 calls |
| 消融汇总 | `runs_ags_verifier_ablation/qwen3_32b/verifier_ablation.csv` + `table_*.tex` | 汇总 `rerank_arm6_*` |

## 5. 不要读的目录(同一棵树下,名字很像)

| 目录 | 为什么不能用 |
|---|---|
| `runs_fintagging_grounding_baseline/qwen3_32b_CFGCHECK` | config gate, not a result |
| `runs_fintagging_grounding_baseline/qwen3_32b_ags_seq` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_ags_seq_random` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_bandit_freeform` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_decomposed_retrieval` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags_DEV` | development-sample run |
| `runs_fintagging_grounding_baseline/qwen3_32b_intrinsic_self_refinement` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_memory_guided_refinement` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_operator_refinement` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_diversity` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_diversity_j2` | interim J=2 probe |
| `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_j2` | interim J=2 probe |
| `runs_fintagging_grounding_baseline/qwen3_32b_retrieval_feedback_refinement` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_rate` | throughput probe, 40 facts |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s1` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s2` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s3` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_smoke` | smoke test, partial by design |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_smoke2` | smoke test, partial by design |
| `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_smoke3` | smoke test, partial by design |
| `runs_ags_verifier_ablation/qwen3_32b/_quarantine_windowbug_20260730` | quarantined: produced under the window for/else bug |
| `runs_ags_verifier_ablation/qwen3_32b/arm_windows_wcov0` | w_cov=0, kept only for the coverage row |
| `runs_ags_verifier_ablation/qwen3_32b/prefix_zerofill_backup` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_hybrid_full` | hybrid verifier: the paper is LLM-only |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_hybrid_full_k10fused` | ask-3 era: superseded by the arm6 runs |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_llm_only` | llm_strict arm, kept for the abstention contrast only |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_llm_only_k10fused` | ask-3 era: superseded by the arm6 runs |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_no_determ` | 上一版或历史 run:不在这一批的 ledger 里 |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_no_determ_k10fused` | ask-3 era: superseded by the arm6 runs |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_no_determ_k20fused` | ask-3 era |
| `runs_ags_verifier_ablation/qwen3_32b/rerank_no_determ_k5fused` | ask-3 era |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_fulltagging` | extractor-driven pipeline, not a reported table |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_fused` | ask-3 verdicts |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_hint` | symbolic-hint probe, a null result the paper does not print |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_hint6` | symbolic-hint probe, a null result the paper does not print |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_judge6` | ask-6 probe under the OLD prompt scope (hypothesis_scope=all) |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_judge6_hint6` | symbolic-hint probe, a null result the paper does not print |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k20_fused` | ask-3 verdicts |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_k5_fused` | ask-3 verdicts |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_m20` | pre-fix window |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_m5` | pre-fix: window cut from the deterministically reranked order |
| `runs_ags_verifier_ablation/qwen3_32b/verdicts_smoke_fused` | smoke test, partial by design |

## 6. 一条判据

拿不准某个目录是不是这一版的,不要看名字或时间,跑:

```
python3 verify_single_code_path.py      # 每个 run 自述的配置 vs 钉住的配置
```

它读的是 run 自己写下的 `ranking_summary.json` / `llm_verifier_summary.json`,漂移会直接报出来。
