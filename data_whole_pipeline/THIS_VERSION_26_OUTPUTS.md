# 这一版的 26 个输出目录

生成时间 2026-07-30 04:57。**目录一栏是用 `scontrol show job <id>` 从每个 job 自己的
环境里读出来的**,不是手写的。批次 = 2026-07-30 03:11 一次性提交的 26 个 job(问6算6)。

路径都相对 `data_whole_pipeline/`。

| # | job | job id | 队列状态 | 输出目录 | 磁盘 | 对应论文位置 |
|--:|---|---|---|---|---|---|
| 1 | `bl_direct_retrieval` | 20488395 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval_wcov1` | 正在写入 | tab:main_results / Direct retrieval |
| 2 | `bl_one_pass_grounding` | 20488396 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding_wcov1` | 正在写入 | tab:main_results / One-pass free-text(query 已播种,w_cov 是唯一变量) |
| 3 | `bl_parallel_sampling_iid` | 20488397 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_wcov1` | 正在写入 | tab:main_results / Parallel stochastic(i.i.d.,T=0.8) |
| 4 | `bl_parallel_diversity` | 20488398 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_parallel_sampling_diversity_wcov1` | 正在写入 | tab:main_results / Parallel diversity |
| 5 | `bl_decomposed` | 20488399 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_decomposed_retrieval_wcov1` | 正在写入 | tab:main_results / Decomposed |
| 6 | `bl_intrinsic` | 20488400 | RUNNING | `runs_fintagging_grounding_baseline/qwen3_32b_intrinsic_self_refinement_wcov1` | 正在写入 | tab:main_results / Intrinsic refine. |
| 7 | `bl_feedback` | 20488401 | PENDING | `runs_fintagging_grounding_baseline/qwen3_32b_retrieval_feedback_refinement_wcov1` | 目录尚未创建 | tab:main_results / Feedback refine. |
| 8 | `vf_full` | 20488402 | RUNNING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_full` | 正在写入 | FHS 自己的问6 verdict → 表2/表3 的 FHS 行、K_v=10 行、verifierfull、verifierbridge |
| 9 | `vf_def_only` | 20488403 | RUNNING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_def_only` | 正在写入 | tab:ablation / − label-form 的 verdict |
| 10 | `vf_lab_only` | 20488404 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_lab_only` | 目录尚未创建 | tab:ablation / − definition-form 的 verdict |
| 11 | `vf_ensemble_idx0` | 20488405 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_ensemble_idx0` | 目录尚未创建 | tab:ablation / − ensemble 的 verdict(一半) |
| 12 | `vf_ensemble_idx1` | 20488406 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_ensemble_idx1` | 目录尚未创建 | tab:ablation / − ensemble 的 verdict(另一半) |
| 13 | `vf_mean_fusion` | 20488407 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_mean_fusion` | 目录尚未创建 | tab:ablation / − summed fusion 的 verdict |
| 14 | `vf_k5` | 20488408 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_k5` | 目录尚未创建 | tab:llm_window_sensitivity / K_v=5 |
| 15 | `vf_k20` | 20488409 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/verdicts_arm6_k20` | 目录尚未创建 | tab:llm_window_sensitivity / K_v=20 |
| 16 | `rr_no_determ` | 20488410 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_full` | 目录尚未创建 | tab:main_results + tab:ablation 的 FHS (full) 行 |
| 17 | `rr_llmonly_raw_scaling` | 20488411 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_raw_scaling` | 目录尚未创建 | tab:ablation / − score norm. |
| 18 | `rr_llmonly_label_form` | 20488412 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_label_form` | 目录尚未创建 | tab:ablation / − label-form |
| 19 | `rr_llmonly_definition_form` | 20488413 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_definition_form` | 目录尚未创建 | tab:ablation / − definition-form |
| 20 | `rr_llmonly_ensemble_idx0` | 20488414 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_ensemble_idx0` | 目录尚未创建 | tab:ablation / − ensemble(与 idx1 取算术平均) |
| 21 | `rr_llmonly_ensemble_idx1` | 20488415 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_ensemble_idx1` | 目录尚未创建 | tab:ablation / − ensemble(与 idx0 取算术平均) |
| 22 | `rr_llmonly_mean_fusion` | 20488416 | PENDING | `runs_ags_verifier_ablation/qwen3_32b/rerank_arm6_llmonly_mean_fusion` | 目录尚未创建 | tab:ablation / − summed fusion |
| 23 | `seqvf_s0` | 20488417 | PENDING | `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s0` | 目录尚未创建 | tab:main_results / FHS-Seq(分片 1/4) |
| 24 | `seqvf_s1` | 20488418 | PENDING | `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s1` | 目录尚未创建 | tab:main_results / FHS-Seq(分片 2/4) |
| 25 | `seqvf_s2` | 20488419 | PENDING | `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s2` | 目录尚未创建 | tab:main_results / FHS-Seq(分片 3/4) |
| 26 | `seqvf_s3` | 20488420 | PENDING | `runs_fintagging_grounding_baseline/qwen3_32b_seq_verifier_s3` | 目录尚未创建 | tab:main_results / FHS-Seq(分片 4/4) |

## 唯一性

- 26 个 job → **26 个互不重复的目录**
- 三类前缀就是这一版的标记:基线 `*_wcov1`、verifier `*arm6*`、顺序臂 `seq_verifier_s{0..3}`

## 一个必须知道的同名陷阱

`_superseded_20260730/` 里有 **10 个与本版完全同名**的目录(8 个 `*_wcov1` + `verdicts_arm6_def_only`
+ `verdicts_arm6_lab_only`),是今晚归档的旧产物。最危险的一个:

```
_superseded_20260730/baselines/qwen3_32b_direct_retrieval_wcov1      metrics.json 有(完整)
runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval_wcov1  还在跑
```

归档目录在 `data_whole_pipeline/` 顶层,**不在任何 `runs_*` 树内部**,所以:

- 只在两棵 runs 树里找 → 只会命中本版(已实测)
- 从顶层 `find .` 无脑找 → 10 个归档全部命中(已实测)

**所以:永远从本文件或 `ask6_batch_20260730_jobids.txt` 解析路径,不要按名字 glob。**
拿不准某个目录属于哪一版,跑 `python3 verify_single_code_path.py` —— 它读每个 run 自己写下的配置,
归档那些是 ask-3 / deterministic,会直接报漂移。
