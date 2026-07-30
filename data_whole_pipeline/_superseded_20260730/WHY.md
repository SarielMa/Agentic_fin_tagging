# 2026-07-30 03:1x — 为什么这些结果被移到这里(移动,不是删除)

用户的判断:一周以来的循环是"发现一个错 → 改代码 → 重跑",每次都给还在跑的东西引入一个版本差,
结果之间开始出现矛盾。所以这一次:**先全停,确认所有方法来自一个框架、所有消融改自一个方法,
再一起提交一批**。这个目录是"停"和"清"的那一步。

判据是 `verify_single_code_path.py`,它不看代码,只读每个 run 自己记录的配置,和一份钉住的
配置(PIN)比。清理前它的输出是:**15 行里只有 5 行在同一条路径上,10 行漂移**。

用户拍板:统一到 **问6算6**(verifier 被问的维度 = 被计分的维度 = 生成的全部六维)。

## 移进来的东西,四类

### 1. 被取消的 job 留下的半成品(绝不能被 `--resume` 或人读到)
- `verdicts_arm6_def_only/`、`verdicts_arm6_lab_only/` — 各 3,520 / 5,018 calls(约 70%)。
  这两份的**生成逻辑与现在的代码是同一份**(用 tokenizer 复原 prompt,fact0 的 prompt_tokens
  1326 / 1227 与现码逐 token 相同),技术上可以 `--resume` 省 3 GPU-h。**仍然作废重跑**:
  这一批的全部意义是"一批一个版本",省 3 小时换回一个说不清的来源不值得。
- `verdicts_k5_ask6/` — 768 calls,来自那个 `UnboundLocalError` 崩掉的 job。

### 2. 证明被污染 / 来源无法复现
- `rerank_llmonly_ask6_score6/` — `--top-m 10` 把 verdict 通过**老 deterministic 窗口**的 call log
  过滤了:50,180 → 42,910 键,前 200 个 fact 里 68% 的 top-10 变了。它的 `llm_verdicts_used`
  = 42910,全库没有任何文件是这个键数,这是唯一的可见症状。
- `rerank_llmonly_raw_scaling/` — 打分集记的是六维,但喂进去的 verdict 是**问3** 的
  `verdicts_k10_fused`(那时 dump 还不记录 verdict 路径,所以 `verify_single_code_path.py` 误判它
  为 ok)。要用 `verdicts_arm6_full` 重出。

### 3. 在 test 上选出来的维度/打分 study — 不进论文
`rerank_no_determ_k10judge6*`(含 score_FAMILY / ROLE / EVENT 三个单维)、
`rerank_neutral_judge6_score3/6`、`rerank_varweight_judge6_score3/6`、
`rerank_no_determ_k10hint`、`rerank_no_determ_k10hint6`。
这些是"问几维、怎么算弃权"的探索,结论(ask6+neutral 比部署配置高 0.0116 MRR)是在 **test** 上
看出来的,不能作为选择依据,也不进附录。部署配置用的是 **drop**(弃权不进分母),即论文一直写的
那条约定,没有被 test 影响过。

### 4. 会被 ask-6 臂取代的 det 打分臂(旧 → 新 映射)
| 移走的 | Table 3 的行 | 取代它的新 run |
|---|---|---|
| `rerank_ensemble_idx0/` | − ensemble (J=1) 的一半 | `rerank_arm6_llmonly_ensemble_idx0/` |
| `rerank_ensemble_idx1/` | − ensemble (J=1) 的另一半 | `rerank_arm6_llmonly_ensemble_idx1/` |
| `rerank_label_form/` | − label-form | `rerank_arm6_llmonly_label_form/` |
| `rerank_definition_form/` | − definition-form | `rerank_arm6_llmonly_definition_form/` |
| `rerank_mean_fusion/` | − summed fusion | `rerank_arm6_llmonly_mean_fusion/` |
| `rerank_raw_scaling/` | − score norm. | `rerank_arm6_llmonly_raw_scaling/` |

另外 `arm_windows/window_wcov0.jsonl` 也移走了:它与 `window_full.jsonl` 逐字节相同
(md5 `51f009f7…`),`stage_arm_windows.py` 的 `ARM_CONFIGS` 里根本没有 wcov0 这一项,拿它当
per-arm 窗口就是把窗口 confound 装回去。

### 5. w_cov=1 基线全部八个
`qwen3_32b_{direct_retrieval,one_pass_grounding,one_pass_structured,parallel_sampling,
parallel_sampling_diversity,decomposed_retrieval,intrinsic_self_refinement,
retrieval_feedback_refinement}_wcov1/`

六个是被取消的半成品(三个留着**半份** `bm25_candidates.jsonl`)。另外两个
(`direct_retrieval_wcov1`、`one_pass_grounding_wcov1`)是**跑完的、而且核对过的**
—— direct 的 0.2399/0.5620/0.1115 与 probe 的 cov-ON raw_context 行逐位相同。它们仍然重跑,
理由只有一个:它们起跑于 07-30 00:46 之前,而其余基线会起跑于之后。那次改动经逐 hunk 分类是
**instrumentation(token 计数)+ metadata(记录实际 w_cov)+ 一个由 `PARALLEL_PROMPT_STYLE`
环境变量门控、默认落回老代码的新分支**,所以数字本应完全一致 —— 但"本应"不是"一批一个版本"。
3.5 GPU-h 换掉这个疑点。

## 没有移走的,以及为什么

`verdicts_k10_fused` / `k5_fused` / `k20_fused`、`rerank_no_determ_k10fused`、
`rerank_no_verifier`、`rerank_no_llm`、`rerank_wcov0`、`rerank_oracle_single` 留在原地:
论文**现在**印的就是它们,新一批落地之前删掉会让表没有出处。它们会被逐个取代:

| 留着的 | 取代它的 |
|---|---|
| `verdicts_k10_fused` + `rerank_no_determ_k10fused` | `verdicts_arm6_full` + `rerank_arm6_full`(FHS 行) |
| `verdicts_k5_fused` / `verdicts_k20_fused` | `verdicts_arm6_k5` / `verdicts_arm6_k20`(K_v 表) |
| `rerank_no_verifier`(β=0)、`rerank_no_llm`(det 打分) | 不需要取代:它们**按定义**不含 LLM verifier,问几维与它们无关 |
| `rerank_wcov0` | 不重跑(用户已定:coverage 行用 caption 说明基准) |
| `rerank_oracle_single` | **不重跑**:oracle 对每个假设各自融合出一个排名(`core._evaluate_oracle`),
  per-arm 窗口文件每个 fact 只存一个窗口,要它入链得改窗口格式。它保持程序化打分,caption 里写明 |

## 要恢复任何一个

`mv _superseded_20260730/verifier_ablation/<dir> runs_ags_verifier_ablation/qwen3_32b/`
`mv _superseded_20260730/baselines/<dir> runs_fintagging_grounding_baseline/`

什么都没有删。磁盘是同一个文件系统,移动不占额外空间(项目盘已用 90%,这一点很重要)。
