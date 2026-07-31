# 主业:全文统一到"问6算6" — 清单与状态

部署配置(2026-07-30 起):verifier 被问的维度 = 被计分的维度 = 生成的全部六维
(FAMILY/ROLE/EVENT/QUALIFIER/SCOPE/TEMPORAL)。理由不是 test 上更好看,而是"问3算3"需要
一个按域的人工判断(哪些维度能靠候选文本判定),该判断不可迁移:ICD-10-CM 把侧别与就诊次序
直接写进码本文本。所有进论文的数字在此设置下重出;ask-3 的产物全部作废。

## 代码/配置(已改完,新 job 起跑前生效)
- [x] `core.LLM_VERIFIER_DIMENSIONS_DEFAULT` -> 六维
- [x] `run_llm_verifier.VERIFIER_DIMENSIONS` -> 六维;新增 `--judge-dimensions legacy` 仅用于读旧文件
- [x] **token 上限改为跟判定集大小走**(六维默认 2816;只有 legacy 用 1536)。原来只在
      `JUDGE_DIMENSIONS=all` 时才升上限,默认改成六维后不修会全量截断成"假弃权"
- [x] `llmonly_*` 臂的 `--llm-verifier-dimensions` pin -> 六维
- [x] `dump_reranked_ranking.py` 的 summary 记录 `llm_verifier_dimensions`(打分集此前无法从元数据还原)
- [ ] `verify_single_code_path.py` 的 PIN 改成六维,并把新记录的维度字段纳入比对
- [!] **prompt 里 `hypothesis_label` 还写着 "FAMILY/ROLE/EVENT only",而 scope="llm" 下
      现在列的是六维**(`run_llm_verifier.py:331`)。这是个 stale label,但**本轮不要改**:
      pending 的 job 起跑时才读代码,改了就会让七个臂分裂成两种 prompt。全文没有任何
      claim 引用这个标签,读者看不到;等这批 verdict 全部落地后再改

## GPU(问6)— job ID 已更新到 2026-07-30 02:00 这一批
20453191-99 与 20463488-97 两批都作废(前者配置未冻结,后者踩了 window for/else bug,
产物在 `_quarantine_windowbug_20260730/`)。现存活的是:
- [ ] 表3 五个按臂 verdict:20477403 def_only(R,+续跑 20482727) / 20477404 lab_only(R,+续跑
      20482728) / 20477405 ensemble_idx0 / 20477406 ensemble_idx1 / 20477407 mean_fusion
      **walltime 教训**:实测 2509 facts 需约 2h25(800 facts / 44 min 生成),原来只给 2h。
      pending 的已用 `scontrol update TimeLimit=3:00:00` 原地改(保住排队资历);running 的改不了
      (Access/permission denied),只能靠 `--resume` 续跑
- [ ] K_v 敏感性:20477408 (K_v=5, 2h) / 20477409 (K_v=20, 3h)
- [x] **FHS 行必须自己生成一份问6 verdict,不能借 `verdicts_k10_judge6`**。已提交
      20482731 `vf6_full`(`--window-tags arm_windows/window_full.jsonl`)+ 20482732 `rr_fhs_arm6`
      (afterok,CPU dump + listwise rerank → `rerank_arm6_full/`)。
      **原因(实测,不是推测)**:用 Qwen3-32B tokenizer 复原 prompt 逐 token 对上了
      —— `verdicts_k10_judge6` 是 `hypothesis_scope="all"`(只列已解析维度,标签写
      "resolved dimensions",fact0 = 1274 tok),而两个在跑的臂是 `scope="llm"`
      (六维全列、未解析的写成 `"UNRESOLVED"`,def_only 1326 / lab_only 1227,逐个精确命中)。
      拿 judge6 当 FHS 行,表3 每个 delta 里就混进一次 prompt 变更
- [ ] 七个臂的 listwise rerank(等各自 verdict 落地)
- [x] **`apply_server_verifier_ablation_rerank.sh` 里 llmonly_* 臂的 `--top-m 10` 全部删掉**。
      它把 verdict 通过 `--calls` 过滤,而这个脚本从不传 `--calls`,于是用了
      `dump_reranked_ranking.py` 的默认值 = 2025-07-25 的 **deterministic window** call log:
      实测 50180 → 42910 键(掉 14.5%),68% 的 fact top-10 变了,等于把 per-arm window 想
      去掉的那个 confound 又装回来。`rerank_llmonly_ask6_score6/` 就是这么来的(已弃用,
      其 rerank job 20463498 已 scancel)。dump 脚本现在会拒绝这种混搭
- [ ] 表12 两次 listwise rerank(不做本轮:用户已指示不追 dense/hybrid 的 FHS 行)
- [ ] 七个臂的 listwise rerank(等各自 verdict 落地)
- [ ] 表12 两次 listwise rerank(等 verdict 落地)
- [x] `seqvf_full` 20438554:起跑时读新常量,自动是问6算6
- [ ] 第二个域:codiesp 也按问6 配置(数据准备 20419632 之后)

## 已撤(方向已错,省下 GPU)
20446872/73/74/75、20448107(五个问3 按臂 verdict)、20440019/20(问3 dense/hybrid)、
20445021(问3 的 score-norm rerank)。

## 正文要改的地方
- [ ] §4.5「It rules on the three label-derived dimensions...」整段:改为验全部六维;删掉
      「QUALIFIER/SCOPE/TEMPORAL 需要结构化元数据而非文本」那条理由(在 ICD-10 上是错的)
- [ ] §4.5 末「scoring abstentions as non-support instead costs 0.002」:按六维重测
- [ ] `tab:schema` caption:六维现在全部被 verifier 判定(词表匹配仍只属符号检查)
- [ ] `tab:hyperparameters`:加/改「verified dimensions = 6」
- [ ] `tab:verifierfull` caption:两层现在覆盖同一组六维,第一次是严格配对比较
- [ ] §Cost / `tab:efficiency`:六维 verdict 的 completion token 约翻倍,7.7 s 要按新 job 的
      elapsed 重算
- [ ] `tab:llm_window_sensitivity`、`tab:ablation`、`tab:main_results` 的 FHS 行、
      `tab:verifierbridge`、`tab:retriever_robustness`:全部换成问6算6 的数
- [ ] `paper/PROVENANCE.md`:每条来源路径改到 ask6 产物,并写明 ask-3 已作废

---

## 2026-07-30 02:xx 这一轮做的事(链路已经全部挂好依赖,不需要人守着提交下一步)

**已提交的完整链条**(afterok 依赖,verdict 一落地 rerank 自己起):

| job | 内容 | 依赖 |
|---|---|---|
| 20477403 / 20477404 | vf6 def_only / lab_only verdict(RUNNING,2h 不够) | — |
| 20482727 / 20482728 | 上面两个的 `--resume` 续跑 | afterany 20477403 / 20477404 |
| 20477405 / 06 / 07 | ensemble_idx0 / idx1 / mean_fusion verdict(已改 3h) | — |
| 20477408 / 20477409 | K_v=5 / K_v=20 verdict(2h / 3h) | — |
| **20482731 `vf6_full`** | **FHS 自己的问6 verdict(`window_full.jsonl`)** | — |
| 20482732 `rr_fhs_arm6` | FHS 行的 CPU dump + listwise rerank → `rerank_arm6_full/` | afterok 20482731 |
| 20483746 | `llmonly_label_form`(−label-form 行,β=0.8) | afterok 20482727 |
| 20483747 | `llmonly_definition_form`(−definition-form 行,β=0.2) | afterok 20482728 |
| 20483748 / 49 | `llmonly_ensemble_idx0` / `idx1`(两者算术平均 = −ensemble 行) | afterok 20477405 / 06 |
| 20483750 | `llmonly_mean_fusion`(−summed fusion 行) | afterok 20477407 |
| 20483751 | `llmonly_raw_scaling`(−score norm. 行;窗口与 FHS 相同,复用 `verdicts_arm6_full`) | afterok 20482731 |

**注意 `llmonly_raw_scaling` 必须显式传 `VERDICTS`**:wrapper 里它是唯一豁免 VERDICTS 检查的臂,
不传就会静默用 `dump_reranked_ranking.py` 的默认 = 2025-07-25 那份 **问3 + det 窗口** 的 verdict。

## 还没有解决的两件事(都不在上面的链条里)

1. **`tab:ablation` 的 `Oracle best single` 行仍是 det 打分**。`stage_arm_windows.py` 的
   `ARM_CONFIGS` 里没有 oracle 条目(它靠 `--oracle-best-single`,每个 fact 挑最好的单假设,
   所以 fused 排名和窗口都和 FHS 不同),批次里也没有它的 verdict job。
   附录 §app:seq 那句「95% of the best-single-hypothesis oracle:0.401 against 0.420 Recall@10」
   因此是 **llm-only 的 FHS 对 det 的 oracle**,而且 0.401 本身会随问6 改。
   **按"便宜自洽"的原则**:等新的 FHS 数落地后,要么把这句改成明确写出 oracle 是程序化打分的上界,
   要么给 oracle 行加一次 verdict+rerank(2 个 GPU job,ARM_CONFIGS 加一行即可)。别默默留着。
2. **§4.4 末句「scoring abstentions as non-support instead costs 0.002 final accuracy」是问3 的数**。
   问6 下弃权多得多(SCOPE 只在 0.227 的机会上开火),这个 0.002 几乎肯定要变。
   `verdicts_arm6_full` 落地后可以**纯 CPU** 复现:同一份 verdict 分别用
   `--verifier-mode llm_drop` 与 `llm_strict` dump 两次,比较 retrieval-stage 的 MRR/R@10;
   要 final accuracy 就还得多一次 listwise rerank。先看 CPU 的差值有多大再决定。

---

## 落地后必须**同一次提交**改完的清单(2026-07-30 核对过依赖关系)

分两组,组内任何一格单独改都会让论文自相矛盾。

### 组 A:w_cov=1 基线(9 个 job)
1. `tab:main_results` 七个基线行 → 各 `runs_fintagging_grounding_baseline/qwen3_32b_<m>_wcov1/metrics.json`
2. `tab:queryform`(表 5)四行 → **cov-ON**
3. §3.3「63.8% / 14.5% / 23.7% / 0.069→0.113 / 58.1%」→ cov-ON 的对应值
4. Limitations 里同一组数字
5. `tab:evidence_type` 九个方法的 R@50 / MRR(重跑的 metrics 里有 `by_input_type`)

**已落地两个,并且有一个可证伪的自检通过**:`direct_retrieval_wcov1` 的
R@10 / R@50 / MRR = 0.2399 / 0.5620 / 0.1115,与 probe 的 cov-ON raw_context 行
(0.240 / 0.562 / 0.111)逐位相同 —— 这条 pipeline 没有 LLM,所以本该逐位相同,确实是。

**但 free-text 那条不会自动对上**:`one_pass_grounding_wcov1` 是
0.3045 / 0.4930 / 0.1702,probe 的 cov-ON freetext 是 0.304 / 0.496 / 0.171 —— R@50 差 0.003。
原因查明:重跑**重新生成了自己的 query description**(`query_descriptions.jsonl` 的 md5 与老 run
不同),所以两张表用的不是同一批 query。修法(不是改措辞):把 probe 的 freetext 行按**新的**
description 重算,已在本地 CPU 跑:
```
python3 run_ags_probe_queryform.py --modality pooled \
  --one-pass-queries runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding_wcov1/query_descriptions.jsonl \
  --output-dir runs_ags_probe_queryform/qwen3_32b_pooled_wcov1queries
```
日志 `probe_queryform_pooled_wcov1queries.log`。落地后 raw_context 行应仍是 0.2399/0.5620/0.1115,
freetext 行应变成与表 2 逐位相同的 0.3045/0.4930/0.1702。**这个自检不过就别改表**。

### 组 B:问6(11 个 job)
1. `tab:main_results` 的 FHS 行 → `rerank_arm6_full/metrics.json`
2. `tab:ablation`:FHS 行同上;`−label-form` / `−definition-form` / `−ensemble`(idx0/idx1 均值)/
   `−summed fusion` / `−score norm.` 五行 → `rerank_arm6_llmonly_*/metrics.json`
3. `tab:llm_window_sensitivity` 三行 → `verdicts_k5_ask6` / `verdicts_arm6_full` / `verdicts_k20_ask6`
   (CPU 重打分;脚本内自检:K_v=10 行必须逐位等于 `tab:ablation` 的 FHS 行)
4. `tab:verifierfull` / `tab:verifierbridge` → 用 `verdicts_arm6_full/llm_verifier_calls.jsonl`
   本地 CPU 重跑(**必须显式传 `--llm-calls`**),caption 里「两层现在覆盖同一组六维」
5. `tab:verifier_reranker_interaction` 的 On/Off 与 On/On → 同一份新 FHS 数
6. `tab:efficiency` 的 FHS 行:verifier 调用数不变(2/fact);completion token **实测 610 → 891**
   (mean,max 765 → 1137,`hit_token_cap` 全程 0),即 +46%,不是之前注释里估的翻倍。
   Time 要按 `vf6_full` 的 sacct elapsed 重算(现值 7.7 = 5.44 + 2.28,后一项是问3 的 verdict 生成)
7. §4.1 的「+0.029 R@10 / +0.054 MRR / +0.023 Acc」、§1 与 §Ablations 的 0.074 / 0.017、
   §app:seq 的「0.401 vs 0.420」——全部依赖 FHS 的新值,逐条重算
8. §4.4 末句的 0.002(见上一节)

### 组 C:FHS-Seq(4 个 shard job)
`tab:main_results` 的 FHS-Seq 行、`tab:seq_outcome` 两格、附录 I 复述段、`tab:efficiency` 的
FHS-Seq 行(现在整行缺)。自检:Rd-1 的 R@50 按构造应落在 FHS 自己的值上(问3 时是 0.543,
问6 后要用新的 FHS R@50)。

### 只剩 std 是"已知不是测量值"
其余每一格落地后都可追到一个文件。std 的 55 格是解析估计,而 §Experiment Settings 仍写着
"mean under 3 runs with different random seeds" —— 这是用户明确决定暂时保留的状态。

## 2026-07-30 02:50 补:verdict 完整性现在是硬失败

`run_llm_verifier.py` 以前从不检查落盘的 verdict 是否覆盖整个窗口 —— 42,910 那次就是这类静默半份
覆盖,而下游 rerank 现在是 `afterok` 挂上去的,半份 verdict 会被照单全收。现在在**两个文件都写完
之后**加了检查(所以失败不丢任何工作,`--resume` 能接着跑),不完整就 exit 1,`afterok` 不会触发。

四个分支都执行验证过,不是只编译:
- `--window-tags` 完整:`completeness: 50180/50180`,exit 0
- `--window-tags` 少 20 键(往窗口文件里塞一个 trace 里不存在的 fact):`INCOMPLETE ... 20 short`,
  exit 1,`llm_verifier_calls.jsonl` / `verdicts.json` / `summary.json` 三个文件都还在
- 无 `--window-tags`(k5/k20 走的分支)完整:`completeness: 2509/2509 facts visited`,exit 0
- 带 `--limit`:检查跳过(smoke 不会假失败)

已同步到 `fhs/src/verifier/run_llm_verifier.py`(合并时有一处冲突残留被 `py_compile` 当场抓到 ——
`compileall` 必须包含 SyntaxError 才看得见这类问题,见记忆 fhs-folder-reorg 第 3 条)。

---

# 2026-07-30 03:11 — 全停 + 一批提交(用户拍板)

用户的判断:改一次代码就给还在跑的东西引入一个版本差,这个循环一周了。所以:先全停,验"一个框架 /
一个方法",再一起提交一批,期间**代码冻结**(见 `CODE_FREEZE.md`)。

## 停之前的机械判据(`verify_single_code_path.py`)
**15 行里 5 行在钉住的配置上,10 行漂移**:7 个消融臂是 det 打分(FHS 行是 llm_drop)、
3 个 verdict 集是问3。框架层是干净的:十个基线都是同一个 `run_fintagging_grounding_baseline.py`
的 `query_mode` 分支;十二个消融臂都对同一份 frozen trace 用同一个 `core.evaluate()` 打分,
没有任何臂换过 `--test-trace`。

## 停了什么
FHS 侧 28 个 job,02:59:10 一次性按 ID 取消(codiesp 与 jupyter 未动;codiesp 那批是
03:08:49 被账号里另一处取消的,不是我)。产物移到 `_superseded_20260730/`(移动不删除,
WHY.md 里有旧→新映射)。

## 这一批(26 个,`ask6_batch_20260730_jobids.txt`)
7 个 w_cov=1 基线 + 8 个问6 verdict + 7 次 rerank(afterok)+ 4 个 FHS-Seq 分片。
`one_pass_structured` **不在**里面:它的配置本来就 pin 了 w_cov=1。

## 这一批修掉的一个隐藏不一致
`one_pass_grounding` 的 w_cov=1 重跑本该"只改 coverage",实际换了 **254/2509(10%)**条 query
—— 因为配方里写的是 `RESUME=0`,进的是空目录,贪心解码在 vLLM 里换个批次组合就有约 10% 的
token 分叉。这一批 phase 0 先把已发表的 `query_descriptions.jsonl` 播种进输出目录并核对 md5,
再用 `RESUME=1`,所以表 2 的 free-text 行与表 5 的 freetext 行会逐位相同。

## 下一批再说的(冻结期内只记录,不动手)
- `tab:ablation` 的 `Oracle best single`:`core._evaluate_oracle` 对每个假设各自融合,窗口是
  per-hypothesis,而 `--window-tags` 每个 fact 只存一个窗口 → 要入链得改窗口格式。本轮用 caption。
- §4.4 末句「abstention as non-support costs 0.002」:等 `verdicts_arm6_full` 落地后纯 CPU 重测。
- `run_llm_verifier.py` 的 `hypothesis_label` 还写着 "FAMILY/ROLE/EVENT only"(实际列六维)。
  **冻结期禁止改**:pending job 起跑才读代码,改了就会把这批劈成两种 prompt。
- `tab:retriever_robustness` 的 dense/hybrid FHS 行(用户已指示本轮不追)。

## 三问的机械答案(2026-07-30 03:2x,证据都在仓库里)

1. **一个版本**:`FREEZE_MANIFEST.txt` 记了这批会读到的 26 个文件的 sha256,**没有一个的 mtime
   晚于提交时刻 03:10:57**。job 全部跑完后重算一次这个 manifest,就能证明期间没人动过。
2. **一个框架**:四类入口(基线 / seq / verdict / rerank)最终都执行
   `run_fintagging_grounding_baseline.py`;`run_llm_verifier.py`、`dump_reranked_ranking.py`、
   `ags_seq_verifier_arm.py`、`ags_frozen_grounding.py`、`core.py` 全部 import 它。同一个索引、
   同一个 tokenizer、同一个 listwise selector。
3. **一个算法**:七个臂(含 FHS 行)都是 `AblationConfig` 的字段改动,过同一个
   `core.evaluate()`,打分同一份 frozen trace,`verifier_mode` 全是 `llm_drop`,
   `llm_unjudged_fill` 七个臂全部显式 `mean`(注意 dataclass 默认是 `zero`,所以必须显式写)。

   **唯一的例外要写进 caption**:`− label-form` 与 `− definition-form` 除了少一种渲染,β 也不同
   (0.8 / 0.2 vs 0.6)。出处 `selected_betas.json`:规则是"只有当臂改变了被融合的 ranking 数量
   或 scaling 时才在开发集上重扫 β",去掉一种渲染正好把 ranking 数量减半,所以这两臂重扫;
   `mean_fusion` / `index_ablation` 按同一条规则**保持** 0.6。选择规则是 dev 上
   `argmax table/recall_at_10`,MRR 破平局,扫 6 个值。
   另外 `selected_betas.json` 里 `raw` 这一项是 0.2,但论文的 `− score norm.` 用的是
   **β=0.6 的"deliberately untuned"那一版**(`raw_at_0.6`),wrapper 里也确实是 0.6 —— 一致,
   但别被那个 0.2 的条目误导。

## 批次进度 — 2026-07-30 12:2x(26 个 job 的第一次盘点)

已完成 11:`w1_direct` 20488395、`w1_freetext` 20488396、`v6_full` 20488402、`v6_def_only` 20488403、
`v6_lab_only` 20488404、`v6_ensemble_idx0` 20488405、`v6_ensemble_idx1` 20488406、
`v6_mean_fusion` 20488407、`v6_k5` 20488408、`r6_no_determ` 20488410,外加批次外的
`codiesp_direct_retrieval_full_wcov0` 20488374。
在跑/排队 10:`v6_k20` 20488409、6 个 `r6_llmonly_*` 20488411-16、4 个 `seqvf_s*` 20488417-20。

**超时 5 个,全在 phase 1 的 w_cov=1 基线**(其余 21 个没有一个超时):

| job | 死在哪 | 已完成 | 续跑 job | 新 walltime |
|---|---|---|---|---|
| 20488397 `w1_par_iid` | rerank | 候选 2509/2509,rerank 1792/2509 | **20559656** | 2:30 |
| 20488399 `w1_decomp` | rerank | 候选 2509/2509,rerank 1792/2509 | **20559657** | 2:30 |
| 20488400 `w1_intrins` | rerank | 候选 2509/2509,rerank 480/2509 | **20559658** | 3:30 |
| 20488401 `w1_feedbk` | rerank | 候选 2509/2509,rerank 992/2509 | **20559659** | 3:00 |
| 20488398 `w1_par_div` | generation | traces 1730/2509,还没建候选 | **20559660** | 8:00 |

**有 resume,所以不重跑**:`RESUME=1` → `run_fintagging_grounding_baseline.sh` 加 `--resume`,
query_descriptions / grounding_traces / qwen_rerank_predictions 三个阶段都按 `example_idx`
跳过已写入的行(`load_existing_predictions` / `load_existing_method_records`),再以 `mode="a"` 追加。
续跑前逐个校验过:五个断点文件**都以完整 JSON 行 + 换行结尾**,追加不会写坏。
四个 rerank 断点额外把 `REUSE_CANDIDATES` 从 0 改成 1,直接吃已完成的 2,509 行
`bm25_candidates.jsonl`;`par_div` 没有候选文件,该开关按
`reuse_candidates and candidates_path.exists()` 自动落空,等价于原样重续 generation。
其余 export 与 `submit_ask6_batch_20260730.sh` 逐字一致(含 par_iid 的
`PARALLEL_PROMPT_STYLE=plain` + `QUERY_TEMPERATURE=0.8`)。

walltime 按实测速率给的:rerank 18-19 条/分(par_iid/decomp 还差 717 条≈0.6h,intrins 2029≈1.8h,
feedbk 1517≈1.4h);par_div generation 5.9 条/分,还差 779 条≈2.2h,再加全量 2,509 条 rerank≈2.2h。

**没有破冻结**:续跑脚本写在会话 scratchpad 里(`resubmit_w1_timeouts.sh`),只调用未改动的 wrapper,
`data_whole_pipeline/` 里没有任何 `.py` / `.sh` 被碰过;job id 已追加进
`ask6_batch_20260730_jobids.txt`。

## 下一批:把 `− label coverage` 行改成"FHS 关掉 cov",而不是另一个脚本的重检索

用户 2026-07-30 的判断:消融行必须是**同一个方法减掉一个组件**,所以这一行应该出自 FHS 本身,
不该是 `run_index_ablation.py` 的独立重检索。同意,现在这行的四个数来自两个不同的 run,
且没有候选级 verifier。

**直接"在 FHS 里把 w_cov 设成 0"是做不到的,有三道硬闸(都不是 bug,是防漂移设计):**
1. `ags_frozen_grounding.py` `frozen_ags_startup_assertions`:`retriever.label_coverage_weight <= 0`
   直接 `AssertionError("frozen_ags requires ... w_cov > 0")`;
2. 同文件 `_FROZEN_VARIANTS[frozen_ags]` 把 `label_coverage_weight` 钉死 1.0,`_assert_frozen`
   查到不等就报 config drifted;
3. `run_fintagging_grounding_baseline.py:3540` 的 `--label-coverage-weight` 覆盖分支,一旦与
   frozen 变体钉住的值冲突就 `SystemExit`。

**不用碰这三道闸也能拿到用户要的东西**:w_cov 只作用在检索打分,假设生成在检索之前,所以
"FHS 关掉 cov"= 拿 **FHS 自己的假设**在 w_cov=0 下重新融合 —— 这正是
`ags_table5_ablation/dump_index_ablation_ranking.py` 的功能,而且它已经具备需要的一切:
`--label-coverage-weight`、`--verifier-mode llm_drop`(默认)、`--llm-unjudged-fill mean`(默认,
注意 AblationConfig 自己的默认是 `zero`)、就地重写 `rounds`(否则会复现 w_cov=1 的融合与窗口),
并且在 w_cov=1 上做过 60/60 tag-for-tag 自检。这也和其余 7 个臂的做法一致:它们都是在同一份
frozen trace 上改打分,而不是重新生成。

**四步(0 处代码改动):**
1. CPU:`dump_index_ablation_ranking.py --label-coverage-weight 0 --output .../trace_wcov0.jsonl`
   (stage 1,不带 verdicts,产出前验证器融合序 + 重写过的 rounds);
2. CPU:`stage_arm_windows.py` → `arm_windows/window_wcov0.jsonl`(该臂**自己的**融合排序);
3. GPU ~2h30:verdict 生成,`--window-tags window_wcov0.jsonl`,问 6,输出 `verdicts_arm6_wcov0/`;
4. GPU ~2h30:listwise rerank → `rerank_arm6_wcov0/metrics.json`,四列同源。

完成后可以撤掉 tab:ablation caption 里那句"该行为检索阶段重检索、需对着 0.383 读"的说明,
并把 `paper/PROVENANCE.md` 的对应条目改掉。参见 [[coverage-row-not-on-deployed-path]]。

**执行时机:等这批 26 个全部排完再做。** 用户明确要求不要影响正在跑的 job,所以连第 1、2 步的
CPU 部分也不提前跑 —— 冻结期内不往 `runs_ags_verifier_ablation/` 里写任何新文件。

## 表格重写时的耦合编辑:sum vs mean fusion 的措辞

问 6 的新数把 `− summed fusion` 与 FHS 的差抹平了,并在两项上翻转:

| | R@10 | MRR | Acc |
|---|---|---|---|
| FHS(sum) | 0.3970 | 0.2573 | 0.2551 |
| − summed fusion(mean) | 0.3938 | **0.2605** | **0.2555** |

论文当前印的是 `0.401 against 0.384`(+0.017);新数是 +0.0032,且 MRR/Acc 上 mean 略高。
两处必须与 tab:ablation 的数在**同一次提交**里改:

1. §方法(`acl_latex_llmonly (2).tex` 第 145 行)"Summation rather than averaging is the one choice
   here that is not standard, and it is deliberate ... the sum carries a multiplicity bonus ...
   Table 5 replaces it with mean RRF and **reports the cost**" —— 现在没有 cost 可报,这句读起来
   像在宣称机制性收益。降调成"开发集选型,测试集上与 mean 无实质差别"。
2. §结果(第 299 行)"Summed rather than mean fusion is worth little (0.401 against 0.384
   Recall@10)" —— 换数,并改成"在噪声内,mean 在 MRR/Acc 上名义更高"。

**框架不用推翻**:开发集附录(第 502 行)已经预先写了"we treat this development contrast as a
selection result rather than a general component claim",新数只让它更成立。

**连带风险**:救回这个框架的那句是 "within test uncertainty",而 uncertainty 来自 std 列 ——
那一列是建模估的、不是三种子实测(见 [[std-column-claims-3-seeds]])。"差异在噪声内"目前没有
实测方差支撑,审稿人可以把这两点串起来。

## review 清单 P0-1 已被问 6 解掉:`− ensemble (J=1)` 不再输给 one-pass structured

reviewer 的矛盾陈述(`review/FHS-revision-checklist.md` 第 10-24 行):J=1 的 FHS 保留了 dual
rendering 和 verifier,理应 ≥ one-pass structured,但旧表里 0.366/0.178 全面低于 0.372/0.203;
而同一张表又说去掉 verifier 会让 MRR 从 0.257 掉到 0.205 —— 两者不能同时成立。

问 6 之后解掉了(J=1 取 idx0/idx1 算术平均):

| | R@10 | MRR | Acc |
|---|---|---|---|
| One-pass, structured | 0.3723 | 0.2027 | 0.2260 |
| − ensemble (J=1) | 0.3824 | 0.2460 | 0.2429 |
| FHS (full) | 0.3970 | 0.2573 | 0.2551 |

清单给的两条出路是"配置有差异就写明"或"配置相同则是 bug 重跑";实际是第三种 —— 旧 verdict 只问
三维,J=1 臂被削得最狠。

**写正文时必须守住的三点:**

1. **这两行的 "J=1" 不是同一个配置**,差三样:解码 T=0(贪心)vs T=0.8;β=0.0 vs 0.6;
   无 verifier vs 有。所以 +0.043 MRR 不是"集成之外还剩什么",而是单假设下 verifier+β 的价值。
   这同时回答了 reviewer 三个"需确认"里的前两个。
2. **集成的是假设,不是验证器。** J=2 确实是每 fact 2 次 verifier 调用(实测 5018/2509=2.0),
   但被融合的 ranking 数量同时从 1 变 2,这个消融分不开两者。J=1→full 的 **+0.011 MRR** 只能
   归给"集成整体",不能归给"更多验证器"。真要分开需要一个新臂:J=2 生成、两份 ranking 照常
   融合、但只消费一个假设的 verdict —— 现在没有。
3. 建议正文分层写:one-pass structured →(+0.043 MRR,verifier+β)→ J=1 →(+0.011 MRR,集成)
   → FHS full。

**还没被证伪的一条**:reviewer 第 22 行 "J=1 时 fused range 极窄,β=0.6 可能被严重错配"。新数里
J=1 不再垫底,但这个机制只是不再表现为异常,没有被排除。可用已有 per-fact 分数在 **CPU** 上直接
查 J=1 的 fused range 分布,不需要 GPU。

---

# 2026-07-30 23:xx — 已把这一版填进论文(FHS-Seq 除外)

活文件:`paper/acl_latex_llmonly (2).tex`。备份 `*.bak-before-ask6-fill-20260730`。
**所有更新过的格子和句子都包在 `\upd{}` 里(红色)**;拿不到的值包在 `\ph{}` 里。
DEV 表(`tab:devsample` / `tab:coveragegain` / `tab:pilot` / `tab:beta` / `tab:configabl`)按用户指示未动。

## 改法
不是按表改,是**先列出这一版会作废的旧值,再对全文 grep 出所有出现位置逐个看**。
这一步抓到了只改表会漏的四处:`0.543` 在附录 615/643/646、`0.401` 在 621、`0.249` 在 887/890、
`0.237` 在 Limitations 319。脚本留在会话 scratchpad。

## 落地的表
`tab:main_results`(7 基线 + FHS)、`tab:ablation`(6 行)、`tab:queryform`、`tab:evidence_type`(9 行)、
`tab:llm_window_sensitivity`、`tab:verifierfull`、`tab:verifierbridge`、
`tab:verifier_reranker_interaction`、`tab:retriever_robustness` 的两个 BM25 行。

## 落地的正文
§1 的 0.074/0.017、§3.3 的四个百分比、§4.1 主结果段(重写)、§4 消融段(重写)、
§4 modality 段(重写)、§4.5 弃权那句、Limitations 的深度权衡、app:queryform 两个配对对比、
app:window 的区间、app:seq 的 0.401→0.397 与 95%→94%。

## 三条自洽性(机械核过,脚本在 scratchpad)
① 一个框架:九个方法七项配置字段逐字相同,最后都调 `run_fintagging_grounding_baseline.py`,
rerank flag 一致;② 一个方法:七个臂全部 `llm_drop` + `mean` + 六维 + `verdicts_arm6_*`,
只有 β 按既定规则在两个臂上不同;③ 一个版本:八份 verdict 全 `dims=6` / `fused` / `parse_rate=1.0`,
opportunity 逐个等于 `2509 × K_v × 2`。

## 这一轮改的代码(都不在冻结集,precheck 过,双分支实测)
- `run_ags_verification_quality.py` / `run_ags_verifier_bridge.py`:`LLM_DIMENSIONS` 由硬编码三维
  改成 `--llm-dimensions {deployed,legacy}`,默认 **deployed=六维**。原来问6 的 verdict 被按三维计分,
  QUALIFIER/SCOPE/TEMPORAL 的 coverage 全是 0.0,读起来像 LLM 弃权。两个分支都跑过:
  deterministic 行逐位不变,只有 LLM 层变。
- `count_placeholders.py`:现在也认 `\ph{--}`,否则整表未填会报 0。两个分支都测过。

## 还欠的
1. **FHS-Seq**:`tab:main_results` 一行 + `tab:seq_outcome` 六格是 `\ph{--}`;
   §app:seq 和 Conclusion 里的结论句已改成"待测"而不是断言。20488418/19/20 明早跑完后填。
2. `tab:efficiency` 的 FHS 延迟:六维 verifier 的 prompt 翻倍,7.7 s 是旧值,caption 里已标注待重测。
3. `std` 列仍是解析估计(见 [[std-column-claims-3-seeds]]),这一轮没动。
4. `verify_single_code_path.py` 的 PIN 还指着 ask-3 的目录名,证不了这批(见本文件上面那条未打勾项)。
5. `paper/PROVENANCE.md` 的来源路径还没改到 arm6。

## 2026-07-31 补:`tab:ablation` 三个未更新行的定性

- `− verifier`:β=**0.0**、`llm_verdicts_used=0` → 重排项被乘掉,是纯融合排序。
  旁证:R@50 / R@200 与 FHS 逐位相同(0.5428 / 0.7055)。问6 不作用于它,**不用改**。
- `Program-driven score`:`verifier_mode=deterministic`、β=0.6、不读 verdict,这就是该行的定义。**不用改**。
- `Oracle best single`:**不是"本来就对"**,是没修的偏差,而且不止打分项:
  它 per-hypothesis 融合,**池子都不同**(R@200 0.7429 vs 其余全部 0.7055)。
  caption 已补一句写明它不是 matched arm、只是假设集上的选择上界。修它要改窗口文件格式,不在这一批。

# 2026-07-31 — Oracle 行修好了,零代码改动、零新 verdict

## 清单里"必须改窗口格式"那句是错的
窗口文件格式**一直有 `hypothesis_indices`**。真正的限制在读取端:
`run_llm_verifier.py:643` 是 `arm_windows[fact_id] = {...}`,同一 fact 的第二行会覆盖第一行。

**但这条限制根本用不着碰**,因为:`core._evaluate_oracle` 对假设 j 走的是
`_rankings_for(fact,[j])` → `fuse` → `rerank(...,[j])`,**和 `ensemble_idx{j}` 臂是同一件事**。
实测(`oracle_window_check.py`):oracle 的每假设 top-10 窗口与 `window_ensemble_idx{j}.jsonl`
**2509/2509 逐个相同,两个假设都是**;`truncate_pool_to_top_k` 从未改变过 top-10。
所以 `verdicts_arm6_ensemble_idx0/idx1` 正好就是 oracle 需要的 verdict。

## 做法
1. CPU:两份 verdict 拼接(键 `(fact_id, hypothesis_idx, tag)`,按 hypothesis_idx 天然不相交,
   已断言无重复)→ `oracle_arm6_llmdrop/llm_verifier_verdicts.json`,同目录 `WHY.md` 写明这是拼接不是生成。
2. CPU 自检:用 `verifier_mode=deterministic` 重跑 oracle,**逐位复现盘上
   `rerank_oracle_single/metrics.json`**(0.1244/0.4205/0.5971/0.2209)—— 证明 CPU 路径与原 GPU run 同源。
3. CPU:`dump_reranked_ranking.py --oracle-best-single --verifier-mode llm_drop`
   (该脚本本来就有这个 flag)→ `rerank_arm6_oracle/bm25_candidates.jsonl`。
   **闸门**:dump 出的排序逐位复现步骤 2 的 llm_drop 数,过了才提交 GPU。
4. GPU:`sbatch ARM=llmonly_oracle_single`(wrapper 里本来就有这个臂)→ job **20682177**,只补 Acc 一列。

## 结果:原来那行是被程序化打分压低了头部,不是"上界偏高"

| | R@1 | R@10 | R@50 | R@200 | MRR |
|---|---|---|---|---|---|
| 旧(程序化打分) | 0.1244 | 0.4205 | 0.5971 | 0.7429 | 0.2209 |
| **新(llm_drop 六维)** | **0.2220** | **0.4368** | 0.5923 | 0.7338 | **0.2950** |

连带改掉:`tab:ablation` 的 oracle 行(Acc 暂为 `\ph{--}`)、caption(不再说它是 det 打分,
改成说明它唯一剩下的差异是 per-hypothesis 融合导致 R@200 0.734 vs 0.705)、
§app:seq 的「95% → 91%」并补了 R@1 0.184 vs 0.222 这个真正的残差位置。

# 2026-07-31 — `− label coverage` 行已入链(job 20684342 -> 20684343)

## 盘上那份 w_cov=0 staged trace 是坏的,已弃用
`rerank_wcov0/bm25_candidates.jsonl`(2026-07-28)的 `rounds` 里
`retrieval_score = bm25_norm + 1.0×(label_cov + qcov)` 逐位成立 —— 还是 **w_cov=1** 的分。
它早于 `dump_index_ablation_ranking.py` 的"就地重写 rounds",summary 里没有 `rounds_rewritten`。
后果:从它 staged 出来的窗口与部署窗口 **2509/2509 相同**,等于根本没关掉 cov。
`arm_windows_wcov0/` 那份 07-29 的窗口同样作废。

## 重做后的两道闸门(都过了)
- 新窗口 vs 部署窗口:**0/2509 相同**
- `rerank_arm6_wcov0/ranking_summary.json`:`rounds_rewritten=true`, `label_coverage_weight=0.0`

## 目录约定
- `wcov0_stage1/` —— **只是 stage 1**,预验证器序、beta=0、无 verdict,**不可 rerank**,
  只作 `stage_arm_windows.py` 和 verdict job 的 `--test-trace`。目录里有 WHY.md 写明。
- `arm_windows_arm6_wcov0/window_full.jsonl` —— 该臂自己的 w_cov=0 窗口
- `verdicts_arm6_wcov0/` —— job 20684342(问6,`--window-tags` 上面那个)
- `rerank_arm6_wcov0/` —— job 20684343(afterok),四列同源的可报告行

## 代码改动(precheck 过,三分支实测)
`apply_server_wcov0_rerank.sh` 加了 `VERDICTS` 透传:
`${VERDICTS:+--llm-verifier-verdicts "${VERDICTS}"}`,并在文件不存在时 exit 1。
不传就退化成原来的"无 verifier 重检索"。测过:空 / 不存在(guard 触发)/ 正常文件。

## 落地后要改的
`tab:ablation` 的 `− label coverage` 四列换成 `rerank_arm6_wcov0/metrics.json`,
并**删掉 caption 里**"\emph{$-$ label coverage} is a re-retrieval without the verifier whose
retrieval and accuracy columns come from separate runs"这句,以及正文里那句
"its retrieval columns belong against the matched w_cov=1 control through that same path
(Recall@10 0.383)"。届时它就是第七个匹配臂。

# 2026-07-31 — 表状态单独成文:`paper/TABLE_STATUS.md`

19 张表(24 个 label 去重后)逐张列了状态、来源、以及红格对应哪个 job。
颜色规则已改成:**黑 = 这一版的真实测量;红 = 不是**(原来是"红 = 我改过",分不清真假)。
全文 160 处 `\upd{}` 已剥掉,只剩 `\ph{}`。

没有 job 的三类,都写在 TABLE_STATUS.md 的 C 节:std 列 26 格、Dense/Hybrid 12 格、
`tab:efficiency` 两格 Time(测不了,verifier 没有计时、one-pass 没有 wall_time 字段)。
