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
