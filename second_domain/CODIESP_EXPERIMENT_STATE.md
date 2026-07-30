# CodiEsp Second-Domain Experiment State

Last updated: 2026-07-30 03:20 America/New_York.

This top section is the handoff for the next session. Lower sections contain
older history and are not the authoritative current queue state.

Operational guardrails:

- Only touch jobs from this CodiEsp experiment, job names beginning `codiesp_`.
- Do not cancel or modify unrelated jobs such as `vf6_*`, `b_*`, `seqvf_*`, or
  Jupyter/other user jobs.
- CPU-only work should run locally in this package; do not use `sbatch` for pure
  CPU validation/submission wrappers.

## Current Revised-Matrix Status

Active input set is the exact-only full CodiEsp diagnosis slice:

- `data/codiesp/facts_test_full_exact.jsonl`: 3144 facts.
- `data/codiesp/evidence_relocations_full_exact.jsonl`: 3144 relocation rows.
- `data/codiesp/stats_full_exact.json`: exact rate 1.0, parse rate 1.0.
- `data/codiesp/test_docs_full_exact.txt`: 250 docs.

Current coverage-expanded matrix. Gold oracle is run once; the other seven
methods are run with both `wcov0` and `wcov1`, for 15 final result units total.

Clean restart note:

- All previous CodiEsp GPU jobs were cancelled or were already terminal.
- `runs_codiesp_grounding_baseline/` and `logs/` were cleared before restart.
- The valid result wave is the current-code submission wave from 2026-07-30
  03:10-03:18 EDT. Do not mix in earlier job outputs.

Method grouping for reporting:

- Independent comparison/baseline methods: `direct_retrieval`,
  `one_pass_grounding`, `parallel_sampling_n2`, and the CPU-only
  `gold_label_definition_retrieval` oracle diagnostic.
- FHS-family ablations: `one_pass_structured`, `frozen_ags`, `fhs_j1`, and
  `fhs_no_verifier`. These all run through the same
  `ags_frozen_grounding.build_frozen_ags_method_record` implementation and
  differ only by frozen config constants (`J`, `beta`, verifier use, and
  coverage setting).

| Method | Coverage | Status | Job | Output |
|---|---|---|---|---|
| `direct_retrieval` | `wcov0` | pending in clean restart | `20488374` | `runs_codiesp_grounding_baseline/qwen3_32b_direct_retrieval_full_wcov0/` |
| `direct_retrieval` | `wcov1` | BM25/candidates rebuilt locally from current code; GPU rerank pending | `20488827` | `runs_codiesp_grounding_baseline/qwen3_32b_direct_retrieval_full_wcov1/` |
| `gold_label_definition_retrieval` | single oracle | final done locally, CPU-only | local | `runs_codiesp_grounding_baseline/qwen3_32b_gold_label_definition_retrieval_full_exact/` |
| `one_pass_grounding` | `wcov0` | pending in clean restart | `20488828` | `runs_codiesp_grounding_baseline/qwen3_32b_one_pass_grounding_full_wcov0/` |
| `one_pass_grounding` | `wcov1` | pending in clean restart | `20488829` | `runs_codiesp_grounding_baseline/qwen3_32b_one_pass_grounding_full_wcov1/` |
| `one_pass_structured` | `wcov0` | pending in clean restart | `20488830` | `runs_codiesp_grounding_baseline/qwen3_32b_one_pass_structured_full_wcov0/` |
| `one_pass_structured` | `wcov1` | pending in clean restart | `20488831` | `runs_codiesp_grounding_baseline/qwen3_32b_one_pass_structured_full_wcov1/` |
| `parallel_sampling`, N=2 | `wcov0` | pending in clean restart; comparison baseline, not an FHS ablation | `20489123` | `runs_codiesp_grounding_baseline/qwen3_32b_parallel_sampling_n2_full_wcov0/` |
| `parallel_sampling`, N=2 | `wcov1` | pending in clean restart; comparison baseline, not an FHS ablation | `20489162` | `runs_codiesp_grounding_baseline/qwen3_32b_parallel_sampling_n2_full_wcov1/` |
| `frozen_ags` / FHS full | `wcov0` | pending in clean restart | `20488832` | `runs_codiesp_grounding_baseline/qwen3_32b_frozen_ags_full_wcov0/` |
| `frozen_ags` / FHS full | `wcov1` | pending in clean restart | `20488833` | `runs_codiesp_grounding_baseline/qwen3_32b_frozen_ags_full_wcov1/` |
| `fhs_j1` | `wcov0` | pending in clean restart; FHS-family ablation | `20489164` | `runs_codiesp_grounding_baseline/qwen3_32b_fhs_j1_full_wcov0/` |
| `fhs_j1` | `wcov1` | pending in clean restart; FHS-family ablation | `20489165` | `runs_codiesp_grounding_baseline/qwen3_32b_fhs_j1_full_wcov1/` |
| `fhs_no_verifier` | `wcov0` | pending in clean restart; FHS-family ablation | `20489163` | `runs_codiesp_grounding_baseline/qwen3_32b_fhs_no_verifier_full_wcov0/` |
| `fhs_no_verifier` | `wcov1` | pending in clean restart; FHS-family ablation | `20489166` | `runs_codiesp_grounding_baseline/qwen3_32b_fhs_no_verifier_full_wcov1/` |

Completed current-code CPU metrics so far:

- #2 oracle retrieval:
  - BM25 top-1 `0.998410`, MRR `0.999205`, Recall@10/50/200 `1.0`.
- Direct retrieval `wcov1` BM25/candidates:
  - BM25 top-1 `0.046120`, MRR `0.087444`, Recall@10 `0.174300`, Recall@50 `0.314249`, Recall@200 `0.466603`.
  - GPU rerank is pending as job `20488827`.

Current partial outputs at handoff:

- `direct_retrieval wcov1`: current-code `bm25_candidates.jsonl` and
  `bm25_metrics.json` exist.
- `gold_label_definition_retrieval`: current-code `bm25_candidates.jsonl` and
  `bm25_metrics.json` exist.
- No current-code GPU outputs have been written yet; all GPU jobs are pending.

Old revised-out `wcov1` duplicate jobs were cancelled:

- `20459407`: direct retrieval `wcov1`, cancelled after GPU-idle warning before writing candidates.
- `20459409`: one-pass free-text `wcov1`, cancelled while pending.
- `20459411`: one-pass structured `wcov1`, cancelled while pending.
- `20459413`: FHS full `wcov1`, cancelled while pending.
- `20473793`: parallel sampling `wcov0`, cancelled while pending because it was submitted without GPU resources.
- `20473794`: FHS J=1 `wcov0`, cancelled while pending because it was submitted without GPU resources.
- `20473795`: FHS no-verifier `wcov0`, failed before work because it was submitted without GPU resources; replaced by `20474738`.

Submission scripts now reflect this split:

- `submit_codiesp_full_local.sh`: validates exact-only data and submits base methods
  for `wcov0` and `wcov1`; direct retrieval `wcov1` generates candidates locally
  before GPU rerank to avoid GPU idle.
- `submit_codiesp_new_requirements.sh`: runs gold oracle once locally and submits
  the independent `parallel_sampling_n2` comparison plus FHS-family ablations
  for both `wcov0` and `wcov1`, with explicit GPU resources.

## Current Goal

Prepare a clean full CodiEsp / ICD-10-CM second-domain dataset, then run the four-method transfer experiment for each label-coverage setting:

- `wcov0`: `LABEL_COVERAGE_WEIGHT=0.0`
- `wcov1`: `LABEL_COVERAGE_WEIGHT=1.0`

Important caveats:

- This is currently one run per method/coverage combination, not three seeds. A three-seed run would require `4 methods x 2 coverage settings x 3 seeds = 24 jobs` and seed-specific output directories.
- The currently submitted jobs use `data/codiesp/facts_test.jsonl`, which was generated with the preparation script default `--target-facts 250`. That is a 22-document / 250-target sample, not the full CodiEsp test split.
- The full CodiEsp test split has 250 documents/cases. For diagnosis grounding, `testX.tsv` contains 3665 `DIAGNOSTICO` rows; after filtering to ICD-10-CM inventory codes and the same dedup key used by the script, there are 3431 diagnosis target facts across all 250 documents.

## Submitted Jobs

These eight sample jobs were submitted on 2026-07-29 and then canceled before running because the input file was only a 22-document / 250-target sample and evidence relocation was not final-clean:

| Job ID | Method | Coverage | Output suffix | Time limit |
|---|---|---|---|---|
| `20411011` | `direct_retrieval` | `wcov0` | `qwen3_32b_direct_retrieval_wcov0` | `3:00:00` |
| `20411012` | `direct_retrieval` | `wcov1` | `qwen3_32b_direct_retrieval_wcov1` | `3:00:00` |
| `20411013` | `one_pass_grounding` | `wcov0` | `qwen3_32b_one_pass_grounding_wcov0` | `4:00:00` |
| `20411014` | `one_pass_grounding` | `wcov1` | `qwen3_32b_one_pass_grounding_wcov1` | `4:00:00` |
| `20411015` | `one_pass_structured` | `wcov0` | `qwen3_32b_one_pass_structured_wcov0` | `4:00:00` |
| `20411016` | `one_pass_structured` | `wcov1` | `qwen3_32b_one_pass_structured_wcov1` | `4:00:00` |
| `20411017` | `frozen_ags` | `wcov0` | `qwen3_32b_frozen_ags_wcov0` | `5:00:00` |
| `20411018` | `frozen_ags` | `wcov1` | `qwen3_32b_frozen_ags_wcov1` | `5:00:00` |

Canceled sample job IDs: `20411011` through `20411018`.

Current data-cleaning job:

| Job ID | Name | Purpose | Time limit | Status at last check |
|---|---|---|---|---|
| `20419632` | `codiesp_reloc` | Run Qwen evidence relocation for the full 250-document / 3431-fact CodiEsp diagnosis test set | `3:00:00` | `PENDING` |

CPU submitter cancellation:

| Job ID | Name | Purpose | Dependency | Status at last check |
|---|---|---|---|---|
| `20437767` | `codiesp_submit_full` | CPU-only Slurm submitter for validation plus GPU job submission | `afterok:20419632` | `CANCELLED` |

Do not use Slurm for CPU-only validation/submission wrappers. Pure CPU checks
should run locally. The local submit path is now
`submit_codiesp_full_local.sh`: it runs `scripts/validate_codiesp_full_outputs.py`
locally before submitting the eight GPU jobs. The validation gate requires:

- `data/codiesp/facts_test_full.jsonl`, `evidence_relocations_full.jsonl`, `stats_full.json`, `test_docs_full.txt`, and `spotcheck_50_full.tsv` to exist and be non-empty.
- `3431` full facts/relocations and `250` full test documents.
- spotcheck TSV to contain its header plus 50 sampled rows.
- `relocation_exact_substring_rate >= 0.95`.
- `relocation_parse_ok_rate >= 0.95`.
- `relocation_counts.fallback_document == 0`.

If the local gate passes, it submits the eight full GPU jobs with:

- `TEST_JSONL=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain/data/codiesp/facts_test_full.jsonl`
- output directories named `runs_codiesp_grounding_baseline/qwen3_32b_${query_mode}_full_${wcov}`
- `--time=1-00:00:00` overriding the shorter per-arm script defaults.

Why this relocation job exists:

- CodiEsp gold offsets are Spanish character offsets.
- The retrieval index, BM25 text, label-coverage term, and dimension vocabularies are English ICD-10-CM.
- Therefore each CodiEsp fact needs a clean English evidence mention before retrieval.
- The spec requires English exact-substring relocation with fallback logging and a 50-row spotcheck.
- Without this step, many current examples use whole sentences or whole documents as `entity`, which makes direct retrieval noisy and unfair to all arms.

## Code Changes Made

- `codiesp_pipeline/`
  - New vendored runtime for the CodiEsp transfer experiment.
  - Contains local copies of:
    - `run_fintagging_grounding_baseline.py`
    - `ags_frozen_grounding.py`
    - `ags_configuration_scoring.py`
    - `ags_symbolic_agreement.py`
  - Contains a CodiEsp-local minimal `run_ags_component_validation.py` with only the rendering helpers needed by `ags_frozen_grounding.py`.
  - The eight CodiEsp jobs should run against this local package, not the upstream/shared package outside `second_domain`.

- `submit_codiesp_four_arms.sh`
  - Expanded from four jobs to eight jobs.
  - Uses `--export` to set `QUERY_MODE`, `LABEL_COVERAGE_WEIGHT`, and coverage-specific `OUTPUT_DIR`.

- `run_codiesp_grounding_baseline.sh`
  - Now sets `SHARED_ROOT` to `second_domain/codiesp_pipeline`.
  - Added optional `LABEL_COVERAGE_WEIGHT`.
  - Passes `--label-coverage-weight` to `scripts/run_codiesp_grounding.py` when set.
  - Logs the selected label-coverage settings.

- `scripts/run_codiesp_grounding.py`
  - Keeps CodiEsp-specific ICD-10-CM prompts.
  - Now imports the vendored runner from `codiesp_pipeline/run_fintagging_grounding_baseline.py`.
  - Adds a local CodiEsp-only override so `one_pass_structured` and `frozen_ags` can run the `w_cov=0.0` ablation. The shared parent implementation normally pins frozen-family methods at `w_cov=1.0` and refuses `w_cov=0`.
  - Sets `one_pass_structured` to `rerank_beta=0.0` and fused-RRF-only ranking.
  - Keeps `frozen_ags`/FHS at `rerank_beta=0.6` and routes it through the local six-dimension candidate-level LLM verifier.

- `codiesp_pipeline/ags_frozen_grounding.py`
  - Adds a CodiEsp-local FHS candidate verifier.
  - FHS verifier judges all six generated dimensions: `FAMILY`, `ROLE`, `EVENT`, `QUALIFIER`, `SCOPE`, `TEMPORAL`.
  - FHS verifier support is averaged over those six dimensions and used in `range_normalized_fused + beta * verifier_support`.
  - `one_pass_structured` remains fused-only and does not run the verifier.

- `codiesp_pipeline/run_fintagging_grounding_baseline.py`
  - Skips the generic final listwise reranker for `frozen_ags`, because FHS's own candidate-level verifier is its final reranking component.
  - Other arms still use the configured final listwise reranker when `RUN_RERANK=1`.

- `apply_server_codiesp_direct_retrieval.sh`
  - Walltime reduced to `3:00:00`.

- `apply_server_codiesp_one_pass_grounding.sh`
  - Walltime reduced to `4:00:00`.

- `apply_server_codiesp_one_pass_structured.sh`
  - Walltime reduced to `4:00:00`.

- `apply_server_codiesp_frozen_ags.sh`
  - Walltime reduced to `5:00:00`.

- `scripts/relocate_codiesp_evidence.py`
  - New two-stage data cleaning script.
  - Imports shared generation helpers from the vendored local `codiesp_pipeline` runner.
  - Builds full CodiEsp diagnosis test targets by default (`3431` facts over `250` documents).
  - Prompts Qwen to return an exact English substring copied from the MT candidate text.
  - Validates that the substring appears in the English text before using it.
  - Writes full clean candidate files:
    - `data/codiesp/evidence_relocations_full.jsonl`
    - `data/codiesp/facts_test_full.jsonl`
    - `data/codiesp/stats_full.json`
    - `data/codiesp/test_docs_full.txt`
    - `data/codiesp/spotcheck_50_full.tsv`

- `apply_server_codiesp_relocate_evidence.sh`
  - New Slurm job wrapper for the Qwen relocation pass.

## Validation Done

- `bash -n submit_codiesp_four_arms.sh`
- `bash -n run_codiesp_grounding_baseline.sh`
- `bash -n apply_server_codiesp_shared.sh`
- `python -m py_compile scripts/run_codiesp_grounding.py`
- Eight `sbatch --test-only` checks passed.
- The real `sbatch` submission succeeded after Slurm access was approved.
- Existing pending jobs were updated with shorter `TimeLimit` using `scontrol update`.
- The eight sample jobs were later canceled with `scancel`.
- `python -m py_compile scripts/relocate_codiesp_evidence.py scripts/prepare_codiesp_domain.py`
- `bash -n apply_server_codiesp_relocate_evidence.sh`
- Relocation dry-run target generation produced `3431` full-test targets.
- `sbatch --test-only apply_server_codiesp_relocate_evidence.sh` passed.
- Relocation job `20419632` was submitted.

## CodiEsp J Definition

Here `J` means the number of structured hypotheses sampled for the AGS-style method before retrieval/fusion/reranking:

- `one_pass_structured`: `J=1`, one greedy structured hypothesis, `beta=0.0`, no consensus rerank term.
- `frozen_ags`: `J=2`, two structured hypotheses, `beta=0.6`, six-dimension candidate-level LLM verifier rerank.

The direct and free-text one-pass baselines do not use structured-hypothesis `J` in this sense.

## CodiEsp Dimension Definitions

The CodiEsp structured prompt asks the model to fill six ICD-10-CM-oriented semantic dimensions. Unsupported fields should be `UNRESOLVED`.

- `FAMILY`: ICD-10-CM chapter or broad clinical family.
- `ROLE`: diagnosis class such as disease/disorder, neoplasm, symptom/sign/abnormal-finding, injury/poisoning, external cause, health-status factor, pregnancy-related, perinatal, or congenital.
- `EVENT`: the specific condition, symptom, finding, injury, or clinical state.
- `QUALIFIER`: modifiers such as acute/chronic, severity, complication status, malignant/benign, type, open/closed, or specified/unspecified.
- `SCOPE`: laterality only, such as right, left, bilateral, unspecified-side, or not-applicable.
- `TEMPORAL`: encounter or extension status, such as initial encounter, subsequent encounter, sequela, healing status, or not-applicable.

The normalization map for CodiEsp lives at `schema/icd10cm/normalization_map.json`. It uses controlled vocabularies for `family`, `qualifier`, `scope`, and `temporal`; `role` and `event` are intentionally token/lexical branches with empty controlled vocabularies.

## Current Next Step

Full relocation job `20419632` produced the full files, but the strict quality gate failed on
the unfiltered output (`3144` exact substring, `120` fallback aligned sentence, `167`
fallback document). The experiment input was filtered to exact-substring evidence only:

- `data/codiesp/facts_test_full_exact.jsonl`: 3144 facts.
- `data/codiesp/evidence_relocations_full_exact.jsonl`: 3144 relocation rows.
- `data/codiesp/stats_full_exact.json`: exact rate 1.0, parse rate 1.0.
- `data/codiesp/spotcheck_50_full_exact.tsv`.
- `data/codiesp/test_docs_full_exact.txt`: 250 docs.

The local validation/submission script now defaults to these `_exact` files:

```bash
bash submit_codiesp_full_local.sh
```

This validates:

- `data/codiesp/stats_full_exact.json`
- `data/codiesp/spotcheck_50_full_exact.tsv`
- `data/codiesp/evidence_relocations_full_exact.jsonl`

The validation gate passed on the exact-only data, and these 8 GPU jobs were submitted:

- `20459406`: `codiesp_direct_retrieval_full_wcov0`
- `20459407`: `codiesp_direct_retrieval_full_wcov1` was cancelled after a GPU-idle warning.
  It was still in CPU candidate generation with `LABEL_COVERAGE_WEIGHT=1.0` and had not
  written candidates. The revised matrix does not require this wcov1 duplicate; if revived,
  split it into CPU candidate generation followed by a GPU rerank-only job.
- `20459408`: `codiesp_one_pass_grounding_full_wcov0`
- `20459409`: `codiesp_one_pass_grounding_full_wcov1`
- `20459410`: `codiesp_one_pass_structured_full_wcov0`
- `20459411`: `codiesp_one_pass_structured_full_wcov1`
- `20459412`: `codiesp_frozen_ags_full_wcov0`
- `20459413`: `codiesp_frozen_ags_full_wcov1`

At submission time all 8 were `PENDING`. Their initial 1-day limits were reduced to improve
queue priority: direct retrieval `3:00:00`, one-pass grounding `4:00:00`, one-pass
structured `4:00:00`, frozen_ags/FHS `6:00:00`.

## Revised Experiment Matrix Additions

The revised CodiEsp matrix added four configurations. They were implemented inside this
package only. Only the FHS-family variants are ablations; `parallel_sampling` is an
independent comparison baseline.

- `gold_label_definition_retrieval`: CPU-only oracle diagnostic that queries BM25 with the
  gold ICD-10-CM code, label, and generated definition text.
- `parallel_sampling`: independent free-text comparison baseline. CodiEsp
  prompt override added; submitted with `RETRIEVAL_ROUNDS=2`, which means two
  independent free-text samples, not iterative rounds.
- `fhs_j1`: FHS-family ablation with `J=1`, `beta=0.6`, candidate-level
  verifier retained.
- `fhs_no_verifier`: FHS-family ablation with `J=2`, `beta=0.0`,
  candidate-level verifier disabled.

The oracle diagnostic was run locally on
`data/codiesp/facts_test_full_exact.jsonl` and wrote
`runs_codiesp_grounding_baseline/qwen3_32b_gold_label_definition_retrieval_full_exact/`.
Metrics: top-1 accuracy `0.998410`, MRR `0.999205`, Recall@10/50/200 `1.0`.

The three added GPU jobs were submitted:

- `20466298`: `codiesp_parallel_sampling_n2_full_exact`
- `20466299`: `codiesp_fhs_j1_full_exact`
- `20466300`: `codiesp_fhs_no_verifier_full_exact`

If the experiment must also be three seeds, expand the submit script before launching more jobs and avoid reusing these output directories for new seeds.

Useful monitoring commands for a fresh session:

```bash
squeue -u lm2445 -o "%.18i %.36j %.9T %.12l %.30R"
sacct -j 20419632 --format=JobID,JobName%36,State,ExitCode,Elapsed,Timelimit,MaxRSS,Reason -P
tail -n 80 logs/*_codiesp_relocate_evidence_qwen3_b200.txt
```

Expected successful relocation outputs:

```text
data/codiesp/evidence_relocations_full.jsonl
data/codiesp/facts_test_full.jsonl
data/codiesp/stats_full.json
data/codiesp/test_docs_full.txt
data/codiesp/spotcheck_50_full.tsv
```

After `20419632` finishes, first check `stats_full.json` and spotcheck quality. Do not submit the four main experiment arms until that looks acceptable.

## Data Cleanliness Caveat

The filtering/cleaning rules from `codiesp-domain-port-spec.md` are partly implemented:

- implemented: use only `DIAGNOSTICO` rows from CodiEsp-X; ignore procedure codes.
- implemented: filter gold codes to the ICD-10-CM FY2018 inventory built as CodiEsp valid diagnosis codes intersected with billable ICD-10-CM FY2018 codes.
- implemented: deduplicate `(document, span, code)`.
- implemented: sample at document level with seed 0.
- implemented: parse discontinuous Spanish offsets and normalize whitespace.
- implemented: emit `spotcheck_50.tsv`.

But the current alignment implementation is not final-clean relative to the spec:

- The spec asked for an LLM-assisted exact English substring relocation inside the aligned English sentence, then fallback to sentence-level locus if that fails.
- The current script does not do the LLM exact-substring relocation. It uses the aligned English sentence as the located span when Spanish/English sentence counts match.
- When sentence counts do not match, the current script falls back to the full English document, not a sentence-level locus.
- On the full CodiEsp test diagnosis inventory, this current locator gives 1024 sentence-level loci and 2407 document-level fallbacks: document fallback rate `0.701545`.

Therefore the dataset is structurally filtered, but it should not be described as fully clean/final for paper results until the English evidence relocation is fixed or the document-level fallback is explicitly accepted as the experimental protocol.

## 配置约定(2026-07-30,与主域一致,提交前必须核对)

第一个域在这一天把三处配置统一了,CodiEsp 的每一次 run 都要用同一套,否则两个域的表不可比:

1. **Parallel sampling 用 N=2**,不是 N=4。预算按**假设数**匹配 FHS 的 J=2,而不是按模型调用数
   匹配——一次 verifier call 携带 K_v 个候选的 label+definition(主域实测 3,459 prompt token/fact),
   把它折算成一次生成调用会低估 FHS 的成本,方向上偏向我们。FHS 的 4 次调用 vs 基线的 2 次要如实报出。
2. **verifier 问全部六维、也按六维计分**(`core.LLM_VERIFIER_DIMENSIONS_DEFAULT`)。原来的三维需要
   "哪些维度能靠候选文本判定"这个按域的人工判断,而它在 ICD-10-CM 上是错的:侧别(scope)和就诊次序
   (temporal)就写在码本描述里。问六维不需要这个判断。六维时输出上限必须 2,816,否则 verdict 数组
   被截断、读起来像"全部弃权"。
3. **生成 token 上限统一 2,048,且只在一处定义**(主域是
   `apply_server_fintagging_direct_retrieval.sh`,所有方法的 wrapper 都委托到它)。主域此前有 128/384/512
   三种并存,导致 one_pass_structured 截断 8.5%、parallel diversity 33.1%,而自家方法在 512 下 0%。
   每个 run 的 `metrics.json` 现在有 `truncation` 段(生成调用数、触顶数、上限),提交前必须是触顶 0。

提交前用主域的两个脚本自查:`check_config_gate.py <run 目录>`(验收配置本身而非"跑完了")和
`verify_single_code_path.py`(逐行比对每个 run 自报的配置)。
