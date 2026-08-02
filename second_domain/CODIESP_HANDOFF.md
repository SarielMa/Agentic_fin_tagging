# CodiEsp Handoff

Updated: 2026-07-30 03:20 EDT.

Use `CODIESP_EXPERIMENT_STATE.md` top section as the authoritative detailed
state. This file is the short version.

Update 2026-08-01 21:09 EDT:

- Generic final Qwen listwise rerank is no longer skipped for AGS/FHS modes when
  `RUN_RERANK=1`.
- Submitted rerank-only continuation jobs with existing candidates reused and
  `--time=03:00:00`:
  - `20929443`: `frozen_ags wcov0`.
  - `20929445`: `frozen_ags wcov1`.
  - `20929444`: `fhs_j1 wcov0`.
  - `20929446`: `fhs_j1 wcov1`.
  - `20929464`: `fhs_no_verifier wcov0`.
  - `20929465`: `fhs_no_verifier wcov1`.
- These jobs should write `qwen_rerank_predictions.jsonl` and `metrics.json`.
- User requested cancelling all `j3`/`j4` tasks. Cancelled active/pending jobs:
  `20929466` (`fhs_j3 wcov1` rerank continuation), `20693295` (`fhs_j4 wcov0`),
  and `20693296` (`fhs_j4 wcov1`). `fhs_j3 wcov0` job `20693293` had already
  failed startup assertions; `fhs_j3 wcov1` job `20693294` had already completed
  candidate generation before cancellation request, but its rerank continuation
  was cancelled.
- Added CodiEsp-local ports of the financial intrinsic self-refinement and
  retrieval-feedback refinement wrappers without modifying `data_whole_pipeline`.
  Submitted `w_cov=1.0` only, with one CPU and final selector enabled
  (`MODE=full`):
  - `20939892`: `intrinsic_self_refinement`, output
    `qwen3_32b_intrinsic_self_refinement_full_wcov1`, `--time=06:00:00`.
  - `20939893`: `retrieval_feedback_refinement`, output
    `qwen3_32b_retrieval_feedback_refinement_full_wcov1`, `--time=06:00:00`.
- Upload policy: the HF artifact currently contains data only. If a separate
  result artifact is later prepared, any method with both `wcov0` and `wcov1`
  outputs should upload only the `wcov1` output; do not upload both coverage
  variants for the same method.
- Submitted extra `wcov1` repeat jobs for computing standard deviations. All
  use `MODE=full`, final selector enabled, and
  `facts_test_full_exact.jsonl`; output directories are suffixed `_r2` and
  `_r3`. The user said `ags_j1`; this was submitted as the existing `fhs_j1`
  method to avoid duplicating `one_pass_structured`.
  - `20949745`: `frozen_ags r2`, output
    `qwen3_32b_frozen_ags_full_wcov1_r2`.
  - `20949748`: `frozen_ags r3`, output
    `qwen3_32b_frozen_ags_full_wcov1_r3`.
  - `20949750`: `fhs_j1 r2`, output
    `qwen3_32b_fhs_j1_full_wcov1_r2`.
  - `20949752`: `fhs_j1 r3`, output
    `qwen3_32b_fhs_j1_full_wcov1_r3`.
  - `20949753`: `one_pass_structured r2`, output
    `qwen3_32b_one_pass_structured_full_wcov1_r2`.
  - `20949756`: `one_pass_structured r3`, output
    `qwen3_32b_one_pass_structured_full_wcov1_r3`.
- Submitted remaining non-`j3`/`j4`, non-oracle `wcov1` repeat jobs for STD.
  All use `MODE=full`, final selector enabled, and
  `facts_test_full_exact.jsonl`; output directories are suffixed `_r2` and
  `_r3`.
  - `20953983`: `direct_retrieval r2`, output
    `qwen3_32b_direct_retrieval_full_wcov1_r2`.
  - `20953984`: `direct_retrieval r3`, output
    `qwen3_32b_direct_retrieval_full_wcov1_r3`.
  - `20953985`: `one_pass_grounding r2`, output
    `qwen3_32b_one_pass_grounding_full_wcov1_r2`.
  - `20953987`: `one_pass_grounding r3`, output
    `qwen3_32b_one_pass_grounding_full_wcov1_r3`.
  - `20953988`: `parallel_sampling_n2 r2`, output
    `qwen3_32b_parallel_sampling_n2_full_wcov1_r2`, with
    `RETRIEVAL_ROUNDS=2`.
  - `20954149`: `parallel_sampling_n2 r3`, output
    `qwen3_32b_parallel_sampling_n2_full_wcov1_r3`, with
    `RETRIEVAL_ROUNDS=2`.
  - `20954150`: `fhs_no_verifier r2`, output
    `qwen3_32b_fhs_no_verifier_full_wcov1_r2`.
  - `20954178`: `fhs_no_verifier r3`, output
    `qwen3_32b_fhs_no_verifier_full_wcov1_r3`.
  - `20954181`: `intrinsic_self_refinement r2`, output
    `qwen3_32b_intrinsic_self_refinement_full_wcov1_r2`.
  - `20954182`: `intrinsic_self_refinement r3`, output
    `qwen3_32b_intrinsic_self_refinement_full_wcov1_r3`.
  - `20954184`: `retrieval_feedback_refinement r2`, output
    `qwen3_32b_retrieval_feedback_refinement_full_wcov1_r2`.
  - `20954186`: `retrieval_feedback_refinement r3`, output
    `qwen3_32b_retrieval_feedback_refinement_full_wcov1_r3`.

Status checkpoint 2026-08-02 02:50 EDT:

- Do not resubmit these jobs tomorrow. They are either running or pending under
  `QOSMaxJobsPerUserLimit`; no duplicate work is needed.
- Running:
  - `20939892` `intrinsic_self_refinement main`: output directory exists,
    `grounding_traces.jsonl` has 3144 rows, `qwen_rerank_predictions.jsonl` has
    608 rows, no `metrics.json` yet.
  - `20939893` `retrieval_feedback_refinement main`: output directory exists,
    `grounding_traces.jsonl` has 2665 rows, no rerank rows and no
    `metrics.json` yet.
  - `20949745` `frozen_ags r2`: output directory exists,
    `grounding_traces.jsonl` has 521 rows, no `metrics.json` yet.
  - `20949748` `frozen_ags r3`: output directory exists,
    `grounding_traces.jsonl` has 503 rows, no `metrics.json` yet.
  - `20949750` `fhs_j1 r2`: output directory exists,
    `grounding_traces.jsonl` has 543 rows, no `metrics.json` yet.
  - `20949753` `one_pass_structured r2`: output directory exists,
    `grounding_traces.jsonl` has 3144 rows, `qwen_rerank_predictions.jsonl`
    has 1856 rows, no `metrics.json` yet.
- Pending with no output directory yet:
  - `20949752` `fhs_j1 r3`.
  - `20949756` `one_pass_structured r3`.
  - `20953983` `direct_retrieval r2`.
  - `20953984` `direct_retrieval r3`.
  - `20953985` `one_pass_grounding r2`.
  - `20953987` `one_pass_grounding r3`.
  - `20953988` `parallel_sampling_n2 r2`.
  - `20954149` `parallel_sampling_n2 r3`.
  - `20954150` `fhs_no_verifier r2`.
  - `20954178` `fhs_no_verifier r3`.
  - `20954181` `intrinsic_self_refinement r2`.
  - `20954182` `intrinsic_self_refinement r3`.
  - `20954184` `retrieval_feedback_refinement r2`.
  - `20954186` `retrieval_feedback_refinement r3`.

Do not touch unrelated jobs. Only operate on this CodiEsp experiment's
`codiesp_*` jobs. In particular, do not cancel or modify `vf6_*`, `b_*`,
`seqvf_*`, Jupyter, or other non-CodiEsp jobs.

Current result target count: gold oracle runs once; the other seven methods run
with `wcov0` and `wcov1`, so 15 final result units total.

Method grouping:

- Independent comparison/baseline methods: `direct_retrieval`,
  `one_pass_grounding`, `parallel_sampling_n2`, and the CPU-only
  `gold_label_definition_retrieval` oracle diagnostic.
- FHS-family ablation ladder: `one_pass_structured`, `frozen_ags`, `fhs_j1`,
  and `fhs_no_verifier`. These all use the same
  `ags_frozen_grounding.build_frozen_ags_method_record` path; only frozen
  config constants change.

Clean restart:

- All previous CodiEsp GPU jobs were cancelled or were already terminal.
- `runs_codiesp_grounding_baseline/` and `logs/` were cleared before the restart.
- New results must come only from the 2026-07-30 03:10-03:18 EDT submission
  wave listed below.

Completed CPU-only results/prework:

- `gold_label_definition_retrieval` oracle diagnostic was rerun locally from
  the current code and wrote
  `runs_codiesp_grounding_baseline/qwen3_32b_gold_label_definition_retrieval_full_exact/`.
- `direct_retrieval wcov1` CPU-only candidates and BM25 metrics were rebuilt
  locally from the current code in
  `runs_codiesp_grounding_baseline/qwen3_32b_direct_retrieval_full_wcov1/`.

Partial result:

- `direct_retrieval wcov1`:
  GPU rerank-only job `20488827` is pending.

Running at handoff:

- None. All current CodiEsp GPU jobs are pending at clean restart.

Pending GPU jobs:

- `20488374`: `direct_retrieval wcov0`.
- `20488827`: `direct_retrieval wcov1` rerank.
- `20488828`: `one_pass_grounding wcov0`.
- `20488829`: `one_pass_grounding wcov1`.
- `20488830`: `one_pass_structured wcov0`.
- `20488831`: `one_pass_structured wcov1`.
- `20488832`: `frozen_ags wcov0`.
- `20488833`: `frozen_ags wcov1`.
- `20489123`: `parallel_sampling_n2 wcov0` comparison baseline.
- `20489162`: `parallel_sampling_n2 wcov1` comparison baseline.
- `20489164`: `fhs_j1 wcov0` FHS-family ablation.
- `20489165`: `fhs_j1 wcov1` FHS-family ablation.
- `20489163`: `fhs_no_verifier wcov0` FHS-family ablation.
- `20489166`: `fhs_no_verifier wcov1` FHS-family ablation.

Important fixes already made:

- Runtime is vendored under `second_domain/codiesp_pipeline`; do not edit
  upstream sibling packages for this experiment.
- FHS verifier sees all six dimensions: `FAMILY`, `ROLE`, `EVENT`,
  `QUALIFIER`, `SCOPE`, `TEMPORAL`.
- FHS verifier max new tokens is `3072`; structured/FHS query generation is
  bumped to `512`; generic rerank max new tokens is `1024`.
- `LABEL_COVERAGE_POOL_MULTIPLIER` default is `10`, not `0`, to avoid cov1
  doing full-taxonomy CPU prework while holding a GPU.
- `submit_codiesp_new_requirements.sh` now requests explicit `gpu_b200`,
  `b200:1`, `8` CPUs, and `256G`.

Useful checks:

```bash
sacct -j 20488374,20488827,20488828,20488829,20488830,20488831,20488832,20488833,20489123,20489162,20489163,20489164,20489165,20489166 --format=JobID,JobName%46,Partition,State,ExitCode,Elapsed,Timelimit,ReqTRES%70 -P
find runs_codiesp_grounding_baseline -maxdepth 2 -type f -printf '%h/%f %s\n' | sort
```
