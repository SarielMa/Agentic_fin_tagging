# CodiEsp Handoff

Updated: 2026-07-30 03:20 EDT.

Use `CODIESP_EXPERIMENT_STATE.md` top section as the authoritative detailed
state. This file is the short version.

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
