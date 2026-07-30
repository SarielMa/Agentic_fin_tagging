# CodiEsp Handoff

Updated: 2026-07-30 01:25 EDT.

Use `CODIESP_EXPERIMENT_STATE.md` top section as the authoritative detailed
state. This file is the short version.

Do not touch unrelated jobs. Only operate on this CodiEsp experiment's
`codiesp_*` jobs. In particular, do not cancel or modify `vf6_*`, `b_*`,
`seqvf_*`, Jupyter, or other non-CodiEsp jobs.

Current result target count: gold oracle runs once; the other seven methods run
with `wcov0` and `wcov1`, so 15 final result units total.

Completed final results:

- `direct_retrieval wcov0`:
  `runs_codiesp_grounding_baseline/qwen3_32b_direct_retrieval_full_wcov0/`
  has 3144 candidates, 3144 rerank predictions, and `metrics.json`.
  Slurm job `20459406` is `FAILED` only because the submitted script copy had a
  post-result shell syntax error after metrics were written.
- `gold_label_definition_retrieval`:
  `runs_codiesp_grounding_baseline/qwen3_32b_gold_label_definition_retrieval_full_exact/`
  has `bm25_metrics.json`.

Partial result:

- `direct_retrieval wcov1`:
  CPU-only candidates and BM25 metrics are complete in
  `runs_codiesp_grounding_baseline/qwen3_32b_direct_retrieval_full_wcov1/`.
  GPU rerank-only job `20474170` is pending.

Running at handoff:

- `20459408` `codiesp_one_pass_grounding_full_wcov0`, running on `gpu_b200`.
  `query_descriptions.jsonl` has 3144 rows; candidates and rerank predictions
  exist and are being written.
- `20459410` `codiesp_one_pass_structured_full_wcov0`, running on `gpu_b200`.
  `grounding_traces.jsonl` exists and is growing. Logs show Qwen3-32B/vLLM GPU
  activity.

Pending GPU jobs:

- `20474170`: `direct_retrieval wcov1` rerank.
- `20473790`: `one_pass_grounding wcov1`.
- `20473792`: `one_pass_structured wcov1`.
- `20474745`: `parallel_sampling_n2 wcov0`.
- `20466298`: `parallel_sampling_n2 wcov1`; output dir still named
  `qwen3_32b_parallel_sampling_n2_full_exact`.
- `20459412`: `frozen_ags wcov0`.
- `20473791`: `frozen_ags wcov1`.
- `20474744`: `fhs_j1 wcov0`.
- `20466299`: `fhs_j1 wcov1`; output dir still named
  `qwen3_32b_fhs_j1_full_exact`.
- `20474738`: `fhs_no_verifier wcov0`.
- `20466300`: `fhs_no_verifier wcov1`; output dir still named
  `qwen3_32b_fhs_no_verifier_full_exact`.

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
sacct -j 20459406,20474170,20459408,20473790,20459410,20473792,20474745,20466298,20459412,20473791,20474744,20466299,20474738,20466300 --format=JobID,JobName%46,Partition,State,ExitCode,Elapsed,Timelimit,ReqTRES%70 -P
find runs_codiesp_grounding_baseline -maxdepth 2 -type f -printf '%h/%f %s\n' | sort
```
