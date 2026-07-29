# FHS — Factorized Hypothesis Search

Code and data for the evidence-to-taxonomy retrieval paper. This is a **self-contained copy** of
`data_whole_pipeline/`, reorganized. The original directory is untouched and still owns every
`runs_*` result and every job currently in the SLURM queue.

## Layout

```
src/          the engine and FHS core
  run_fintagging_grounding_baseline.py   all query modes live here (~4k lines)
  ags_frozen_grounding.py                FHS config; w_cov = 1.0 is pinned here
  ags_seq_verifier_arm.py                FHS-Seq
  ags_symbolic_agreement.py              the program-driven dimension check
  verifier/                              LLM verifier: verdict generation, rerank dumps
  efficiency/                            per-fact cost and retriever-robustness harness
analysis/     one script per appendix table (probes, ablations, sweeps, table builders)
scripts/
  run_fintagging_grounding_baseline.sh   pipeline driver, builds the python argv
  slurm/                                 one sbatch wrapper per arm
    fulltagging/                         end-to-end variant, kept for the removed section
  stage/                                 local CPU staging + submit helpers
tests/        unit tests
tools/        dataset construction, split stats, paper table utilities
data/
  test/test.jsonl                        2,509 facts / 191 contexts  = Table 1
  dev/sample_facts.jsonl                 661 facts / 70 contexts     = Appendix A
  taxonomy/                              US-GAAP 2024 enriched, 17,388 concepts
  source/                                HF datasets the dev sample is drawn from
attic/        SFT / value-type / context-extraction branch, unused by this paper
runs/         outputs land here (not copied from the original; regenerate as needed)
```

## Running

Every script locates the repo from its own path, so the folder can be moved or renamed:

```bash
FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
```

The one exception is `#SBATCH --output=`, which sbatch parses before any shell runs and therefore
carries a literal path. **If you move this folder, rewrite those lines** or the jobs will fail at
launch with no output file.

Retrieval only, no GPU:

```bash
python3 src/run_fintagging_grounding_baseline.py \
  --query-mode direct_retrieval --limit 20 --output-dir runs/smoke
```

A full arm, GPU:

```bash
sbatch scripts/slurm/apply_server_fintagging_frozen_ags.sh
```

## Table 2 method names

The paper renamed AGS to FHS; **the code did not follow**. In `--query-mode`:

| paper row | query_mode |
|---|---|
| FHS (full) | `frozen_ags` |
| FHS-Seq | `ags_seq` |
| One-pass grounding, structured | `one_pass_structured` |
| One-pass grounding, free-text | `one_pass_grounding` |
| Direct retrieval | `direct_retrieval` |

## The label-coverage term

`TaxonomyRetriever` defaults to `label_coverage_weight = 0.0`, and only the frozen family
(`frozen_ags`, `one_pass_structured`, `ags_seq`) assigns a non-zero value. Every other baseline
therefore ran **without** the term. `--label-coverage-weight` forces it for any mode; it refuses to
override the frozen family, whose value is pinned. The shell passes it through as
`LABEL_COVERAGE_WEIGHT`.

`retrieve()` returns early when the weight is `<= 0`, before the pool multiplier is read, so at
`w_cov=0` the multiplier has no effect.
