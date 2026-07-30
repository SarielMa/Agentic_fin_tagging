# FHS — Factorized Hypothesis Search

Code and data for *Factorized Hypothesis Search for Evidence-to-Taxonomy Retrieval*.

This tree contains the code that produces the numbers reported in the paper, and nothing else.
Exploratory studies that no reported result depends on are not included; see
[What is deliberately absent](#what-is-deliberately-absent).

## Layout

```
src/          engine and FHS core
  run_fintagging_grounding_baseline.py   every query mode lives here (~4k lines)
  ags_frozen_grounding.py                FHS config; w_cov = 1.0 is pinned here
  ags_seq_verifier_arm.py                FHS-Seq, the matched sequential control
  ags_sequential_arms.py                 shared loop machinery for the sequential arms
  ags_symbolic_agreement.py              the program-driven dimension check (the rejected score)
  verifier/                              candidate-level verifier: verdict generation, offline
                                         re-scoring, per-arm rerank dumps
  efficiency/                            per-fact cost harness, dense/hybrid retriever harness
analysis/     one script per table or diagnostic; also the table builders
scripts/
  run_fintagging_grounding_baseline.sh   pipeline driver, builds the python argv
  slurm/                                 one sbatch wrapper per arm
  stage/                                 local CPU staging and submit helpers
tests/        unit tests (all pass; two print "ALL N CHECKS PASSED" instead of unittest's OK)
tools/        dataset construction and split statistics
data/
  test/test.jsonl                        2,509 facts / 191 contexts   -> Table 1
  dev/sample_facts.jsonl                 661 facts / 70 contexts      -> Appendix A
  taxonomy/                              US-GAAP 2024 enriched, 17,388 concepts
  source/                                the HF datasets the splits are drawn from
runs/         all outputs land here (regenerate; run artifacts are not shipped)
```

## Order of operations

Nothing downstream works before its input exists, and the trace is the input to everything:

1. **Index and splits** — `tools/build_us_gaap_2024_retrieval_dataset.py`,
   `tools/make_fintagging_train_test_split.py`, `tools/analyze_fintagging_split_stats.py`.
2. **An arm** — `sbatch scripts/slurm/apply_server_fintagging_<arm>.sh`. The FHS arm
   (`frozen_ags`) writes the trace every analysis below reads:
   `runs/runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl`,
   whose `rounds` field carries the per-hypothesis, per-rendering candidate lists.
3. **Verifier verdicts (GPU)** — `scripts/slurm/apply_server_ags_table5_llm_verifier.sh`.
   Always at `WINDOW_SOURCE=fused`: the window must come from the fused retrieval score, or the
   deterministic check decides what the LLM is allowed to see, including in the arms that claim
   to have removed it.
4. **Offline re-scoring (CPU)** — `src/verifier/dump_reranked_ranking.py` materialises an arm's
   ranking; `scripts/slurm/apply_server_verifier_ablation_rerank.sh` then runs the shared
   listwise selector over it to get the final-accuracy column.
5. **Tables** — the analysis scripts below, then
   `analysis/build_verifier_ablation_table.py`.

Retrieval-only smoke, no GPU:

```bash
python3 src/run_fintagging_grounding_baseline.py \
  --query-mode direct_retrieval --limit 20 --output-dir runs/smoke
```

## Reproducing each table

| paper table | script |
|---|---|
| Table 1 (data statistics) | `tools/analyze_fintagging_split_stats.py` |
| Table 2 (main results) | one arm per row, step 2 above; FHS row from steps 3–4 with `ARM=no_determ` |
| Table 3 (component ablations) | `apply_server_verifier_ablation_rerank.sh` per `ARM=` (see below) |
| Table 5 (retrieval readiness) | `analysis/run_ags_probe_queryform.py --modality pooled` |
| Label-coverage diagnostics | `analysis/run_ags_coverage_query_form_interaction.py`, `analysis/run_ags_coverage_pilot.py` |
| Development design study | `analysis/run_ags_config_ablation.py`, `analysis/run_ags_component_validation.py` |
| Rerank-weight sensitivity | `analysis/run_ags_beta_sweep.py`, `analysis/run_ags_beta_sweep_extended.py` |
| Verifier window sensitivity | `analysis/run_verifier_window_sensitivity.py --verifier-mode llm_drop` |
| Verification quality | `analysis/run_ags_verification_quality.py --llm-calls <fused verdicts>` |
| Candidate-level behaviour | `analysis/run_ags_verifier_bridge.py --llm-calls <fused verdicts>` |
| Sequential control | `analysis/compute_round1_vs_full.py`, `analysis/compute_ags_best_prefix_r50.py`, `analysis/compute_ags_null_search_permutation.py` |
| Retriever robustness | `src/efficiency/run_t28_retriever_robustness.py`, `src/efficiency/build_t28_deployed_candidates.py` |
| Inference cost | `src/efficiency/run_t7_efficiency.py` |
| Paired confidence intervals quoted in the text | `analysis/paired_final_bootstrap.py` |

## Three things that silently produce wrong numbers

**1. `--llm-calls` defaults to the wrong verdicts.** `run_ags_verifier_bridge.py` and
`run_ags_verification_quality.py` both default to a verdict file generated before the window fix.
Pass `--llm-calls .../verdicts_k10_fused/llm_verifier_calls.jsonl` explicitly. Without it the two
verifier layers are compared over different windows and the comparison is no longer paired.

**2. `llm_unjudged_fill` defaults to `zero`, and the paper's arms use `mean`.** Any script calling
`verifier.core.evaluate` directly must pass `llm_unjudged_fill="mean"`. The failure is nearly
invisible: Recall@1, Recall@50 and MRR all reproduce, and only Recall@10 moves (by 0.022). Add an
assertion that reproduces a published row before trusting a new driver.

**3. An ablation arm needs its OWN verifier window.** `run_llm_verifier.py` cuts the window from
the trace's stored fused score, which is FHS's — measured coverage of another arm's head is
0.76–0.84 (`analysis/check_ablation_window_coverage.py`). For the LLM-only ablation arms:

```bash
python3 analysis/stage_arm_windows.py --arm mean_fusion \
  --verify-against runs/.../verdicts_k10_fused/llm_verifier_verdicts.json
sbatch --export=ALL,WINDOW_TAGS=runs/.../arm_windows/window_mean_fusion.jsonl,... \
  scripts/slurm/apply_server_ags_table5_llm_verifier.sh
sbatch --export=ALL,ARM=llmonly_mean_fusion,VERDICTS=<that arm's verdicts>,... \
  scripts/slurm/apply_server_verifier_ablation_rerank.sh
```

`stage_arm_windows.py --arm full --verify-against <deployed verdicts>` is the self-check: the
window it computes for the deployed arm must already be judged in full (25,090/25,090 keys), and it
exits non-zero otherwise. The rerank wrapper refuses an `llmonly_*` arm that was handed FHS's
verdicts, because that failure mode returns plausible numbers instead of an error.

Removing range normalization is the one exception: it is a monotone transform of the fused score,
so that arm's window is identical to FHS's and the deployed verdicts apply unchanged.

## Which dimensions the verifier judges

Asked = scored = all six the generator emits (FAMILY, ROLE, EVENT, QUALIFIER, SCOPE, TEMPORAL).
`verifier/run_llm_verifier.py` reads the set from one constant, `VERIFIER_DIMENSIONS`, and
`verifier/core.py` scores over `LLM_VERIFIER_DIMENSIONS_DEFAULT`; both are the six. Two consequences
worth knowing before running anything:

- **The completion-token cap follows the size of the judged set, not the name of the flag**, because
  an undersized cap truncates responses mid-array and the parser reports that as total abstention
  rather than as an error. The wrapper defaults to 2816 and only `--judge-dimensions legacy` may use
  1536. Measured over the finished ask-6 runs (7,770 calls at `top_m=10`): mean completion 887-891
  tokens, max 1,137, `hit_token_cap` 0, against 610 mean / 765 max at ask-3 -- so six dimensions cost
  about 46% more completion, and 2816 is headroom rather than a requirement. Size this from
  `hit_token_cap` in the call log, not from an estimate.
- `--judge-dimensions legacy` exists to *re-read* verdict files generated before 2026-07-30, when the
  asked set was FAMILY/ROLE/EVENT only. It is not a configuration to report.

`dump_reranked_ranking.py` records `llm_verifier_dimensions` (the scored set), `llm_verdicts_path`
and `llm_verdicts_dropped_by_top_m` in each arm's `ranking_summary.json`, because the scored set is
not recoverable from the verdicts file -- that file records only what was asked.

**`--top-m` is a trap and now refuses to be one.** It restricts the verdicts through `--calls`, so
the two files must come from the same generation run. Left at its default, `--calls` pointed at a
2026-07-25 deterministic-window call log; pairing that with a fused-window verdicts file silently
dropped 14.5% of the keys (50,180 -> 42,910) and changed the top ten on 68% of facts, i.e. it put the
window confound back. The script now refuses a mixed pair, and refuses any restriction that drops
keys when `--top-m` equals the window the verdicts were generated at. Verdicts already carry their
window: unless you mean to shrink one, do not pass `--top-m`.

## Method names in the code

The paper renamed AGS to FHS; the code did not follow. In `--query-mode`:

| paper row | query_mode |
|---|---|
| FHS (full) | `frozen_ags` |
| FHS-Seq | `seq_verifier` (the matched control; `ags_seq` is the earlier, unmatched arm) |
| One-pass grounding, structured | `one_pass_structured` |
| One-pass grounding, free-text | `one_pass_grounding` |
| Direct retrieval | `direct_retrieval` |

## The label-coverage term

`TaxonomyRetriever` defaults to `label_coverage_weight = 0.0`, and only the frozen family
(`frozen_ags`, `one_pass_structured`, the sequential arms) assigns a non-zero value, so a baseline
run without an explicit flag runs **without** the term. `--label-coverage-weight`
(`LABEL_COVERAGE_WEIGHT` in the shell) forces it for any mode; it refuses to override the frozen
family, whose value is pinned. `retrieve()` returns early when the weight is `<= 0`, before the
pool multiplier is read, so at `w_cov=0` the multiplier has no effect.

## Paths

Every script locates the tree from its own path, so the folder can be moved or renamed. The one
exception is `#SBATCH --output=`, which sbatch parses before any shell runs and therefore carries a
literal path: **if you move this folder, rewrite those lines**, or jobs fail at launch with no
output file.

## What is deliberately absent

- The `std` columns in the paper are **analytic estimates**, not multi-seed measurements: there is
  no seed loop to find here. Each reported point estimate comes from a single run; paired
  context-clustered bootstrap intervals (2,000 resamples, resampled at the source-context level)
  are what the paper uses for significance, and those are reproducible from
  `analysis/paired_final_bootstrap.py` and the `ci_low`/`ci_high` columns the analysis scripts write.
- An SFT / value-type / context-extraction branch, and an extractor-driven full-tagging pipeline.
  Neither supports a reported result.
- Exploratory verifier studies that were measured and then left out of the paper: a sweep of the
  rerank weight on the verifier term, a contradicted-dimension weight sweep, and variants of the
  abstention rule and of which dimensions the verifier is asked about.
- Revision-stage diagnostics for the sequential arm, and an end-to-end tagging comparison. Both
  belonged to sections the paper does not contain.
