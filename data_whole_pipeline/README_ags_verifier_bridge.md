# Verifier bridge: reconciling Table 3 with the candidate-level LLM reranking row

Run of 2026-07-26, frozen test split. **All results reconstructed from completed logs — no
experiment was rerun, no LLM calls were regenerated, no GPU was used.** The single exception is
the end-to-end cell, which is reported as BLOCKED rather than estimated; see
[Known gap](#known-gap-end-to-end-tagging-accuracy).

## Terminology

Used consistently across code, outputs, and manuscript:

| Term | Meaning |
|---|---|
| **deterministic dimension verifier** | produces absolute dimension-level feedback for revision |
| **LLM dimension-feedback verifier** | the LLM version evaluated in Table 3 |
| **candidate-level LLM reranker** | the LLM scoring component in the ablation table |

The candidate-level reranker is **never** called an "LLM verifier". The old label
`+ LLM verification layer` has been renamed to `+ candidate-level LLM reranking` in
`../appendix_component_ablation.tex` (backup at `.tex.bak`); a grep for the old phrases returns
nothing.

## Exact commands

```bash
cd /nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline

./run_ags_verification_quality.sh     # Table 3 / Table 13          (~5 min, CPU)
./run_ags_verifier_bridge.sh          # bridge Panels A and B       (~4 min, CPU)
EMIT_LATEX=1 ./run_ags_verifier_bridge.sh   # also writes bridge_table.tex

# smoke tests
LIMIT=100 ./run_ags_verifier_bridge.sh
```

Neither script allocates a GPU; both set `CUDA_VISIBLE_DEVICES=""`. Unit tests run first unless
`SKIP_TESTS=1` (28 tests for the verification-quality module, 21 for the bridge).

## Inputs

| Path | Role |
|---|---|
| `runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl` | AGS test trace: J=2 hypotheses, fused ranking, gold tags |
| `runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl` | 5,018 per-candidate LLM judgements |
| `runs_ags_table5_ablation/qwen3_32b/llm_verifier_summary.json` | `top_m`, parse rate, call counts |
| `runs_ags_table5_ablation/qwen3_32b/ablation.csv` | AGS (full) paired baseline |
| `runs_ags_table5_ablation/qwen3_32b/llm_verifier_row.csv` | reranker arm + paired CIs |
| `runs_ags_table5_ablation/qwen3_32b/verifier_top20_overlap.json` | top-20 handoff diagnostic |
| `runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json` | deployed-stage without-reranking arm |
| `retrieval_data/us_gaap_2024_enriched/..._retrieval.jsonl` | taxonomy (17,388 concepts) |

## Outputs

Written to `runs_ags_verifier_bridge/qwen3_32b/`. **No existing experiment output is
overwritten** — this is a new directory.

```
bridge_candidate_discrimination.csv   Panel A, all calls and gold-in-window scopes
bridge_hypothesis_calibration.csv     Panel B, flat scalar row
bridge_threshold_sweep.csv            D- firing sweep, both layers, 8 thresholds
final_reranker_comparison.csv         with/without, retrieval stage, paired CIs
bootstrap_results.json                all context-level bootstrap intervals
bridge_summary.json                   everything above plus validation and config
bridge_table.tex                      LaTeX rows (only with EMIT_LATEX=1)
```

Table 3 outputs (regenerated with the new F1/coverage columns and paired contrasts) stay in
`runs_ags_verification_quality/qwen3_32b/`.

## Counts and validation

| Quantity | Expected | Observed | Match |
|---|---|---|---|
| Facts | 2,509 | 2,509 | yes |
| Source contexts | 191 | 191 | yes |
| LLM calls (usable) | 5,018 | 5,018 | yes |
| LLM calls unusable | 0 | 0 | yes |
| Hypotheses without a call | 0 | 0 | yes |
| Dimension observations | — | 25,699 | — |

No mismatches. The script prints a WARNING and records `facts_match` / `contexts_match` in
`bridge_summary.json → validation` if any count diverges, and aborts up front if `TOP_M`
disagrees with the logged verifier `top_m` (which would mean the two layers were scored on
different windows).

## Results

### Table 3 — hypothesis-level absolute calibration

| Verifier | Prec. | Rec. | F1 | Coverage | Prec.−base |
|---|---|---|---|---|---|
| Deterministic | 0.968 | 0.620 | 0.756 | 0.860 | +0.297 † |
| LLM | 0.569 | 0.163 | 0.254 | 0.454 | −0.103 |
| Merged | 0.947 | 0.647 | 0.769 | 0.917 | +0.276 † |

n = 25,699 dimension verdicts; base rate 0.671. Coverage is non-abstention. Paired
context-level bootstrap (2000 resamples), all excluding zero:

```
deterministic − LLM  F1      +0.5004  [+0.4547, +0.5460]
deterministic − LLM  recall  +0.4544  [+0.4078, +0.5012]
merged − deterministic F1    +0.0133  [+0.0091, +0.0184]
```

### Bridge Panel A — candidate-level discrimination

| | |
|---|---|
| LLM calls | 5,018 |
| Calls with gold in assessed window | 1,764 (35.2%) |
| Support rate, gold candidate | 0.948 |
| Support rate, distractors | 0.491 |
| Gold − distractor gap | +0.457, CI [+0.409, +0.501] |
| Mean per-call gap | +0.437 |
| Calls favouring gold | 87.1% |

### Bridge Panel B — hypothesis-level calibration

| | |
|---|---|
| Dimension observations | 25,699 |
| True disagreement base rate | 0.671 |
| LLM non-abstention rate | 0.563 |
| Base rate among judged observations | 0.568 |
| Mean support \| hypothesis wrong | 0.535 |
| Mean support \| hypothesis right | 0.547 |
| Difference | −0.012, CI [−0.065, +0.038] |
| AUROC of D− score, LLM | 0.514, CI on AUROC−0.5 [−0.024, +0.055] |
| AUROC of D− score, deterministic | 0.900 |
| Average precision, LLM / deterministic | 0.614 / 0.951 |

**Read Table 3's −0.103 carefully.** It is partly an abstention effect. Among the 56.3% of
opportunities on which the LLM does issue a verdict, its precision (0.569) sits at the
corresponding base rate (0.568) — that is *at chance*, not below it. The accurate scoped
statement is: the LLM dimension-feedback verifier lacks sufficient hypothesis-level calibration
for reliable sequential revision.

### Threshold sweep (supplementary)

Full split, `bridge_threshold_sweep.csv`. LLM precision stays within ±0.01 of its base rate at
every threshold except unanimous contradiction:

```
layer          thr   fired    prec  recall  vs base
llm           0.00    1253   0.760   0.116   +0.192
llm           0.25    4953   0.569   0.342   +0.001   <- Table 3 operating point
llm           0.90   10160   0.576   0.711   +0.008
deterministic 0.25   11050   0.968   0.620   +0.297
```

So Table 3's LLM row is a property of the signal, not of the operating point inherited from the
deterministic layer.

### Task 5 — with vs without candidate-level LLM reranking

Retrieval stage, everything else held fixed (same split, hypotheses, deterministic verifier,
candidate pool, fusion, normalisation, β=0.6, bootstrap seed):

| Metric | Without | With | Δ | 95% CI | Excl. 0 |
|---|---|---|---|---|---|
| Recall@10 | 0.3830 | 0.3994 | +0.0163 | [+0.0100, +0.0244] | yes |
| Recall@50 | 0.5496 | 0.5496 | +0.0000 | — | no |
| Recall@200 | 0.7182 | 0.7182 | +0.0000 | — | no |
| MRR | 0.1867 | 0.2518 | +0.0651 | [+0.0473, +0.0856] | yes |
| Top-1 accuracy | 0.0953 | 0.1714 | +0.0761 | [+0.0530, +0.1039] | yes |

Operational: 5,018 LLM calls, 2.00 per fact, parse success 1.000, all `clean`. Latency and
inference cost were not logged by the verifier run and are **not** reported rather than
estimated.

## Known gap: end-to-end tagging accuracy

**The decision rule cannot be resolved from existing logs, and the script says so rather than
substituting a retrieval-stage number.**

Every row above is scored *before* the listwise reranker the deployed pipeline applies to the
top 20. The without-reranking arm's deployed numbers exist
(`qwen_reranked.accuracy = 0.2375`, MRR 0.3176, R@10 0.4480); the with-reranking arm's do not.

The top-20 handoff diagnostic bounds how much can survive:

```
candidate set identical on         55.9% of facts
gold enters / leaves top 20        9 / 2  (of 2,509)
Recall@20                          44.80% -> 45.08%
gold's rank changes within top 20  26% of facts
```

Most of the retrieval-stage gain is reordering inside a window the listwise stage re-sorts anyway.

**Current ruling: PROVISIONAL rule 2 — optional retrieval-stage enhancement.** Rules 1 and 3
remain open pending measurement.

To unblock:

```bash
sbatch apply_server_ags_bridge_deployed_rerank.sh   # GPU, ~8h
./run_ags_verifier_bridge.sh                        # picks the new metrics.json up automatically
```

That script fails fast with instructions if its input ranking has not been materialised: the
candidate-level-reranked per-fact ranking is computed today by
`ags_table5_ablation/run_verifier_row_only.py` but only its aggregates are kept, so that script
needs a small CPU-only change to persist the ranking first.

## Manuscript status

The paper source was located at `../comparing_methods/paper_src/` (extracted from the uploaded
zip). All edits are applied directly to `acl_latex.tex`; backup at `acl_latex.tex.bak`.

| Change | Location |
|---|---|
| Abstract: dev numbers → test; LLM claim scoped; complementarity added | abstract |
| Intro role sentence, stage instrumentation | §1 |
| Contribution bullet rewritten; author's `\textbf{this claim might be wrong}` note removed | §1 |
| Related work, verifier design section | §2, §4.4 |
| Table 3: test numbers + F1 + Coverage columns + paired contrasts in caption | `tab:verifier` |
| Table 13: all rows to test numbers; caption rewritten | `tab:verifierfull` |
| Ablation row renamed `+ candidate-level LLM reranking` | `tab:ablation` |
| New §"Reconciling verification and reranking" + bridge table | `sec:bridge`, `tab:verifierbridge` |
| Deterministic-verifier caveat | `sec:bridge` paragraph + Limitations |
| Table 14: Panels A and B to test numbers; `[will run on test]` removed | `tab:revisiondiag` |
| Appendix G prose + §5 + intro revision claims updated | Appendix G, §5, §1 |
| Conclusion: scoped claim + complementarity | Conclusion |
| Limitations: retrieval-stage-only caveat for the reranker | Limitations |

Config-table row `LLM verification layer & disabled` was renamed to **`LLM dimension-feedback
verifier`**, not to "candidate-level LLM reranking" — it refers to the sequential control's
disabled feedback layer, which is a different component. A blind find-and-replace would have
been wrong there.

Verified: environments balanced (14 `table`, 9 `table*`, 24 `tabular`), no undefined `\ref`,
column counts match the tabular specs for all three edited tables. **Not compile-verified** —
no LaTeX toolchain on this machine.

`../verifier_bridge.tex` was written as a standalone fragment before the source was available.
Its content is now integrated into `acl_latex.tex`; keep it only as a reference copy, and do not
`\input` it or the tables will be duplicated.

Tables 3, 13 and 14 no longer carry `[will run on test]`. The only remaining marker is on the
retrieval-readiness diagnostics table (line 818), which genuinely has not been run on test; the
sentence at line 794 explaining the convention is kept for that reason.

**Table 14 changed two findings relative to the development placeholders.** Targeted-dimension
improvement is now 0.214 against a 0.146 control with a CI excluding zero (dev: 0.182 vs 0.143,
spanning zero), and \textsc{Family} is now significant (dev: indistinguishable). Revision now
beats random replacement on all three label-derived dimensions. The negative result is
unaffected — each stage functioning while the product stays negligible is the same argument, and
is if anything better supported.
