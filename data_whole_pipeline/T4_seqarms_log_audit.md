# Log audit: T4 (verifier accuracy) and T22-T26 (sequential arms)

Audit 2026-07-24. Read-only inspection of existing logs and running jobs, plus the fix applied
to the LLM verifier after job 19506429 was cancelled.

---

## CHECK 1 — T4 verifier accuracy: needs a new run

### 1. Does anything log per-dimension verdicts + gold on the full test split?

**No.** The right data structure exists, but only at the old diagnostic scope.

`runs_ags_feedback_verdict_accuracy/qwen3_32b/per_verdict.jsonl` has exactly what T4 needs,
one record per (fact, dimension, arm):

```
fact_id, context_key, arm, round_idx, modality, dimension,
truth_layer, truth_reason, truth_disagrees,      <- gold-derived ground truth
d_minus_symbolic, d_minus_llm, d_minus_merged,   <- symbolic AND LLM D- verdicts
d_plus_symbolic,  d_plus_llm,  d_plus_merged,
gold_rank_before, feedback_top_m, gold_in_feedback_window, has_symbolic_feedback
```

Scope confirms the hole:

| property | value |
|---|---|
| unique facts | **250** (fact_id 0-249, contiguous) |
| modality | **table 250 / text 0** |
| records | 5,765 (fact x dimension x arm) |
| arms | bandit 2,891 / random 2,874 |
| dimensions | FAMILY 1160, ROLE 1157, EVENT 1145, TEMPORAL 1132, QUALIFIER 678, SCOPE 493 |
| truth_layer | exact 2,615 / lexical 3,150 |
| **missing field** | **no `confidence`** on verdicts |

Symbolic and LLM rows both exist, both tabular-only, both 250 facts, neither carries the
per-verdict LLM confidence T4 asks for.

### 2. Does the LLM verifier run anywhere on full test?

Job 19506429 was running it over the full split. **Its output was 100% unusable**, and the job
has since been cancelled.

```
records:               2,375
parse_ok == true:      0        (0.0%)
empty verdicts_by_tag: 2,375    (100.0%)
unique facts:          1,188 of 2,509  (contiguous 0-1,187)
```

**Root cause: output truncation.** Raw outputs were well-formed JSON stopping mid-object:

```
...onfidence": 0.2\n    },\n    {\n      "tag": "us-gaap:AdvertisingBarterTransactions
```

- lengths min 1,530 / median 1,666 / max 1,895 chars — a tight band, the signature of a hard
  token cap
- **0 of 2,373 outputs ended in `}` or a closing fence**

The cap was `--query-max-new-tokens=512`. That is the arg that bounds the response:
`QueryGenerator.generate_many` passes it as vLLM's `max_tokens`
(run_fintagging_grounding_baseline.py:1331). `--max-new-tokens` was a red herring — it only
sizes `max_model_len`. Every generation hit the cap, so `parse_json_object` failed and
`parse_verifier_output` returned empty.

**Knock-on:** Table 5 row 3.9 ("+ LLM verification layer") reads the same file, so that row was
blocked too.

### Fix applied (2026-07-24)

Four changes to `ags_table5_ablation/run_llm_verifier.py`, verified by
`ags_table5_ablation/test_llm_verifier.py` (9 checks, no GPU needed):

1. **Token cap raised — the actual bug.** `--query-max-new-tokens` 512 -> **1536**. One verdict
   entry is ~110-130 tokens, so top_m=10 needs ~1,100-1,300 plus wrapper. Raise it further if
   `--top-m` is ever increased.

2. **Salvage parser** (`salvage_verdict_entries`). A truncated
   `{"verdicts": [ {...}, {...},` now yields its complete entries instead of nothing.
   Verified against the 2,436 real failed responses: **100% now yield verdicts, median 8 of
   10** (19,245 recovered). Defence in depth only — see the warning below.

3. **Resume no longer skips failures.** `load_existing` previously keyed on
   `(fact_id, hypothesis_idx)` regardless of outcome, so `RESUME=1` (the sbatch default) would
   have treated all 2,436 empty rows as done and baked the failure in permanently. It now
   reuses a row only if it carries verdicts, and reports how many are regenerated.

4. **Fail-fast guard.** `--abort-check-after 25` / `--abort-if-parse-rate-below 0.5`: aborts
   after 25 calls if fewer than half produced verdicts, naming truncation and the flag to
   raise. Verified to abort at exactly 25 calls rather than 2,369.

`llm_verifier_summary.json` now also carries `parse_rate`, `parse_modes`, `calls_generated`.
The existing "near-zero firing rate is the finding, not a bug" note now states that this
reading holds **only when `parse_rate` is high** — a parse failure and genuine abstention both
drive the firing rate to zero and mean opposite things. That note is precisely what would have
rationalised this bug's output.

**Do NOT reuse the salvaged old data for the final table.** Salvage recovers a median 8 of 10
candidates, and the missing ones are systematically the lowest-ranked, since truncation cuts
the tail. That is a biased sample of the top-M window. The dead file is archived at
`runs_ags_table5_ablation/qwen3_32b/archive_truncated_20260724/llm_verifier_calls.jsonl` so the
rerun starts clean.

### 3. Scope of the rerun

- Calls are **per (fact, hypothesis)**, not per fact — the log shows `hypothesis_idx` 0 and 1.
  So **~5,018 calls**, not ~2.5k.
- Observed throughput 2,375 calls / 4.6h ~= 520 calls/h -> **~9.7h**; the fixed version emits
  longer outputs, so budget **~10-12h on one B200**.
- No hypothesis generation (reused from the frozen AGS trace); retrieval already logged.

```bash
sbatch apply_server_ags_table5_llm_verifier.sh
```

Watch for `Parse-rate guard passed: ...% of the first 25 calls produced verdicts` in the log
within the first few minutes. If it aborts instead, the message says what to raise.

**Still to confirm before T4 itself:** T4 derives `truth_disagrees` from per-dimension gold
attributes. That machinery lives in `runs_ags_gold_attribute_audit/` and is wired into the
250-fact script, but currently covers only the audited subset. Verify it extends to all 2,509
test facts, or the LLM row will have verdicts with no ground truth to score against.

---

## CHECK 2 — sequential arms scope: correct, no resubmission needed

| question | finding |
|---|---|
| 1. how many facts | **Full test split.** No `LIMIT` set; 1,125/2,509 written, contiguous 0-1,124 |
| 2. text included? | **Yes.** 1,040 table / **85 text** so far |
| 3. stratified? | **No — first-N in file order.** Does not matter here; see below |
| 4. T22-T26 fields | **All present** |

### On ordering

The order is contiguous, not stratified — the same property that made the old 250-fact run
tabular-only. **The difference is that this job has no `LIMIT`, so it covers the entire split.**
Text is interleaved throughout rather than clustered at the end:

```
text share so far:  85 / 1,125  = 7.6%
text share overall: 168 / 2,509 = 6.7%
```

All 168 text facts will be covered at completion.

**Caveat:** because the order is non-stratified, a partial trace is not a valid sample. If the
job dies before finishing, the partial output under-covers text and must not be used as-is.

### Per-fact logging supports T22-T26 without a re-run

Per round, inside `ags_seq_rounds[]`:
`round_idx`, `candidate_list`, `reward`, `delta_y`, `delta_replay`, `utility_before`,
`utility_after`, `rank_before`, `rank_after`, `selected_operator`, `psi`, `D_plus`, `D_minus`,
`neighborhood_novelty_n`

Per fact:
`round1_candidates`, `round1_rank_gold`, `round1_retrieval_metrics`, `final_rank_gold`,
`candidate_union_tags`, `realized_rounds`, `stop_reason`, `ags_seq_round1_parity`

Covers round-1-vs-full (T22), AULC (T24), reward density (T25), search behavior (T26).

### Paired random arm is aligned (permutation null, T23)

```
qwen3_32b_ags_seq         1,125 facts   table 1,040 / text 85
qwen3_32b_ags_seq_random  1,128 facts   table 1,040 / text 88
fact-id aligned:          yes
```

---

## Bottom line

- **Check 2 needs nothing.**
- **Check 1's blocker is fixed**; the verifier rerun is ready to submit. Confirm gold-attribute
  coverage over the full split before treating T4 itself as unblocked.
