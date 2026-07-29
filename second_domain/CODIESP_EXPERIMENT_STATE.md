# CodiEsp Second-Domain Experiment State

Last updated: 2026-07-29 18:32 America/New_York.

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

Why this relocation job exists:

- CodiEsp gold offsets are Spanish character offsets.
- The retrieval index, BM25 text, label-coverage term, and dimension vocabularies are English ICD-10-CM.
- Therefore each CodiEsp fact needs a clean English evidence mention before retrieval.
- The spec requires English exact-substring relocation with fallback logging and a 50-row spotcheck.
- Without this step, many current examples use whole sentences or whole documents as `entity`, which makes direct retrieval noisy and unfair to all arms.

## Code Changes Made

- `submit_codiesp_four_arms.sh`
  - Expanded from four jobs to eight jobs.
  - Uses `--export` to set `QUERY_MODE`, `LABEL_COVERAGE_WEIGHT`, and coverage-specific `OUTPUT_DIR`.

- `run_codiesp_grounding_baseline.sh`
  - Added optional `LABEL_COVERAGE_WEIGHT`.
  - Passes `--label-coverage-weight` to `scripts/run_codiesp_grounding.py` when set.
  - Logs the selected label-coverage settings.

- `scripts/run_codiesp_grounding.py`
  - Keeps CodiEsp-specific ICD-10-CM prompts.
  - Adds a local CodiEsp-only override so `one_pass_structured` and `frozen_ags` can run the `w_cov=0.0` ablation. The shared parent implementation normally pins frozen-family methods at `w_cov=1.0` and refuses `w_cov=0`.

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
- `frozen_ags`: `J=2`, two structured hypotheses, `beta=0.6`, consensus agreement contributes to the final AGS score.

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

Wait for relocation job `20419632` to finish. Then inspect:

- `data/codiesp/stats_full.json`
- `data/codiesp/spotcheck_50_full.tsv`
- `data/codiesp/evidence_relocations_full.jsonl`

If the relocation exact-substring rate and spotcheck look acceptable, submit the four-method full experiment using `TEST_JSONL=data/codiesp/facts_test_full.jsonl`. Revisit walltime first, because full test is `3431` facts rather than the old `250`-fact sample.

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
