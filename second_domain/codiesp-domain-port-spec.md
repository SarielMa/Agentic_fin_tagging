# Implementation Spec: Porting FHS to a Second Domain (CodiEsp / ICD-10-CM)

**Audience:** coding agent (Claude Code / Codex) working inside the existing FHS repository.
**Goal:** add a second evaluation domain — Spanish clinical case reports (CodiEsp) grounded to
ICD-10-CM — reusing the existing FHS implementation *unchanged*, to support a
domain-generality claim in an ACL submission.

**The single most important constraint:** this is a *transfer* experiment, not a tuning
experiment. Every hyperparameter, every shared component, and every prompt template stays
as it is for US-GAAP. The only new things you write are (a) a domain adapter that loads
CodiEsp facts, (b) an ICD-10-CM index builder, (c) a dimension schema definition, and
(d) an evaluation script. If you find yourself editing a file under the shared/core path,
stop and flag it.

---

## 0. Verify-before-trust

This spec names XML element names, TSV column layouts, and directory names from
documentation rather than from the actual archives. **Before writing parsing code, download
the files and print the real structure**, then reconcile against this spec and record any
discrepancy in `NOTES.md`. Do not silently code around a mismatch.

Specifically verify:
- ICD-10-CM FY2018 tabular XML: element names (`chapter`, `section`, `diag`, `name`, `desc`,
  `inclusionTerm`, `includes`, `excludes1`, `excludes2`, `sevenChrDef`, `extension`).
- CodiEsp `*X.tsv` column order and whether span offsets are single or `;`-separated
  (discontinuous references exist in this corpus).
- Whether the English machine-translated directory is present in the release you download.

---

## 1. Data acquisition

| Artifact | Source | Notes |
|---|---|---|
| CodiEsp corpus (train/dev/test + gold) | Zenodo DOI `10.5281/zenodo.3693570` | Also check `10.5281/zenodo.3837305` for the latest version; record which you used |
| CodiEsp valid code list | Zenodo DOI `10.5281/zenodo.3706838` | Defines the candidate inventory; also documents the Spanish→English description mapping |
| ICD-10-CM FY2018 | CDC archive, comprehensive ICD-10-CM file listing (`archive.cdc.gov`, NCHS) | Need: tabular XML, alphabetic index XML, and `icd10cm_codes_2018.txt` |

**Why FY2018 specifically:** the CodiEsp valid-code list was built against the 2018
ICD-10-CM code file. Using any later fiscal year introduces code churn (deletions,
replacements) that would require GEM/conversion-table remapping. Freeze FY2018. Record the
exact filenames and download dates in `NOTES.md`.

Record each source's license verbatim from its landing page (ICD-10-CM is US public domain;
CodiEsp carries a Creative Commons license — copy the exact variant, do not assume).

Use only `CodiEsp-D` (diagnoses). Ignore `CodiEsp-P` (procedures / ICD-10-PCS) entirely for
now; it is a possible later arm and out of scope here.

---

## 2. Language handling (read carefully — this supersedes an earlier decision)

The retrieval index must be **English**, because the label-coverage term, the tokenizer, and
the dimension vocabularies are all lexical and all built around English ICD-10-CM
descriptions. Cross-lingual BM25 is not viable.

The complication: CodiEsp-X span offsets are character offsets into the **Spanish** source
text, so they do not transfer to the English machine translation.

**Protocol (Option A — implement this):**

1. Use the English machine-translated document as the source context `X`.
2. Re-locate each evidence span in the English document:
   - Extract the Spanish evidence text from the `reference` column of the `*X.tsv` file.
   - Determine which Spanish sentence(s) the span offsets fall inside.
   - Sentence-align the Spanish and English documents by index (MT output is normally
     sentence-parallel; assert equal sentence counts and log every document where it fails).
   - Within the aligned English sentence, locate the span with one LLM call using the same
     backbone as the rest of the pipeline. Prompt it to return the exact substring, then
     assert that substring occurs in the English sentence.
3. Fall back to **sentence-level locus** (the whole aligned English sentence as `ℓ`) for any
   fact where step 2 fails the assertion. Log the fallback rate.
4. **Emit a spot-check file** of 50 randomly sampled aligned spans (Spanish original,
   English sentence, located English span, gold code, gold description) as a TSV for human
   review. Do not proceed to the full run until the human confirms this file.

Do **not** translate the ICD-10-CM side. Do **not** run the pipeline on Spanish text with an
English index.

Record the alignment failure rate and the fallback rate; both go in the paper.

---

## 3. Test-set sampling

- Sample from the **official CodiEsp test set** (250 clinical cases with gold annotations).
- **Sample at the document level, never at the fact level.** All facts from one clinical case
  must land in the same split. This mirrors the existing US-GAAP protocol.
- Draw documents with a fixed seed (`seed=0`, record it) until the accumulated diagnosis-fact
  count reaches **≥ 250**. Expect roughly 15–20 documents (the corpus averages ~16 diagnosis
  spans per case).
- Write the selected document IDs to `data/codiesp/test_docs.txt` and treat that file as
  frozen. Every later run reads it. Do not resample.
- Deduplicate: if the same `(document, span, code)` triple appears twice, keep one.
- **Do not construct a development split.** No configuration selection happens in this
  domain (see §7).

Produce a statistics table matching the format of the existing US-GAAP Table 1:
source contexts, target facts, unique gold concepts, facts per context.

---

## 4. ICD-10-CM index construction

Build the index with the **existing** index builder, only supplying different field values.
The retriever, BM25 parameters, `w_cov`, tokenizer, and the label-coverage term (`cov(q,c)`)
must be untouched.

**Inventory:** all codes present in the CodiEsp valid diagnosis code list, intersected with
valid billable ICD-10-CM FY2018 codes. Report the resulting inventory size (this is the
analogue of "17,388 concepts" for US-GAAP).

**Per-entry fields:**

| FHS field | ICD-10-CM source |
|---|---|
| `canonical_label` | Code description from `icd10cm_codes_2018.txt` (the long description) |
| `definition` | Concatenate, in this order: (1) `inclusionTerm` notes on the code, (2) `includes` notes inherited from the nearest ancestor `diag`/`section`/`chapter`, (3) the hierarchy path text `chapter desc > section desc > parent diag desc`, (4) alphabetic-index lead terms and sub-terms that resolve to this code |
| `structural_metadata` | Chapter number, section range, code character length, 7th-character extension letter (if present), and the extension's own definition string |
| datatype-analogue filter key | Diagnosis-vs-other code class (see below) |

**On the `definition` field:** ICD-10-CM has no per-code definition paragraph, unlike
US-GAAP documentation labels. The concatenation above is the substitute. This is a
documented design substitution and must be surfaced in the paper — write it into `NOTES.md`
with the exact concatenation order you implemented.

`excludes1` / `excludes2` notes: keep them in a **separate** field, not inside `definition`.
They name *near neighbours that are not this code*, so putting them in the retrieval text
would actively pull wrong candidates in. Store them for possible later analysis only.

**Deterministic pre-filter (the analogue of US-GAAP datatype filtering):** restrict candidates
to diagnosis codes that are billable/valid at full character length. Implement it through the
same pre-filter hook the US-GAAP path uses.

---

## 5. Dimension schema (M = 6)

Same six dimension *names* as US-GAAP, re-instantiated. Three vocabulary-backed and
label/text-exposed (verified by the candidate-level verifier), three derived from structural
metadata (rendered into queries but not verified) — the same 3/3 split as the existing
implementation, so `§4.5`'s verifier needs no code change.

| Dimension | Verified? | Vocabulary | Derivation |
|---|---|---|---|
| `FAMILY` | yes | closed, 21 | The 21 ICD-10-CM chapters. Auto-generate from `<chapter><desc>`; do not hand-write |
| `ROLE` | yes | closed, ~9 | Condition class from code range: disease/disorder, neoplasm, symptom/sign/abnormal-finding (R), injury/poisoning (S–T), external cause (V–Y), health-status factor (Z), pregnancy-related (O), perinatal (P), congenital (Q). Deterministic from the code prefix |
| `EVENT` | yes | none | The specific condition. Falls back to token overlap against the candidate's concatenated label + definition, exactly as US-GAAP's vocabulary-free dimensions do |
| `QUALIFIER` | no | curated, target ~18 | Type/severity/manifestation modifiers: acute, chronic, acute-on-chronic, with-complication, without-complication, mild, moderate, severe, displaced, nondisplaced, primary, secondary, malignant, benign, in-situ, uncertain-behavior, controlled, uncontrolled, unspecified, type-1, type-2, open, closed. Build by mining description n-grams across the inventory, then curate down to ~18. Ship a normalization map |
| `SCOPE` | no | closed, 5 | **Laterality only**: right, left, bilateral, unspecified-side, not-applicable. Anatomical site goes into `EVENT`, not here. This keeps `SCOPE` vocabulary-matched, as in US-GAAP |
| `TEMPORAL` | no | closed, target ~11 | **Auto-derive**: extract every `<sevenChrDef><extension char="…">` definition string in the tabular XML, then cluster the distinct strings into ~10–12 categories (initial-encounter, subsequent-encounter, sequela, subsequent-routine-healing, subsequent-delayed-healing, subsequent-nonunion, subsequent-malunion, not-applicable, …). Emit the raw extraction *and* the clustering as separate files so the mapping is auditable |

**Auto-derive wherever possible.** `FAMILY`, `ROLE`, and `TEMPORAL` should all be generated
from the FY2018 files by script, not typed by hand. Commit the generated vocabularies as
data files, and commit the generator script.

**Unmatched-value logging:** reuse the existing mechanism — values that fail to normalize get
logged, and the log is how the vocabulary is extended. Report the final unmatched rate.

**Generator prompt:** reuse the existing template. Replace only the per-dimension one-line
definitions and the few-shot examples (if any exist, use ICD-10 examples drawn from the
**train** split, never from test). The instruction to emit `null` rather than guess stays
verbatim. The generator must not see the taxonomy or any candidate codes.

---

## 6. Arms to run

Only four. Do not port the other seven.

1. `direct_retrieval` — the located fact and its raw context as the query
2. `one_pass_free_text` — single free-text retrieval-ready description (HyDE-style)
3. `one_pass_structured` — a single factorized hypothesis through the same renderer
4. `fhs_full` — the deployed configuration

Rationale: (2) vs (3) isolates factorization, which is the paper's load-bearing claim;
(1) is the floor; (4) is the method. The iterative block is not needed for a
generality claim.

Rendering: US-GAAP uses dual rendering (label-form + definition-form) on tabular evidence and
definition-form only on narrative. CodiEsp is narrative, so **use definition-form only**,
consistent with the existing narrative branch. Do not add a new rendering mode. Note that
this means ~1 retrieval per hypothesis, so cost per fact will be lower than the US-GAAP
tabular numbers.

---

## 7. Hyperparameters: frozen, no exceptions

Use the US-GAAP-selected values verbatim:

```
J        = 2
beta     = 0.6
K_v      = 10
kappa    = 60
w_cov    = 1.0
K        = 200
fusion   = summed RRF
scaling  = range-normalized
rendering = definition-form only (narrative branch)
```

**Run no sweep. Build no development split. Select nothing.** If a result looks poor, report
it as-is and flag it — do not tune. The transfer claim is worth more than a better number,
and a reviewer who sees per-domain tuning will discount the entire section.

Write a hard assertion into the eval entrypoint that fails if any of these differs from the
US-GAAP config values, and have it print the comparison.

---

## 8. Evaluation

**Metrics.** Recall@10, Recall@50, MRR as primary; Recall@200 as secondary. `Acc.` = top-1
after the shared listwise selector, measured downstream of it. Recall and MRR are measured at
end of retrieval, before the selector — match the existing convention exactly and reuse the
existing metric code.

**Seeds.** 3 seeds, report mean and standard deviation. Use the same seed values as the
US-GAAP runs.

**Confidence intervals.** Report paired bootstrap CIs for the two contrasts that matter:
`fhs_full − one_pass_structured` and `one_pass_structured − one_pass_free_text`, on both MRR
and Acc. Resample at the **document level**, 2,000 iterations, contrasts paired per fact —
identical to the existing US-GAAP bootstrap procedure. Reuse that code.

**Main output table** — same column layout as the existing US-GAAP main results table:

| Method | R@10 | R@50 | MRR | Acc. | std |

**Specificity breakdown table (required — this is the scientifically interesting output).**
Group facts by how much dimensional specificity the gold code demands, and report R@10, MRR,
and Acc per group for all four arms:

| Group | Definition |
|---|---|
| 3-char | gold code is 3 characters (category-level, minimal specificity) |
| 4–5 char | gold code is 4–5 characters |
| 6–7 char | gold code is 6–7 characters (requires laterality and/or 7th-character extension) |

**The prediction to test:** FHS's margin over `one_pass_free_text` should be *larger* in the
6–7 char group than in the 3-char group, because that is where the evidence span names the
condition but leaves `SCOPE` and `TEMPORAL` unresolved. Report the result whether or not the
prediction holds. A clean falsification is a publishable finding here; a fabricated
confirmation is not. Include per-group `n` — the 3-char group may be small.

**Also report:** span-alignment failure rate, sentence-locus fallback rate, vocabulary
unmatched rate, inventory size, and per-fact model calls / retrievals / wall-clock latency
(same format as the existing cost table).

---

## 9. Do not

- Do not modify anything under the shared/core path: retriever, index scorer, `cov(q,c)`,
  fusion, normalization, candidate-level verifier, listwise selector, metric code, bootstrap
  code. Add a domain config; do not fork a code path. If two reported numbers could come from
  divergent code paths, that is a bug.
- Do not re-tune any hyperparameter, and do not add a development split.
- Do not translate ICD-10-CM descriptions, and do not build a Spanish index.
- Do not put `excludes1` / `excludes2` text into the retrieval `definition` field.
- Do not use test-split facts for prompt examples or vocabulary curation. Train split only.
- Do not touch, re-run, or overwrite any existing US-GAAP result.
- Do not include the `CodiEsp-P` / ICD-10-PCS arm.
- Do not "fix" a disappointing number. Report and flag it.

---

## 10. Execution order

1. Download all three artifacts. Print real file structures. Reconcile with §0. Write `NOTES.md`.
2. Build the ICD-10-CM FY2018 index (§4). Report inventory size.
3. **Sanity probe first:** query the index with each gold code's own label + definition and
   verify it returns that code at the top for nearly every code. This is the analogue of the
   existing retrievability probe. If it does not, the index is broken — stop and report,
   do not proceed.
4. Auto-generate the three derivable vocabularies; curate `QUALIFIER` (§5).
5. Implement span alignment (§2). Emit the 50-span spot-check file. **Halt for human review.**
6. Freeze the test sample (§3). Emit the statistics table.
7. Smoke test: run all four arms on 20 facts, 1 seed. Verify end-to-end plumbing, inspect
   a few hypotheses by hand, confirm the frozen-hyperparameter assertion fires correctly.
8. Full run: 4 arms × 3 seeds over the frozen test set.
9. Emit both tables plus the diagnostics in §8.
10. Write `RESULTS.md`: the tables, every logged rate, every discrepancy found in step 1,
    and an explicit statement of what was substituted for the missing `definition` field.

Halt and ask at step 3 (probe failure), step 5 (spot-check review), and any point where
following this spec would require editing a shared component.

---

## 11. Deliverables

```
data/codiesp/
  test_docs.txt              # frozen document IDs
  facts_test.jsonl           # (locus, span text, English context, gold code)
  spotcheck_50.tsv           # human-reviewed alignment sample
index/icd10cm_fy2018/        # built index + inventory manifest
schema/icd10cm/
  vocab_family.json          # auto-generated
  vocab_role.json            # auto-generated
  vocab_temporal_raw.json    # auto-extracted extension definitions
  vocab_temporal.json        # clustered categories
  vocab_qualifier.json       # curated + normalization map
  generate_vocabs.py
results/codiesp/
  main_table.md
  specificity_table.md
  diagnostics.json
NOTES.md
RESULTS.md
```
