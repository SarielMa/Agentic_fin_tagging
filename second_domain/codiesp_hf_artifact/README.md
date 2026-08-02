# CodiEsp Diagnosis Grounding Data

This repository contains the prepared data files needed to reproduce the
CodiEsp-D diagnosis grounding experiments. It is a data artifact only: it does
not include experiment code, Slurm scripts, model outputs, logs, or raw source
archives.

## Directory Layout

```text
data/codiesp/
  facts_test_full_exact.jsonl
  evidence_relocations_full_exact.jsonl
  stats_full_exact.json
  test_docs_full_exact.txt
  spotcheck_50_full_exact.tsv

index/icd10cm_fy2018/
  icd10cm_fy2018_retrieval.jsonl
  code_metadata.jsonl
  inventory_manifest.json
  self_retrieval_probe.json

schema/icd10cm/
  normalization_map.json
  vocab_family.json
  vocab_role.json
  vocab_qualifier.json
  vocab_scope.json
  vocab_temporal.json
  vocab_temporal_raw.json
```

## Main Evaluation File

```text
data/codiesp/facts_test_full_exact.jsonl
```

This is the prepared CodiEsp-D test set used by the second-domain experiments.
It contains the exact-English-relocation slice of the official CodiEsp test
split.

Summary from `data/codiesp/stats_full_exact.json`:

- Source documents: 250
- Target facts: 3,144
- Unique gold diagnosis codes: 958
- Exact relocation parse rate: 1.0
- Exact substring rate: 1.0

Each JSONL row contains the clinical mention, English context fields, document
identifier, one gold diagnosis code, and relocation metadata.

## Relocation Files

```text
data/codiesp/evidence_relocations_full_exact.jsonl
data/codiesp/spotcheck_50_full_exact.tsv
```

These files document how Spanish CodiEsp diagnosis spans were relocated into
the English context used for retrieval. The evaluation JSONL keeps only facts
with exact English substring relocation.

## Retrieval Inventory

```text
index/icd10cm_fy2018/icd10cm_fy2018_retrieval.jsonl
```

This is the candidate label inventory used by retrieval. It contains 71,344
candidate diagnosis-code entries. Each entry contains a code identifier,
canonical English label, retrieval/documentation text, and structural metadata.

Inventory construction metadata is in:

```text
index/icd10cm_fy2018/inventory_manifest.json
```

## Schema Files

```text
schema/icd10cm/
```

This directory contains the controlled vocabularies and normalization map used
by the CodiEsp structured grounding schema.

## Not Included

The following are intentionally excluded:

- Experiment code and Slurm scripts.
- Model outputs and intermediate run files.
- Slurm logs.
- Raw downloaded CodiEsp or diagnosis-code archives.
- Python bytecode caches.

`MANIFEST.tsv` lists the files in this data artifact with byte sizes.
