# FinAI Agentic XBRL Tagging

This repository implements an agentic pipeline for target-centered XBRL concept tagging on the FinTagging FinCL data.

The system maps a financial fact in context to a US-GAAP taxonomy tag:

```text
context + entity + entity_type -> us-gaap tag
```

The current implementation focuses on the reconstructed FinCL task. Each data row provides the original report context, a target entity, its XBRL value type, and the gold US-GAAP tag.

## Data

Expected input files are in `data/`:

```text
data/FinCL-eval-subset-clean-memory.csv
data/FinCL-eval-subset-clean-test.csv
data/us_gaap_2024_BM25.jsonl
```

The clean FinCL CSVs must contain:

| Column | Description |
|---|---|
| `context` | Original text or serialized HTML table context |
| `category` | `text` or `table` |
| `entity` | Target financial fact value |
| `entity_type` | XBRL value type, e.g. `monetaryItemType` |
| `query` | Original FinCL query field; not used by this system |
| `answer` | Gold US-GAAP tag |

The taxonomy JSONL must contain:

| Field | Description |
|---|---|
| `us_gaap_tag` | Taxonomy concept name without `us-gaap:` prefix |
| `entity_type` | XBRL value type |
| `text` | Human-readable concept text |

## Pipeline

The upgraded system uses two agents:

1. **Tag Selector Agent**
   - Retrieves taxonomy candidates.
   - Uses selector memory, error-book memory, and table-pattern memory.
   - Selects a candidate tag.

2. **Validator-Corrector Agent**
   - Checks the selector output.
   - Can keep, correct, retry, or flag the prediction.
   - Controls writes to long-term memory.

Long-term memory is append-only and stored under the run output directory:

```text
outputs/<run>/ltm/
  selector_memory.jsonl
  error_book.jsonl
  table_context_patterns.jsonl
```

## Run Modes

### Offline Evaluation

Offline mode first builds memory from the memory/training split using gold labels, then freezes memory and evaluates on the held-out test split.

```bash
python scripts/run_agentic_fincl_experiment.py \
  --mode offline \
  --memory-build data/FinCL-eval-subset-clean-memory.csv \
  --test data/FinCL-eval-subset-clean-test.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/offline_agentic_run
```

### Online With Ground Truth

Online-with-GT mode processes one stream. After each prediction, the gold tag is available and can update memory for future samples.

```bash
python scripts/run_agentic_fincl_experiment.py \
  --mode online_with_gt \
  --stream data/FinCL-eval-subset-clean-memory.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/online_gt_run
```

### Online Without Ground Truth

Online-without-GT mode processes one stream without exposing gold labels to the agents. Memory updates are conservative and only happen for low-risk accepted predictions.

```bash
python scripts/run_agentic_fincl_experiment.py \
  --mode online_without_gt \
  --stream data/FinCL-eval-subset-clean-test.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/online_no_gt_run
```

## Model Configuration

The selector and validator can use different backends and models.

Retrieval/rule baseline:

```bash
python scripts/run_agentic_fincl_experiment.py \
  --mode offline \
  --output-dir outputs/offline_retrieval_rule
```

Llama selector and Llama validator:

```bash
python scripts/run_agentic_fincl_experiment.py \
  --mode offline \
  --selector-backend llama \
  --selector-model meta-llama/Llama-3.2-3B-Instruct \
  --validator-backend llama \
  --validator-model meta-llama/Llama-3.2-3B-Instruct \
  --output-dir outputs/offline_llama_agents
```

The Llama backend uses Hugging Face `local_files_only=True`, so model weights should already be available in the local cache or at the provided model path.

## Outputs

Each run writes:

```text
outputs/<run>/summary.json
outputs/<run>/ltm/*.jsonl
outputs/<run>/<phase>/predictions.jsonl
outputs/<run>/<phase>/metrics.json
outputs/<run>/<phase>/breakdown.json
```

For offline mode, phases are:

```text
memory_build/
test/
```

For online modes, the phase is:

```text
online_with_gt/
online_without_gt/
```

Each prediction record contains:

```json
{
  "gold": {"Fact": "...", "Type": "...", "Tag": "..."},
  "prediction": {"Fact": "...", "Type": "...", "Tag": "..."},
  "correct": true,
  "stm": {
    "evidence": "...",
    "top_k": [],
    "memory_hits": {},
    "attempts": [],
    "final_action": "keep|correct|retry|flag"
  }
}
```

## Evaluation Metrics

The main metrics are:

| Metric | Meaning |
|---|---|
| `tag_accuracy` | Fraction of examples where final predicted tag equals `answer` |
| `recall_at_k` | Whether the gold tag appears in the retrieved top-k candidate list |
| `flag_rate` | Fraction of examples where the final validator action is `flag` |
| `action_counts` | Counts of final validator actions: `keep`, `correct`, `flag`, etc. |

Breakdowns are reported by:

```text
category: text/table
entity_type: monetaryItemType, percentItemType, sharesItemType, perShareItemType, integerItemType
```

## Utility Scripts

Clean the original FinCL subset:

```bash
python scripts/create_clean_fincl_dataset.py
```

Split the clean set into memory and test splits:

```bash
python scripts/split_clean_fincl_dataset.py
```

Run the older fixed-memory baseline:

```bash
python scripts/run_agentic_fincl_pipeline.py
```

