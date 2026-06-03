# FinAI Agentic XBRL Tagging

This repository implements an agentic pipeline for target-centered XBRL concept tagging on the FinTagging FinCL data.

The system maps a target financial fact in context to a US-GAAP taxonomy tag:

```text
context + category + entity + entity_type -> us-gaap tag
```

The current implementation focuses on the reconstructed FinCL task. Each data row provides the original report context, a target entity, its XBRL value type, and the gold US-GAAP tag.

## Task Input and Output

This is a **target-centered tagging system**, not a full all-number extraction system.

For each example, the agent receives:

| Input Field | Used by Agents? | Description |
|---|---:|---|
| `context` | Yes | Original text paragraph or serialized HTML table |
| `category` | Yes | Whether the context is `text` or `table`; used to build different evidence for each format |
| `entity` | Yes | The target value to tag, e.g. `46`, `21.6`, `two` |
| `entity_type` | Yes | The target XBRL value type, e.g. `monetaryItemType` |
| `answer` | Training only / scoring only | Gold US-GAAP tag; used during memory-build or evaluation, not shown to agents in held-out test mode |
| `query` | No | Original FinCL serialized query; ignored because some table rows have broken local context |

The model prediction output is one final tag:

```json
{"Tag": "us-gaap:RevenueNotFromContractWithCustomerOther"}
```

The run also writes audit information, including retrieved candidates, memory hits, ReAct attempts, validation actions, and metrics. Those are logs, not the main prediction target.

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

### ReAct-Style Loop

For each sample, the system runs a small selector-validator loop:

```text
Initialize STM with:
  context, category, entity, entity_type, localized evidence

For attempt = 1..max_iters:
  1. Retrieve candidates from taxonomy + current LTM.
  2. Agent 1 selects a tag.
  3. Agent 2 validates the selected tag.

  If Agent 2 returns keep:
      final_tag = Agent 1 tag
      optionally write approved memory
      stop loop

  If Agent 2 returns correct:
      final_tag = Agent 2 corrected tag
      optionally write correction memory
      stop loop

  If Agent 2 returns retry:
      update STM with Agent 2 feedback
      continue loop

  If Agent 2 returns flag:
      final_tag = current tag, marked as flagged
      stop loop
```

STM is short-term state for the current sample. It can change within the loop, mainly by adding validator feedback such as:

```text
The previous selection was low confidence or inconsistent; reconsider candidates. Risk signals: low_top2_gap.
```

LTM is durable memory across samples. It is updated only after a final accepted or corrected decision, not after every retry. Therefore, LTM updates affect later samples, while STM feedback affects the next loop attempt for the same sample.

## Run Modes

### Offline Evaluation

Offline mode is the main held-out evaluation setting. It has two phases:

```text
Phase 1: memory-build set
  gold labels are hidden during the Agent1-Agent2 loop
  after the loop produces a final tag, gold is used for supervision
  LTM is updated from the post-loop supervised keep/correction

Phase 2: held-out test set
  gold labels are hidden from both agents
  LTM is frozen
  final predictions are scored against gold labels after inference
```

Pseudo-code:

```text
for sample in memory_build:
    run Agent1-Agent2 loop without gold
    compare final_tag with answer after the loop
    update LTM using post-loop supervision

freeze LTM

for sample in test:
    run Agent1-Agent2 loop without gold
    do not update LTM
    evaluate final_tag against answer
```

```bash
python scripts/run_two_agent_system.py \
  --mode offline \
  --memory-build data/FinCL-eval-subset-clean-memory.csv \
  --test data/FinCL-eval-subset-clean-test.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/offline_agentic_run
```

### Online With Ground Truth

Online-with-GT mode processes one stream. There is no separate train/test split. For each sample, gold is not shown inside the Agent1-Agent2 loop. After the loop produces a final tag, gold is used to supervise memory updates for future samples.

Pseudo-code:

```text
for sample in stream:
    run Agent1-Agent2 loop without gold
    compare final_tag with answer after the loop
    update LTM using post-loop supervision
```

The update is for later samples only. The current sample is not rerun after its LTM write.

```bash
python scripts/run_two_agent_system.py \
  --mode online_with_gt \
  --stream data/FinCL-eval-subset-clean-memory.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/online_gt_run
```

### Online Without Ground Truth

Online-without-GT mode processes one stream without exposing gold labels to the agents. Agent 2 uses risk signals rather than correctness labels.

Pseudo-code:

```text
for sample in stream:
    run Agent1-Agent2 loop without gold

    if final action is low-risk keep:
        update selector_memory
        update table_context_patterns if the sample is a table

    if final action is retry:
        update STM only, then continue loop

    if final action is flag:
        do not update LTM
```

In this mode, `error_book` is not updated because there is no trusted correction. Flagged cases can be treated as a review queue in future work.

```bash
python scripts/run_two_agent_system.py \
  --mode online_without_gt \
  --stream data/FinCL-eval-subset-clean-test.csv \
  --taxonomy data/us_gaap_2024_BM25.jsonl \
  --output-dir outputs/online_no_gt_run
```

## Model Configuration

The selector and validator can use different backends and models.

Retrieval/rule baseline:

```bash
python scripts/run_two_agent_system.py \
  --mode offline \
  --output-dir outputs/offline_retrieval_rule
```

Llama selector and Llama validator:

```bash
python scripts/run_two_agent_system.py \
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

Run the fixed-memory baseline:

```bash
python scripts/run_fixed_memory_baseline.py
```
