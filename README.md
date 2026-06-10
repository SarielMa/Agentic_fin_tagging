# FinAI Tagging Baseline

This repository contains one minimal three-agent baseline for US-GAAP tag selection.

## Baseline

Agent 1 is the retriever. It uses no LLM and no table heuristic:

1. Filter taxonomy concepts by `entity_type`.
2. Rank the filtered concepts with BM25 between the input context and taxonomy `text`.
3. Return the top 200 candidates.

Agent 2 is the selector. It uses an LLM to select one tag from Agent 1 candidates. Before prompting, it retrieves LTM memories by:

1. Filtering both `correct_book` and `error_book` by `entity_type`.
2. Ranking each book with BM25 between the input context and memory `context`.
3. Inserting the top 5 memories from each book into the selector prompt.

Agent 3 is the validator. It uses an LLM and is the only agent that writes LTM.

- Testing mode: no ground truth access. The validator reads only `error_book`, gives feedback, and the selector tries once more. Total iterations: 2.
- Memory-build mode: ground truth access. The selector runs once, then the validator writes one memory. Total iterations: 1.

LTM has exactly two JSONL books:

- `correct_book.jsonl`: `entity_type`, `context`, `tag`, `comment`
- `error_book.jsonl`: `entity_type`, `context`, `predicted_tag`, `correct_tag`, `comment`

## Modules

- `scripts/agentic_fincl/retrieval.py`: BM25 retrieval for taxonomy and memory.
- `scripts/agentic_fincl/agents.py`: Agent 1 retriever, Agent 2 selector, Agent 3 validator.
- `scripts/agentic_fincl/pipeline.py`: offline, online_gt, and online_wo_gt pipeline modes.
- `scripts/agentic_fincl/evaluation.py`: accuracy and retrieval recall metrics.
- `scripts/agentic_fincl/experiment_cli.py`: command-line interface.
- `scripts/agentic_fincl/single_llm.py`: memoryless single-LLM testing baseline.

## Run

```sh
sh run_baseline_tests.sh
```

Default models:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-14B`

Default modes:

- `single_llm`
- `offline`
- `online_gt`
- `online_wo_gt`

Useful overrides:

```sh
LIMIT=10 sh run_baseline_tests.sh
CLEAN_OUTPUTS=1 sh run_baseline_tests.sh
RUN_SINGLE_LLM=0 sh run_baseline_tests.sh
RUN_ONLINE_GT=0 RUN_ONLINE_WO_GT=0 sh run_baseline_tests.sh
```

`LIMIT=0` means use the full CSV.

## Single LLM Test

This test uses Agent 1 retrieval as input, then asks one LLM to select one tag. It has no memory, no validator, and no retry loop.

```sh
RUN_OFFLINE=0 RUN_ONLINE_GT=0 RUN_ONLINE_WO_GT=0 sh run_baseline_tests.sh
```

## Outputs

Outputs are written under `outputs/baseline_three_agent/` by default.

Each run contains:

- `ltm/correct_book.jsonl`
- `ltm/error_book.jsonl`
- `*/predictions.jsonl`
- `*/metrics.json`
- `summary.json`
