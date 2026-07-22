# FinTagging Grounding Experiments

This evaluates the context-aware tag grounding step.

Direct retrieval:

1. Use `(original_context, entity, type)` from
   `FinTagging_800_200_grounding_test_JSON/data/test.jsonl` as the query.
2. Retrieve top-200 candidates from the enriched US-GAAP 2024 retrieval dataset.
3. Rerank those candidates with `Qwen/Qwen3-32B` using vLLM.
4. Record `recall_at_10`, `recall_at_50`, `recall_at_200`, `mrr`, `accuracy`,
   and `search_coverage`.

One-pass grounding:

1. Qwen generates a brief retrieval query from `(entity, type, original_context)`.
2. BM25 retrieves top-200 candidates using `entity + type + generated_query`.
3. The same Qwen reranker and evaluator are used as direct retrieval.

Metrics are reported overall and separately under `by_input_type.table` and
`by_input_type.text`.

Run direct retrieval on the cluster:

```bash
sbatch data_whole_pipeline/apply_server_fintagging_direct_retrieval.sh
```

Run one-pass grounding:

```bash
sbatch data_whole_pipeline/apply_server_fintagging_one_pass_grounding.sh
```

Useful variants:

```bash
sbatch --export=ALL,MODE=retrieval data_whole_pipeline/apply_server_fintagging_direct_retrieval.sh
sbatch --export=ALL,LIMIT=20 data_whole_pipeline/apply_server_fintagging_direct_retrieval.sh
sbatch --export=ALL,QUERY_MODE=one_pass_grounding data_whole_pipeline/apply_server_fintagging_direct_retrieval.sh
sbatch --export=ALL,RERANK_BACKEND=transformers data_whole_pipeline/apply_server_fintagging_direct_retrieval.sh
```

The default rerank backend is `vllm` with batched generation. This should be
substantially faster than one-prompt-at-a-time Transformers inference while
keeping the prompt, parser, and metrics fixed across methods.

`REUSE_CANDIDATES=1` is the default, so rerank runs reuse the fixed
`bm25_candidates.jsonl` file if it already exists instead of rebuilding the
retrieval set.

For `QUERY_MODE=one_pass_grounding`, Qwen first writes
`query_descriptions.jsonl`, then BM25 uses `entity + type + query_description`
as the retrieval query. The output directory defaults to
`runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding`, separate from
direct retrieval.

Outputs:

- `data_whole_pipeline/runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval/bm25_candidates.jsonl`
- `data_whole_pipeline/runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval/bm25_metrics.json`
- `data_whole_pipeline/runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval/qwen_rerank_predictions.jsonl`
- `data_whole_pipeline/runs_fintagging_grounding_baseline/qwen3_32b_direct_retrieval/metrics.json`

For this single-round baseline, `search_coverage` is equivalent to whether a gold
concept appears in the top-200 retrieved candidate set. It is kept as a separate
metric because multi-round self-reflection experiments can accumulate a larger
candidate union across rounds.
