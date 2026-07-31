# Results package

Every number in the paper's tables, with the artifact it was read from.

Assembled 2026-07-31 from `data_whole_pipeline/`, which is where the runs executed. This
package is a **selection**, not a copy: the small artifacts (`metrics.json`,
`ranking_summary.json`, `llm_verifier_summary.json`, csv) are here in full; the multi-GB
candidate lists and generation traces stay in the run tree and are listed in
`MANIFEST.tsv` by path and byte count so they can be located, not guessed at.

## Layout

```
results/
  MANIFEST.tsv                 every source artifact: path, copied/left-in-tree, bytes, sha256
  tables/
    table01_main_results.tsv   one row per paper row, values read from the source
    table02_ablation.tsv       same, plus the config each arm actually ran under
    verdict_provenance.tsv     judged dimensions / window source / parse rate per verdict set
  sources/                     the copied artifacts, under their original relative paths
```

## How to read it

`tables/*.tsv` carry a `source` column naming the directory under
`data_whole_pipeline/`. The same relative path appears under `sources/` here and as a row
in `MANIFEST.tsv`. Nothing in the tsv files was typed by hand: the build script reads each
`metrics.json` and formats it, so a mismatch between a tsv and its source is a build bug,
not a transcription error.

## Column conventions

`R@1`, `R@10`, `R@50`, `R@200` and `MRR` are the **retrieval stage**
(`metrics.json` -> `bm25_retrieval`). `Acc` is **after the shared listwise selector**
(`metrics.json` -> `qwen_reranked`). `R@1` is `bm25_retrieval.accuracy`, which is
precision at rank one. This split is the paper's convention and is easy to get wrong when
reading `metrics.json` directly, because both blocks carry an `accuracy` key.

## Two rows that are not single runs

- **FHS-Seq** ran as four test shards (`qwen3_32b_seq_verifier_s0..s3`, 628/628/628/625 =
  2,509). The row in `table01_main_results.tsv` is the fact-weighted pooling of the four
  `metrics.json` files. Each shard restarts `example_idx` at 0; the shards are contiguous
  slices of the test set in the same order, verified by matching `(context_id, entity)`
  against the FHS run fact for fact.
- **`- ensemble (J=1)`** is the arithmetic mean of the `idx0` and `idx1` arms, which are
  listed separately here rather than pre-averaged.

## Arms that are not matched to the deployed method

`table02_ablation.tsv` prints `verifier_mode`, `beta` and `judged_dims` per arm so this is
visible rather than asserted:

- `- verifier` runs at `beta=0`, so no verdict enters the score.
- `Program-driven score` is `verifier_mode=deterministic`: the symbolic check replaces the
  candidate-level verifier, which is the point of the row.
- `- factorization` is a baseline run (`qwen3_32b_parallel_sampling_wcov1_j2`), so it has
  no `ranking_summary.json` and its config columns read `-`. It is a free-text ensemble
  with no dimensions to verify.
- `- label coverage` shows `judged_dims = -` because the wrapper that produced it does not
  record the verdict path in `ranking_summary.json`. It **is** a matched arm: the job log
  shows `50180 verdicts, verifier_mode=llm_drop beta=0.6 llm_unjudged_fill=mean` read from
  `verdicts_arm6_wcov0`, whose `llm_verifier_summary.json` (copied here) reports all six
  judged dimensions, `window_source=fused`, `parse_rate=1.0`. The missing field is a
  logging gap, not a configuration difference.

## Still open

`tab:retriever_robustness` dense/hybrid **accuracy** — four rerank jobs were submitted
2026-07-31 (`t28_dense_AGS`, `t28_dense_one_pass`, `t28_hybrid_AGS`, `t28_hybrid_one_pass`).
Their Recall@200 is already final and is in `sources/.../t28_stage1/*_manifest.json`:
Recall@200 is pool membership, so the candidate-level verifier cannot change it, and the
rebuild reproduced the published values exactly. Only the accuracy column needs the GPU.

The `std` column in the paper is an analytic estimate, `sqrt(p(1-p)/n + g^2)`, not a
multi-seed measurement. No artifact here backs it and none is claimed to.
