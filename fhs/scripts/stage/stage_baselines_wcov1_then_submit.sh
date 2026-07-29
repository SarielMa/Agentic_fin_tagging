#!/bin/bash
# Re-run the two non-structured baselines with the label-coverage term ENABLED.
#
# WHY: the paper states the coverage term "is a shared index property, enabled identically for
# every method", and app:coverage's Panel A measures its gain for exactly these two query forms
# (Raw context +0.231 R@10, Free-text +0.175). But in run_fintagging_grounding_baseline.py the
# retriever's label_coverage_weight is only assigned inside the frozen_ags-family branches, so
# `direct_retrieval` and `one_pass_grounding` ran at the constructor default w_cov=0.0 while
# frozen_ags and one_pass_structured ran at w_cov=1.0. tab:main_results therefore compares FHS
# (coverage on) against two baselines (coverage off), and the paper's fairness claim does not
# hold as run. The coverage term is worth R@10 0.401 -> 0.258 on FHS itself (tab:ablation), so
# this is not a rounding-level difference.
#
# The fix is to re-run those two arms with the term on, not to reword the claim. --label-coverage-
# weight was added for this: it applies to any query mode and defaults to None, so every existing
# run's behaviour is unchanged.
#
# Queries are held FIXED. one_pass_grounding reuses its stored query_descriptions.jsonl, so no
# LLM regenerates anything and the coverage term is the only variable. direct_retrieval never
# used an LLM for queries at all.
#
# Stage 1 (retrieval) is pure CPU and runs locally per the project rule. Stage 2 (the Qwen3-32B
# listwise rerank that produces the Acc./std columns) is sbatched once its candidates are on disk.
#
# NOTE the two stages land in the paper together or not at all: tab:main_results reports R@10,
# R@50, MRR *and* Acc. for these rows. Updating retrieval alone would leave Acc. describing a
# different retrieval configuration -- a worse inconsistency than the one being fixed.
set -uo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${FHS_ROOT}"
RUNS="${REPO}/runs/runs_fintagging_grounding_baseline"
SCRATCH="/nfs/roberts/scratch/pi_sjf37/lm2445/runs_fintagging_grounding_baseline"
LOG="${REPO}/scripts/stage/stage_baselines_wcov1.log"
cd "${REPO}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

say "=== stage 1: retrieval at w_cov=1.0, CPU, local ==="

# --- direct_retrieval -------------------------------------------------------
OUT_DR="${RUNS}/qwen3_32b_direct_retrieval_wcov1"
if [[ -f "${OUT_DR}/bm25_metrics.json" ]]; then
  say "direct_retrieval already staged, skipping"
else
  say "direct_retrieval: retrieving with w_cov=1.0"
  python run_fintagging_grounding_baseline.py \
    --query-mode direct_retrieval \
    --output-dir "${OUT_DR}" \
    --label-coverage-weight 1.0 \
    --top-k 200 \
    >> "${LOG}" 2>&1 || { say "direct_retrieval FAILED"; exit 1; }
  say "direct_retrieval done"
fi

# --- one_pass_grounding -----------------------------------------------------
# Reuse the frozen query descriptions so the LLM is not re-invoked and the queries are identical
# to the published run; only w_cov changes.
OUT_OP="${RUNS}/qwen3_32b_one_pass_grounding_wcov1"
QDESC_SRC="${SCRATCH}/qwen3_32b_one_pass_grounding/query_descriptions.jsonl"
QDESC="${OUT_OP}/query_descriptions.jsonl"
if [[ ! -f "${QDESC_SRC}" ]]; then
  say "MISSING stored queries: ${QDESC_SRC} -- cannot hold queries fixed, aborting"
  exit 1
fi
if [[ -f "${OUT_OP}/bm25_metrics.json" ]]; then
  say "one_pass_grounding already staged, skipping"
else
  # Work on a COPY. generate_query_descriptions_* opens the path in append mode whenever any
  # example is missing, so pointing at the published run's file risks mutating a published
  # artifact if the cache turns out to be incomplete.
  mkdir -p "${OUT_OP}"
  cp -n "${QDESC_SRC}" "${QDESC}"
  SRC_N=$(wc -l < "${QDESC_SRC}")
  say "one_pass_grounding: copied ${SRC_N} stored queries; retrieving with w_cov=1.0"
  # --resume is what makes the copy actually load: without it the cache is ignored, every query
  # counts as missing, and the run would try to load an LLM (and fail, on CPU).
  python run_fintagging_grounding_baseline.py \
    --query-mode one_pass_grounding \
    --output-dir "${OUT_OP}" \
    --query-description-path "${QDESC}" \
    --query-generation-backend vllm \
    --query-generation-model Qwen/Qwen3-32B \
    --resume \
    --label-coverage-weight 1.0 \
    --top-k 200 \
    >> "${LOG}" 2>&1 || { say "one_pass_grounding FAILED"; exit 1; }
  # If the cache had been short, the LLM path would have rewritten this file. Catch it.
  NEW_N=$(wc -l < "${QDESC}")
  if [[ "${SRC_N}" != "${NEW_N}" ]]; then
    say "WARNING: query file changed ${SRC_N} -> ${NEW_N}; queries were NOT held fixed"
  fi
  say "one_pass_grounding done"
fi

# --- report the retrieval-stage deltas before anything touches the paper ----
say "--- retrieval stage, w_cov=0 (published) vs w_cov=1.0 (this run) ---"
python - <<'PY' 2>&1 | tee -a "${LOG}"
import json
from pathlib import Path
RUNS = Path("/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs/runs/runs_fintagging_grounding_baseline")
pairs = [("direct retrieval", "qwen3_32b_direct_retrieval", "qwen3_32b_direct_retrieval_wcov1"),
         ("one-pass free-text", "qwen3_32b_one_pass_grounding", "qwen3_32b_one_pass_grounding_wcov1")]
for label, old, new in pairs:
    o = json.load(open(RUNS / old / "bm25_metrics.json"))["bm25_retrieval"]
    n = json.load(open(RUNS / new / "bm25_metrics.json"))["bm25_retrieval"]
    print(f"\n{label}")
    for k in ("recall_at_10", "recall_at_50", "mrr"):
        print(f"   {k:14s} {o.get(k):.4f} -> {n.get(k):.4f}   ({n.get(k)-o.get(k):+.4f})")
PY

say "=== stage 2: submitting the GPU rerank (Acc./std columns) ==="
JID1=$(sbatch --parsable "${REPO}/scripts/slurm/apply_server_baselines_wcov1_rerank.sh" direct_retrieval "${OUT_DR}")
say "submitted direct_retrieval rerank: ${JID1}"
JID2=$(sbatch --parsable "${REPO}/scripts/slurm/apply_server_baselines_wcov1_rerank.sh" one_pass_grounding "${OUT_OP}")
say "submitted one_pass_grounding rerank: ${JID2}"
say "both queued; the paper is NOT touched until both metrics.json carry qwen_reranked"
