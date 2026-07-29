#!/bin/bash
#SBATCH --job-name=fintag_full_frozen_ags
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_fintag_fulltagging_frozen_ags_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# FullTagging frozen_ags grounding:
#   extractor predictions first, then the frozen AGS single-pass grounding method
#   (J=2 structured hypotheses -> def/lab render -> coverage retrieval -> sum-RRF ->
#    range-normalize -> agree rerank), then the listwise rerank on top so the method is scored
#    at the same stage as every other query mode (rerank_gold_entity_scope in
#    fulltagging_metrics.json).

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_fulltagging}"
export EXTRACTOR_TAG="${EXTRACTOR_TAG:-qwen2.5_14b_extractors}"
export QUERY_MODE="${QUERY_MODE:-frozen_ags}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${EXTRACTOR_TAG}/qwen3_32b_frozen_ags}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
# Full eval: extraction + AGS retrieval stage + listwise rerank, same as the other methods.
# REUSE_CANDIDATES=1 reuses the existing bm25_candidates.jsonl, so a rerun is rerank-only.
export RUN_RERANK="${RUN_RERANK:-1}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/fulltagging/apply_server_fintagging_fulltagging_direct_retrieval.sh"
