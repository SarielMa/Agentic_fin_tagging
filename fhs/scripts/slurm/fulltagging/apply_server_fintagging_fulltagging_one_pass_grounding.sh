#!/bin/bash
#SBATCH --job-name=fintag_full_onepass
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_fintag_fulltagging_one_pass_grounding_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# FullTagging one-pass grounding:
#   extractor predictions first, then the same one-pass grounding pipeline.

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_fulltagging}"
export EXTRACTOR_TAG="${EXTRACTOR_TAG:-qwen2.5_14b_extractors}"
export QUERY_MODE="${QUERY_MODE:-one_pass_grounding}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${EXTRACTOR_TAG}/qwen3_32b_one_pass_grounding}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/fulltagging/apply_server_fintagging_fulltagging_direct_retrieval.sh"
