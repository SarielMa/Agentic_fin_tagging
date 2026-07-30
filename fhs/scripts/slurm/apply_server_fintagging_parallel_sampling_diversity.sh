#!/bin/bash
#SBATCH --job-name=fintag_parallel_div
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs/logs/%j_fintag_parallel_sampling_diversity_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Parallel sampling diversity:
#   B sequential-exclusion grounding hypotheses, each retrieved with q_lab/q_def
#   dual surfaces and fused with the same RRF path as other grounding methods.

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-parallel_sampling_diversity}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_parallel_sampling_diversity}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.8}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
