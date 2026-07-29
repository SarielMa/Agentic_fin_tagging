#!/bin/bash
#SBATCH --job-name=fintag_bandit_free
#SBATCH --mail-type=ALL
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_fintag_bandit_freeform_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Bandit-guided free-form rewriting.
# For the matched-arm variant, submit with QUERY_MODE=bandit_freeform_10arm.

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-bandit_freeform}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_${QUERY_MODE}}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export BANDIT_INITIAL_GROUNDINGS="${BANDIT_INITIAL_GROUNDINGS:-3}"
export BANDIT_REPLAY="${BANDIT_REPLAY:-1}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.8}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
