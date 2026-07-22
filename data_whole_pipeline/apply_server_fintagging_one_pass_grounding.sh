#!/bin/bash
#SBATCH --job-name=fintag_one_pass
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_fintag_one_pass_grounding_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# One-pass grounding comparison:
#   Qwen generates a brief retrieval query from (entity, type, context).
#   BM25 retrieves top-200 using (generated query, entity, type).
#   The same Qwen reranker and evaluator as direct retrieval are then used.

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"

export QUERY_MODE="${QUERY_MODE:-one_pass_grounding}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${SCRIPT_DIR}/apply_server_fintagging_direct_retrieval.sh"
