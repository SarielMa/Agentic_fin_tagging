#!/bin/bash
#SBATCH --job-name=fintag_full_par_div
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_fintag_fulltagging_parallel_sampling_diversity_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# FullTagging parallel sampling diversity:
#   extractor predictions first, then diversity-prompted dual-surface grounding.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs_fintagging_fulltagging}"
export EXTRACTOR_TAG="${EXTRACTOR_TAG:-qwen2.5_14b_extractors}"
export QUERY_MODE="${QUERY_MODE:-parallel_sampling_diversity}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${EXTRACTOR_TAG}/qwen3_32b_parallel_sampling_diversity}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.8}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/apply_server_fintagging_fulltagging_direct_retrieval.sh"
