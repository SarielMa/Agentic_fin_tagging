#!/bin/bash
#SBATCH --job-name=fintag_ags_seq_random
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_fintag_ags_seq_random_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# AGS+Seq-random: the control for the control (ags_seq_arms_spec.md section 2.3).
#
# Identical to apply_server_fintagging_ags_seq.sh in every respect except directive
# selection: the directive is drawn uniformly from the same admissible slate. The
# posteriors are still computed and updated -- they are simply not consulted -- which is
# what makes "the posteriors moved" and "consulting them helped" separable claims.
#
# Run with the same TEST_JSONL and instance order as the Thompson arm; the metrics script
# pairs the two per fact and reports any instance-order mismatch.
#
# Same cost profile as the Thompson arm (8 LLM calls, up to 16 retrievals per episode);
# RESUME=1 makes a wall-clock timeout recoverable by resubmitting.
#
# Examples:
#   sbatch apply_server_fintagging_ags_seq_random.sh
#   sbatch --export=ALL,LIMIT=20 apply_server_fintagging_ags_seq_random.sh

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-ags_seq_random}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_ags_seq_random}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RUN_RERANK="${RUN_RERANK:-1}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

export AGS_SEQ_MAX_ROUNDS="${AGS_SEQ_MAX_ROUNDS:-4}"
export AGS_SEQ_NOVELTY_GATE="${AGS_SEQ_NOVELTY_GATE:-0}"
export AGS_SEQ_SEED="${AGS_SEQ_SEED:-20260724}"

bash "${REPO_ROOT}/apply_server_fintagging_direct_retrieval.sh"
