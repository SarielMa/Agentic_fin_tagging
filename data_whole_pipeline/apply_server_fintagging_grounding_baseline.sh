#!/bin/bash
#SBATCH --job-name=fintag_direct
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_fintag_direct_retrieval_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Compatibility wrapper. Prefer apply_server_fintagging_direct_retrieval.sh.

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
export QUERY_MODE="${QUERY_MODE:-direct_retrieval}"

bash "${SCRIPT_DIR}/apply_server_fintagging_direct_retrieval.sh"
