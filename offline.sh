#!/usr/bin/env sh

# ---- CUDA toolkit for DeepSpeed ops ----
if [ -z "${CONDA_PREFIX:-}" ]; then
  module purge
fi
module load StdEnv || true
module load CUDA/12.6.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER:-user}}"
mkdir -p "$TRITON_CACHE_DIR"


python scripts/run_two_agent_system.py \
  --mode offline \
  --table-evidence-backend llama \
  --table-evidence-model meta-llama/Llama-3.2-3B-Instruct \
  --selector-backend llama \
  --selector-model meta-llama/Llama-3.2-3B-Instruct \
  --validator-backend llama \
  --validator-model meta-llama/Llama-3.2-3B-Instruct \
  --output-dir outputs/offline_llama_agents_TTT
