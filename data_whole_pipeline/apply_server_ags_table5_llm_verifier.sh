#!/bin/bash
#SBATCH --job-name=ags_table5_llm_verifier
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_ags_table5_llm_verifier_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Table 5 row 3.9: + LLM verification layer (comparing_methods/ags_table5_ablation_spec.md).
# The only row needing new generation: ~2 x 2,509 batched calls re-judging FAMILY/ROLE/EVENT
# on AGS's own top-M=10 cluster representatives per hypothesis.
#
# Standalone by design, modeled on apply_server_fintagging_frozen_ags.sh's environment setup
# but calling ags_table5_ablation/run_llm_verifier.py directly -- it does not go through
# run_fintagging_grounding_baseline.sh's --query-mode dispatch, which other experiments
# (e.g. the AGS+Seq sequential-arm jobs) run against concurrently. Reads AGS's already-
# computed test-split trace read-only; writes only to its own runs_ags_table5_ablation/ tree.
#
# After this completes, re-run apply_server_ags_table5_offline.sh (or just
# ags_table5_ablation/run_test_rows.py) so ablation.csv picks up the verdicts file and folds
# row 3.9 into the table.
#
# Examples:
#   sbatch apply_server_ags_table5_llm_verifier.sh
#   sbatch --export=ALL,LIMIT=20 apply_server_ags_table5_llm_verifier.sh   # smoke test

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

conda activate finben_b200

which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export TOP_M="${TOP_M:-10}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_table5_ablation/qwen3_32b}"
RESUME_ARGS=()
if [[ "${RESUME:-1}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "QUERY_GENERATION_MODEL=${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND=${QUERY_GENERATION_BACKEND}"
echo "TOP_M=${TOP_M}"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

python -m py_compile ags_table5_ablation/run_llm_verifier.py

python ags_table5_ablation/run_llm_verifier.py \
  --output-dir "${OUTPUT_DIR}" \
  --top-m "${TOP_M}" \
  --query-generation-model "${QUERY_GENERATION_MODEL}" \
  --query-generation-backend "${QUERY_GENERATION_BACKEND}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --vllm-batch-size "${VLLM_BATCH_SIZE}" \
  "${RESUME_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"
