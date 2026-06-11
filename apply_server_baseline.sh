#!/bin/bash
#SBATCH --job-name=finai_baseline
#SBATCH --mail-type=ALL
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=128G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_finai_baseline_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# This repo runs Qwen3-14B / Llama-3.2-3B through Hugging Face transformers with
# device_map="auto" (fp16). A 14B in fp16 is ~28GB, so one B200 is plenty; there
# is no vLLM and no tensor-parallel config to set, unlike the SFT-eval pipeline.
REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic"
TEST_SH="${REPO_ROOT}/run_baseline_tests.sh"
CONDA_ENV="${CONDA_ENV:-finben_b200}"

# ---- Clean any inherited conda state so `module --force purge` is safe --------
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

# ---- CUDA toolkit ------------------------------------------------------------
module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# ---- conda env ---------------------------------------------------------------
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

conda activate "${CONDA_ENV}"

which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd "${REPO_ROOT}"
[[ -f "${TEST_SH}" ]] || { echo "Missing test script: ${TEST_SH}" >&2; exit 1; }

# ---- baseline run settings ---------------------------------------------------
# The sbatch already configured CUDA, so tell run_baseline_tests.sh not to redo
# `module load` (its setup_cuda would otherwise reload a different CUDA module).
export SKIP_CUDA_SETUP=1

# Focus this run on the cap fix: compare single_llm vs offline. Override any of
# these on submission, e.g.  sbatch --export=ALL,RUN_ONLINE_GT=1 apply_server_baseline.sh
export RUN_SINGLE_LLM="${RUN_SINGLE_LLM:-1}"
export RUN_OFFLINE="${RUN_OFFLINE:-1}"
export RUN_ONLINE_GT="${RUN_ONLINE_GT:-1}"
export RUN_ONLINE_WO_GT="${RUN_ONLINE_WO_GT:-1}"

# Raised input cap (also the new code default) — keep it explicit and overridable.
export FINAI_MAX_INPUT_TOKENS="${FINAI_MAX_INPUT_TOKENS:-32768}"

# Use already-downloaded model caches; flip to 0 to allow downloads.
export FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-1}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TEST_SH=${TEST_SH}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "FINAI_MAX_INPUT_TOKENS=${FINAI_MAX_INPUT_TOKENS}"
echo "RUN_SINGLE_LLM=${RUN_SINGLE_LLM} RUN_OFFLINE=${RUN_OFFLINE} RUN_ONLINE_GT=${RUN_ONLINE_GT} RUN_ONLINE_WO_GT=${RUN_ONLINE_WO_GT}"

sh "${TEST_SH}"
