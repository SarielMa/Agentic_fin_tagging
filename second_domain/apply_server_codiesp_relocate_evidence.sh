#!/bin/bash
#SBATCH --job-name=codiesp_reloc
#SBATCH --mail-type=ALL
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain/logs/%j_codiesp_relocate_evidence_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

DOMAIN_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain"

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

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${TRITON_CACHE_DIR}"

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

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

cd "${DOMAIN_ROOT}"

export RELOCATION_MODEL="${RELOCATION_MODEL:-Qwen/Qwen3-32B}"
export RELOCATION_BATCH_SIZE="${RELOCATION_BATCH_SIZE:-64}"
export RELOCATION_MAX_INPUT_TOKENS="${RELOCATION_MAX_INPUT_TOKENS:-12000}"
export RELOCATION_MAX_NEW_TOKENS="${RELOCATION_MAX_NEW_TOKENS:-96}"
export BF16="${BF16:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export RESUME="${RESUME:-1}"
export LIMIT="${LIMIT:-}"

ARGS=(
  --model "${RELOCATION_MODEL}"
  --batch-size "${RELOCATION_BATCH_SIZE}"
  --max-input-tokens "${RELOCATION_MAX_INPUT_TOKENS}"
  --max-new-tokens "${RELOCATION_MAX_NEW_TOKENS}"
  --query-max-new-tokens "${RELOCATION_MAX_NEW_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --log-every 256
)

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python -m py_compile scripts/relocate_codiesp_evidence.py scripts/prepare_codiesp_domain.py
python scripts/relocate_codiesp_evidence.py "${ARGS[@]}"
