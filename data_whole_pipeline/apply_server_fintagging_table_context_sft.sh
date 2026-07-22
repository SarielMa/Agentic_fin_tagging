#!/bin/bash
#SBATCH --job-name=fintag_table_ctx_sft
#SBATCH --mail-type=ALL
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_fintag_table_context_sft_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# What to run:
#   full         = train + merge + predict + evaluate
#   train        = train only
#   merge        = merge only
#   predict      = predict only
#   eval         = evaluate only
#   predict_eval = predict + evaluate
#   dryrun       = check environment and paths only
#
# Example:
#   sbatch --export=ALL,MODE=full apply_server_fintagging_table_context_sft.sh
#   sbatch --export=ALL,MODE=train,MODEL=meta-llama/Llama-3.3-70B-Instruct apply_server_fintagging_table_context_sft.sh
MODE="${MODE:-full}"

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
PIPELINE_SH="${REPO_ROOT}/run_fintagging_table_context_sft.sh"

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

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
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
[[ -f "${PIPELINE_SH}" ]] || { echo "Missing pipeline script: ${PIPELINE_SH}" >&2; exit 1; }

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi

if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
  echo "Unable to detect available GPUs." >&2
  exit 1
fi

export NUM_GPUS

export MODELS="${MODELS:-Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.3-70B-Instruct}"
if [[ -n "${MODEL:-}" ]]; then
  export MODEL
fi
export DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/FinTagging_800_200_table_context_sft_arrow}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/runs_fintagging_table_context}"
export TP="${TP:-${NUM_GPUS}}"
export MAX_LEN="${MAX_LEN:-16384}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export CONTEXT_MATCH="${CONTEXT_MATCH:-relaxed}"
export JACCARD_THRESHOLD="${JACCARD_THRESHOLD:-0.6}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export EPOCHS="${EPOCHS:-3}"
export LR="${LR:-2e-4}"
export BF16="${BF16:-1}"
export USE_QLORA="${USE_QLORA:-1}"

case "${MODE}" in
  full)
    export TRAIN=1 MERGE=1 PREDICT=1 EVAL=1
    ;;
  train)
    export TRAIN=1 MERGE=0 PREDICT=0 EVAL=0
    ;;
  merge)
    export TRAIN=0 MERGE=1 PREDICT=0 EVAL=0
    ;;
  predict)
    export TRAIN=0 MERGE=0 PREDICT=1 EVAL=0
    ;;
  eval)
    export TRAIN=0 MERGE=0 PREDICT=0 EVAL=1
    ;;
  predict_eval)
    export TRAIN=0 MERGE=0 PREDICT=1 EVAL=1
    ;;
  dryrun)
    export TRAIN=0 MERGE=0 PREDICT=0 EVAL=0
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected full|train|merge|predict|eval|predict_eval|dryrun." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PIPELINE_SH=${PIPELINE_SH}"
echo "MODE=${MODE}"
echo "MODEL=${MODEL:-<unset; using MODELS>}"
echo "MODELS=${MODELS}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "TP=${TP}"
echo "MAX_LEN=${MAX_LEN}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "CONTEXT_MATCH=${CONTEXT_MATCH}"
echo "JACCARD_THRESHOLD=${JACCARD_THRESHOLD}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"
echo "BF16=${BF16}"
echo "USE_QLORA=${USE_QLORA}"
echo "============================================================"

bash "${PIPELINE_SH}"
