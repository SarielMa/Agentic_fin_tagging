#!/bin/bash
#SBATCH --job-name=fintag_direct
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/scratch/pi_sjf37/lm2445/logs/%j_fintag_direct_retrieval_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Direct retrieval grounding method.
#
# What to run:
#   full      = BM25 retrieval + Qwen rerank + metrics
#   retrieval = BM25 retrieval only
#   dryrun    = check environment and paths only
#
# Examples:
#   sbatch apply_server_fintagging_direct_retrieval.sh
#   sbatch --export=ALL,MODE=retrieval apply_server_fintagging_direct_retrieval.sh
#   sbatch --export=ALL,LIMIT=20 apply_server_fintagging_direct_retrieval.sh
MODE="${MODE:-full}"

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
PIPELINE_SH="${REPO_ROOT}/run_fintagging_grounding_baseline.sh"

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
[[ -f "${PIPELINE_SH}" ]] || { echo "Missing pipeline script: ${PIPELINE_SH}" >&2; exit 1; }

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

export TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/FinTagging_800_200_grounding_test_JSON/data/test.jsonl}"
export TAXONOMY_JSONL="${TAXONOMY_JSONL:-/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
export QUERY_MODE="${QUERY_MODE:-direct_retrieval}"
export RUNS_ROOT="${RUNS_ROOT:-${SCRATCH_ROOT}/runs_fintagging_grounding_baseline}"
if [[ "${QUERY_MODE}" == "one_pass_grounding" || "${QUERY_MODE}" == "llm_description" ]]; then
  DEFAULT_OUTPUT_DIR="${RUNS_ROOT}/qwen3_32b_one_pass_grounding"
else
  DEFAULT_OUTPUT_DIR="${RUNS_ROOT}/qwen3_32b_direct_retrieval"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
export TOP_K="${TOP_K:-200}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export TYPE_FILTER="${TYPE_FILTER:-1}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
export QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
export QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-128}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.0}"
export QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
export BF16="${BF16:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-12000}"
export CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
export RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export RESUME="${RESUME:-1}"
export LOG_EVERY="${LOG_EVERY:-25}"

case "${MODE}" in
  full)
    export RUN_RERANK=1
    ;;
  retrieval)
    export RUN_RERANK=0
    ;;
  dryrun)
    export RUN_RERANK=0
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected full|retrieval|dryrun." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PIPELINE_SH=${PIPELINE_SH}"
echo "MODE=${MODE}"
echo "TEST_JSONL=${TEST_JSONL}"
echo "TAXONOMY_JSONL=${TAXONOMY_JSONL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "QUERY_MODE=${QUERY_MODE}"
echo "REUSE_CANDIDATES=${REUSE_CANDIDATES}"
echo "TYPE_FILTER=${TYPE_FILTER}"
echo "RUN_RERANK=${RUN_RERANK}"
echo "RERANK_MODEL=${RERANK_MODEL}"
echo "RERANK_BACKEND=${RERANK_BACKEND}"
echo "QUERY_GENERATION_MODEL=${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND=${QUERY_GENERATION_BACKEND}"
echo "QUERY_MAX_NEW_TOKENS=${QUERY_MAX_NEW_TOKENS}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE}"
echo "CONTEXT_MAX_CHARS=${CONTEXT_MAX_CHARS}"
echo "CANDIDATE_DOC_MAX_CHARS=${CANDIDATE_DOC_MAX_CHARS}"
echo "MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "RESUME=${RESUME}"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

python -m py_compile "${REPO_ROOT}/run_fintagging_grounding_baseline.py"
[[ "${MODE}" == "dryrun" ]] && exit 0

bash "${PIPELINE_SH}"
