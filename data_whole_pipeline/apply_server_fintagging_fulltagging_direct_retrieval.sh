#!/bin/bash
#SBATCH --job-name=fintag_full_direct
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_fintag_fulltagging_direct_retrieval_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# FullTagging direct retrieval:
#   run extractor predictions from original contexts, then call the same
#   direct retrieval/rerank grounding pipeline with extracted entity/type inputs.

MODE="${MODE:-full}"

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
PIPELINE_SH="${REPO_ROOT}/run_fintagging_fulltagging.sh"

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

export QUERY_MODE="${QUERY_MODE:-direct_retrieval}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs_fintagging_fulltagging}"
export EXTRACTOR_TAG="${EXTRACTOR_TAG:-qwen2.5_14b_extractors}"
case "${QUERY_MODE}" in
  direct|direct_retrieval)
    DEFAULT_METHOD_DIR="qwen3_32b_direct_retrieval"
    ;;
  llm_description|one_pass_grounding)
    DEFAULT_METHOD_DIR="qwen3_32b_one_pass_grounding"
    ;;
  intrinsic|self_refinement|intrinsic_self_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_intrinsic_self_refinement"
    ;;
  feedback|retrieval_feedback|retrieval_feedback_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_retrieval_feedback_refinement"
    ;;
  parallel|parallel_sampling)
    DEFAULT_METHOD_DIR="qwen3_32b_parallel_sampling"
    ;;
  decomposed|decomposed_retrieval)
    DEFAULT_METHOD_DIR="qwen3_32b_decomposed_retrieval"
    ;;
  operator|operator_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_operator_refinement"
    ;;
  memory|memory_guided_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_memory_guided_refinement"
    ;;
  *)
    DEFAULT_METHOD_DIR="qwen3_32b_${QUERY_MODE}"
    ;;
esac
DEFAULT_OUTPUT_DIR="${RUNS_ROOT}/${EXTRACTOR_TAG}/${DEFAULT_METHOD_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

export RUN_EXTRACTION="${RUN_EXTRACTION:-1}"
export TEXT_EXTRACTOR_MODEL="${TEXT_EXTRACTOR_MODEL:-${REPO_ROOT}/runs_fintagging_text_context/qwen2.5_14b_instruct/sft_3ep/merged}"
export TABLE_EXTRACTOR_MODEL="${TABLE_EXTRACTOR_MODEL:-${REPO_ROOT}/runs_fintagging_table_context/qwen2.5_14b_instruct/sft_3ep/merged}"
export TEXT_EXTRACTION_PREDICTIONS="${TEXT_EXTRACTION_PREDICTIONS:-${REPO_ROOT}/runs_fintagging_text_context/qwen2.5_14b_instruct/sft_3ep/predictions/test_predictions.jsonl}"
export TABLE_EXTRACTION_PREDICTIONS="${TABLE_EXTRACTION_PREDICTIONS:-${REPO_ROOT}/runs_fintagging_table_context/qwen2.5_14b_instruct/sft_3ep/predictions/test_predictions.jsonl}"
export EXTRACTION_BACKEND="${EXTRACTION_BACKEND:-vllm}"
export EXTRACTION_VLLM_BATCH_SIZE="${EXTRACTION_VLLM_BATCH_SIZE:-16}"
export EXTRACTION_TENSOR_PARALLEL_SIZE="${EXTRACTION_TENSOR_PARALLEL_SIZE:-1}"
export EXTRACTION_GPU_MEMORY_UTILIZATION="${EXTRACTION_GPU_MEMORY_UTILIZATION:-0.9}"
export EXTRACTION_MAX_MODEL_LEN="${EXTRACTION_MAX_MODEL_LEN:-16384}"
export EXTRACTION_MAX_NUM_SEQS="${EXTRACTION_MAX_NUM_SEQS:-8}"
export TOP_K="${TOP_K:-200}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export FEEDBACK_CANDIDATE_COUNT="${FEEDBACK_CANDIDATE_COUNT:-10}"
export RRF_KAPPA="${RRF_KAPPA:-60.0}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
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
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
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
  prepare)
    export RUN_RERANK=0
    ;;
  dryrun)
    export RUN_RERANK=0
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected full|retrieval|prepare|dryrun." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PIPELINE_SH=${PIPELINE_SH}"
echo "MODE=${MODE}"
echo "QUERY_MODE=${QUERY_MODE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_EXTRACTION=${RUN_EXTRACTION}"
echo "TEXT_EXTRACTOR_MODEL=${TEXT_EXTRACTOR_MODEL}"
echo "TABLE_EXTRACTOR_MODEL=${TABLE_EXTRACTOR_MODEL}"
echo "EXTRACTION_BACKEND=${EXTRACTION_BACKEND}"
echo "EXTRACTION_VLLM_BATCH_SIZE=${EXTRACTION_VLLM_BATCH_SIZE}"
echo "EXTRACTION_TENSOR_PARALLEL_SIZE=${EXTRACTION_TENSOR_PARALLEL_SIZE}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "EXTRACTOR_TAG=${EXTRACTOR_TAG}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "TOP_K=${TOP_K}"
echo "RETRIEVAL_ROUNDS=${RETRIEVAL_ROUNDS}"
echo "FEEDBACK_CANDIDATE_COUNT=${FEEDBACK_CANDIDATE_COUNT}"
echo "RRF_KAPPA=${RRF_KAPPA}"
echo "MEMORY_TOP_K=${MEMORY_TOP_K}"
echo "REUSE_CANDIDATES=${REUSE_CANDIDATES}"
echo "RUN_RERANK=${RUN_RERANK}"
echo "RERANK_MODEL=${RERANK_MODEL}"
echo "QUERY_GENERATION_MODEL=${QUERY_GENERATION_MODEL}"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

python -m py_compile "${REPO_ROOT}/generate_fintagging_fulltagging_extractions.py"
python -m py_compile "${REPO_ROOT}/build_fintagging_fulltagging_grounding_input.py"
python -m py_compile "${REPO_ROOT}/evaluate_fintagging_fulltagging_pipeline.py"
python -m py_compile "${REPO_ROOT}/run_fintagging_grounding_baseline.py"
[[ "${MODE}" == "dryrun" ]] && exit 0

bash "${PIPELINE_SH}"
