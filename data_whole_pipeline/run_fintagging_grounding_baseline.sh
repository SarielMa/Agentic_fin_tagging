#!/usr/bin/env bash
set -euo pipefail

# Grounding experiments:
#   Method-specific query generation -> BM25 candidates -> shared Qwen reranking/eval.

REPO_ROOT="$(readlink -f "$(dirname "$0")")"
PROJECT_ROOT="$(readlink -f "${REPO_ROOT}/..")"

TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/FinTagging_800_200_grounding_test_JSON/data/test.jsonl}"
TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
QUERY_MODE="${QUERY_MODE:-direct_retrieval}"
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
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/runs_fintagging_grounding_baseline/${DEFAULT_METHOD_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

TOP_K="${TOP_K:-200}"
RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
FEEDBACK_CANDIDATE_COUNT="${FEEDBACK_CANDIDATE_COUNT:-10}"
RRF_KAPPA="${RRF_KAPPA:-60.0}"
MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
TYPE_FILTER="${TYPE_FILTER:-1}"
RUN_RERANK="${RUN_RERANK:-1}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
QUERY_DESCRIPTION_PATH="${QUERY_DESCRIPTION_PATH:-}"
QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-128}"
QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.0}"
QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
BF16="${BF16:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-12000}"
CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RESUME="${RESUME:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
LIMIT="${LIMIT:-}"

require_path () {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${REPO_ROOT}/run_fintagging_grounding_baseline.py" "grounding baseline script"
require_path "${TEST_JSONL}" "context tagging test JSONL"
require_path "${TAXONOMY_JSONL}" "enriched retrieval taxonomy"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

ARGS=(
  --test-jsonl "${TEST_JSONL}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --top-k "${TOP_K}"
  --retrieval-rounds "${RETRIEVAL_ROUNDS}"
  --feedback-candidate-count "${FEEDBACK_CANDIDATE_COUNT}"
  --rrf-kappa "${RRF_KAPPA}"
  --memory-top-k "${MEMORY_TOP_K}"
  --query-mode "${QUERY_MODE}"
  --rerank-model "${RERANK_MODEL}"
  --rerank-backend "${RERANK_BACKEND}"
  --query-generation-model "${QUERY_GENERATION_MODEL}"
  --query-generation-backend "${QUERY_GENERATION_BACKEND}"
  --query-context-max-chars "${QUERY_CONTEXT_MAX_CHARS}"
  --query-max-input-tokens "${QUERY_MAX_INPUT_TOKENS}"
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS}"
  --query-temperature "${QUERY_TEMPERATURE}"
  --query-top-p "${QUERY_TOP_P}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --vllm-batch-size "${VLLM_BATCH_SIZE}"
  --context-max-chars "${CONTEXT_MAX_CHARS}"
  --candidate-doc-max-chars "${CANDIDATE_DOC_MAX_CHARS}"
  --rerank-list-size "${RERANK_LIST_SIZE}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --log-every "${LOG_EVERY}"
)

if [[ "${REUSE_CANDIDATES}" == "1" ]]; then
  ARGS+=(--reuse-candidates)
fi

if [[ -n "${QUERY_DESCRIPTION_PATH}" ]]; then
  ARGS+=(--query-description-path "${QUERY_DESCRIPTION_PATH}")
fi

if [[ "${TYPE_FILTER}" == "1" ]]; then
  ARGS+=(--type-filter)
else
  ARGS+=(--no-type-filter)
fi

if [[ "${RUN_RERANK}" == "1" ]]; then
  ARGS+=(--run-rerank)
fi

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "TASK                 : grounding_experiment"
echo "TEST_JSONL           : ${TEST_JSONL}"
echo "TAXONOMY_JSONL       : ${TAXONOMY_JSONL}"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "HF_HOME              : ${HF_HOME}"
echo "HF_HUB_CACHE         : ${HF_HUB_CACHE}"
echo "TOP_K                : ${TOP_K}"
echo "RETRIEVAL_ROUNDS     : ${RETRIEVAL_ROUNDS}"
echo "FEEDBACK_CANDIDATE_COUNT: ${FEEDBACK_CANDIDATE_COUNT}"
echo "RRF_KAPPA            : ${RRF_KAPPA}"
echo "MEMORY_TOP_K         : ${MEMORY_TOP_K}"
echo "QUERY_MODE           : ${QUERY_MODE}"
echo "REUSE_CANDIDATES     : ${REUSE_CANDIDATES}"
echo "TYPE_FILTER          : ${TYPE_FILTER}"
echo "RUN_RERANK           : ${RUN_RERANK}"
echo "RERANK_MODEL         : ${RERANK_MODEL}"
echo "RERANK_BACKEND       : ${RERANK_BACKEND}"
echo "QUERY_GENERATION_MODEL: ${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND: ${QUERY_GENERATION_BACKEND}"
echo "QUERY_DESCRIPTION_PATH: ${QUERY_DESCRIPTION_PATH:-<default>}"
echo "QUERY_MAX_NEW_TOKENS : ${QUERY_MAX_NEW_TOKENS}"
echo "TENSOR_PARALLEL_SIZE : ${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "MAX_NUM_SEQS         : ${MAX_NUM_SEQS}"
echo "VLLM_BATCH_SIZE      : ${VLLM_BATCH_SIZE}"
echo "CONTEXT_MAX_CHARS    : ${CONTEXT_MAX_CHARS}"
echo "CANDIDATE_DOC_MAX_CHARS: ${CANDIDATE_DOC_MAX_CHARS}"
echo "MAX_INPUT_TOKENS     : ${MAX_INPUT_TOKENS}"
echo "MAX_NEW_TOKENS       : ${MAX_NEW_TOKENS}"
echo "RESUME               : ${RESUME}"
echo "LIMIT                : ${LIMIT:-<none>}"
echo "============================================================"

python "${REPO_ROOT}/run_fintagging_grounding_baseline.py" "${ARGS[@]}"
