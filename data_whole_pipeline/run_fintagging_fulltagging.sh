#!/usr/bin/env bash
set -euo pipefail

# FullTagging experiment:
#   original FinTagging context -> text/table extractor predictions ->
#   weak grounding input -> existing direct/one-pass grounding pipeline.

REPO_ROOT="$(readlink -f "$(dirname "$0")")"
PROJECT_ROOT="$(readlink -f "${REPO_ROOT}/..")"

ORIGINAL_TEST_PARQUET="${ORIGINAL_TEST_PARQUET:-${REPO_ROOT}/FinTagging_800_200_HF/data/test.parquet}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs_fintagging_fulltagging}"
EXTRACTOR_TAG="${EXTRACTOR_TAG:-qwen2.5_14b_extractors}"
QUERY_MODE="${QUERY_MODE:-direct_retrieval}"
if [[ "${QUERY_MODE}" == "one_pass_grounding" || "${QUERY_MODE}" == "llm_description" ]]; then
  DEFAULT_OUTPUT_DIR="${RUNS_ROOT}/${EXTRACTOR_TAG}/qwen3_32b_one_pass_grounding"
else
  DEFAULT_OUTPUT_DIR="${RUNS_ROOT}/${EXTRACTOR_TAG}/qwen3_32b_direct_retrieval"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

MODE="${MODE:-full}"
RUN_EXTRACTION="${RUN_EXTRACTION:-1}"
TEXT_EXTRACTOR_MODEL="${TEXT_EXTRACTOR_MODEL:-${REPO_ROOT}/runs_fintagging_text_context/qwen2.5_14b_instruct/sft_3ep/merged}"
TABLE_EXTRACTOR_MODEL="${TABLE_EXTRACTOR_MODEL:-${REPO_ROOT}/runs_fintagging_table_context/qwen2.5_14b_instruct/sft_3ep/merged}"
EXTRACTION_PREDICTIONS="${EXTRACTION_PREDICTIONS:-${OUTPUT_DIR}/extraction_predictions.jsonl}"
TEXT_EXTRACTION_PREDICTIONS="${TEXT_EXTRACTION_PREDICTIONS:-${REPO_ROOT}/runs_fintagging_text_context/qwen2.5_14b_instruct/sft_3ep/predictions/test_predictions.jsonl}"
TABLE_EXTRACTION_PREDICTIONS="${TABLE_EXTRACTION_PREDICTIONS:-${REPO_ROOT}/runs_fintagging_table_context/qwen2.5_14b_instruct/sft_3ep/predictions/test_predictions.jsonl}"
FULLTAGGING_INPUT_JSONL="${FULLTAGGING_INPUT_JSONL:-${OUTPUT_DIR}/extracted_grounding_input.jsonl}"
FULLTAGGING_METADATA_JSON="${FULLTAGGING_METADATA_JSON:-${OUTPUT_DIR}/fulltagging_input_metadata.json}"
FULLTAGGING_EVAL_JSON="${FULLTAGGING_EVAL_JSON:-${OUTPUT_DIR}/fulltagging_metrics.json}"
EXTRACTION_METADATA_JSON="${EXTRACTION_METADATA_JSON:-${OUTPUT_DIR}/extraction_prediction_metadata.json}"

TEXT_MAX_NEW_TOKENS="${TEXT_MAX_NEW_TOKENS:-2048}"
TABLE_MAX_NEW_TOKENS="${TABLE_MAX_NEW_TOKENS:-4096}"
EXTRACTION_TEMPERATURE="${EXTRACTION_TEMPERATURE:-0.0}"
EXTRACTION_TOP_P="${EXTRACTION_TOP_P:-1.0}"
EXTRACTION_BACKEND="${EXTRACTION_BACKEND:-vllm}"
EXTRACTION_VLLM_BATCH_SIZE="${EXTRACTION_VLLM_BATCH_SIZE:-16}"
EXTRACTION_TENSOR_PARALLEL_SIZE="${EXTRACTION_TENSOR_PARALLEL_SIZE:-1}"
EXTRACTION_GPU_MEMORY_UTILIZATION="${EXTRACTION_GPU_MEMORY_UTILIZATION:-0.9}"
EXTRACTION_MAX_MODEL_LEN="${EXTRACTION_MAX_MODEL_LEN:-16384}"
EXTRACTION_MAX_NUM_SEQS="${EXTRACTION_MAX_NUM_SEQS:-8}"
EXTRACTION_TRUST_REMOTE_CODE="${EXTRACTION_TRUST_REMOTE_CODE:-0}"
EXTRACTION_ENFORCE_EAGER="${EXTRACTION_ENFORCE_EAGER:-0}"
BF16="${BF16:-1}"
RESUME="${RESUME:-1}"
CONTEXT_LIMIT="${CONTEXT_LIMIT:-${LIMIT:-}}"
GROUNDING_LIMIT="${GROUNDING_LIMIT:-}"

TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"

require_path () {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${ORIGINAL_TEST_PARQUET}" "original FinTagging test parquet"
require_path "${REPO_ROOT}/build_fintagging_fulltagging_grounding_input.py" "fullTagging input builder"
require_path "${REPO_ROOT}/evaluate_fintagging_fulltagging_pipeline.py" "fullTagging evaluator"
require_path "${REPO_ROOT}/run_fintagging_grounding_baseline.sh" "grounding pipeline"
require_path "${TAXONOMY_JSONL}" "enriched retrieval taxonomy"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

case "${MODE}" in
  full)
    RUN_GROUNDING=1
    export RUN_RERANK="${RUN_RERANK:-1}"
    ;;
  retrieval)
    RUN_GROUNDING=1
    export RUN_RERANK="${RUN_RERANK:-0}"
    ;;
  prepare)
    RUN_GROUNDING=0
    export RUN_RERANK="${RUN_RERANK:-0}"
    ;;
  dryrun)
    RUN_GROUNDING=0
    export RUN_RERANK="${RUN_RERANK:-0}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected full|retrieval|prepare|dryrun." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "TASK                       : fulltagging"
echo "MODE                       : ${MODE}"
echo "ORIGINAL_TEST_PARQUET      : ${ORIGINAL_TEST_PARQUET}"
echo "OUTPUT_DIR                 : ${OUTPUT_DIR}"
echo "QUERY_MODE                 : ${QUERY_MODE}"
echo "RUN_EXTRACTION             : ${RUN_EXTRACTION}"
echo "TEXT_EXTRACTOR_MODEL       : ${TEXT_EXTRACTOR_MODEL}"
echo "TABLE_EXTRACTOR_MODEL      : ${TABLE_EXTRACTOR_MODEL}"
echo "EXTRACTION_BACKEND         : ${EXTRACTION_BACKEND}"
echo "EXTRACTION_VLLM_BATCH_SIZE : ${EXTRACTION_VLLM_BATCH_SIZE}"
echo "EXTRACTION_TENSOR_PARALLEL_SIZE: ${EXTRACTION_TENSOR_PARALLEL_SIZE}"
echo "EXTRACTION_MAX_MODEL_LEN   : ${EXTRACTION_MAX_MODEL_LEN}"
echo "EXTRACTION_PREDICTIONS     : ${EXTRACTION_PREDICTIONS}"
echo "TEXT_EXTRACTION_PREDICTIONS: ${TEXT_EXTRACTION_PREDICTIONS}"
echo "TABLE_EXTRACTION_PREDICTIONS: ${TABLE_EXTRACTION_PREDICTIONS}"
echo "FULLTAGGING_INPUT_JSONL    : ${FULLTAGGING_INPUT_JSONL}"
echo "FULLTAGGING_EVAL_JSON      : ${FULLTAGGING_EVAL_JSON}"
echo "TAXONOMY_JSONL             : ${TAXONOMY_JSONL}"
echo "HF_HOME                    : ${HF_HOME}"
echo "HF_HUB_CACHE               : ${HF_HUB_CACHE}"
echo "RUN_GROUNDING              : ${RUN_GROUNDING}"
echo "RUN_RERANK                 : ${RUN_RERANK}"
echo "CONTEXT_LIMIT              : ${CONTEXT_LIMIT:-<none>}"
echo "GROUNDING_LIMIT            : ${GROUNDING_LIMIT:-<none>}"
echo "============================================================"

LIMIT_ARGS=()
if [[ -n "${CONTEXT_LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${CONTEXT_LIMIT}")
fi

python -m py_compile "${REPO_ROOT}/generate_fintagging_fulltagging_extractions.py"
python -m py_compile "${REPO_ROOT}/build_fintagging_fulltagging_grounding_input.py"
python -m py_compile "${REPO_ROOT}/evaluate_fintagging_fulltagging_pipeline.py"
python -m py_compile "${REPO_ROOT}/run_fintagging_grounding_baseline.py"
if [[ "${MODE}" == "dryrun" ]]; then
  exit 0
fi

if [[ "${RUN_EXTRACTION}" == "1" ]]; then
  require_path "${TEXT_EXTRACTOR_MODEL}" "text extractor model"
  require_path "${TABLE_EXTRACTOR_MODEL}" "table extractor model"
  EXTRACTION_RESUME_ARGS=()
  if [[ "${RESUME}" == "1" ]]; then
    EXTRACTION_RESUME_ARGS=(--resume)
  fi
  EXTRACTION_BF16_ARGS=()
  if [[ "${BF16}" == "1" ]]; then
    EXTRACTION_BF16_ARGS=(--bf16)
  fi
  EXTRACTION_REMOTE_CODE_ARGS=()
  if [[ "${EXTRACTION_TRUST_REMOTE_CODE}" == "1" ]]; then
    EXTRACTION_REMOTE_CODE_ARGS=(--trust-remote-code)
  fi
  EXTRACTION_EAGER_ARGS=()
  if [[ "${EXTRACTION_ENFORCE_EAGER}" == "1" ]]; then
    EXTRACTION_EAGER_ARGS=(--enforce-eager)
  fi
  EXTRACTION_COMMON_ARGS=(
    --original-test-parquet "${ORIGINAL_TEST_PARQUET}"
    --text-extractor-model "${TEXT_EXTRACTOR_MODEL}"
    --table-extractor-model "${TABLE_EXTRACTOR_MODEL}"
    --output-jsonl "${EXTRACTION_PREDICTIONS}"
    --text-max-new-tokens "${TEXT_MAX_NEW_TOKENS}"
    --table-max-new-tokens "${TABLE_MAX_NEW_TOKENS}"
    --temperature "${EXTRACTION_TEMPERATURE}"
    --top-p "${EXTRACTION_TOP_P}"
    --backend "${EXTRACTION_BACKEND}"
    --batch-size "${EXTRACTION_VLLM_BATCH_SIZE}"
    --tensor-parallel-size "${EXTRACTION_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${EXTRACTION_GPU_MEMORY_UTILIZATION}"
    --max-model-len "${EXTRACTION_MAX_MODEL_LEN}"
    --max-num-seqs "${EXTRACTION_MAX_NUM_SEQS}"
    "${EXTRACTION_BF16_ARGS[@]}"
    "${EXTRACTION_REMOTE_CODE_ARGS[@]}"
    "${EXTRACTION_EAGER_ARGS[@]}"
    "${EXTRACTION_RESUME_ARGS[@]}"
    "${LIMIT_ARGS[@]}"
  )

  if [[ "${EXTRACTION_BACKEND}" == "vllm" ]]; then
    python "${REPO_ROOT}/generate_fintagging_fulltagging_extractions.py" \
      "${EXTRACTION_COMMON_ARGS[@]}" \
      --metadata-json "${EXTRACTION_METADATA_JSON%.json}_text.json" \
      --task text

    EXTRACTION_APPEND_ARGS=()
    if [[ -f "${EXTRACTION_PREDICTIONS}" ]]; then
      EXTRACTION_APPEND_ARGS=(--append)
    fi
    python "${REPO_ROOT}/generate_fintagging_fulltagging_extractions.py" \
      "${EXTRACTION_COMMON_ARGS[@]}" \
      --metadata-json "${EXTRACTION_METADATA_JSON%.json}_table.json" \
      --task table \
      "${EXTRACTION_APPEND_ARGS[@]}"
  else
    python "${REPO_ROOT}/generate_fintagging_fulltagging_extractions.py" \
      "${EXTRACTION_COMMON_ARGS[@]}" \
      --metadata-json "${EXTRACTION_METADATA_JSON}" \
      --task all
  fi

  python "${REPO_ROOT}/build_fintagging_fulltagging_grounding_input.py" \
    --original-test-parquet "${ORIGINAL_TEST_PARQUET}" \
    --extraction-predictions "${EXTRACTION_PREDICTIONS}" \
    --output-jsonl "${FULLTAGGING_INPUT_JSONL}" \
    --metadata-json "${FULLTAGGING_METADATA_JSON}" \
    "${LIMIT_ARGS[@]}"
else
  require_path "${TEXT_EXTRACTION_PREDICTIONS}" "text extraction predictions"
  require_path "${TABLE_EXTRACTION_PREDICTIONS}" "table extraction predictions"
  python "${REPO_ROOT}/build_fintagging_fulltagging_grounding_input.py" \
    --original-test-parquet "${ORIGINAL_TEST_PARQUET}" \
    --text-predictions "${TEXT_EXTRACTION_PREDICTIONS}" \
    --table-predictions "${TABLE_EXTRACTION_PREDICTIONS}" \
    --output-jsonl "${FULLTAGGING_INPUT_JSONL}" \
    --metadata-json "${FULLTAGGING_METADATA_JSON}" \
    "${LIMIT_ARGS[@]}"
fi

export TEST_JSONL="${FULLTAGGING_INPUT_JSONL}"
export TAXONOMY_JSONL
export OUTPUT_DIR
export QUERY_MODE
export LIMIT="${GROUNDING_LIMIT}"

if [[ "${RUN_GROUNDING}" == "1" ]]; then
  bash "${REPO_ROOT}/run_fintagging_grounding_baseline.sh"

  CANDIDATE_JSONL="${OUTPUT_DIR}/bm25_candidates.jsonl"
  RERANK_JSONL="${OUTPUT_DIR}/qwen_rerank_predictions.jsonl"
  require_path "${CANDIDATE_JSONL}" "BM25 candidate output"
  FULLTAGGING_EVAL_ARGS=(
    --original-test-parquet "${ORIGINAL_TEST_PARQUET}"
    --grounding-input-jsonl "${FULLTAGGING_INPUT_JSONL}"
    --candidate-jsonl "${CANDIDATE_JSONL}"
    --output-json "${FULLTAGGING_EVAL_JSON}"
  )
  if [[ -n "${CONTEXT_LIMIT}" ]]; then
    FULLTAGGING_EVAL_ARGS+=(--limit "${CONTEXT_LIMIT}")
  fi
  if [[ -f "${RERANK_JSONL}" ]]; then
    FULLTAGGING_EVAL_ARGS+=(--rerank-predictions-jsonl "${RERANK_JSONL}")
  fi
  python "${REPO_ROOT}/evaluate_fintagging_fulltagging_pipeline.py" "${FULLTAGGING_EVAL_ARGS[@]}"
fi
