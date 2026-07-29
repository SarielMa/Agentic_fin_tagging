#!/usr/bin/env bash
set -euo pipefail

# End-to-end SFT pipeline for the FinTagging value/type extraction subtask.
# This mirrors the PV Miner flow, but replaces FinBen with local generation
# and evaluate_fintagging_value_type.py.

REPO_ROOT="$(readlink -f "$(dirname "$0")")"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/FinTagging_800_200_value_type_sft_arrow}"
# Qwen/Qwen2.5-14B-Instruct already completed (job 17415324); only the Llama leg remains.
DEFAULT_MODELS="meta-llama/Llama-3.3-70B-Instruct"
if [[ -n "${MODEL:-}" ]]; then
  MODELS="${MODEL}"
else
  MODELS="${MODELS:-${DEFAULT_MODELS}}"
fi

TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BF16="${BF16:-1}"
USE_QLORA="${USE_QLORA:-1}"

TRAIN="${TRAIN:-1}"
MERGE="${MERGE:-1}"
PREDICT="${PREDICT:-1}"
EVAL="${EVAL:-1}"

PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/runs/runs_fintagging_value_type}"

model_slug () {
  local model_basename="$1"
  model_basename="${model_basename,,}"
  model_basename="${model_basename/llama-/llama}"
  model_basename="${model_basename//-/_}"
  printf '%s' "${model_basename}"
}

require_path () {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

# Ensure gated HF repos (e.g. meta-llama/*) are accessible: the pipeline overrides
# HF_HOME, so the token at ~/.cache/huggingface/token is not auto-discovered.
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

require_path "${DATASET_PATH}" "SFT dataset"
require_path "${REPO_ROOT}/attic/sft_value_type_peft_ddp.py" "SFT script"
require_path "${REPO_ROOT}/attic/merge_lora_value_type.py" "merge script"
require_path "${REPO_ROOT}/attic/generate_fintagging_value_type_predictions.py" "generation script"
require_path "${REPO_ROOT}/attic/evaluate_fintagging_value_type.py" "evaluation script"

mkdir -p "${PIPELINE_ROOT}"
PIPELINE_ROOT="$(readlink -f "${PIPELINE_ROOT}")"

for MODEL in ${MODELS}; do
MODEL_TAG="$(basename "${MODEL}")"
MODEL_SLUG="$(model_slug "${MODEL_TAG}")"
RUN_TAG="sft_${EPOCHS}ep"

MODEL_DIR="${PIPELINE_ROOT}/${MODEL_SLUG}/${RUN_TAG}"
ADAPTER_OUT="${MODEL_DIR}/sft_adapter"
ADAPTER_PATH="${ADAPTER_OUT}/lora_adapter"
MERGED_DIR="${MODEL_DIR}/merged"
PRED_DIR="${MODEL_DIR}/predictions"
EVAL_DIR="${MODEL_DIR}/eval"
LOG_DIR="${MODEL_DIR}/logs"

mkdir -p "${ADAPTER_OUT}" "${MERGED_DIR}" "${PRED_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "DATASET_PATH : ${DATASET_PATH}"
echo "MODELS       : ${MODELS}"
echo "MODEL        : ${MODEL}"
echo "MODEL_DIR    : ${MODEL_DIR}"
echo "TP           : ${TP}"
echo "MAX_LEN      : ${MAX_LEN}"
echo "EPOCHS       : ${EPOCHS}"
echo "LR           : ${LR}"
echo "USE_QLORA    : ${USE_QLORA}"
echo "BF16         : ${BF16}"
echo "============================================================"

BF16_FLAG=()
if [[ "${BF16}" == "1" ]]; then
  BF16_FLAG=(--bf16)
fi

QLORA_FLAG=()
if [[ "${USE_QLORA}" == "1" ]]; then
  QLORA_FLAG=(--use_qlora)
fi

if [[ "${TRAIN}" == "1" ]]; then
  torchrun --nproc_per_node="${TP}" "${REPO_ROOT}/attic/sft_value_type_peft_ddp.py" \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${ADAPTER_OUT}" \
    --max_length "${MAX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    "${BF16_FLAG[@]}" \
    "${QLORA_FLAG[@]}" \
    2>&1 | tee "${LOG_DIR}/sft.log"
fi

if [[ "${MERGE}" == "1" ]]; then
  require_path "${ADAPTER_PATH}" "LoRA adapter"
  python "${REPO_ROOT}/attic/merge_lora_value_type.py" \
    --base "${MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --out "${MERGED_DIR}" \
    --dtype bf16 \
    2>&1 | tee "${LOG_DIR}/merge.log"
fi

if [[ "${PREDICT}" == "1" ]]; then
  require_path "${MERGED_DIR}" "merged model"
  python "${REPO_ROOT}/attic/generate_fintagging_value_type_predictions.py" \
    --model "${MERGED_DIR}" \
    --dataset-path "${DATASET_PATH}" \
    --split test \
    --output-jsonl "${PRED_DIR}/test_predictions.jsonl" \
    --bf16 \
    2>&1 | tee "${LOG_DIR}/predict.log"
fi

if [[ "${EVAL}" == "1" ]]; then
  require_path "${PRED_DIR}/test_predictions.jsonl" "prediction file"
  python "${REPO_ROOT}/attic/evaluate_fintagging_value_type.py" \
    --gold-dataset "${DATASET_PATH}" \
    --split test \
    --predictions "${PRED_DIR}/test_predictions.jsonl" \
    --output-json "${EVAL_DIR}/test_metrics.json" \
    --per-row-csv "${EVAL_DIR}/test_per_row.csv" \
    2>&1 | tee "${LOG_DIR}/eval.log"
fi

echo
echo "Done. Outputs under: ${MODEL_DIR}"
done
