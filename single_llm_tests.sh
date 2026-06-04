#!/usr/bin/env sh
set -eu

TOTAL_JOBS=2
JOB_INDEX=0
MODEL_CACHE_MISSING=0
FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-0}"
TABLE_EVIDENCE_BACKEND="${TABLE_EVIDENCE_BACKEND:-llama}"
LIMIT="${LIMIT:-0}"
export FINAI_LOCAL_FILES_ONLY

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

hub_cache_dir() {
  if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    printf '%s\n' "$HUGGINGFACE_HUB_CACHE"
  elif [ -n "${TRANSFORMERS_CACHE:-}" ]; then
    printf '%s\n' "$TRANSFORMERS_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then
    printf '%s\n' "$HF_HOME/hub"
  else
    printf '%s\n' "$HOME/.cache/huggingface/hub"
  fi
}

model_cache_dir() {
  cache_name=$(printf '%s' "$1" | sed 's|/|--|g')
  printf '%s/models--%s\n' "$(hub_cache_dir)" "$cache_name"
}

check_model_cache() {
  model="$1"
  cache_dir=$(model_cache_dir "$model")
  if [ -d "$cache_dir" ]; then
    echo "Found local model cache: ${model}"
  elif is_truthy "$FINAI_LOCAL_FILES_ONLY"; then
    echo "Missing local model cache: ${model} (${cache_dir})"
    MODEL_CACHE_MISSING=1
  elif is_truthy "${HF_HUB_OFFLINE:-0}" || is_truthy "${TRANSFORMERS_OFFLINE:-0}"; then
    echo "Missing local model cache while Hugging Face offline mode is enabled: ${model} (${cache_dir})"
    MODEL_CACHE_MISSING=1
  else
    echo "Local model cache missing; will download if Hugging Face access is available: ${model}"
  fi
}

check_model_caches() {
  check_model_cache "Qwen/Qwen3-14B"
  check_model_cache "Qwen/Qwen3-32B"
  # check_model_cache "meta-llama/Llama-3.2-3B-Instruct"
  # check_model_cache "meta-llama/Llama-3.1-8B-Instruct"

  if [ "$MODEL_CACHE_MISSING" -ne 0 ]; then
    echo "One or more required model caches are missing while offline/local-only mode is enabled."
    echo "Unset FINAI_LOCAL_FILES_ONLY, HF_HUB_OFFLINE, and TRANSFORMERS_OFFLINE to allow downloads."
    exit 1
  fi
}

announce_job() {
  JOB_INDEX=$((JOB_INDEX + 1))
  echo "[$JOB_INDEX/$TOTAL_JOBS] Running single-LLM baseline for $1"
}

run_single_llm_baseline() {
  model="$1"
  suffix="$2"

  announce_job "$model"
  python scripts/run_single_llm_baseline.py \
    --test data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --model "$model" \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$model" \
    --limit "$LIMIT" \
    --output-dir "outputs/single_llm_${suffix}_baseline"
}

if [ "${SKIP_MODEL_CACHE_CHECK:-0}" != "1" ]; then
  check_model_caches
fi

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
  echo "Precheck complete."
  exit 0
fi

# ---- CUDA toolkit for Hugging Face generation ----
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

run_single_llm_baseline "Qwen/Qwen3-14B" "qwen3_14b"
# run_single_llm_baseline "Qwen/Qwen3-32B" "qwen3_32b"
run_single_llm_baseline "meta-llama/Llama-3.2-3B-Instruct" "llama3_2_3b"
# run_single_llm_baseline "meta-llama/Llama-3.1-8B-Instruct" "llama3_1_8b"
