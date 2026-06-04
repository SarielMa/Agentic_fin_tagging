#!/usr/bin/env sh
set -eu

MODEL="meta-llama/Llama-3.2-3B-Instruct"
SUFFIX="llama3_2_3b"
TOTAL_JOBS=3
JOB_INDEX=0
MODEL_CACHE_MISSING=0
FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-0}"
TABLE_EVIDENCE_BACKEND="${TABLE_EVIDENCE_BACKEND:-llama}"
LIMIT="${LIMIT:-0}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
export FINAI_LOCAL_FILES_ONLY

OFFLINE_OUTPUT_DIR="outputs/offline_${SUFFIX}_TTT"
ONLINE_GT_OUTPUT_DIR="outputs/online_gt_${SUFFIX}_TTT"
ONLINE_NO_GT_OUTPUT_DIR="outputs/online_without_gt_${SUFFIX}_TTT"

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
  check_model_cache "$MODEL"

  if [ "$MODEL_CACHE_MISSING" -ne 0 ]; then
    echo "Required model cache is missing while offline/local-only mode is enabled."
    echo "Unset FINAI_LOCAL_FILES_ONLY, HF_HUB_OFFLINE, and TRANSFORMERS_OFFLINE to allow downloads."
    exit 1
  fi
}

prepare_output_dir() {
  output_dir="$1"
  if [ ! -d "$output_dir" ]; then
    return 0
  fi

  if is_truthy "$CLEAN_OUTPUTS"; then
    echo "Removing existing output directory: ${output_dir}"
    rm -rf "$output_dir"
    return 0
  fi

  echo "Output directory already exists: ${output_dir}"
  echo "Set CLEAN_OUTPUTS=1 to remove it before running, or move it aside manually."
  exit 1
}

prepare_output_dirs() {
  prepare_output_dir "$OFFLINE_OUTPUT_DIR"
  prepare_output_dir "$ONLINE_GT_OUTPUT_DIR"
  prepare_output_dir "$ONLINE_NO_GT_OUTPUT_DIR"
}

announce_job() {
  JOB_INDEX=$((JOB_INDEX + 1))
  echo "[$JOB_INDEX/$TOTAL_JOBS] $1 for $MODEL"
}

run_offline() {
  announce_job "Running offline TTT"
  python scripts/run_two_agent_system.py \
    --mode offline \
    --memory-build data/FinCL-eval-subset-clean-memory.csv \
    --test data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend llama \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend llama \
    --validator-model "$MODEL" \
    --limit "$LIMIT" \
    --output-dir "$OFFLINE_OUTPUT_DIR"
}

run_online_with_gt() {
  announce_job "Running online_with_gt TTT"
  python scripts/run_two_agent_system.py \
    --mode online_with_gt \
    --stream data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend llama \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend llama \
    --validator-model "$MODEL" \
    --limit "$LIMIT" \
    --output-dir "$ONLINE_GT_OUTPUT_DIR"
}

run_online_without_gt() {
  announce_job "Running online_without_gt TTT"
  python scripts/run_two_agent_system.py \
    --mode online_without_gt \
    --stream data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend llama \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend llama \
    --validator-model "$MODEL" \
    --limit "$LIMIT" \
    --output-dir "$ONLINE_NO_GT_OUTPUT_DIR"
}

if [ "${SKIP_MODEL_CACHE_CHECK:-0}" != "1" ]; then
  check_model_caches
fi

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
  echo "Precheck complete."
  exit 0
fi

prepare_output_dirs

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

run_offline
run_online_with_gt
run_online_without_gt
