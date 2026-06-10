#!/usr/bin/env sh
set -eu

MODEL="meta-llama/Llama-3.2-3B-Instruct"
SUFFIX="llama3_2_3b"
QWEN_MODEL="Qwen/Qwen3-14B"
QWEN_SUFFIX="qwen3_14b"
TOTAL_JOBS=8
JOB_INDEX=0
MODEL_CACHE_MISSING=0
FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-0}"
TABLE_EVIDENCE_BACKEND="${TABLE_EVIDENCE_BACKEND:-heuristic}"
LIMIT="${LIMIT:-0}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
SUPERVISED_MEMORY_ITERS="${SUPERVISED_MEMORY_ITERS:-2}"
TOP_K="${TOP_K:-200}"
RERANK_K="${RERANK_K:-200}"
SAVE_TOP_K="${SAVE_TOP_K:-200}"
RECALL_K="${RECALL_K:-1 5 10 20 50 100 200}"
BM25_WEIGHT="${BM25_WEIGHT:-1.0}"
DENSE_WEIGHT="${DENSE_WEIGHT:-0.0}"
DENSE_MODEL="${DENSE_MODEL:-}"
export FINAI_LOCAL_FILES_ONLY

BM25_OFFLINE_OUTPUT_DIR="outputs/offline_${SUFFIX}_bm25_200_gan"
BM25_ONLINE_GT_OUTPUT_DIR="outputs/online_gt_${SUFFIX}_bm25_200_gan"
ONLINE_NO_GT_OUTPUT_DIR="outputs/online_without_gt_${SUFFIX}_bm25_200_rule"
SINGLE_OUTPUT_DIR="outputs/single_llm_${SUFFIX}_bm25_200"

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
  check_model_cache "$QWEN_MODEL"

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
  prepare_output_dir "$BM25_OFFLINE_OUTPUT_DIR"
  prepare_output_dir "$BM25_ONLINE_GT_OUTPUT_DIR"
  prepare_output_dir "$ONLINE_NO_GT_OUTPUT_DIR"
  prepare_output_dir "$SINGLE_OUTPUT_DIR"
  prepare_output_dir "outputs/offline_${QWEN_SUFFIX}_bm25_200_gan"
  prepare_output_dir "outputs/online_gt_${QWEN_SUFFIX}_bm25_200_gan"
  prepare_output_dir "outputs/online_without_gt_${QWEN_SUFFIX}_bm25_200_rule"
  prepare_output_dir "outputs/single_llm_${QWEN_SUFFIX}_bm25_200"
}

announce_job() {
  JOB_INDEX=$((JOB_INDEX + 1))
  echo "[$JOB_INDEX/$TOTAL_JOBS] $1 for $MODEL"
}

run_single_llm() {
  announce_job "Running single_llm bm25_200 baseline"
  python scripts/run_single_llm_baseline.py \
    --test data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --model "$MODEL" \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$MODEL" \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --limit "$LIMIT" \
    --output-dir "$SINGLE_OUTPUT_DIR"
}

run_offline() {
  announce_job "Running offline bm25_200_gan with rule+GT critic"
  python scripts/run_two_agent_system.py \
    --mode offline \
    --memory-build data/FinCL-eval-subset-clean-memory.csv \
    --test data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend rule \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --supervised-memory-iters "$SUPERVISED_MEMORY_ITERS" \
    --limit "$LIMIT" \
    --output-dir "$BM25_OFFLINE_OUTPUT_DIR"
}

run_online_with_gt() {
  announce_job "Running online_with_gt bm25_200_gan with rule+GT critic"
  python scripts/run_two_agent_system.py \
    --mode online_with_gt \
    --stream data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend rule \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --supervised-memory-iters "$SUPERVISED_MEMORY_ITERS" \
    --limit "$LIMIT" \
    --output-dir "$BM25_ONLINE_GT_OUTPUT_DIR"
}

run_online_without_gt() {
  announce_job "Running online_without_gt bm25_200 with rule-only critic"
  python scripts/run_two_agent_system.py \
    --mode online_without_gt \
    --stream data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$MODEL" \
    --selector-backend llama \
    --selector-model "$MODEL" \
    --validator-backend rule \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --supervised-memory-iters 0 \
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
# run_online_without_gt
run_single_llm

# ---- Qwen3-14B ----
MODEL="$QWEN_MODEL"
SUFFIX="$QWEN_SUFFIX"
BM25_OFFLINE_OUTPUT_DIR="outputs/offline_${SUFFIX}_bm25_200_gan"
BM25_ONLINE_GT_OUTPUT_DIR="outputs/online_gt_${SUFFIX}_bm25_200_gan"
ONLINE_NO_GT_OUTPUT_DIR="outputs/online_without_gt_${SUFFIX}_bm25_200_rule"
SINGLE_OUTPUT_DIR="outputs/single_llm_${SUFFIX}_bm25_200"

run_offline
run_online_with_gt
# run_online_without_gt
run_single_llm
