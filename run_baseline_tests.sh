#!/usr/bin/env sh
set -eu

# Minimal baseline tests.
# Modes: single_llm, offline, online_gt, online_wo_gt
# Models: Llama-3.2-3B, Llama-3.3-70B, and Qwen3-14B by default

# One model per line, format: <hf_model_id>:<output_suffix>
# Add or remove models by editing this list (comment out with # to disable).
# # meta-llama/Llama-3.3-70B-Instruct:llama3_3_70b
# Qwen/Qwen3-14B:qwen3_14b
MODEL_RUNS="${MODEL_RUNS:-$(cat <<'EOF'
meta-llama/Llama-3.2-3B-Instruct:llama3_2_3b
EOF
)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baseline_three_agent}"

# Readable timestamp shared by every output dir in this invocation so reruns
# never collide (and the pipeline never halts on an existing dir). Override by
# exporting RUN_TIMESTAMP to reuse a specific run folder.
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"

TAXONOMY_JSONL="${TAXONOMY_JSONL:-data/us_gaap_2024_BM25.jsonl}"
MEMORY_BUILD_CSV="${MEMORY_BUILD_CSV:-data/FinCL-eval-subset-clean-memory.csv}"
TEST_CSV="${TEST_CSV:-data/FinCL-eval-subset-clean-test.csv}"
STREAM_CSV="${STREAM_CSV:-$TEST_CSV}"

TOP_K="${TOP_K:-200}"
MEMORY_K="${MEMORY_K:-5}"
RECALL_K="${RECALL_K:-1 5 10 20 50 100 200}"
LIMIT="${LIMIT:-0}"
# Memory-build critic retry: 'hint' (non-leaking corrective hint) or 'oracle' (reveal GT tag).
COACH_MODE="${COACH_MODE:-hint}"
# Max coached retries before fallback, and whether a final miss still banks the GT exemplar.
COACH_RETRIES="${COACH_RETRIES:-2}"
ORACLE_FALLBACK="${ORACLE_FALLBACK:-1}"

RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_ONLINE_GT="${RUN_ONLINE_GT:-1}"
RUN_ONLINE_WO_GT="${RUN_ONLINE_WO_GT:-1}"
RUN_SINGLE_LLM="${RUN_SINGLE_LLM:-1}"

CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
SKIP_CUDA_SETUP="${SKIP_CUDA_SETUP:-0}"
SKIP_MODEL_CACHE_CHECK="${SKIP_MODEL_CACHE_CHECK:-0}"
FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-0}"

export FINAI_LOCAL_FILES_ONLY

# Drop blank lines and #-comments so models can be toggled with a leading #.
MODEL_RUNS=$(printf '%s\n' "$MODEL_RUNS" | sed 's/#.*//' | grep -v '^[[:space:]]*$' || true)

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

check_model_cache() {
  model="$1"
  cache_name=$(printf '%s' "$model" | sed 's|/|--|g')
  cache_dir="$(hub_cache_dir)/models--${cache_name}"

  if [ -d "$cache_dir" ]; then
    echo "Found local model cache: $model"
  elif is_truthy "$FINAI_LOCAL_FILES_ONLY" || is_truthy "${HF_HUB_OFFLINE:-0}" || is_truthy "${TRANSFORMERS_OFFLINE:-0}"; then
    echo "Missing local model cache: $model ($cache_dir)"
    exit 1
  else
    echo "Local model cache missing; will download if access is available: $model"
  fi
}

setup_cuda() {
  if is_truthy "$SKIP_CUDA_SETUP"; then
    return 0
  fi

  if command -v module >/dev/null 2>&1; then
    if [ -z "${CONDA_PREFIX:-}" ]; then
      module purge || true
    fi
    module load StdEnv || true
    module load CUDA/12.6.0 || true
  fi

  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi

  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER:-user}}"
  export TRITON_CACHE_DIR
  mkdir -p "$TRITON_CACHE_DIR"
}

prepare_output_dir() {
  output_dir="$1"
  if [ ! -d "$output_dir" ]; then
    return 0
  fi

  if is_truthy "$CLEAN_OUTPUTS"; then
    echo "Removing existing output directory: $output_dir"
    rm -rf "$output_dir"
    return 0
  fi

  echo "Output directory already exists: $output_dir"
  echo "Use OUTPUT_ROOT to choose a new path, or set CLEAN_OUTPUTS=1."
  exit 1
}

run_mode() {
  mode="$1"
  model="$2"
  suffix="$3"
  output_dir="$OUTPUT_ROOT/${suffix}_${mode}_${RUN_TIMESTAMP}"

  prepare_output_dir "$output_dir"
  echo "=== $suffix | $mode | limit=$LIMIT ==="

  python scripts/run_three_agent_system.py \
    --mode "$mode" \
    --taxonomy "$TAXONOMY_JSONL" \
    --memory-build "$MEMORY_BUILD_CSV" \
    --test "$TEST_CSV" \
    --stream "$STREAM_CSV" \
    --output-dir "$output_dir" \
    --selector-model "$model" \
    --validator-model "$model" \
    --top-k "$TOP_K" \
    --memory-k "$MEMORY_K" \
    --recall-k $RECALL_K \
    --coach-mode "$COACH_MODE" \
    --coach-retries "$COACH_RETRIES" \
    $(is_truthy "$ORACLE_FALLBACK" && echo --oracle-fallback || echo --no-oracle-fallback) \
    --limit "$LIMIT"
}

run_single_llm() {
  model="$1"
  suffix="$2"
  output_dir="$OUTPUT_ROOT/${suffix}_single_llm_${RUN_TIMESTAMP}"

  prepare_output_dir "$output_dir"
  echo "=== $suffix | single_llm | limit=$LIMIT ==="

  python scripts/run_single_llm_test.py \
    --taxonomy "$TAXONOMY_JSONL" \
    --test "$TEST_CSV" \
    --output-dir "$output_dir" \
    --model "$model" \
    --top-k "$TOP_K" \
    --recall-k $RECALL_K \
    --limit "$LIMIT"
}

echo "Baseline test settings:"
echo "  retrieval: type filter + BM25(context, taxonomy.text), top_k=$TOP_K"
echo "  memory: correct_book/error_book BM25 top $MEMORY_K"
echo "  single_llm: no memory, no validator, no retry"
echo "  limit=$LIMIT"

if [ "$SKIP_MODEL_CACHE_CHECK" != "1" ]; then
  for run in $MODEL_RUNS; do
    check_model_cache "${run%%:*}"
  done
fi

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
  echo "Precheck complete."
  exit 0
fi

setup_cuda

for run in $MODEL_RUNS; do
  model="${run%%:*}"
  suffix="${run#*:}"
  if is_truthy "$RUN_SINGLE_LLM"; then
    run_single_llm "$model" "$suffix"
  fi
  if is_truthy "$RUN_OFFLINE"; then
    run_mode offline "$model" "$suffix"
  fi
  if is_truthy "$RUN_ONLINE_GT"; then
    run_mode online_gt "$model" "$suffix"
  fi
  if is_truthy "$RUN_ONLINE_WO_GT"; then
    run_mode online_wo_gt "$model" "$suffix"
  fi
done
