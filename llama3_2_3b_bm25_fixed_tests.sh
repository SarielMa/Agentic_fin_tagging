#!/usr/bin/env sh
set -eu

# Small fixed-BM25 agentic test for Llama-3.2-3B and Qwen3-14B.
# BM25 is fixed by using:
#   entity_type filter + BM25 over taxonomy text only
#   no dense retrieval
#   heuristic table evidence

MODEL_RUNS="${MODEL_RUNS:-meta-llama/Llama-3.2-3B-Instruct:llama3_2_3b Qwen/Qwen3-14B:qwen3_14b}"
LIMIT="${LIMIT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fixed_bm25_small}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
SKIP_CUDA_SETUP="${SKIP_CUDA_SETUP:-0}"
SKIP_MODEL_CACHE_CHECK="${SKIP_MODEL_CACHE_CHECK:-0}"
FINAI_LOCAL_FILES_ONLY="${FINAI_LOCAL_FILES_ONLY:-0}"

RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_ONLINE_GT="${RUN_ONLINE_GT:-1}"
RUN_SINGLE="${RUN_SINGLE:-0}"

TOP_K="${TOP_K:-200}"
RERANK_K="${RERANK_K:-200}"
SAVE_TOP_K="${SAVE_TOP_K:-200}"
RECALL_K="${RECALL_K:-1 5 10 20 50 100 200}"
SUPERVISED_MEMORY_ITERS="${SUPERVISED_MEMORY_ITERS:-0}"

TABLE_EVIDENCE_BACKEND="heuristic"
TAXONOMY_DOC_MODE="text"
BM25_WEIGHT="1.0"
DENSE_WEIGHT="0.0"
DENSE_MODEL=""
SELECTOR_BACKEND="llama"
VALIDATOR_BACKEND="llama"

export FINAI_LOCAL_FILES_ONLY
export PYTHONHASHSEED=0

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
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
    return 0
  fi

  if is_truthy "$FINAI_LOCAL_FILES_ONLY" || is_truthy "${HF_HUB_OFFLINE:-0}" || is_truthy "${TRANSFORMERS_OFFLINE:-0}"; then
    echo "Missing local model cache: $model ($cache_dir)"
    exit 1
  fi

  echo "Local model cache missing; will download if access is available: $model"
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

  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER:-user}}"
  mkdir -p "$TRITON_CACHE_DIR"
}

verify_fixed_bm25_metrics() {
  metrics_path="$1"
  python - "$metrics_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    metrics = json.load(f)

expected = {
    "retrieval": "bm25",
    "bm25_weight": 1.0,
    "dense_weight": 0.0,
    "dense_model": None,
    "taxonomy_doc_mode": "text",
    "table_evidence_backend": "heuristic",
}

errors = []
for key, value in expected.items():
    if metrics.get(key) != value:
        errors.append(f"{key}={metrics.get(key)!r}")
if "recall_at_k" not in metrics:
    errors.append("missing recall_at_k")

if errors:
    raise SystemExit(f"{path} is not fixed BM25: " + ", ".join(errors))

print(f"Verified fixed BM25: {path}")
PY
}

run_two_agent() {
  mode="$1"
  model="$2"
  suffix="$3"
  output_dir="$4"

  if [ "$mode" = "offline" ]; then
    python scripts/run_two_agent_system.py \
      --mode offline \
      --memory-build data/FinCL-eval-subset-clean-memory.csv \
      --test data/FinCL-eval-subset-clean-test.csv \
      --taxonomy data/us_gaap_2024_BM25.jsonl \
      --output-dir "$output_dir" \
      --selector-backend "$SELECTOR_BACKEND" \
      --selector-model "$model" \
      --validator-backend "$VALIDATOR_BACKEND" \
      --validator-model "$model" \
      --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
      --table-evidence-model "$model" \
      --taxonomy-doc-mode "$TAXONOMY_DOC_MODE" \
      --bm25-weight "$BM25_WEIGHT" \
      --dense-weight "$DENSE_WEIGHT" \
      --dense-model "$DENSE_MODEL" \
      --top-k "$TOP_K" \
      --rerank-k "$RERANK_K" \
      --save-top-k "$SAVE_TOP_K" \
      --recall-k $RECALL_K \
      --supervised-memory-iters "$SUPERVISED_MEMORY_ITERS" \
      --limit "$LIMIT"
    verify_fixed_bm25_metrics "$output_dir/memory_build/metrics.json"
    verify_fixed_bm25_metrics "$output_dir/test/metrics.json"
    return 0
  fi

  python scripts/run_two_agent_system.py \
    --mode online_with_gt \
    --stream data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --output-dir "$output_dir" \
    --selector-backend "$SELECTOR_BACKEND" \
    --selector-model "$model" \
    --validator-backend "$VALIDATOR_BACKEND" \
    --validator-model "$model" \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$model" \
    --taxonomy-doc-mode "$TAXONOMY_DOC_MODE" \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --supervised-memory-iters "$SUPERVISED_MEMORY_ITERS" \
    --limit "$LIMIT"
  verify_fixed_bm25_metrics "$output_dir/online_with_gt/metrics.json"
}

run_single_llm() {
  model="$1"
  output_dir="$2"

  python scripts/run_single_llm_baseline.py \
    --test data/FinCL-eval-subset-clean-test.csv \
    --taxonomy data/us_gaap_2024_BM25.jsonl \
    --output-dir "$output_dir" \
    --model "$model" \
    --table-evidence-backend "$TABLE_EVIDENCE_BACKEND" \
    --table-evidence-model "$model" \
    --taxonomy-doc-mode "$TAXONOMY_DOC_MODE" \
    --bm25-weight "$BM25_WEIGHT" \
    --dense-weight "$DENSE_WEIGHT" \
    --dense-model "$DENSE_MODEL" \
    --top-k "$TOP_K" \
    --rerank-k "$RERANK_K" \
    --save-top-k "$SAVE_TOP_K" \
    --recall-k $RECALL_K \
    --limit "$LIMIT"
  verify_fixed_bm25_metrics "$output_dir/single_llm/metrics.json"
}

echo "Fixed BM25 settings:"
echo "  taxonomy_doc_mode=$TAXONOMY_DOC_MODE"
echo "  table_evidence_backend=$TABLE_EVIDENCE_BACKEND"
echo "  bm25_weight=$BM25_WEIGHT dense_weight=$DENSE_WEIGHT"
echo "  limit=$LIMIT"

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
  echo "Precheck complete."
  exit 0
fi

setup_cuda

for spec in $MODEL_RUNS; do
  model="${spec%:*}"
  suffix="${spec#*:}"
  echo "=== $suffix: $model ==="

  if [ "$SKIP_MODEL_CACHE_CHECK" != "1" ]; then
    check_model_cache "$model"
  fi

  if is_truthy "$RUN_OFFLINE"; then
    output_dir="${OUTPUT_ROOT}/${suffix}_offline"
    prepare_output_dir "$output_dir"
    run_two_agent offline "$model" "$suffix" "$output_dir"
  fi

  if is_truthy "$RUN_ONLINE_GT"; then
    output_dir="${OUTPUT_ROOT}/${suffix}_online_gt"
    prepare_output_dir "$output_dir"
    run_two_agent online_with_gt "$model" "$suffix" "$output_dir"
  fi

  if is_truthy "$RUN_SINGLE"; then
    output_dir="${OUTPUT_ROOT}/${suffix}_single_llm"
    prepare_output_dir "$output_dir"
    run_single_llm "$model" "$output_dir"
  fi
done
