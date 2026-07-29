#!/bin/bash
#SBATCH --job-name=ags_component
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_component_validation_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
PROJECT_ROOT="$(readlink -f "${REPO_ROOT}/..")"

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

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_component_validation/qwen3_32b}"
export SAMPLE_PATH="${SAMPLE_PATH:-${REPO_ROOT}/data/dev/sample_facts.jsonl}"
export TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
export NORMALIZATION_MAP="${NORMALIZATION_MAP:-${REPO_ROOT}/src/ags_symbolic_normalization_map.yaml}"
export TOP_K="${TOP_K:-200}"
export DEPTHS="${DEPTHS:-10,50,200}"
export HYPOTHESES_PER_FACT="${HYPOTHESES_PER_FACT:-3}"
export HYPOTHESIS_TEMPERATURE="${HYPOTHESIS_TEMPERATURE:-0.8}"
export QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
export RRF_KAPPA="${RRF_KAPPA:-60.0}"
export DUAL_RRF_KAPPA="${DUAL_RRF_KAPPA:-60.0}"
export BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260724}"
export GATE_MIN_EFFECT="${GATE_MIN_EFFECT:-0.03}"
export AGREEMENT_TOP_M="${AGREEMENT_TOP_M:-10}"
export AGREEMENT_DEBUG_FACTS="${AGREEMENT_DEBUG_FACTS:-50}"
export RANDOM_INIT_DRAWS="${RANDOM_INIT_DRAWS:-100}"
export INIT_SELECTION_K="${INIT_SELECTION_K:-50}"
export CONSENSUS_BETAS="${CONSENSUS_BETAS:-0.05,0.1,0.2,0.4}"
export LABEL_COVERAGE_WEIGHT="${LABEL_COVERAGE_WEIGHT:-1.0}"
export LABEL_COVERAGE_POOL_MULTIPLIER="${LABEL_COVERAGE_POOL_MULTIPLIER:-0}"
export TYPE_FILTER="${TYPE_FILTER:-1}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
export QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
export QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-512}"
export BF16="${BF16:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RESUME="${RESUME:-1}"
export REFRESH_RETRIEVALS="${REFRESH_RETRIEVALS:-0}"
export REFRESH_AGREEMENT_DEBUG="${REFRESH_AGREEMENT_DEBUG:-0}"
export OVERWRITE="${OVERWRITE:-0}"
export DRY_RUN_NO_LLM="${DRY_RUN_NO_LLM:-0}"
export LOG_EVERY="${LOG_EVERY:-25}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --sample-path "${SAMPLE_PATH}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
  --normalization-map "${NORMALIZATION_MAP}"
  --top-k "${TOP_K}"
  --depths "${DEPTHS}"
  --hypotheses-per-fact "${HYPOTHESES_PER_FACT}"
  --hypothesis-temperature "${HYPOTHESIS_TEMPERATURE}"
  --query-top-p "${QUERY_TOP_P}"
  --rrf-kappa "${RRF_KAPPA}"
  --dual-rrf-kappa "${DUAL_RRF_KAPPA}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --gate-min-effect "${GATE_MIN_EFFECT}"
  --agreement-top-m "${AGREEMENT_TOP_M}"
  --agreement-debug-facts "${AGREEMENT_DEBUG_FACTS}"
  --random-init-draws "${RANDOM_INIT_DRAWS}"
  --init-selection-k "${INIT_SELECTION_K}"
  --consensus-betas "${CONSENSUS_BETAS}"
  --label-coverage-weight "${LABEL_COVERAGE_WEIGHT}"
  --label-coverage-pool-multiplier "${LABEL_COVERAGE_POOL_MULTIPLIER}"
  --query-generation-model "${QUERY_GENERATION_MODEL}"
  --query-generation-backend "${QUERY_GENERATION_BACKEND}"
  --query-context-max-chars "${QUERY_CONTEXT_MAX_CHARS}"
  --query-max-input-tokens "${QUERY_MAX_INPUT_TOKENS}"
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --vllm-batch-size "${VLLM_BATCH_SIZE}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --log-every "${LOG_EVERY}"
)

if [[ "${TYPE_FILTER}" == "1" ]]; then
  ARGS+=(--type-filter)
else
  ARGS+=(--no-type-filter)
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
else
  ARGS+=(--no-resume)
fi

if [[ "${REFRESH_RETRIEVALS}" == "1" ]]; then
  ARGS+=(--refresh-retrievals)
fi

if [[ "${REFRESH_AGREEMENT_DEBUG}" == "1" ]]; then
  ARGS+=(--refresh-agreement-debug)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "${DRY_RUN_NO_LLM}" == "1" ]]; then
  ARGS+=(--dry-run-no-llm)
fi

echo "============================================================"
echo "TASK                       : ags_component_validation"
echo "OUTPUT_DIR                 : ${OUTPUT_DIR}"
echo "SAMPLE_PATH                : ${SAMPLE_PATH}"
echo "TAXONOMY_JSONL             : ${TAXONOMY_JSONL}"
echo "NORMALIZATION_MAP          : ${NORMALIZATION_MAP}"
echo "TOP_K                      : ${TOP_K}"
echo "DEPTHS                     : ${DEPTHS}"
echo "HYPOTHESES_PER_FACT        : ${HYPOTHESES_PER_FACT}"
echo "HYPOTHESIS_TEMPERATURE     : ${HYPOTHESIS_TEMPERATURE}"
echo "GATE_MIN_EFFECT            : ${GATE_MIN_EFFECT}"
echo "QUERY_GENERATION_MODEL     : ${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND   : ${QUERY_GENERATION_BACKEND}"
echo "BOOTSTRAP_SAMPLES          : ${BOOTSTRAP_SAMPLES}"
echo "LABEL_COVERAGE_WEIGHT      : ${LABEL_COVERAGE_WEIGHT}"
echo "LABEL_COVERAGE_POOL_MULT   : ${LABEL_COVERAGE_POOL_MULTIPLIER}"
echo "RESUME                     : ${RESUME}"
echo "REFRESH_RETRIEVALS         : ${REFRESH_RETRIEVALS}"
echo "REFRESH_AGREEMENT_DEBUG    : ${REFRESH_AGREEMENT_DEBUG}"
echo "OVERWRITE                  : ${OVERWRITE}"
echo "DRY_RUN_NO_LLM             : ${DRY_RUN_NO_LLM}"
echo "HF_HOME                    : ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES       : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -m py_compile \
  "${REPO_ROOT}/src/ags_symbolic_agreement.py" \
  "${REPO_ROOT}/analysis/run_ags_component_validation.py"
python -m unittest \
  "${REPO_ROOT}/tests/test_ags_symbolic_agreement.py" \
  "${REPO_ROOT}/tests/test_label_coverage_retriever.py"
python "${REPO_ROOT}/analysis/run_ags_component_validation.py" "${ARGS[@]}"
