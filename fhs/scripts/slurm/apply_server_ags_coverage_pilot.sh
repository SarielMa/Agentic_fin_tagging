#!/bin/bash
#SBATCH --job-name=ags_cov_pilot
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_coverage_pilot_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# AGS coverage headroom pilot:
#   A  = one-pass greedy reference
#   B  = same one-pass prompt, temperature sampling
#   B' = existing explicit diversity-prompt sampling
#   C  = dimension-directed complete hypotheses

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

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_coverage_pilot/qwen3_32b}"
export TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
export SPLIT="${SPLIT:-train}"
export TABLE_CONTEXTS="${TABLE_CONTEXTS:-30}"
export TEXT_CONTEXTS="${TEXT_CONTEXTS:-40}"
export TARGET_FACTS="${TARGET_FACTS:-600}"
export TARGET_FACTS_WEIGHT="${TARGET_FACTS_WEIGHT:-1.0}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260723}"
export SAMPLE_ATTEMPTS="${SAMPLE_ATTEMPTS:-2000}"
export BUDGET="${BUDGET:-4}"
export TOP_K="${TOP_K:-200}"
export DEPTHS="${DEPTHS:-10,50,200}"
export RRF_KAPPA="${RRF_KAPPA:-60.0}"
export TYPE_FILTER="${TYPE_FILTER:-1}"
export BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260724}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
export QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
export QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-128}"
export ARM_C_MAX_NEW_TOKENS="${ARM_C_MAX_NEW_TOKENS:-512}"
export QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
export ARM_A_TEMPERATURE="${ARM_A_TEMPERATURE:-0.0}"
export ARM_B_TEMPERATURE_SCHEDULE="${ARM_B_TEMPERATURE_SCHEDULE:-0.8,1.0,1.2}"
export ARM_B_JACCARD_THRESHOLD="${ARM_B_JACCARD_THRESHOLD:-0.8}"
export ARM_BPRIME_TEMPERATURE="${ARM_BPRIME_TEMPERATURE:-0.0}"
export ARM_C_TEMPERATURE="${ARM_C_TEMPERATURE:-0.0}"
export PROBE_MIN_R200="${PROBE_MIN_R200:-0.95}"
export ARM_A_REFERENCE_R200="${ARM_A_REFERENCE_R200:-0.69}"
export ARM_A_REFERENCE_TOLERANCE="${ARM_A_REFERENCE_TOLERANCE:-0.10}"
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
export OVERWRITE="${OVERWRITE:-0}"
export DRY_RUN_NO_LLM="${DRY_RUN_NO_LLM:-0}"
export LOG_EVERY="${LOG_EVERY:-25}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
  --split "${SPLIT}"
  --table-contexts "${TABLE_CONTEXTS}"
  --text-contexts "${TEXT_CONTEXTS}"
  --target-facts "${TARGET_FACTS}"
  --target-facts-weight "${TARGET_FACTS_WEIGHT}"
  --sample-seed "${SAMPLE_SEED}"
  --sample-attempts "${SAMPLE_ATTEMPTS}"
  --budget "${BUDGET}"
  --top-k "${TOP_K}"
  --depths "${DEPTHS}"
  --rrf-kappa "${RRF_KAPPA}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --query-generation-model "${QUERY_GENERATION_MODEL}"
  --query-generation-backend "${QUERY_GENERATION_BACKEND}"
  --query-context-max-chars "${QUERY_CONTEXT_MAX_CHARS}"
  --query-max-input-tokens "${QUERY_MAX_INPUT_TOKENS}"
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS}"
  --arm-c-max-new-tokens "${ARM_C_MAX_NEW_TOKENS}"
  --query-top-p "${QUERY_TOP_P}"
  --arm-a-temperature "${ARM_A_TEMPERATURE}"
  --arm-b-temperature-schedule "${ARM_B_TEMPERATURE_SCHEDULE}"
  --arm-b-jaccard-threshold "${ARM_B_JACCARD_THRESHOLD}"
  --arm-bprime-temperature "${ARM_BPRIME_TEMPERATURE}"
  --arm-c-temperature "${ARM_C_TEMPERATURE}"
  --probe-min-r200 "${PROBE_MIN_R200}"
  --arm-a-reference-r200 "${ARM_A_REFERENCE_R200}"
  --arm-a-reference-tolerance "${ARM_A_REFERENCE_TOLERANCE}"
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

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "${DRY_RUN_NO_LLM}" == "1" ]]; then
  ARGS+=(--dry-run-no-llm)
fi

echo "============================================================"
echo "TASK                         : ags_coverage_pilot"
echo "OUTPUT_DIR                   : ${OUTPUT_DIR}"
echo "TAXONOMY_JSONL               : ${TAXONOMY_JSONL}"
echo "SPLIT                        : ${SPLIT}"
echo "TABLE_CONTEXTS               : ${TABLE_CONTEXTS}"
echo "TEXT_CONTEXTS                : ${TEXT_CONTEXTS}"
echo "TARGET_FACTS                 : ${TARGET_FACTS}"
echo "TARGET_FACTS_WEIGHT          : ${TARGET_FACTS_WEIGHT}"
echo "SAMPLE_SEED                  : ${SAMPLE_SEED}"
echo "SAMPLE_ATTEMPTS              : ${SAMPLE_ATTEMPTS}"
echo "BUDGET                       : ${BUDGET}"
echo "TOP_K                        : ${TOP_K}"
echo "DEPTHS                       : ${DEPTHS}"
echo "ARM_B_TEMPERATURE_SCHEDULE   : ${ARM_B_TEMPERATURE_SCHEDULE}"
echo "ARM_B_JACCARD_THRESHOLD      : ${ARM_B_JACCARD_THRESHOLD}"
echo "PROBE_MIN_R200               : ${PROBE_MIN_R200}"
echo "ARM_A_REFERENCE_R200         : ${ARM_A_REFERENCE_R200}"
echo "ARM_A_REFERENCE_TOLERANCE    : ${ARM_A_REFERENCE_TOLERANCE}"
echo "QUERY_GENERATION_MODEL       : ${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND     : ${QUERY_GENERATION_BACKEND}"
echo "TENSOR_PARALLEL_SIZE         : ${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION       : ${GPU_MEMORY_UTILIZATION}"
echo "MAX_NUM_SEQS                 : ${MAX_NUM_SEQS}"
echo "VLLM_BATCH_SIZE              : ${VLLM_BATCH_SIZE}"
echo "BOOTSTRAP_SAMPLES            : ${BOOTSTRAP_SAMPLES}"
echo "RESUME                       : ${RESUME}"
echo "OVERWRITE                    : ${OVERWRITE}"
echo "DRY_RUN_NO_LLM               : ${DRY_RUN_NO_LLM}"
echo "HF_HOME                      : ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES         : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -m py_compile "${REPO_ROOT}/analysis/run_ags_coverage_pilot.py"
python "${REPO_ROOT}/analysis/run_ags_coverage_pilot.py" "${ARGS[@]}"
