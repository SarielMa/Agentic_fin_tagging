#!/bin/bash
#SBATCH --job-name=ags_reward_diag
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_ags_reward_diagnostic_qwen3_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
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

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_reward_diagnostic/qwen3_32b}"
export COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR:-${REPO_ROOT}/runs_ags_component_validation/qwen3_32b}"
export SAMPLE_PATH="${SAMPLE_PATH:-${REPO_ROOT}/runs_ags_coverage_pilot/qwen3_32b/sample_facts.jsonl}"
export TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
export NORMALIZATION_MAP="${NORMALIZATION_MAP:-${REPO_ROOT}/ags_symbolic_normalization_map.yaml}"
export TOP_K="${TOP_K:-200}"
export DEPTHS="${DEPTHS:-10,50,200}"
export STREAM_FACTS="${STREAM_FACTS:-250}"
export ROUNDS="${ROUNDS:-4}"
export RRF_KAPPA="${RRF_KAPPA:-60.0}"
export FEEDBACK_CANDIDATE_COUNT="${FEEDBACK_CANDIDATE_COUNT:-10}"
export CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
export AGREEMENT_TOP_M="${AGREEMENT_TOP_M:-10}"
export POSTERIOR_SNAPSHOT_EVERY="${POSTERIOR_SNAPSHOT_EVERY:-25}"
export POSTERIOR_ALPHA="${POSTERIOR_ALPHA:-0.75}"
export POSTERIOR_RIDGE="${POSTERIOR_RIDGE:-1.0}"
export INFORMATIVE_DELTA="${INFORMATIVE_DELTA:-0.01}"
export NOVELTY_THRESHOLD="${NOVELTY_THRESHOLD:-0.02}"
export LIVE_CONSENSUS_BETA="${LIVE_CONSENSUS_BETA:-}"
export DIAGNOSTIC_CONSENSUS_BETAS="${DIAGNOSTIC_CONSENSUS_BETAS:-}"
export LABEL_COVERAGE_WEIGHT="${LABEL_COVERAGE_WEIGHT:-1.0}"
export LABEL_COVERAGE_POOL_MULTIPLIER="${LABEL_COVERAGE_POOL_MULTIPLIER:-0}"
export SEED="${SEED:-20260727}"
export TYPE_FILTER="${TYPE_FILTER:-1}"
export ACKNOWLEDGE_NEGATIVE_RENDERING_GATE="${ACKNOWLEDGE_NEGATIVE_RENDERING_GATE:-0}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
export QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
export QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-512}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.8}"
export QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
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
export LOG_EVERY="${LOG_EVERY:-10}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --component-output-dir "${COMPONENT_OUTPUT_DIR}"
  --sample-path "${SAMPLE_PATH}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
  --normalization-map "${NORMALIZATION_MAP}"
  --top-k "${TOP_K}"
  --depths "${DEPTHS}"
  --stream-facts "${STREAM_FACTS}"
  --rounds "${ROUNDS}"
  --rrf-kappa "${RRF_KAPPA}"
  --feedback-candidate-count "${FEEDBACK_CANDIDATE_COUNT}"
  --candidate-doc-max-chars "${CANDIDATE_DOC_MAX_CHARS}"
  --agreement-top-m "${AGREEMENT_TOP_M}"
  --posterior-snapshot-every "${POSTERIOR_SNAPSHOT_EVERY}"
  --posterior-alpha "${POSTERIOR_ALPHA}"
  --posterior-ridge "${POSTERIOR_RIDGE}"
  --informative-delta "${INFORMATIVE_DELTA}"
  --novelty-threshold "${NOVELTY_THRESHOLD}"
  --label-coverage-weight "${LABEL_COVERAGE_WEIGHT}"
  --label-coverage-pool-multiplier "${LABEL_COVERAGE_POOL_MULTIPLIER}"
  --seed "${SEED}"
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
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${LIVE_CONSENSUS_BETA}" ]]; then
  ARGS+=(--live-consensus-beta "${LIVE_CONSENSUS_BETA}")
fi

if [[ -n "${DIAGNOSTIC_CONSENSUS_BETAS}" ]]; then
  ARGS+=(--diagnostic-consensus-betas "${DIAGNOSTIC_CONSENSUS_BETAS}")
fi

if [[ "${TYPE_FILTER}" == "1" ]]; then
  ARGS+=(--type-filter)
else
  ARGS+=(--no-type-filter)
fi

if [[ "${ACKNOWLEDGE_NEGATIVE_RENDERING_GATE}" == "1" ]]; then
  ARGS+=(--acknowledge-negative-rendering-gate)
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
echo "TASK                           : ags_reward_diagnostic"
echo "OUTPUT_DIR                     : ${OUTPUT_DIR}"
echo "COMPONENT_OUTPUT_DIR           : ${COMPONENT_OUTPUT_DIR}"
echo "SAMPLE_PATH                    : ${SAMPLE_PATH}"
echo "STREAM_FACTS                   : ${STREAM_FACTS}"
echo "ROUNDS                         : ${ROUNDS}"
echo "TOP_K                          : ${TOP_K}"
echo "POSTERIOR_SNAPSHOT_EVERY       : ${POSTERIOR_SNAPSHOT_EVERY}"
echo "ACKNOWLEDGE_NEGATIVE_GATE      : ${ACKNOWLEDGE_NEGATIVE_RENDERING_GATE}"
echo "LIVE_CONSENSUS_BETA            : ${LIVE_CONSENSUS_BETA:-<A3 recommendation>}"
echo "DIAGNOSTIC_CONSENSUS_BETAS     : ${DIAGNOSTIC_CONSENSUS_BETAS:-<A3 candidates>}"
echo "LABEL_COVERAGE_WEIGHT          : ${LABEL_COVERAGE_WEIGHT}"
echo "LABEL_COVERAGE_POOL_MULTIPLIER : ${LABEL_COVERAGE_POOL_MULTIPLIER}"
echo "QUERY_GENERATION_MODEL         : ${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND       : ${QUERY_GENERATION_BACKEND}"
echo "RESUME                         : ${RESUME}"
echo "OVERWRITE                      : ${OVERWRITE}"
echo "DRY_RUN_NO_LLM                 : ${DRY_RUN_NO_LLM}"
echo "HF_HOME                        : ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES           : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -m py_compile \
  "${REPO_ROOT}/ags_symbolic_agreement.py" \
  "${REPO_ROOT}/run_ags_reward_diagnostic.py"
python -m unittest \
  "${REPO_ROOT}/test_ags_symbolic_agreement.py" \
  "${REPO_ROOT}/test_label_coverage_retriever.py"
python "${REPO_ROOT}/run_ags_reward_diagnostic.py" "${ARGS[@]}"
