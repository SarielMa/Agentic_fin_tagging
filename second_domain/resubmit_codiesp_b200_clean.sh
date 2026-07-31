#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"

cd "${DOMAIN_ROOT}"
mkdir -p logs "${RUNS_ROOT}"

submit_arm() {
  local query_mode="$1"
  local output_name="$2"
  local time_limit="$3"
  local label_coverage_weight="$4"
  local suffix="$5"
  local extra_exports="$6"
  shift 6

  sbatch --parsable \
    "$@" \
    --partition=gpu_b200 \
    --gpus=b200:1 \
    --nodes=1 \
    --cpus-per-task=1 \
    --mem=256G \
    --time="${time_limit}" \
    --job-name="codiesp_${query_mode}_full_${suffix}" \
    --output="${DOMAIN_ROOT}/logs/%j_codiesp_${query_mode}_full_${suffix}.txt" \
    --export="ALL,TEST_JSONL=${TEST_JSONL},QUERY_MODE=${query_mode},OUTPUT_DIR=${RUNS_ROOT}/${output_name},LABEL_COVERAGE_WEIGHT=${label_coverage_weight},RESUME=1,REUSE_CANDIDATES=1,${extra_exports}" \
    "${DOMAIN_ROOT}/apply_server_codiesp_shared.sh"
}

submit_arm frozen_ags qwen3_32b_frozen_ags_full_wcov0 10:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_arm frozen_ags qwen3_32b_frozen_ags_full_wcov1 10:00:00 1.0 wcov1 "RUN_RERANK=1"

submit_arm parallel_sampling qwen3_32b_parallel_sampling_n2_full_wcov0 08:00:00 0.0 wcov0 "RETRIEVAL_ROUNDS=2,RUN_RERANK=1"
submit_arm parallel_sampling qwen3_32b_parallel_sampling_n2_full_wcov1 08:00:00 1.0 wcov1 "RETRIEVAL_ROUNDS=2,RUN_RERANK=1"

submit_arm fhs_j1 qwen3_32b_fhs_j1_full_wcov0 08:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_arm fhs_j1 qwen3_32b_fhs_j1_full_wcov1 08:00:00 1.0 wcov1 "RUN_RERANK=1"
submit_arm fhs_no_verifier qwen3_32b_fhs_no_verifier_full_wcov0 08:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_arm fhs_no_verifier qwen3_32b_fhs_no_verifier_full_wcov1 08:00:00 1.0 wcov1 "RUN_RERANK=1"

j3_wcov0_id="$(
  submit_arm fhs_j3_wcov0 qwen3_32b_fhs_j3_full_wcov0 20:00:00 0.0 wcov0 \
    "RUN_RERANK=1,QUERY_MAX_NEW_TOKENS=512,FHS_VERIFIER_MAX_NEW_TOKENS=3072"
)"
echo "${j3_wcov0_id}"

j3_wcov1_id="$(
  submit_arm fhs_j3_wcov1 qwen3_32b_fhs_j3_full_wcov1 20:00:00 1.0 wcov1 \
    "RUN_RERANK=1,QUERY_MAX_NEW_TOKENS=512,FHS_VERIFIER_MAX_NEW_TOKENS=3072" \
    --dependency="afterany:${j3_wcov0_id}"
)"
echo "${j3_wcov1_id}"

j4_wcov0_id="$(
  submit_arm fhs_j4_wcov0 qwen3_32b_fhs_j4_full_wcov0 20:00:00 0.0 wcov0 \
    "RUN_RERANK=1,QUERY_MAX_NEW_TOKENS=512,FHS_VERIFIER_MAX_NEW_TOKENS=3072" \
    --dependency="afterany:${j3_wcov1_id}"
)"
echo "${j4_wcov0_id}"

j4_wcov1_id="$(
  submit_arm fhs_j4_wcov1 qwen3_32b_fhs_j4_full_wcov1 20:00:00 1.0 wcov1 \
    "RUN_RERANK=1,QUERY_MAX_NEW_TOKENS=512,FHS_VERIFIER_MAX_NEW_TOKENS=3072" \
    --dependency="afterany:${j4_wcov0_id}"
)"
echo "${j4_wcov1_id}"
