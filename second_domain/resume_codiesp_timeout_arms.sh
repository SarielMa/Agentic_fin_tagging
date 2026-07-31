#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"

cd "${DOMAIN_ROOT}"
mkdir -p logs "${RUNS_ROOT}"

submit_resume_arm() {
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
    --job-name="codiesp_resume_${query_mode}_${suffix}" \
    --output="${DOMAIN_ROOT}/logs/%j_codiesp_resume_${query_mode}_${suffix}.txt" \
    --export="ALL,TEST_JSONL=${TEST_JSONL},QUERY_MODE=${query_mode},OUTPUT_DIR=${RUNS_ROOT}/${output_name},LABEL_COVERAGE_WEIGHT=${label_coverage_weight},RESUME=1,REUSE_CANDIDATES=1,${extra_exports}" \
    "${DOMAIN_ROOT}/apply_server_codiesp_shared.sh"
}

# Remaining work estimated from the timed-out trace counts on 2026-07-31:
# frozen_ags wcov0: 2679/3144 in 10h -> ~1.8h remaining; request 5h.
# frozen_ags wcov1: 2548/3144 in 10h -> ~2.4h remaining; request 5h.
# fhs_j1 wcov0: 2413/3144 in 8h -> ~2.5h remaining; request 6h.
submit_resume_arm frozen_ags qwen3_32b_frozen_ags_full_wcov0 05:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_resume_arm frozen_ags qwen3_32b_frozen_ags_full_wcov1 05:00:00 1.0 wcov1 "RUN_RERANK=1"
submit_resume_arm fhs_j1 qwen3_32b_fhs_j1_full_wcov0 06:00:00 0.0 wcov0 "RUN_RERANK=1"

# The original fhs_j1 wcov1 job is still running; resume only after it exits so
# the two processes never append to the same grounding_traces.jsonl concurrently.
submit_resume_arm fhs_j1 qwen3_32b_fhs_j1_full_wcov1 06:00:00 1.0 wcov1 "RUN_RERANK=1" \
  --dependency=afterany:20693290
