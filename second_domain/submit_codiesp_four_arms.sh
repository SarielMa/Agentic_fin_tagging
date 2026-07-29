#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"

submit_arm () {
  local query_mode="$1"
  local script="$2"
  local label_coverage_weight="$3"
  local suffix="$4"
  local run_dir="${RUNS_ROOT}/qwen3_32b_${query_mode}_${suffix}"

  sbatch \
    --job-name="codiesp_${query_mode}_${suffix}" \
    --export="ALL,QUERY_MODE=${query_mode},LABEL_COVERAGE_WEIGHT=${label_coverage_weight},OUTPUT_DIR=${run_dir}" \
    "${script}"
}

submit_arm direct_retrieval apply_server_codiesp_direct_retrieval.sh 0.0 wcov0
submit_arm direct_retrieval apply_server_codiesp_direct_retrieval.sh 1.0 wcov1
submit_arm one_pass_grounding apply_server_codiesp_one_pass_grounding.sh 0.0 wcov0
submit_arm one_pass_grounding apply_server_codiesp_one_pass_grounding.sh 1.0 wcov1
submit_arm one_pass_structured apply_server_codiesp_one_pass_structured.sh 0.0 wcov0
submit_arm one_pass_structured apply_server_codiesp_one_pass_structured.sh 1.0 wcov1
submit_arm frozen_ags apply_server_codiesp_frozen_ags.sh 0.0 wcov0
submit_arm frozen_ags apply_server_codiesp_frozen_ags.sh 1.0 wcov1
