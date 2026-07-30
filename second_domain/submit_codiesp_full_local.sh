#!/usr/bin/env bash
# Validate full CodiEsp relocation outputs locally, then submit the eight GPU
# grounding jobs.
set -euo pipefail

DOMAIN_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/second_domain"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"
STATS_JSON="${STATS_JSON:-${DOMAIN_ROOT}/data/codiesp/stats_full_exact.json}"
RELOCATIONS_JSONL="${RELOCATIONS_JSONL:-${DOMAIN_ROOT}/data/codiesp/evidence_relocations_full_exact.jsonl}"
SPOTCHECK_TSV="${SPOTCHECK_TSV:-${DOMAIN_ROOT}/data/codiesp/spotcheck_50_full_exact.tsv}"
DOCS_TXT="${DOCS_TXT:-${DOMAIN_ROOT}/data/codiesp/test_docs_full_exact.txt}"
EXPECTED_FACTS="${EXPECTED_FACTS:-3144}"

cd "${DOMAIN_ROOT}"
mkdir -p logs "${RUNS_ROOT}"

python scripts/validate_codiesp_full_outputs.py \
  --facts-jsonl "${TEST_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --relocations-jsonl "${RELOCATIONS_JSONL}" \
  --spotcheck-tsv "${SPOTCHECK_TSV}" \
  --docs-txt "${DOCS_TXT}" \
  --expected-facts "${EXPECTED_FACTS}" \
  --min-exact-rate 1.0 \
  --min-parse-rate 1.0

submit_arm () {
  local query_mode="$1"
  local script="$2"
  local label_coverage_weight="$3"
  local suffix="$4"
  local time_limit="$5"
  local run_dir="${RUNS_ROOT}/qwen3_32b_${query_mode}_full_${suffix}"

  sbatch \
    --time="${time_limit}" \
    --job-name="codiesp_${query_mode}_full_${suffix}" \
    --export="ALL,TEST_JSONL=${TEST_JSONL},QUERY_MODE=${query_mode},LABEL_COVERAGE_WEIGHT=${label_coverage_weight},OUTPUT_DIR=${run_dir}" \
    "${script}"
}

submit_arm direct_retrieval apply_server_codiesp_direct_retrieval.sh 0.0 wcov0 03:00:00
submit_arm direct_retrieval apply_server_codiesp_direct_retrieval.sh 1.0 wcov1 03:00:00
submit_arm one_pass_grounding apply_server_codiesp_one_pass_grounding.sh 0.0 wcov0 04:00:00
submit_arm one_pass_grounding apply_server_codiesp_one_pass_grounding.sh 1.0 wcov1 04:00:00
submit_arm one_pass_structured apply_server_codiesp_one_pass_structured.sh 0.0 wcov0 04:00:00
submit_arm one_pass_structured apply_server_codiesp_one_pass_structured.sh 1.0 wcov1 04:00:00
submit_arm frozen_ags apply_server_codiesp_frozen_ags.sh 0.0 wcov0 06:00:00
submit_arm frozen_ags apply_server_codiesp_frozen_ags.sh 1.0 wcov1 06:00:00
