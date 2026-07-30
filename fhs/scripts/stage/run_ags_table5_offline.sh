#!/usr/bin/env bash
set -euo pipefail

# Direct (non-sbatch) runner for the Table 5 ablation's no-GPU half: dev beta sweep, every
# offline test row (ags_table5_ablation/run_test_rows.py, the 10 methods with progress bars),
# and the w_cov=0 index ablation panel. See apply_server_ags_table5_offline.sh for the SLURM
# version; this one just runs in the current shell/conda env with no module purge.
#
# Writes to its own OUTPUT_DIR (default: a fresh qwen3_32b_rerun/ dir) so it never overwrites
# the existing runs_ags_table5_ablation/qwen3_32b/ results.
#
# Examples:
#   ./run_ags_table5_offline.sh
#   OUTPUT_DIR=runs_ags_table5_ablation/qwen3_32b_2026-07-24 ./run_ags_table5_offline.sh
#   LIMIT=50 ./run_ags_table5_offline.sh   # smoke test

REPO_ROOT="$(readlink -f "$(dirname "$0")")"
cd "${REPO_ROOT}"

BASE_OUTPUT_DIR="${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}_rerun}"
LIMIT="${LIMIT:-}"
SKIP_BETA_SWEEP="${SKIP_BETA_SWEEP:-1}"

mkdir -p "${OUTPUT_DIR}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

SKIP_BETA_SWEEP_ARGS=()
if [[ "${SKIP_BETA_SWEEP}" == "1" ]]; then
  if [[ ! -f "${OUTPUT_DIR}/selected_betas.json" && -f "${BASE_OUTPUT_DIR}/selected_betas.json" ]]; then
    # Same frozen beta selection either way; reuse it instead of redoing the ~5min dev
    # sweep just because the output moved to a new directory.
    cp "${BASE_OUTPUT_DIR}/selected_betas.json" "${OUTPUT_DIR}/selected_betas.json"
  fi
  if [[ -f "${OUTPUT_DIR}/selected_betas.json" ]]; then
    SKIP_BETA_SWEEP_ARGS=(--skip-beta-sweep)
  fi
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LIMIT=${LIMIT:-<none>}"
echo "SKIP_BETA_SWEEP=${SKIP_BETA_SWEEP}"
echo "============================================================"

python src/verifier/run_all_offline.py --output-dir "${OUTPUT_DIR}" "${LIMIT_ARGS[@]}" "${SKIP_BETA_SWEEP_ARGS[@]}"
