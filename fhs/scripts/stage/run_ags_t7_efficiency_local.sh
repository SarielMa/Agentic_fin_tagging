#!/bin/bash
# T7 efficiency table -- LOCAL runner (no Slurm).
#
# Same job as apply_server_ags_t7_efficiency.sh, minus the sbatch/module machinery: T7 is pure
# offline instrumentation of runs that already exist (no generation, no retrieval, no GPU), so
# there is nothing a compute node provides that a login/interactive shell does not.
#
# The only model artifact touched is the Qwen3-32B *tokenizer*, already cached under HF_HOME,
# used to recover completion-token counts for the two legacy traces (direct_retrieval,
# one_pass_grounding) whose logs stored raw_output but no token count. NO_TOKENIZER=1 skips
# that and leaves those cells unmeasured.
#
# Reads the other runs read-only; writes only to runs_ags_t7_t28/.
#
# Usage:
#   ./run_ags_t7_efficiency_local.sh
#   LIMIT=50 ./run_ags_t7_efficiency_local.sh              # smoke test
#   REPORT_WALLCLOCK=1 ./run_ags_t7_efficiency_local.sh    # see note below
#   CONDA_ENV=finben ./run_ags_t7_efficiency_local.sh      # override the env
#
# REPORT_WALLCLOCK is off by default on purpose: only four of six methods logged a per-fact
# timer, and one-pass grounding batches 32 prompts per call, so the rows were not timed under
# identical conditions -- the spec's rule for that case is to drop the column.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs}"
CONDA_ENV="${CONDA_ENV:-finben_b200}"

# ---- conda ---------------------------------------------------------------------------------
# Activate by sourcing conda.sh if available, else fall back to the env's bin on PATH. Both
# finben and finben_b200 satisfy this script's only import needs (transformers + tqdm).
if [[ -z "${CONDA_EXE:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
fi
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
fi
if ! python -c "import transformers" >/dev/null 2>&1; then
  CANDIDATE="${HOME}/.conda/envs/${CONDA_ENV}/bin"
  if [[ -x "${CANDIDATE}/python" ]]; then
    export PATH="${CANDIDATE}:${PATH}"
  else
    echo "Could not activate ${CONDA_ENV}; activate it yourself and re-run." >&2
    exit 1
  fi
fi

echo "python: $(command -v python)"
python --version

# ---- caches --------------------------------------------------------------------------------
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
mkdir -p "${HF_HOME}"
# The tokenizer is already cached. Left offline by default so a missing cache fails loudly
# instead of silently pulling from the network; set HF_HUB_OFFLINE=0 to allow a fetch.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${REPO_ROOT}"

# T7-specific by default, NOT the shared runs_ags_t7_t28/qwen3_32b that the sbatch scripts use.
# T7 and T28 both write a file called metrics.json, so pointing them at one directory means
# whichever runs second silently destroys the other's metadata. Keep them apart.
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_t7_t28/qwen3_32b_t7}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${NO_TOKENIZER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-tokenizer)
fi
if [[ "${REPORT_WALLCLOCK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-wallclock)
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "LIMIT=${LIMIT:-<none>}"
echo "REPORT_WALLCLOCK=${REPORT_WALLCLOCK:-0}"
echo "============================================================"

python -m py_compile src/efficiency/efficiency.py src/efficiency/run_t7_efficiency.py

python src/efficiency/run_t7_efficiency.py --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"
