#!/bin/bash
#SBATCH --job-name=ags_t7_efficiency
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_t7_efficiency.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# T7 efficiency table (comparing_methods/ags_t7_t28_spec.md). Pure offline instrumentation of
# runs that already exist: no generation, no retrieval, no GPU. It streams each method's trace
# and writes efficiency.csv + metrics.json.
#
# The only model artifact it touches is the Qwen3-32B *tokenizer* (already cached under
# HF_HOME), used to recover completion-token counts for the two legacy traces
# (direct_retrieval, one_pass_grounding) whose logs stored raw_output but no token count. Pass
# NO_TOKENIZER=1 to skip that and leave those cells unmeasured.
#
# Reads other experiments' run directories read-only and writes only to its own
# runs_ags_t7_t28/ tree, so it is safe to run alongside the AGS-Seq jobs.
#
# NOTE: AGS-Seq's own run must have FINISHED for its row to cover all 2,509 facts. If only
# grounding_traces.jsonl exists, this script still reports the row but marks it incomplete in
# metrics.json and prints a warning -- re-run once that job finishes.
#
# Examples:
#   sbatch apply_server_ags_t7_efficiency.sh
#   sbatch --export=ALL,LIMIT=50 apply_server_ags_t7_efficiency.sh          # smoke test
#   sbatch --export=ALL,REPORT_WALLCLOCK=1 apply_server_ags_t7_efficiency.sh

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
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
which python
python --version

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
mkdir -p "${HF_HOME}"
# The tokenizer is already cached; do not reach for the network on a compute node.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_t7_t28/qwen3_32b}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${NO_TOKENIZER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-tokenizer)
fi
# Off by default on purpose: only four of six methods logged a per-fact timer, and one-pass
# grounding batches 32 prompts per call, so the rows were not timed under identical
# conditions -- the spec's own rule for that case is to drop the column.
if [[ "${REPORT_WALLCLOCK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-wallclock)
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LIMIT=${LIMIT:-<none>}"
echo "REPORT_WALLCLOCK=${REPORT_WALLCLOCK:-0}"
echo "============================================================"

python -m py_compile ags_t7_t28/efficiency.py ags_t7_t28/run_t7_efficiency.py

python ags_t7_t28/run_t7_efficiency.py --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"
