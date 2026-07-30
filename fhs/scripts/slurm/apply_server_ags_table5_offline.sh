#!/bin/bash
#SBATCH --job-name=ags_table5_offline
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=day
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs/logs/%j_ags_table5_offline.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Table 5 ablation, offline half (comparing_methods/ags_table5_ablation_spec.md): the dev
# beta sweep, every offline test row, the freetext-vs-parallel_sampling equality assertion,
# and the w_cov=0 index ablation panel. No GPU, no new generation, no new retrieval beyond
# CPU-only BM25 replay -- this is a deliberately separate job from
# apply_server_ags_table5_llm_verifier.sh (the one row that needs a GPU), and from
# apply_server_fintagging_frozen_ags.sh / the shared run_fintagging_grounding_baseline.py
# pipeline other experiments run against concurrently: this script only reads their already-
# logged traces and writes to its own runs_ags_table5_ablation/ tree.
#
# Examples:
#   sbatch apply_server_ags_table5_offline.sh
#   sbatch --export=ALL,LIMIT=50 apply_server_ags_table5_offline.sh   # smoke test

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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

cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi
# selected_betas.json is already on disk from the completed dev sweep; skip re-deriving it
# (the ~5min dev-sample replay) unless SKIP_BETA_SWEEP=0 is set explicitly.
SKIP_BETA_SWEEP_ARGS=()
if [[ "${SKIP_BETA_SWEEP:-1}" == "1" && -f "${OUTPUT_DIR}/selected_betas.json" ]]; then
  SKIP_BETA_SWEEP_ARGS=(--skip-beta-sweep)
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

python -m py_compile src/verifier/core.py src/verifier/data_prep.py \
  src/verifier/run_beta_sweep.py src/verifier/run_test_rows.py \
  src/verifier/run_index_ablation.py

python src/verifier/run_all_offline.py --output-dir "${OUTPUT_DIR}" "${LIMIT_ARGS[@]}" "${SKIP_BETA_SWEEP_ARGS[@]}"
