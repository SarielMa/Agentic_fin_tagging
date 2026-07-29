#!/bin/bash
#SBATCH --job-name=ags_verification_quality
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=day
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_verification_quality.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Tables 3 and 13 -- verification quality -- on the FROZEN TEST SPLIT, replacing the
# development-set placeholders in both captions.
#
#   sbatch apply_server_ags_verification_quality.sh
#   sbatch --export=ALL,LIMIT=25 apply_server_ags_verification_quality.sh   # smoke test, ~2 min
#   sbatch --export=ALL,ARM_COMPARISON=0 apply_server_ags_verification_quality.sh
#
# NO --gpus LINE, DELIBERATELY. Every input already exists; this job only reads them.
# Allocating an idle GPU is what the auto-cancel policy kills, and there is nothing here
# for it to do -- no generation, no retrieval, no model load.
#
# Test split: the same one apply_server_fintagging_frozen_ags.sh runs, i.e.
#   FinTagging_800_200_grounding_test_JSON/data/test.jsonl  -> 2,509 scored facts.
# It is read through that script's own output trace rather than re-derived, so there is no
# way for this job to drift onto a different split.
#
# Inputs (all present, none regenerated):
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl
#       AGS's test-split run: J=2 hypotheses and the fused ranking per fact. The
#       deterministic verifier is replayed over its top-M=10 cluster representatives.
#       Replay is exact -- symbolic_feedback_from_candidates is a pure function of the
#       candidates -- so this reproduces the verdicts generation time issued.
#   runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl
#       The full-test LLM verifier rerun after the truncation fix: 5,018 calls
#       (2,509 facts x 2 hypotheses), parse_rate 1.0, top_m 10, judging FAMILY/ROLE/EVENT
#       over those SAME representatives. Both layers therefore see one window and one
#       hypothesis set, which is what makes "on identical dimensions" a paired claim.
#   runs_fintagging_grounding_baseline/qwen3_32b_ags_seq{,_random}/grounding_traces.jsonl
#       Only for Table 13's closing note (learned vs random operator selection). Two ~4.8GB
#       streams; set ARM_COMPARISON=0 to skip them and finish much faster.
#
# Runtime is dominated by single-threaded Python -- JSON parsing of a 3.6GB trace plus
# ~300k symbolic profile parses -- so raising --cpus-per-task past 8 buys nothing. Budget
# ~1-2h without the arm comparison and ~3-5h with it; `day` allows 24h.
#
# Outputs land in runs_ags_verification_quality/qwen3_32b/:
#   table3.csv, table13.csv    the two paper tables
#   verification_quality.csv   every slice x layer with context-bootstrap CIs
#   per_verdict.jsonl          one row per scored (fact, hypothesis, dimension)
#   metrics.json               all of the above plus coverage and config

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

CONDA_ENV="${CONDA_ENV:-finben}"
conda activate "${CONDA_ENV}"
which python
python --version

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
mkdir -p "${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
# No accelerator is allocated; make that unambiguous rather than relying on the node.
export CUDA_VISIBLE_DEVICES=""

cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_verification_quality/qwen3_32b}"
TEST_TRACE="${TEST_TRACE:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl}"
LLM_CALLS="${LLM_CALLS:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl}"
SEQ_TRACE_LEARNED="${SEQ_TRACE_LEARNED:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq/grounding_traces.jsonl}"
SEQ_TRACE_RANDOM="${SEQ_TRACE_RANDOM:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq_random/grounding_traces.jsonl}"
TOP_M="${TOP_M:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260726}"
ARM_COMPARISON="${ARM_COMPARISON:-1}"

for path in "${TEST_TRACE}" "${LLM_CALLS}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 1; }
done

# TOP_M must match the LLM verifier run, or the two layers would be scored on different
# windows and the comparison would stop being paired.
LOGGED_TOP_M="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['top_m'])" \
  "$(dirname "${LLM_CALLS}")/llm_verifier_summary.json" 2>/dev/null || echo "")"
if [[ -n "${LOGGED_TOP_M}" && "${LOGGED_TOP_M}" != "${TOP_M}" ]]; then
  echo "TOP_M=${TOP_M} does not match the LLM verifier run's top_m=${LOGGED_TOP_M}." >&2
  echo "The two layers must score the same assessed window. Aborting." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${ARM_COMPARISON}" == "1" ]]; then
  EXTRA_ARGS+=(--arm-comparison --seq-trace-learned "${SEQ_TRACE_LEARNED}" --seq-trace-random "${SEQ_TRACE_RANDOM}")
else
  EXTRA_ARGS+=(--no-arm-comparison)
fi

echo "============================================================"
echo "Tables 3 and 13 -- verification quality on the TEST split (CPU, no GPU allocated)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TEST_TRACE=${TEST_TRACE}"
echo "LLM_CALLS=${LLM_CALLS}"
echo "TOP_M=${TOP_M} (logged verifier top_m=${LOGGED_TOP_M:-<unknown>})"
echo "BOOTSTRAP=${BOOTSTRAP_SAMPLES} iterations, seed ${BOOTSTRAP_SEED}, unit=context"
echo "ARM_COMPARISON=${ARM_COMPARISON}"
echo "LIMIT=${LIMIT:-<none>}   THREADS=${OMP_NUM_THREADS}"
echo "============================================================"

python -m py_compile run_ags_verification_quality.py
python -m unittest test_ags_verification_quality -v

python run_ags_verification_quality.py \
  --output-dir "${OUTPUT_DIR}" \
  --test-trace "${TEST_TRACE}" \
  --llm-calls "${LLM_CALLS}" \
  --top-m "${TOP_M}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  "${EXTRA_ARGS[@]}"

echo "Done. Tables in ${OUTPUT_DIR}/table3.csv and ${OUTPUT_DIR}/table13.csv"
