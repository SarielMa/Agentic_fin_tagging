#!/bin/bash
# Tables 3 and 13 -- verification quality -- on the FROZEN TEST SPLIT.
#
# Local runner. No sbatch, no GPU, no module loading: every input already exists on disk and
# this only reads them. Measured end to end at roughly 6 minutes (see TIMING below).
#
#   ./run_ags_verification_quality.sh                  # the whole experiment
#   LIMIT=25 ./run_ags_verification_quality.sh         # smoke test, ~10 seconds
#   ARM_COMPARISON=0 ./run_ags_verification_quality.sh # skip Table 13's learned-vs-random note
#   CONDA_ENV=finben ./run_ags_verification_quality.sh # activate an env first
#   SKIP_TESTS=1 ./run_ags_verification_quality.sh     # skip the unit tests
#
# Test split: the same one apply_server_fintagging_frozen_ags.sh runs, i.e.
#   FinTagging_800_200_grounding_test_JSON/data/test.jsonl -> 2,509 scored facts.
# It is read through that script's own output trace rather than re-derived, so there is no
# way for this to drift onto a different split.
#
# INPUTS (all present, none regenerated)
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl
#       AGS's test-split run: J=2 hypotheses and the fused ranking per fact. The deterministic
#       verifier is replayed over its top-M=10 cluster representatives. Replay is exact --
#       symbolic_feedback_from_candidates is a pure function of the candidates -- so this
#       reproduces the verdicts generation time issued.
#   runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl
#       The full-test LLM verifier rerun after the truncation fix: 5,018 calls (2,509 facts x
#       2 hypotheses), parse_rate 1.0, top_m 10, judging FAMILY/ROLE/EVENT over those SAME
#       representatives. That shared window is what makes "on identical dimensions" a paired
#       claim. This is the one GPU step in the experiment and it has already run.
#   runs_fintagging_grounding_baseline/qwen3_32b_ags_seq{,_random}/grounding_traces.jsonl
#       Only for Table 13's closing note (learned vs random operator selection).
#
# TIMING, measured on this machine rather than estimated:
#   taxonomy load + imports                     3 s
#   LLM verdict load (23 MB)                    5 s
#   frozen-AGS pass, 2,509 facts              ~4 min   <- dominates; 90 ms/fact of symbolic
#                                                         profile parsing, single-threaded
#   bootstrap, 39 scored rows x 2,000 iters    10 s
#   both sequential arms (9.6 GB streamed)    ~1 min
#   ------------------------------------------------
#   total                                     ~6 min   (~5 min with ARM_COMPARISON=0)
# Trace I/O is not a factor: the store reads at ~2.4 GB/s, so all 13 GB streams in seconds.
# Runtime is single-threaded Python, so more cores buy nothing.
#
# OUTPUTS in runs_ags_verification_quality/qwen3_32b/
#   table3.csv, table13.csv    the two paper tables
#   verification_quality.csv   every slice x layer with context-bootstrap CIs
#   per_verdict.jsonl          one row per scored (fact, hypothesis, dimension)
#   metrics.json               all of the above plus coverage and config

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Optional: only touch conda if the caller asked for a specific env. Running in an already
# active env is the common case and must not be disturbed.
if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    source "$(cd "$(dirname "$(command -v conda)")/.." && pwd)/etc/profile.d/conda.sh"
  else
    echo "CONDA_ENV=${CONDA_ENV} was requested but conda is not on PATH." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
fi

# Nothing here uses an accelerator; say so rather than relying on the machine.
export CUDA_VISIBLE_DEVICES=""

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_verification_quality/qwen3_32b}"
TEST_TRACE="${TEST_TRACE:-${REPO_ROOT}/runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl}"
LLM_CALLS="${LLM_CALLS:-${REPO_ROOT}/runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl}"
SEQ_TRACE_LEARNED="${SEQ_TRACE_LEARNED:-${REPO_ROOT}/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq/grounding_traces.jsonl}"
SEQ_TRACE_RANDOM="${SEQ_TRACE_RANDOM:-${REPO_ROOT}/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq_random/grounding_traces.jsonl}"
TOP_M="${TOP_M:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260726}"
ARM_COMPARISON="${ARM_COMPARISON:-1}"

echo "Python: $(command -v python)"
python --version

for path in "${TEST_TRACE}" "${LLM_CALLS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

# TOP_M must match the LLM verifier run, or the two layers get scored on different windows
# and the comparison silently stops being paired.
LLM_SUMMARY="$(dirname "${LLM_CALLS}")/llm_verifier_summary.json"
LOGGED_TOP_M=""
if [[ -f "${LLM_SUMMARY}" ]]; then
  LOGGED_TOP_M="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['top_m'])" "${LLM_SUMMARY}")"
  if [[ "${LOGGED_TOP_M}" != "${TOP_M}" ]]; then
    echo "TOP_M=${TOP_M} does not match the LLM verifier run's top_m=${LOGGED_TOP_M}." >&2
    echo "Both layers must score the same assessed window. Aborting." >&2
    exit 1
  fi
  PARSE_RATE="$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('parse_rate'))" "${LLM_SUMMARY}")"
  echo "LLM verifier run: top_m=${LOGGED_TOP_M}, parse_rate=${PARSE_RATE}"
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
echo "Tables 3 and 13 -- verification quality on the TEST split (CPU only)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TEST_TRACE=${TEST_TRACE}"
echo "LLM_CALLS=${LLM_CALLS}"
echo "TOP_M=${TOP_M}"
echo "BOOTSTRAP=${BOOTSTRAP_SAMPLES} iterations, seed ${BOOTSTRAP_SEED}, unit=context"
echo "ARM_COMPARISON=${ARM_COMPARISON}"
echo "LIMIT=${LIMIT:-<none>}"
echo "Expected runtime: ~6 min (~5 min with ARM_COMPARISON=0)"
echo "============================================================"

python -m py_compile run_ags_verification_quality.py

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "--- unit tests (CPU, no data files) ---"
  python -m unittest test_ags_verification_quality
fi

START_TS=$(date +%s)

python run_ags_verification_quality.py \
  --output-dir "${OUTPUT_DIR}" \
  --test-trace "${TEST_TRACE}" \
  --llm-calls "${LLM_CALLS}" \
  --top-m "${TOP_M}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  "${EXTRA_ARGS[@]}"

echo
echo "Finished in $(( ($(date +%s) - START_TS) / 60 ))m $(( ($(date +%s) - START_TS) % 60 ))s"
echo "Tables:  ${OUTPUT_DIR}/table3.csv"
echo "         ${OUTPUT_DIR}/table13.csv"
echo "Details: ${OUTPUT_DIR}/metrics.json"
