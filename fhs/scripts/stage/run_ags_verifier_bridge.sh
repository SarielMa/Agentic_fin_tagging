#!/bin/bash
# Bridge diagnostic: reconciling Table 3 with the candidate-level LLM reranking ablation row.
#
# Local runner. No sbatch, no GPU: every input already exists on disk and this only reads
# them. Measured end to end at roughly 6 minutes.
#
#   ./run_ags_verifier_bridge.sh                    # the whole diagnostic
#   LIMIT=100 ./run_ags_verifier_bridge.sh          # smoke test, ~20 seconds
#   EMIT_LATEX=1 ./run_ags_verifier_bridge.sh       # also write bridge_table.tex
#   CONDA_ENV=finben ./run_ags_verifier_bridge.sh   # activate an env first
#   SKIP_TESTS=1 ./run_ags_verifier_bridge.sh       # skip the unit tests
#
# WHAT THIS ANSWERS
#   Table 3 says the LLM dimension-feedback verifier scores below the disagreement base rate.
#   The ablation row says candidate-level LLM reranking significantly improves MRR and top-1.
#   Both are correct because they measure different capabilities:
#     Panel A -- candidate-level relative discrimination (what reranking needs)
#     Panel B -- hypothesis-level absolute calibration    (what D- feedback needs)
#
# INPUTS (all present, none regenerated)
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl
#   runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl
#   runs_ags_table5_ablation/qwen3_32b/llm_verifier_summary.json
#   runs_ags_table5_ablation/qwen3_32b/ablation.csv           (AGS (full) baseline)
#   runs_ags_table5_ablation/qwen3_32b/llm_verifier_row.csv   (reranker arm + paired CIs)
#   runs_ags_table5_ablation/qwen3_32b/verifier_top20_overlap.json
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json
#
# SCOPE LIMIT YOU MUST KNOW
#   The with/without reranker comparison is complete at the RETRIEVAL stage only. Final
#   tagging accuracy is the deployed pipeline's post-listwise-rerank accuracy and is NOT
#   reconstructible from these logs -- it needs a GPU rerun of the reranker over the
#   candidate-level-reranked ranking. The script reports this as BLOCKED rather than
#   substituting the retrieval-stage number, and the decision rule returns PROVISIONAL.
#   See apply_server_ags_bridge_deployed_rerank.sh to unblock it.
#
# OUTPUTS in runs_ags_verifier_bridge/qwen3_32b/
#   bridge_candidate_discrimination.csv
#   bridge_hypothesis_calibration.csv
#   bridge_threshold_sweep.csv
#   final_reranker_comparison.csv
#   bootstrap_results.json
#   bridge_summary.json
#   bridge_table.tex             (only with EMIT_LATEX=1)
#
# Existing completed outputs are never overwritten: this writes to its own directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

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

export CUDA_VISIBLE_DEVICES=""

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_verifier_bridge/qwen3_32b}"
TEST_TRACE="${TEST_TRACE:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl}"
LLM_CALLS="${LLM_CALLS:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl}"
LLM_SUMMARY="${LLM_SUMMARY:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/llm_verifier_summary.json}"
ABLATION_CSV="${ABLATION_CSV:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/ablation.csv}"
RERANKER_ROW_CSV="${RERANKER_ROW_CSV:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/llm_verifier_row.csv}"
TOP20_OVERLAP="${TOP20_OVERLAP:-${REPO_ROOT}/runs/runs_ags_table5_ablation/qwen3_32b/verifier_top20_overlap.json}"
DEPLOYED_METRICS="${DEPLOYED_METRICS:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json}"
TOP_M="${TOP_M:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260726}"

echo "Python: $(command -v python)"
python --version

MISSING=0
for path in "${TEST_TRACE}" "${LLM_CALLS}" "${LLM_SUMMARY}" "${ABLATION_CSV}" "${RERANKER_ROW_CSV}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    MISSING=1
  fi
done
[[ "${MISSING}" == "0" ]] || { echo "Aborting: required inputs are missing." >&2; exit 1; }

for path in "${TOP20_OVERLAP}" "${DEPLOYED_METRICS}"; do
  [[ -f "${path}" ]] || echo "NOTE: optional input absent, its block will be omitted: ${path}" >&2
done

# The assessed window must match the LLM verifier run, or Panels A and B would be scored on a
# different candidate set than the model actually saw.
LOGGED_TOP_M="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['top_m'])" "${LLM_SUMMARY}")"
if [[ "${LOGGED_TOP_M}" != "${TOP_M}" ]]; then
  echo "TOP_M=${TOP_M} does not match the LLM verifier run's top_m=${LOGGED_TOP_M}. Aborting." >&2
  exit 1
fi

EXTRA_ARGS=()
[[ -n "${LIMIT:-}" ]] && EXTRA_ARGS+=(--limit "${LIMIT}")
[[ "${EMIT_LATEX:-0}" == "1" ]] && EXTRA_ARGS+=(--emit-latex)

echo "============================================================"
echo "Verifier bridge diagnostic -- TEST split (CPU only, no GPU allocated)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TOP_M=${TOP_M} (matches logged verifier top_m)"
echo "BOOTSTRAP=${BOOTSTRAP_SAMPLES} iterations, seed ${BOOTSTRAP_SEED}, unit=context"
echo "LIMIT=${LIMIT:-<none>}   EMIT_LATEX=${EMIT_LATEX:-0}"
echo "Expected runtime: ~6 min"
echo "============================================================"

python -m py_compile run_ags_verifier_bridge.py

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "--- unit tests (CPU, no data files) ---"
  python -m unittest test_ags_verifier_bridge
fi

START_TS=$(date +%s)

python run_ags_verifier_bridge.py \
  --output-dir "${OUTPUT_DIR}" \
  --test-trace "${TEST_TRACE}" \
  --llm-calls "${LLM_CALLS}" \
  --llm-summary "${LLM_SUMMARY}" \
  --ablation-csv "${ABLATION_CSV}" \
  --reranker-row-csv "${RERANKER_ROW_CSV}" \
  --top20-overlap "${TOP20_OVERLAP}" \
  --deployed-metrics "${DEPLOYED_METRICS}" \
  --top-m "${TOP_M}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  "${EXTRA_ARGS[@]}"

echo
echo "Finished in $(( ($(date +%s) - START_TS) / 60 ))m $(( ($(date +%s) - START_TS) % 60 ))s"
echo "Outputs: ${OUTPUT_DIR}"
