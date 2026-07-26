#!/bin/bash
# Table 14 (Appendix G) -- revision-stage diagnostics -- on the FROZEN TEST SPLIT.
#
# Local runner. No sbatch, no GPU, no module loading: every input already exists on disk and
# this only reads them. Measured end to end at roughly 4 minutes (see TIMING below).
#
#   ./run_ags_revision_diagnostics.sh                    # the whole experiment
#   LIMIT=100 ./run_ags_revision_diagnostics.sh          # smoke test, ~15 seconds
#   ARMS=learned ./run_ags_revision_diagnostics.sh       # one arm only
#   CONDA_ENV=finben ./run_ags_revision_diagnostics.sh   # activate an env first
#   SKIP_TESTS=1 ./run_ags_revision_diagnostics.sh       # skip the unit tests
#
# Test split: the same one apply_server_fintagging_frozen_ags.sh runs, i.e.
#   FinTagging_800_200_grounding_test_JSON/data/test.jsonl -> 2,509 facts.
# It is read through the AGS-Seq runs' own output traces rather than re-derived.
#
# INPUTS (all present, none regenerated)
#   runs_fintagging_grounding_baseline/qwen3_32b_ags_seq/grounding_traces.jsonl
#   runs_fintagging_grounding_baseline/qwen3_32b_ags_seq_random/grounding_traces.jsonl
#       The two sequential arms over all 2,509 test facts at B=4 rounds. Their
#       `ags_seq_rounds` log the verdicts, which dimensions D- fired on, the operator and
#       its targeted dimension, and the post-revision hypothesis. Both arms are pooled for
#       the paper rows; per-arm rows are emitted alongside in revision_effectiveness.csv.
#
# The paper's Table 14 reports one feedback layer here, not two. The test AGS-Seq runs set
# llm_feedback_enabled=False, so the dev tables' "merged" column would be the deterministic
# column relabelled; it is reported once, as `deterministic`.
#
# TIMING, measured rather than estimated:
#   taxonomy load + imports                 3 s
#   round pass, both arms (9.6 GB)        ~1.2 min   (14 ms/fact)
#   event construction + null draws       ~1.5 min   (18 ms/fact)
#   bootstrap grid                        ~1 min
#   ------------------------------------------------
#   total                                 ~4 min
# Single-threaded Python throughout, so more cores buy nothing.
#
# OUTPUTS in runs_ags_revision_diagnostics/qwen3_32b/
#   panel_a_recoverability.csv   Panel A
#   panel_b_effectiveness.csv    Panel B, the paper's row set
#   revision_effectiveness.csv   the full arm x dimension x target-group grid
#   per_event.jsonl              one row per scored (arm, fact, round, dimension)
#   metrics.json                 the above plus the appendix's prose statistics

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Optional: only touch conda if the caller asked for a specific env.
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

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_revision_diagnostics/qwen3_32b}"
SEQ_TRACE_LEARNED="${SEQ_TRACE_LEARNED:-${REPO_ROOT}/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq/grounding_traces.jsonl}"
SEQ_TRACE_RANDOM="${SEQ_TRACE_RANDOM:-${REPO_ROOT}/runs_fintagging_grounding_baseline/qwen3_32b_ags_seq_random/grounding_traces.jsonl}"
ARMS="${ARMS:-learned,random}"
GOLD_CANDIDATE_FIELDS="${GOLD_CANDIDATE_FIELDS:-compact}"
RANDOM_DRAWS="${RANDOM_DRAWS:-20}"
RANDOM_SEED="${RANDOM_SEED:-20260728}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260726}"

echo "Python: $(command -v python)"
python --version

case ",${ARMS}," in
  *,learned,*) [[ -f "${SEQ_TRACE_LEARNED}" ]] || { echo "Missing: ${SEQ_TRACE_LEARNED}" >&2; exit 1; } ;;
esac
case ",${ARMS}," in
  *,random,*) [[ -f "${SEQ_TRACE_RANDOM}" ]] || { echo "Missing: ${SEQ_TRACE_RANDOM}" >&2; exit 1; } ;;
esac

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "Table 14 -- revision-stage diagnostics on the TEST split (CPU only)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ARMS=${ARMS}"
echo "SEQ_TRACE_LEARNED=${SEQ_TRACE_LEARNED}"
echo "SEQ_TRACE_RANDOM=${SEQ_TRACE_RANDOM}"
echo "GOLD_CANDIDATE_FIELDS=${GOLD_CANDIDATE_FIELDS} (deployed; the other is reported alongside)"
echo "NULL=${RANDOM_DRAWS} draws/event, seed ${RANDOM_SEED}"
echo "BOOTSTRAP=${BOOTSTRAP_SAMPLES} iterations, seed ${BOOTSTRAP_SEED}, unit=context"
echo "LIMIT=${LIMIT:-<none>}"
echo "Expected runtime: ~4 min"
echo "============================================================"

python -m py_compile run_ags_revision_diagnostics.py

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "--- unit tests (CPU, no data files) ---"
  python -m unittest test_ags_revision_diagnostics
fi

START_TS=$(date +%s)

python run_ags_revision_diagnostics.py \
  --output-dir "${OUTPUT_DIR}" \
  --seq-trace-learned "${SEQ_TRACE_LEARNED}" \
  --seq-trace-random "${SEQ_TRACE_RANDOM}" \
  --arms "${ARMS}" \
  --gold-candidate-fields "${GOLD_CANDIDATE_FIELDS}" \
  --random-draws "${RANDOM_DRAWS}" \
  --random-seed "${RANDOM_SEED}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  "${EXTRA_ARGS[@]}"

echo
echo "Finished in $(( ($(date +%s) - START_TS) / 60 ))m $(( ($(date +%s) - START_TS) % 60 ))s"
echo "Panels:  ${OUTPUT_DIR}/panel_a_recoverability.csv"
echo "         ${OUTPUT_DIR}/panel_b_effectiveness.csv"
echo "Details: ${OUTPUT_DIR}/metrics.json"
