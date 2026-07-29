#!/bin/bash
# Full, honest re-runs of the seven remaining coverage-OFF baselines at w_cov=1.0.
#
# WHY. tab:main_results has 12 method rows. Nine of them ran with the label-coverage term OFF,
# because run_fintagging_grounding_baseline.py assigns retriever.label_coverage_weight only inside
# the frozen-family branches; TaxonomyRetriever defaults to 0.0. Coverage was ON only for
# one_pass_structured, FHS (frozen_ags) and FHS-Seq (ags_seq) -- the paper's own family. Sec. 4 and
# app:coverage both claim the term is "enabled identically for every method", which is therefore
# false as run. stage_baselines_wcov1_then_submit.sh already closed two of the nine
# (direct_retrieval, one_pass_grounding). This script closes the other seven, after which the
# sentence becomes true with no rewording.
#
# WHY FULL RE-RUNS AND NOT REPLAY. Every one of these arms has grounding_traces.jsonl on disk, and
# the python side has a --resume path that reads it back, so re-scoring stored generations against
# the new index would be cheap. That is NOT what this script does. For the three feedback arms
# (retrieval_feedback, operator, memory_guided) generation is conditioned on the previous round's
# retrieval, so replaying stored queries would score old-index queries against a new index -- not a
# run of that method under the new index. Rather than mix faithful and approximate rows in one
# table, every arm here regenerates: RESUME=0 and REUSE_CANDIDATES=0.
#
# SETTINGS MATCH FHS EXACTLY. w_cov=1.0 is the value frozen_ags pins (ags_frozen_grounding.py:95);
# the pool multiplier stays at its default 0, "score the full type-filtered pool", which is what
# FHS, one_pass_structured and the two already-submitted arms all used. The rerank model and
# backend come from each arm's own apply_server script, unchanged, so the Acc. column stays
# comparable to the rows that are not being re-run.
#
# OUTPUT DIRS ARE NEW (_wcov1 suffix). Nothing published is touched, so the paper keeps compiling
# off the current numbers until every arm has landed.
#
# BOTH STAGES MUST LAND TOGETHER. Each job runs generation, retrieval AND rerank, so its
# metrics.json carries R@10/R@50/MRR and Acc./std from the same configuration. Do not copy the
# retrieval columns into the paper before the rerank of the same arm has finished.
#
# COUPLED EDIT, DO NOT FORGET. Table 5's raw-context and free-text rows currently print the
# coverage-OFF numbers precisely so they match tab:main_results' two baselines to the printed
# digits. The moment those baselines are updated to w_cov=1, Table 5 must be flipped to its
# coverage-ON numbers in the same edit, or the two tables disagree again. The cov-ON values are
# in runs_ags_probe_queryform/qwen3_32b_pooled/query_form_metrics.csv.
set -uo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${FHS_ROOT}"
RUNS="${REPO}/runs/runs_fintagging_grounding_baseline"
LOG="${REPO}/scripts/stage/stage_remaining_baselines_wcov1.log"
cd "${REPO}"

# Ordered fastest-first, from the historical elapsed times of the published runs, so that a chain
# that breaks late still delivers most of the table.
ARMS=(
  "decomposed_retrieval:fintagging_decomposed_retrieval:12:00:00"
  "parallel_sampling:fintagging_parallel_sampling:12:00:00"
  "intrinsic_self_refinement:fintagging_intrinsic_self_refinement:12:00:00"
  "retrieval_feedback_refinement:fintagging_retrieval_feedback_refinement:12:00:00"
  "parallel_sampling_diversity:fintagging_parallel_sampling_diversity:12:00:00"
  "operator_refinement:fintagging_operator_refinement:24:00:00"
  "memory_guided_refinement:fintagging_memory_guided_refinement:24:00:00"
)

# CHAIN=1 serializes with --dependency=afterok: one queue slot at a time, but the total wall clock
# is the sum of every arm plus a fresh queue wait before each. CHAIN=0 submits them independently
# so they pack into whatever slots free up. The arms are independent, so CHAIN=0 is not less
# honest, only less orderly.
CHAIN="${CHAIN:-0}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

say "=== submitting ${#ARMS[@]} full re-runs at w_cov=1.0 (CHAIN=${CHAIN}) ==="

PREV_JOB=""
SUBMITTED=()

for entry in "${ARMS[@]}"; do
  MODE="${entry%%:*}"
  REST="${entry#*:}"
  SCRIPT="apply_server_${REST%%:*}.sh"
  TIME="${entry#*:*:}"
  OUT="${RUNS}/qwen3_32b_${MODE}_wcov1"

  if [[ ! -f "${REPO}/${SCRIPT}" ]]; then
    say "MISSING ${SCRIPT}, skipping ${MODE}"; continue
  fi
  if [[ -f "${OUT}/metrics.json" ]]; then
    say "${MODE}: metrics.json already present, skipping"; continue
  fi

  DEP=()
  if [[ "${CHAIN}" == "1" && -n "${PREV_JOB}" ]]; then
    DEP=(--dependency="afterok:${PREV_JOB}")
  fi

  JOB=$(sbatch --parsable \
    --job-name="wcov1_${MODE:0:12}" \
    --time="${TIME}" \
    "${DEP[@]}" \
    --export=ALL,QUERY_MODE="${MODE}",OUTPUT_DIR="${OUT}",LABEL_COVERAGE_WEIGHT=1.0,RESUME=0,REUSE_CANDIDATES=0,RUN_RERANK=1 \
    "${REPO}/${SCRIPT}" 2>&1)

  if [[ ! "${JOB}" =~ ^[0-9]+$ ]]; then
    say "${MODE}: SUBMIT FAILED -- ${JOB}"; continue
  fi

  say "${MODE} -> job ${JOB} (time ${TIME}, out ${OUT})"
  SUBMITTED+=("${JOB}")
  PREV_JOB="${JOB}"
done

say "submitted ${#SUBMITTED[@]}: ${SUBMITTED[*]:-none}"
say "NOTHING in the paper changes until every arm's metrics.json exists AND Table 5 is flipped to"
say "its coverage-ON numbers in the same edit."
