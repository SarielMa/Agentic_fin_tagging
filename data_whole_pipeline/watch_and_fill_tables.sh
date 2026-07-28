#!/bin/bash
# Poll for finished jobs, run whatever CPU stage their output unblocks, and inject into the
# paper. Idempotent: every step below either no-ops or rewrites the same cells, so re-running
# after a partial pass is safe.
#
# CPU only, runs on the login node -- SLURM is for the GPU stages that feed this.

set -uo pipefail
REPO="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
RUN="${REPO}/runs_ags_verifier_ablation/qwen3_32b"
PAPER="${REPO}/../comparing_methods/acl_latex.tex"
LOG="${REPO}/watch_and_fill.log"
cd "${REPO}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# Marker files, so a completed stage is never redone on the next poll.
mkdir -p "${RUN}/.done"
done_marker() { [[ -f "${RUN}/.done/$1" ]]; }
mark() { touch "${RUN}/.done/$1"; }

say "watcher started; polling every 10 min"

while true; do
  # ---- clean K_v=10 verdicts -> re-run the verifier ablation, replacing the provisional rows
  V10="${RUN}/verdicts_k10_fused/llm_verifier_verdicts.json"
  if [[ -f "${V10}" ]] && ! done_marker ablation_clean; then
    say "K_v=10 fused verdicts landed -> re-running verifier ablation (~30 min)"
    if python run_ags_verifier_ablation.py --verdicts "${V10}" \
         --calls "${RUN}/verdicts_k10_fused/llm_verifier_calls.jsonl" \
         --output-dir "${RUN}" >> "${LOG}" 2>&1; then
      mark ablation_clean
      say "verifier ablation re-run complete"
    else
      say "verifier ablation re-run FAILED (see ${LOG})"
    fi
  fi

  # ---- any fused window verdicts -> refresh the K_v sensitivity rows
  if compgen -G "${RUN}/verdicts_k*_fused/llm_verifier_summary.json" > /dev/null; then
    ARGS=()
    for d in "${RUN}"/verdicts_k*_fused; do
      k="$(basename "$d")"; k="${k#verdicts_k}"; k="${k%_fused}"
      [[ -f "$d/llm_verifier_summary.json" ]] && ARGS+=(--window "${k}=${d}")
    done
    if [[ ${#ARGS[@]} -gt 0 ]]; then
      say "refreshing K_v sensitivity for: ${ARGS[*]}"
      python run_verifier_window_sensitivity.py --run-dir "${RUN}" "${ARGS[@]}" >> "${LOG}" 2>&1
    fi
  fi

  # ---- fulltagging hybrid -> tab:end_to_end's last row
  FT="${REPO}/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_hybrid_verification"
  if [[ -f "${FT}/fulltagging_metrics.json" ]] && ! done_marker end_to_end; then
    say "fulltagging hybrid landed -> filling tab:end_to_end"
    if python fill_end_to_end_hybrid.py --run-dir "${FT}" --paper "${PAPER}" >> "${LOG}" 2>&1; then
      mark end_to_end
      say "tab:end_to_end hybrid row filled"
    else
      say "tab:end_to_end fill FAILED (see ${LOG})"
    fi
  fi

  # ---- rebuild fragments and inject everything currently available
  python build_verifier_ablation_table.py --paper --appendix >> "${LOG}" 2>&1

  REMAINING=$(python "${REPO}/count_placeholders.py" "${PAPER}" 2>/dev/null || echo "?")
  say "placeholder rows remaining: ${REMAINING}"
  if [[ "${REMAINING}" == "0" ]]; then
    say "ALL TABLES FILLED -- watcher exiting"
    exit 0
  fi

  sleep 600
done
