#!/bin/bash
# CPU-only. Regenerate verifier_ablation.csv / _cost.csv / _summary.json with the neutral
# unjudged fill, then rebuild the LaTeX fragments.
#
# WHY: run_ags_verifier_ablation.py grew llm_unjudged_fill="mean" (and records it in the
# summary), but the CSVs on disk are from 02:43, before that landed -- their summary has no
# llm_unjudged_fill key at all. So the two LLM-only arms in verifier_ablation.csv are the
# zero-fill variant, ~0.028 Recall@10 low, and build_verifier_ablation_table.py now refuses to
# emit from them. Re-running is what unblocks the table build.
#
# Waits for the ranking dumps to finish first: 4 cores at load ~60 already, and these two jobs
# contend for the same symbolic-profile parsing.

set -uo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
RUN_DIR="${REPO_ROOT}/runs/runs_ags_verifier_ablation/qwen3_32b"

while pgrep -f dump_reranked_ranking.py > /dev/null; do
  echo "[$(date +%T)] waiting for ranking dumps to finish"
  sleep 60
done
echo "[$(date +%T)] dumps clear, starting ablation re-run"

# Keep the pre-fix CSVs so the two fills can be compared rather than just overwritten.
BACKUP="${RUN_DIR}/prefix_zerofill_backup"
mkdir -p "${BACKUP}"
for f in verifier_ablation.csv verifier_ablation_cost.csv verifier_ablation_summary.json \
         interaction_retrieval_stage.csv; do
  [[ -f "${RUN_DIR}/${f}" && ! -f "${BACKUP}/${f}" ]] && cp "${RUN_DIR}/${f}" "${BACKUP}/${f}"
done

# No --sensitivity-top-m: truncated windows are a diagnostic only, and the real K_v=5/20 rows
# come from the two generation runs still on GPU (19848027 / 19817128).
python "${REPO_ROOT}/analysis/run_ags_verifier_ablation.py" --output-dir "${RUN_DIR}"
rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "[$(date +%T)] ablation re-run FAILED (rc=${rc})" >&2
  exit ${rc}
fi

echo "[$(date +%T)] rebuilding LaTeX fragments"
python "${REPO_ROOT}/analysis/build_verifier_ablation_table.py" --run-dir "${RUN_DIR}"
echo "[$(date +%T)] done"
