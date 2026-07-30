#!/bin/bash
# CPU-only. Re-run the verifier bridge now that its end-to-end cell is no longer blocked.
#
# apply_server_ags_bridge_deployed_rerank.sh landed at 22:04 on 2026-07-26, writing
# runs_ags_verifier_bridge/qwen3_32b/deployed_rerank/metrics.json. run_ags_verifier_bridge.py
# had no argument to read it -- the with-reranking cell was hardcoded None with a BLOCKED
# status -- so the result sat unused. That argument (--deployed-metrics-with-llm) now exists
# and defaults to that path.
#
# Waits for the ablation re-run first; 4 cores at load ~60.

set -uo pipefail
FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

while pgrep -f run_ags_verifier_ablation.py > /dev/null; do
  echo "[$(date +%T)] waiting for the ablation re-run"
  sleep 60
done
echo "[$(date +%T)] ablation clear, re-running the bridge"

BACKUP="${REPO_ROOT}/runs/runs_ags_verifier_bridge/qwen3_32b/blocked_backup"
mkdir -p "${BACKUP}"
for f in bridge_summary.json final_reranker_comparison.csv bridge_table.tex; do
  src="${REPO_ROOT}/runs/runs_ags_verifier_bridge/qwen3_32b/${f}"
  [[ -f "${src}" && ! -f "${BACKUP}/${f}" ]] && cp "${src}" "${BACKUP}/${f}"
done

EMIT_LATEX=1 "${REPO_ROOT}/scripts/stage/run_ags_verifier_bridge.sh"
echo "[$(date +%T)] done"
