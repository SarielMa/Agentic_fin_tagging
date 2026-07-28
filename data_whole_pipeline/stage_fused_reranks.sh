#!/bin/bash
# Re-do the Final Acc. column of tab:ablation's LLM-consuming arms against CLEAN verdicts.
#
# WHY: run_ags_verifier_ablation.py regenerated the retrieval-stage columns from
# verdicts_k10_fused, but the Final Acc. column comes from rerank_<arm>/metrics.json, and every
# one of those dirs was built by dump_reranked_ranking.py against its DEFAULT verdicts file
# (runs_ags_table5_ablation/qwen3_32b/llm_verifier_verdicts.json, 2026-07-25). That file has no
# window_source field -- it predates the --window-source fused fix -- so the Final column stayed
# confounded while the retrieval columns became clean. Mixed provenance inside one row.
#
# Only the three arms that actually read verdicts need this. no_llm and no_verifier are
# verifier_mode=deterministic (run_ags_verifier_ablation.py:72-78, uses_llm=False) and never
# open the verdicts file, so their Final numbers are already correct.
#
# New output dirs, so the old confounded dirs stay on disk for comparison and nothing is
# destroyed if a job fails. Stage 1 (CPU dump) runs locally per the project rule; stage 2 (GPU
# listwise rerank) is sbatched the moment its input lands.
set -uo pipefail

REPO="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
RUN="${REPO}/runs_ags_verifier_ablation/qwen3_32b"
VERDICTS="${RUN}/verdicts_k10_fused/llm_verifier_verdicts.json"
cd "${REPO}"

if [[ ! -f "${VERDICTS}" ]]; then
  echo "clean verdicts missing: ${VERDICTS}" >&2; exit 1
fi

for ARM in hybrid_full no_determ llm_only; do
  OUT="${RUN}/rerank_${ARM}_k10fused"
  RANKING="${OUT}/bm25_candidates.jsonl"
  mkdir -p "${OUT}"

  case "${ARM}" in
    hybrid_full) MODE=hybrid ;;
    no_determ)   MODE=llm_drop ;;
    llm_only)    MODE=llm_strict ;;
  esac

  if [[ ! -f "${RANKING}" ]]; then
    echo "[$(date +%H:%M:%S)] dumping ${ARM} (mode=${MODE}) against clean k10 fused verdicts"
    python "${REPO}/ags_table5_ablation/dump_reranked_ranking.py" \
      --verifier-mode "${MODE}" \
      --beta 0.6 \
      --verdicts "${VERDICTS}" \
      --output "${RANKING}.local_partial" \
      --summary "${OUT}/ranking_summary.json"
    if [[ $? -ne 0 ]]; then
      echo "[$(date +%H:%M:%S)] dump FAILED for ${ARM}, skipping" >&2; continue
    fi
    mv "${RANKING}.local_partial" "${RANKING}"
  else
    echo "[$(date +%H:%M:%S)] ${ARM} ranking already staged, reusing"
  fi

  echo "[$(date +%H:%M:%S)] submitting GPU rerank for ${ARM}"
  sbatch --job-name="rr_${ARM}_fused" \
    --export=ALL,ARM="${ARM}",OUTPUT_DIR="${OUT}",VERDICTS="${VERDICTS}" \
    "${REPO}/apply_server_verifier_ablation_rerank.sh"
done

echo "[$(date +%H:%M:%S)] all three arms staged and submitted"
