#!/bin/bash
# CPU-only stage 1 for the two verifier-ablation arms that have no ranking dumped yet.
#
# hybrid_full / no_llm / no_verifier were dumped and submitted in the previous session
# (19871431 / 19871432 / 19871433). no_determ and llm_only are the two remaining rows of
# tab:ablation's "Verification architecture" block that still show -- in the Final Acc. column.
#
# Each arm: dump to <path>.partial, mv into place, then sbatch its GPU rerank immediately,
# so the GPU half queues the moment its input exists rather than after both dumps finish.
# The login node is oversubscribed (load ~60 on 4 cores), so the dumps run sequentially.
#
# hybrid_m5 is deliberately NOT dumped: tab:llm_window_sensitivity is retrieval-stage only
# ("the shared listwise reranker is run end-to-end for the default K_v=10 alone"), and its
# M=5 / M=20 rows come from the two generation runs already on GPU (19848027 / 19817128).

set -uo pipefail

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
OUT_BASE="${REPO_ROOT}/runs_ags_verifier_ablation/qwen3_32b"

# arm -> dump flags, copied from apply_server_verifier_ablation_rerank.sh's case block so the
# two stay in agreement.
declare -A MODES=( [no_determ]=llm_drop [llm_only]=llm_strict )

for ARM in no_determ llm_only; do
  OUTPUT_DIR="${OUT_BASE}/rerank_${ARM}"
  RANKING="${OUTPUT_DIR}/bm25_candidates.jsonl"
  mkdir -p "${OUTPUT_DIR}"

  if [[ -f "${RANKING}" ]]; then
    echo "[$(date +%T)] ${ARM}: ranking already present, skipping dump"
  else
    echo "[$(date +%T)] ${ARM}: dumping ranking (verifier_mode=${MODES[$ARM]}, beta=0.6)"
    # Write to .partial and mv, so the GPU job's `[[ ! -f ${RANKING} ]]` check can never see a
    # half-written 2.8 GB file. dump_reranked_ranking.py does not do this itself.
    # --llm-unjudged-fill mean matches run_ags_verifier_ablation.py. Without it these two arms'
    # final accuracy would come from a zero-fill ranking while their retrieval-stage row in the
    # same table is mean-fill (a 0.028 Recall@10 gap), so the row would not be internally paired.
    python "${REPO_ROOT}/ags_table5_ablation/dump_reranked_ranking.py" \
      --verifier-mode "${MODES[$ARM]}" \
      --beta 0.6 \
      --llm-unjudged-fill mean \
      --output "${RANKING}.partial" \
      --summary "${OUTPUT_DIR}/ranking_summary.json"
    rc=$?
    if [[ ${rc} -ne 0 || ! -s "${RANKING}.partial" ]]; then
      echo "[$(date +%T)] ${ARM}: DUMP FAILED (rc=${rc}), not submitting" >&2
      continue
    fi
    mv "${RANKING}.partial" "${RANKING}"
    echo "[$(date +%T)] ${ARM}: dump done -> $(stat -c%s "${RANKING}") bytes"
  fi

  echo "[$(date +%T)] ${ARM}: submitting GPU rerank"
  sbatch --job-name="vabl_${ARM}" --export=ALL,ARM="${ARM}" \
    "${REPO_ROOT}/apply_server_verifier_ablation_rerank.sh"
done

echo "[$(date +%T)] all done"
