#!/bin/bash
# Re-run the K_v sensitivity table under the LLM-only verifier.
#
# WHY. tab:llm_window_sensitivity's K_v=10 row is 0.402 / 0.550 / 0.705 / 0.247 / 0.167, which
# matches rerank_hybrid_full_k10fused to every digit. That is the HYBRID arm (deterministic core
# plus LLM). The paper was rewritten to the LLM-only method, whose FHS row is
# rerank_no_determ_k10fused = 0.401 / 0.543 / 0.705 / 0.257 / 0.183. K_v=10 is the deployed
# setting, so the appendix table and tab:ablation must agree there and currently do not: MRR is
# off by 0.010 and R@50 by 0.007. A reader comparing the two tables can see it.
#
# The k10 LLM-only rerank already exists; only the k5 and k20 ends of the sweep are missing.
# Stage 1 dumps the reranked ranking on CPU from the stored verdicts (project rule: CPU work runs
# locally, never sbatch). Stage 2 sbatches the Qwen3-32B listwise rerank that produces the
# accuracy column.
#
# MODE=llm_drop matches rerank_no_determ_k10fused, so the three windows differ only in K_v.
set -uo pipefail
REPO="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
RUN="${REPO}/runs_ags_verifier_ablation/qwen3_32b"
LOG="${REPO}/stage_window_llmonly.log"
cd "${REPO}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

for K in 5 20; do
  VERD="${RUN}/verdicts_k${K}_fused/llm_verifier_verdicts.json"
  OUT="${RUN}/rerank_no_determ_k${K}fused"
  RANK="${OUT}/bm25_candidates.jsonl"
  [[ -f "${VERD}" ]] || { say "k${K}: verdicts missing, skipping"; continue; }
  mkdir -p "${OUT}"
  if [[ ! -f "${RANK}" ]]; then
    say "k${K}: dumping no_determ ranking (llm_drop) from verdicts_k${K}_fused"
    python "${REPO}/ags_table5_ablation/dump_reranked_ranking.py" \
      --verifier-mode llm_drop --beta 0.6 \
      --verdicts "${VERD}" \
      --output "${RANK}.local_partial" \
      --summary "${OUT}/ranking_summary.json" || { say "k${K}: dump FAILED"; continue; }
    mv "${RANK}.local_partial" "${RANK}"
    say "k${K}: ranking staged"
  else
    say "k${K}: ranking already staged, reusing"
  fi
  say "k${K}: submitting GPU rerank"
  sbatch --job-name="rr_nodet_k${K}" \
    --export=ALL,ARM=no_determ,OUTPUT_DIR="${OUT}",VERDICTS="${VERD}" \
    "${REPO}/apply_server_verifier_ablation_rerank.sh" | tee -a "${LOG}"
done
say "done; tab:llm_window_sensitivity must be regenerated only after BOTH land"
