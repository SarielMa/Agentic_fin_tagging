#!/bin/bash
# Rerank the symbolic-hint arm, so the "verdicts as evidence" claim has a ranking measurement.
#
# WHY: the abstract claims that supplying the deterministic verdicts to the LLM as prompt
# context "separates gold from distractors and improves ranking". The separation half comes from
# tab:verifierbridge Panel A, which bridge_summary.json computes from
# runs_ags_table5_ablation/qwen3_32b/llm_verifier_calls.jsonl -- a run with NO symbolic hint, so
# it measures the LLM without any verdicts supplied. The ranking half has no measurement at all:
# verdicts_k10_hint exists (job vf_hint 19986335, parse_rate 1.0, window_source=fused,
# symbolic_hint=True) but was never reranked.
#
# This closes the ranking half. MODE=llm_drop matches tab:ablation's "AGS (full)" row
# (rerank_no_determ_k10fused), so the two are directly comparable and the hint is the only
# variable. Scoring with llm_drop also keeps the symbolic layer out of the score entirely: the
# verdicts enter as prompt context and nothing else, which is what the claim says.
#
# CAUTION on the expected result. verdicts_k10_hint's own summary note records a near-zero
# firing rate (Appendix H found 5/1,160 FAMILY opportunities), and predicts "little difference
# from AGS full for a real reason -- abstention, not disagreement". So a null result here is
# informative, not a bug. Do not read a null as a failed run without checking parse_rate first.
#
# Stage 1 (CPU dump) runs locally per the project rule; stage 2 (GPU rerank) is sbatched once
# its input is on disk.
set -uo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${FHS_ROOT}"
RUN="${REPO}/runs/runs_ags_verifier_ablation/qwen3_32b"
VERDICTS="${RUN}/verdicts_k10_judge6/llm_verifier_verdicts.json"
ARM=no_determ
OUT="${RUN}/rerank_no_determ_k10judge6"
RANKING="${OUT}/bm25_candidates.jsonl"
cd "${REPO}"

if [[ ! -f "${VERDICTS}" ]]; then
  echo "hint verdicts missing: ${VERDICTS}" >&2; exit 1
fi

mkdir -p "${OUT}"

if [[ ! -f "${RANKING}" ]]; then
  echo "[$(date +%H:%M:%S)] dumping judge6 arm (all 6 dims judged) (mode=llm_drop) against verdicts_k10_judge6"
  python "${REPO}/src/verifier/dump_reranked_ranking.py" \
    --verifier-mode llm_drop \
    --beta 0.6 \
    --verdicts "${VERDICTS}" \
    --output "${RANKING}.local_partial" \
    --summary "${OUT}/ranking_summary.json"
  if [[ $? -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] dump FAILED, not submitting" >&2; exit 1
  fi
  mv "${RANKING}.local_partial" "${RANKING}"
  echo "[$(date +%H:%M:%S)] ranking staged -> ${RANKING}"
else
  echo "[$(date +%H:%M:%S)] ranking already staged, reusing"
fi

echo "[$(date +%H:%M:%S)] submitting GPU rerank"
sbatch --job-name="rr_judge6_k10" \
  --export=ALL,ARM="${ARM}",OUTPUT_DIR="${OUT}",VERDICTS="${VERDICTS}" \
  "${REPO}/scripts/slurm/apply_server_verifier_ablation_rerank.sh"
