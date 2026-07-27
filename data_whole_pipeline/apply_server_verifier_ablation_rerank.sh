#!/bin/bash
#SBATCH --job-name=verifier_abl_rerank
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_verifier_abl_%x.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# FINAL RERANKED ACCURACY FOR ONE VERIFIER ABLATION ARM.
#
# run_ags_verifier_ablation.py settles every arm at the RETRIEVAL stage on CPU. This job adds
# the column that stage cannot produce: accuracy after the listwise reranker the deployed
# pipeline applies to the top 20. One submission per arm.
#
#   sbatch --job-name=hybrid_full  --export=ALL,ARM=hybrid_full  apply_server_verifier_ablation_rerank.sh
#   sbatch --job-name=no_llm       --export=ALL,ARM=no_llm       apply_server_verifier_ablation_rerank.sh
#   sbatch --job-name=no_determ    --export=ALL,ARM=no_determ    apply_server_verifier_ablation_rerank.sh
#   sbatch --job-name=llm_only     --export=ALL,ARM=llm_only     apply_server_verifier_ablation_rerank.sh
#   sbatch --job-name=no_verifier  --export=ALL,ARM=no_verifier  apply_server_verifier_ablation_rerank.sh
#   sbatch --job-name=hybrid_m5    --export=ALL,ARM=hybrid_m5    apply_server_verifier_ablation_rerank.sh
#
# Smoke test first:  sbatch --export=ALL,ARM=hybrid_full,LIMIT=50 ...
#
# THE 2x2 INTERACTION STUDY IS A SUBSET OF THESE ARMS, NOT EXTRA RUNS
#   reranker off, LLM off -> no_llm's retrieval-stage row (CPU, already computed)
#   reranker off, LLM on  -> hybrid_full's retrieval-stage row (CPU, already computed)
#   reranker on,  LLM off -> this job with ARM=no_llm
#   reranker on,  LLM on  -> this job with ARM=hybrid_full
#
# PAIRING
#   Every arm's ranking is dumped from the SAME frozen-AGS trace, so all arms share one
#   hypothesis set and one candidate pool per fact and differ only in the rerank term. That
#   pairing does NOT extend to runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags: the
#   dump re-fuses the full logged pool through the ablation core, which reaches R@200 0.7182
#   against the deployed run's 0.7055. Compare arms to each other here, not to the deployed
#   AGS row in the main table.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
ARM="${ARM:?set ARM to one of hybrid_full|no_llm|no_determ|llm_only|no_verifier|hybrid_m5}"

# arm -> dump flags. Two flags carry the whole ablation: which verifier supplies the rerank
# term, and whether the term is applied at all (beta).
case "${ARM}" in
  hybrid_full) MODE=hybrid;        BETA=0.6; TOP_M_FLAG="" ;;
  no_llm)      MODE=deterministic; BETA=0.6; TOP_M_FLAG="" ;;
  no_determ)   MODE=llm_drop;      BETA=0.6; TOP_M_FLAG="" ;;
  llm_only)    MODE=llm_strict;    BETA=0.6; TOP_M_FLAG="" ;;
  no_verifier) MODE=deterministic; BETA=0.0; TOP_M_FLAG="" ;;
  hybrid_m5)   MODE=hybrid;        BETA=0.6; TOP_M_FLAG="--top-m 5" ;;
  *) echo "Unknown ARM=${ARM}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_verifier_ablation/qwen3_32b/rerank_${ARM}}"
RANKING="${OUTPUT_DIR}/bm25_candidates.jsonl"

export TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/FinTagging_800_200_grounding_test_JSON/data/test.jsonl}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"
export CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
export CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-12000}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Verifier ablation arm: ${ARM}"
echo "  verifier_mode=${MODE}  beta=${BETA}  ${TOP_M_FLAG:-(window: as generated)}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  LIMIT=${LIMIT:-<none>}"
echo "============================================================"

# Stage 1 (CPU): materialise this arm's ranking directly at the path the reranker reads.
# Skipped if it already exists, so a requeued job does not redo it.
if [[ ! -f "${RANKING}" ]]; then
  echo "--- dumping ranking for ${ARM} ---"
  python "${REPO_ROOT}/ags_table5_ablation/dump_reranked_ranking.py" \
    --verifier-mode "${MODE}" \
    --beta "${BETA}" \
    ${TOP_M_FLAG} \
    --output "${RANKING}" \
    --summary "${OUTPUT_DIR}/ranking_summary.json" \
    ${LIMIT:+--limit "${LIMIT}"}
else
  echo "--- ranking already staged at ${RANKING}, reusing ---"
fi

# Stage 2 (GPU): the identical listwise reranker every other reported number uses.
# --query-mode must match the staged file's own query_mode tag, which the dump preserves.
echo "--- listwise rerank for ${ARM} ---"
python "${REPO_ROOT}/run_fintagging_grounding_baseline.py" \
  --test-jsonl "${TEST_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --query-mode frozen_ags \
  --reuse-candidates \
  --run-rerank \
  --rerank-model "${RERANK_MODEL}" \
  --rerank-backend "${RERANK_BACKEND}" \
  --rerank-list-size "${RERANK_LIST_SIZE}" \
  ${LIMIT:+--limit "${LIMIT}"}

echo "Done: ${ARM} -> ${OUTPUT_DIR}/metrics.json"
