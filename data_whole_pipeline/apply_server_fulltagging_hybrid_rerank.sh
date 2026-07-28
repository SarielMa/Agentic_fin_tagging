#!/bin/bash
#SBATCH --job-name=full_hybrid_rerank
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_full_hybrid_rerank.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# tab:end_to_end's "AGS (hybrid verification)" Full-pipeline row -- the one cell in that table
# with no measurement behind it.
#
# The pipeline never applies the verifier itself (run_fintagging_grounding_baseline.py contains
# the string "verifier" zero times). The deployed hybrid numbers are produced the same way the
# bridge produced its gold-instance ones: apply the verifier on CPU to build a reranked
# candidate file, then feed that back through --reuse-candidates --run-rerank. This does that
# for the extractor-driven (fulltagging) pool instead of the gold-instance pool.
#
# Stage 0 (separate job, this one depends on it): verdicts over the fulltagging candidates.
#   The existing verdicts cover the gold-instance trace and are NOT reusable here -- different
#   candidates, different hypotheses.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/.bashrc" ]]; then set +u; source "${HOME}/.bashrc"; set -u; fi
conda activate finben_b200

SRC="${SRC:-${REPO_ROOT}/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_frozen_ags}"
VERDICTS_DIR="${VERDICTS_DIR:-${REPO_ROOT}/runs_ags_verifier_ablation/qwen3_32b/verdicts_fulltagging}"
OUT="${OUT:-${REPO_ROOT}/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_hybrid_verification}"

VERDICTS="${VERDICTS_DIR}/llm_verifier_verdicts.json"
if [[ ! -f "${VERDICTS}" ]]; then
  echo "MISSING: ${VERDICTS} -- stage 0 (vf_full) has not produced verdicts." >&2
  exit 1
fi

mkdir -p "${OUT}"
RANKING="${OUT}/bm25_candidates.jsonl"

# Stage 1 (CPU): materialise the verifier-reranked ranking at the path the reranker reads.
# .partial + mv so a requeue can never see a half-written multi-GB file as complete.
if [[ ! -f "${RANKING}" ]]; then
  echo "--- dumping hybrid-verified ranking over the fulltagging pool ---"
  python "${REPO_ROOT}/ags_table5_ablation/dump_reranked_ranking.py" \
    --test-trace "${SRC}/bm25_candidates.jsonl" \
    --verdicts "${VERDICTS}" \
    --verifier-mode hybrid \
    --beta 0.6 \
    --llm-unjudged-fill mean \
    --output "${RANKING}.partial" \
    --summary "${OUT}/ranking_summary.json"
  mv "${RANKING}.partial" "${RANKING}"
else
  echo "--- ranking already staged, reusing ---"
fi

# The extractor outputs must travel with the ranking: the fulltagging scorer scores against
# gold entities, and its denominator comes from these, not from the candidate file.
for f in extraction_predictions.jsonl extracted_grounding_input.jsonl \
         fulltagging_input_metadata.json extraction_prediction_metadata_table.json \
         extraction_prediction_metadata_text.json; do
  [[ -f "${SRC}/${f}" && ! -f "${OUT}/${f}" ]] && cp "${SRC}/${f}" "${OUT}/${f}"
done

# Stage 2 (GPU): the identical listwise reranker every other row of tab:end_to_end uses.
echo "--- fulltagging listwise rerank ---"
export RUNS_ROOT="${REPO_ROOT}/runs_fintagging_fulltagging"
export EXTRACTOR_TAG="qwen2.5_14b_extractors"
export QUERY_MODE="frozen_ags"
export OUTPUT_DIR="${OUT}"
export RUN_RERANK=1
export REUSE_CANDIDATES=1
export RESUME=1
bash "${REPO_ROOT}/apply_server_fintagging_fulltagging_direct_retrieval.sh"

echo "Done -> ${OUT}/fulltagging_metrics.json"
