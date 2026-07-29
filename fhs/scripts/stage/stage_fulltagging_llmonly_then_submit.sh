#!/bin/bash
# Stage 1 of the fulltagging LLM-only arm locally (CPU-only verifier rescoring), then submit
# stage 2 (GPU listwise rerank). Keeps the CPU half off the GPU queue, where it would otherwise
# hold a B200 idle for its duration.
#
# The dump writes to a temp path and is mv'd into place, so the sbatch'd job either finds a
# complete ranking and reuses it, or finds nothing and redoes it -- never a partial file.
set -euo pipefail

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"
cd "${REPO_ROOT}"

SRC="${REPO_ROOT}/runs/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_frozen_ags"
VERDICTS_DIR="${REPO_ROOT}/runs/runs_ags_verifier_ablation/qwen3_32b/verdicts_fulltagging"
OUT="${REPO_ROOT}/runs/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_llmonly_verification"
RANKING="${OUT}/bm25_candidates.jsonl"
mkdir -p "${OUT}"

if [[ ! -f "${RANKING}" ]]; then
  echo "[$(date +%H:%M:%S)] rescoring fulltagging pool at verifier_mode=llm_drop (CPU, local)"
  python "${REPO_ROOT}/src/verifier/dump_reranked_ranking.py" \
    --test-trace "${SRC}/bm25_candidates.jsonl" \
    --verdicts "${VERDICTS_DIR}/llm_verifier_verdicts.json" \
    --verifier-mode llm_drop \
    --beta 0.6 \
    --llm-unjudged-fill mean \
    --output "${RANKING}.local_partial" \
    --summary "${OUT}/ranking_summary.json"
  mv "${RANKING}.local_partial" "${RANKING}"
  echo "[$(date +%H:%M:%S)] ranking staged -> ${RANKING}"
else
  echo "[$(date +%H:%M:%S)] ranking already present, skipping rescoring"
fi

# The extractor outputs must travel with the ranking: the fulltagging scorer's denominator
# comes from these, not from the candidate file.
for f in extraction_predictions.jsonl extracted_grounding_input.jsonl \
         fulltagging_input_metadata.json extraction_prediction_metadata_table.json \
         extraction_prediction_metadata_text.json; do
  [[ -f "${SRC}/${f}" && ! -f "${OUT}/${f}" ]] && cp "${SRC}/${f}" "${OUT}/${f}"
done

echo "[$(date +%H:%M:%S)] submitting GPU rerank"
sbatch "${REPO_ROOT}/scripts/slurm/fulltagging/apply_server_fulltagging_llmonly_rerank.sh"
