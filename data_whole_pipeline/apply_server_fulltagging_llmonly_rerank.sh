#!/bin/bash
#SBATCH --job-name=full_llmonly_rerank
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_full_llmonly_rerank.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# tab:end_to_end's "AGS (full)" Full-pipeline row -- the one cell in that table with no
# measurement behind it.
#
# REPLACES apply_server_fulltagging_hybrid_rerank.sh (job 19968199), which was wrong twice:
#
#  1. It scored with --verifier-mode hybrid. Since the paper was rewritten around the LLM-only
#     result, tab:end_to_end's "AGS (full)" is the same configuration as tab:ablation's
#     "AGS (full)" row -- rerank_no_determ_k10fused, i.e. verifier_mode=llm_drop over a fused
#     window (R@1 0.183 / R@10 0.401 / MRR 0.257 / Acc 0.249). Filling that row from a hybrid
#     run would put a number under a label that does not describe it.
#  2. Its environment block was `source ~/.bashrc; conda activate finben_b200`. conda is not a
#     command on a compute node, so under `set -euo pipefail` that exits 127 in seconds -- the
#     identical failure that killed abl_wcov0 (19968987). The bootstrap below is the working one,
#     copied from apply_server_verifier_ablation_rerank.sh.
#
# The verdicts (verdicts_fulltagging, job vf_full 19968178) are reused unchanged: they already
# carry window_source=fused and parse_rate 1.0, so only the scoring mode differs.
#
# Stage 1 (the CPU dump) is staged locally by stage_fulltagging_llmonly_then_submit.sh; this
# script reuses the ranking if it is already on disk and otherwise rebuilds it.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"

# Environment, copied verbatim from apply_server_verifier_ablation_rerank.sh. The B200 needs
# the finben_b200 conda env: the default finben env's PyTorch is built for sm_50-sm_90 and
# aborts on this node's sm_100 during vLLM engine init. --export=ALL inherits the submitting
# shell's env, which is NOT enough -- the activation must happen inside the job.
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

module load miniconda
if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(cd "$(dirname "$(command -v conda)")/.." && pwd)/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi
conda activate finben_b200

cd "${REPO_ROOT}"

SRC="${SRC:-${REPO_ROOT}/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_frozen_ags}"
VERDICTS_DIR="${VERDICTS_DIR:-${REPO_ROOT}/runs_ags_verifier_ablation/qwen3_32b/verdicts_fulltagging}"
OUT="${OUT:-${REPO_ROOT}/runs_fintagging_fulltagging/qwen2.5_14b_extractors/qwen3_32b_llmonly_verification}"

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
  echo "--- dumping LLM-only-verified ranking over the fulltagging pool ---"
  python "${REPO_ROOT}/ags_table5_ablation/dump_reranked_ranking.py" \
    --test-trace "${SRC}/bm25_candidates.jsonl" \
    --verdicts "${VERDICTS}" \
    --verifier-mode llm_drop \
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
