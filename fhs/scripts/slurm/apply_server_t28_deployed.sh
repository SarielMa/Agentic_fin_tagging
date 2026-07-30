#!/bin/bash
#SBATCH --job-name=t28_deployed
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs/logs/%j_t28_deployed_%x.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# T28 retriever robustness, matched to the MAIN RESULTS TABLE.
#
#   sbatch --export=ALL,RETRIEVER=dense,METHOD=AGS       apply_server_t28_deployed.sh
#   sbatch --export=ALL,RETRIEVER=dense,METHOD=one_pass  apply_server_t28_deployed.sh
#   sbatch --export=ALL,RETRIEVER=hybrid,METHOD=AGS      apply_server_t28_deployed.sh
#   sbatch --export=ALL,RETRIEVER=hybrid,METHOD=one_pass apply_server_t28_deployed.sh
#
# BM25 is NOT run here: the deployed runs already exist, so those rows are read verbatim from
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json
#   runs_fintagging_grounding_baseline/qwen3_32b_one_pass_grounding/metrics.json
# and match the main table by construction.
#
# TWO STAGES, both on this node:
#   1. CPU-ish: replay the logged hypotheses (AGS) or logged query (one-pass) against the
#      dense/hybrid retriever via the DEPLOYED frozen_ags_rankings -> frozen_ags_rerank path.
#      No LLM generation. Uses the GPU for dense encoding.
#   2. GPU: the same listwise reranker the main table used, over that ranking.
# Only after stage 2 are the numbers comparable to the main table; stage 1 alone is the
# retrieval stage, which is exactly the mistake the previous T28 made.
#
# Coverage-term asymmetry is inherited from the deployed config on purpose: AGS runs at
# w_cov=1.0, one-pass at w_cov=0.0. That is what the main table's methods are. It is recorded
# in the stage-1 manifest.

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

RETRIEVER="${RETRIEVER:?set RETRIEVER=dense|hybrid}"
METHOD="${METHOD:?set METHOD=AGS|one_pass}"
case "${RETRIEVER}" in dense|hybrid) ;; *) echo "RETRIEVER must be dense|hybrid (bm25 is read from existing runs)" >&2; exit 1;; esac
case "${METHOD}" in AGS) QUERY_MODE=frozen_ags ;; one_pass) QUERY_MODE=one_pass_grounding ;; *) echo "METHOD must be AGS|one_pass" >&2; exit 1;; esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_t28_deployed/${RETRIEVER}_${METHOD}}"
EMBED_CACHE="${EMBED_CACHE:-${REPO_ROOT}/runs/runs_ags_t7_t28/qwen3_32b/dense_embeddings.pt}"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
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
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then unset -f conda || true; unset _FAKE_CONDA_FOR_PURGE; fi

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
  echo "Failed to initialize conda." >&2; exit 1
fi
conda activate finben_b200

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "T28 deployed-settings run"
echo "RETRIEVER=${RETRIEVER}  METHOD=${METHOD}  QUERY_MODE=${QUERY_MODE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EMBED_CACHE=${EMBED_CACHE}"
echo "Compare against the main results table (post-listwise-rerank)."
echo "============================================================"

# ---- stage 1: build the ranking with the deployed AGS / one-pass path -----------------
STAGED="${OUTPUT_DIR}/bm25_candidates.jsonl"
if [[ ! -s "${STAGED}" ]]; then
  echo ">>> stage 1: building ${RETRIEVER}/${METHOD} ranking"
  python src/efficiency/build_t28_deployed_candidates.py \
    --retriever-kind "${RETRIEVER}" \
    --method "${METHOD}" \
    --embedding-cache "${EMBED_CACHE}" \
    --device cuda \
    --output "${STAGED}" \
    ${LIMIT:+--limit "${LIMIT}"}
else
  echo ">>> stage 1 skipped: ${STAGED} already present"
fi

# ---- stage 2: the deployed listwise reranker -----------------------------------------
echo ">>> stage 2: listwise rerank (this is what makes it comparable to the main table)"
python run_fintagging_grounding_baseline.py \
  --test-jsonl "${REPO_ROOT}/data/test/test.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --query-mode "${QUERY_MODE}" \
  --reuse-candidates \
  --run-rerank \
  --rerank-model "${RERANK_MODEL:-Qwen/Qwen3-32B}" \
  --rerank-backend "${RERANK_BACKEND:-vllm}" \
  --rerank-list-size "${RERANK_LIST_SIZE:-20}" \
  ${LIMIT:+--limit "${LIMIT}"}

echo "Done: ${OUTPUT_DIR}/metrics.json  (read qwen_reranked for the main-table-comparable row)"
