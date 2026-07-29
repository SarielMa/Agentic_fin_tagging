#!/bin/bash
#SBATCH --job-name=ags_t28_retriever_robustness_cpu
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=day
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_t28_retriever_robustness_cpu.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# T28 stage B of two: the full 8-row grid, GPU-free. REQUIRES stage A
# (apply_server_ags_t28_embed_cache.sh) to have written the embedding cache first.
#
# NO --gpus LINE, DELIBERATELY. The idle-GPU auto-cancel policy is what killed the single-job
# version, and it was a fair kill: measured on job 19529572, every grid row is CPU-bound.
#   bm25/one_pass    6:34     dense/one_pass    0:01 (pure GPU retrieval, all 2509 facts)
#   bm25/AGS        37:06     dense/AGS/cov=off 14:37 (same retrieval + CPU consolidation)
#                             dense/AGS/cov=on  39:04
# The gap between 0:01 and 14:37 is the whole story: >99% of an AGS row is core.evaluate
# consolidation, the Eq. 10 coverage rescore and BM25 -- all single-threaded Python. Moving
# that to a CPU node changes the runtime by minutes and removes the cancellation risk entirely.
#
# The dense rows still do exact cosine search, now as a CPU matmul: 17k x 1024 per query, a few
# ms, against a per-entity-type submatrix sliced once at startup instead of re-sliced per call.
#
# `day` allows 24h and this needs ~3h. Runtime is dominated by single-threaded Python, so
# raising --cpus-per-task past 8 buys almost nothing; it only feeds the torch matmuls.
#
#   sbatch apply_server_ags_t28_retriever_robustness_cpu.sh
#   sbatch --export=ALL,LIMIT=25 apply_server_ags_t28_retriever_robustness_cpu.sh   # smoke test
#
# Everything scientific is unchanged from the GPU version: same queries replayed verbatim from
# the logged traces, same core.evaluate at the frozen config, same beta (not re-tuned), same
# RRF kappa, same bootstrap seed. Embeddings are the identical fp32 tensors stage A produced,
# so the dense and hybrid numbers are the same ones the GPU run was computing.

FHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${FHS_ROOT}"

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
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-finben}"
conda activate "${CONDA_ENV}"
which python
python --version

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
mkdir -p "${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Keep torch's thread pool inside the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
# No accelerator is allocated; make that unambiguous to torch rather than relying on the node.
export CUDA_VISIBLE_DEVICES=""

cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_t7_t28/qwen3_32b}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-${OUTPUT_DIR}/dense_embeddings.pt}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-large-en-v1.5}"
RRF_KAPPA="${RRF_KAPPA:-60.0}"
BETA="${BETA:-0.6}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

if [[ ! -f "${EMBEDDING_CACHE}" ]]; then
  echo "Missing embedding cache ${EMBEDDING_CACHE}." >&2
  echo "Run stage A first:  sbatch apply_server_ags_t28_embed_cache.sh" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${SKIP_BM25:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-bm25)
fi
if [[ -n "${QUERY_INSTRUCTION:-}" ]]; then
  EXTRA_ARGS+=(--query-instruction "${QUERY_INSTRUCTION}")
fi

echo "============================================================"
echo "STAGE B (CPU, no GPU allocated)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EMBEDDING_CACHE=${EMBEDDING_CACHE}"
echo "RRF_KAPPA=${RRF_KAPPA}   BETA=${BETA} (not re-tuned, per spec)"
echo "LIMIT=${LIMIT:-<none>}   THREADS=${OMP_NUM_THREADS}"
echo "============================================================"

python -m py_compile ags_t7_t28/dense_index.py ags_t7_t28/run_t28_retriever_robustness.py

python ags_t7_t28/run_t28_retriever_robustness.py \
  --output-dir "${OUTPUT_DIR}" \
  --dense-model "${DENSE_MODEL}" \
  --device cpu \
  --embedding-cache "${EMBEDDING_CACHE}" \
  --rrf-kappa "${RRF_KAPPA}" \
  --beta "${BETA}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  "${EXTRA_ARGS[@]}"
