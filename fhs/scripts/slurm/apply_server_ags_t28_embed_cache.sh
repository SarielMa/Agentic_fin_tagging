#!/bin/bash
#SBATCH --job-name=ags_t28_embed_cache
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_h200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs/logs/%j_ags_t28_embed_cache.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# T28 stage A of two: embed the taxonomy and the replayed queries, write the cache, exit.
#
# WHY THIS IS A SEPARATE JOB. The single-job version of T28 held an H200 for ~3 hours and used
# it for ~30 seconds -- 0:20 to embed 17,388 concepts, 0:09 to embed 11,746 unique queries, and
# nothing after that. Every one of the 8 grid rows is CPU-bound (BM25 scoring, the Eq. 10
# coverage rescore, core.evaluate consolidation, the bootstrap); the measured proof is that
# dense/one_pass, which is pure GPU retrieval over all 2509 facts, finished in 1 SECOND while
# dense/AGS, doing the identical retrieval, took 14m37s. Cluster policy auto-cancels jobs that
# leave a GPU idle for 2 hours, and it was right to: the accelerator was doing nothing.
#
# So this job does the GPU half alone. It runs a few minutes, at real utilization, and exits.
# Stage B (apply_server_ags_t28_retriever_robustness_cpu.sh) loads the cache this writes and
# runs the whole grid on a CPU partition, where no idle-GPU rule applies and there is no
# accelerator to waste.
#
#   sbatch apply_server_ags_t28_embed_cache.sh
#   # then, once it succeeds:
#   sbatch apply_server_ags_t28_retriever_robustness_cpu.sh
#
# Build the cache with NO --limit even if stage B will be limited: the cache is keyed by query
# text, so a full cache serves every subset, while a limited one would force stage B to load
# sentence-transformers and encode the misses on CPU.
#
# Staging the model still applies -- compute nodes run HF_HUB_OFFLINE=1. From a login node:
#   export HF_HOME=/nfs/roberts/scratch/pi_sjf37/lm2445/.cache/huggingface
#   python -c "from sentence_transformers import SentenceTransformer; \
#              SentenceTransformer('BAAI/bge-large-en-v1.5')"

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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

# sentence-transformers lives in `finben`, not `finben_b200`. finben's torch is sm_90, which is
# exactly what an H200 (Hopper) is -- see the long note in the original T28 script.
CONDA_ENV="${CONDA_ENV:-finben}"
conda activate "${CONDA_ENV}"
which python
python --version

export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
mkdir -p "${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/runs_ags_t7_t28/qwen3_32b}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-${OUTPUT_DIR}/dense_embeddings.pt}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-large-en-v1.5}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "============================================================"
echo "STAGE A (GPU): embed only, then exit"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EMBEDDING_CACHE=${EMBEDDING_CACHE}"
echo "DENSE_MODEL=${DENSE_MODEL}"
echo "============================================================"

python -m py_compile src/efficiency/dense_index.py src/efficiency/run_t28_retriever_robustness.py

python src/efficiency/run_t28_retriever_robustness.py \
  --output-dir "${OUTPUT_DIR}" \
  --dense-model "${DENSE_MODEL}" \
  --embed-batch-size "${EMBED_BATCH_SIZE}" \
  --embedding-cache "${EMBEDDING_CACHE}" \
  --build-cache-only

echo "Stage A done. Now: sbatch apply_server_ags_t28_retriever_robustness_cpu.sh"
