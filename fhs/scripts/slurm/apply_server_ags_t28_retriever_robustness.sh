#!/bin/bash
#SBATCH --job-name=ags_t28_retriever_robustness
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu_h200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs%j_ags_t28_retriever_robustness.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# T28 retriever robustness (comparing_methods/ags_t7_t28_spec.md): does AGS's advantage survive
# a non-sparse retriever? Builds a dense index and a hybrid (BM25 + dense RRF) index over the
# same 17,388-concept taxonomy, then replays one-pass grounding's and AGS's ALREADY-LOGGED
# query strings against them. No generation of any kind -- only retrieval and consolidation.
#
# The GPU is used solely to embed 17k concepts and the reused query strings (minutes). The
# consolidation path is ags_table5_ablation.core.evaluate at the frozen AGS config, unchanged.
#
# IMPORTANT -- stage the embedding model first. Compute nodes here run with HF_HUB_OFFLINE=1,
# and no BGE/E5 model is cached yet. From a login node (which does have network), run once:
#
#   export HF_HOME=/nfs/roberts/scratch/pi_sjf37/lm2445/.cache/huggingface
#   python -c "from sentence_transformers import SentenceTransformer; \
#              SentenceTransformer('BAAI/bge-large-en-v1.5')"
#
# then submit this script. Set DENSE_MODEL to use a different one (pick one, do not sweep --
# the spec is explicit about that).
#
# WHY h200 AND NOT b200: sentence-transformers lives only in the `finben` env, whose torch
# (2.7.1+cu126) is compiled for sm_90 and nothing newer -- verified against libtorch_cuda.so.
# A B200 is sm_100, so `finben` cannot run on one; that mismatch is the whole reason the
# separate finben_b200 env (torch 2.9.0+cu128, sm_100/sm_120) exists, and that env in turn has
# no sentence-transformers. An H200 is Hopper = sm_90, so `finben` runs on it unmodified. This
# also keeps the job off the b200 queue the other AGS jobs are contending for.
# If you would rather run on b200, install into that env instead -- every dependency is
# already present there, so it needs no resolver work and touches nothing else:
#   conda activate finben_b200 && pip install --no-deps sentence-transformers
# then set PARTITION/GRES back to b200 and CONDA_ENV=finben_b200.
#
# Reads other experiments' traces read-only; writes only to its own runs_ags_t7_t28/ tree.
#
# Examples:
#   sbatch apply_server_ags_t28_retriever_robustness.sh
#   sbatch --export=ALL,LIMIT=25 apply_server_ags_t28_retriever_robustness.sh   # smoke test
#   sbatch --export=ALL,DENSE_MODEL=BAAI/bge-small-en-v1.5 apply_server_ags_t28_retriever_robustness.sh

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

# sentence-transformers lives in `finben`, not `finben_b200`.
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
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-large-en-v1.5}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
RRF_KAPPA="${RRF_KAPPA:-60.0}"
BETA="${BETA:-0.6}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

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
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DENSE_MODEL=${DENSE_MODEL}"
echo "RRF_KAPPA=${RRF_KAPPA}   BETA=${BETA} (not re-tuned, per spec)"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

python -m py_compile ags_t7_t28/dense_index.py ags_t7_t28/run_t28_retriever_robustness.py

python ags_t7_t28/run_t28_retriever_robustness.py \
  --output-dir "${OUTPUT_DIR}" \
  --dense-model "${DENSE_MODEL}" \
  --embed-batch-size "${EMBED_BATCH_SIZE}" \
  --rrf-kappa "${RRF_KAPPA}" \
  --beta "${BETA}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  "${EXTRA_ARGS[@]}"
