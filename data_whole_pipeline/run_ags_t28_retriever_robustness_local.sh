#!/bin/bash
# T28 retriever robustness, stage B (the full 8-row grid) -- LOCAL runner (no Slurm).
#
# NO GPU. Stage A (apply_server_ags_t28_embed_cache.sh) already wrote the embedding cache, and
# with a populated cache the dense index never loads -- never even imports -- the 335M-parameter
# sentence-transformer. Dense search is an exact cosine matmul on CPU (17k x 1024 per query,
# a few ms). >99% of the runtime is core.evaluate consolidation, the Eq. 10 coverage rescore
# and BM25: all single-threaded Python.
#
# RUNTIME IS ~3 HOURS. That is a long time to hold a login/interactive shell, and some clusters
# reap long-running login-node processes. Prefer detaching:
#
#   nohup ./run_ags_t28_retriever_robustness_local.sh > t28.log 2>&1 &
#   tail -f t28.log
#
# ...or just use the Slurm version (apply_server_ags_t28_retriever_robustness_cpu.sh), which is
# the same job with the same parameters.
#
# Every parameter is baked in below, so this runs with no prefix and no arguments:
#
#   ./run_ags_t28_retriever_robustness_local.sh
#   LIMIT=25 ./run_ags_t28_retriever_robustness_local.sh    # ~1 min smoke test
#
# Scientific settings are identical to the Slurm script: same queries replayed verbatim from
# the logged traces, same core.evaluate at the frozen config, beta NOT re-tuned, same RRF kappa,
# same bootstrap seed.
#
# OVERWRITES retriever_robustness.csv, retriever_robustness_deltas.csv and metrics.json in
# OUTPUT_DIR. Back them up first if you want to diff against the previous four-metric run.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline}"
CONDA_ENV="${CONDA_ENV:-finben}"

# ---- baked-in parameters (match apply_server_ags_t28_retriever_robustness_cpu.sh) ----------
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_t7_t28/qwen3_32b}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-${OUTPUT_DIR}/dense_embeddings.pt}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-large-en-v1.5}"
RRF_KAPPA="${RRF_KAPPA:-60.0}"
BETA="${BETA:-0.6}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

# ---- conda ---------------------------------------------------------------------------------
if [[ -z "${CONDA_EXE:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
fi
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  CANDIDATE="${HOME}/.conda/envs/${CONDA_ENV}/bin"
  if [[ -x "${CANDIDATE}/python" ]]; then
    export PATH="${CANDIDATE}:${PATH}"
  else
    echo "Could not activate ${CONDA_ENV}; activate it yourself and re-run." >&2
    exit 1
  fi
fi

echo "python: $(command -v python)"
python --version

# ---- caches and threading ------------------------------------------------------------------
export SCRATCH_ROOT="${SCRATCH_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
mkdir -p "${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Only feeds the torch matmuls; the dominant cost is single-threaded Python, so raising this
# buys almost nothing.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

cd "${REPO_ROOT}"

if [[ ! -f "${EMBEDDING_CACHE}" ]]; then
  echo "Missing embedding cache ${EMBEDDING_CACHE}." >&2
  echo "Run stage A first (needs a GPU): sbatch apply_server_ags_t28_embed_cache.sh" >&2
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
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EMBEDDING_CACHE=${EMBEDDING_CACHE}"
echo "CONDA_ENV=${CONDA_ENV}   DEVICE=cpu (no GPU)"
echo "RRF_KAPPA=${RRF_KAPPA}   BETA=${BETA} (not re-tuned, per spec)"
echo "BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES}"
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
