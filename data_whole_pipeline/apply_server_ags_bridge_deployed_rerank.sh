#!/bin/bash
#SBATCH --job-name=ags_bridge_deployed_rerank
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline/%j_ags_bridge_deployed_rerank.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# THE ONE GPU STEP THE BRIDGE ANALYSIS CANNOT DO WITHOUT.
#
# run_ags_verifier_bridge.sh settles everything at the RETRIEVAL stage from existing logs, and
# reports the end-to-end cell as BLOCKED. This job unblocks it.
#
# WHY IT IS NEEDED
#   The Task 5 decision rule requires final tagging accuracy, not retrieval MRR. The deployed
#   pipeline applies a listwise reranker to the top 20; every number in the component-ablation
#   table is measured before that stage. The top-20 handoff diagnostic
#   (runs_ags_table5_ablation/qwen3_32b/verifier_top20_overlap.json) shows candidate-level LLM
#   reranking leaves the set the listwise stage receives identical on 55.9% of facts and moves
#   gold across the boundary on 11 of 2,509 -- so most of the +0.076 retrieval-stage top-1 gain
#   is reordering the later stage redoes. Whether any of it survives is an empirical question
#   only this rerun answers.
#
# WHAT IT DOES
#   Reruns the listwise reranker over the candidate-level-reranked ranking, holding everything
#   else fixed, and writes a metrics.json whose "qwen_reranked" block is directly comparable to
#   the without-reranking arm already in
#     runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json
#       (final tagging accuracy 0.2375, MRR 0.3176, R@10 0.4480)
#
#   sbatch apply_server_ags_bridge_deployed_rerank.sh
#   sbatch --export=ALL,LIMIT=50 apply_server_ags_bridge_deployed_rerank.sh   # smoke test
#
# AFTER IT FINISHES
#   Rerun ./run_ags_verifier_bridge.sh -- it picks the new metrics.json up via
#   --deployed-metrics and the decision rule moves off PROVISIONAL automatically.
#
# PREREQUISITE
#   The reranked ranking must exist. This script fails fast if it does not, naming the step
#   that produces it, rather than silently reranking the un-reranked pool and reporting a null.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"

RERANKED_RANKING="${RERANKED_RANKING:-${REPO_ROOT}/runs_ags_verifier_bridge/qwen3_32b/candidate_level_reranked_candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_verifier_bridge/qwen3_32b/deployed_rerank}"

if [[ ! -f "${RERANKED_RANKING}" ]]; then
  cat >&2 <<EOF
============================================================
MISSING INPUT: ${RERANKED_RANKING}

This job reranks the candidate-level-LLM-reranked ranking, but that ranking has not been
materialised yet. It is produced by re-running the offline ablation's evaluate() under
AblationConfig(llm_verifier_verdicts=...) and persisting the per-fact ranking rather than
only the aggregate metrics -- ags_table5_ablation/run_verifier_row_only.py computes exactly
this ranking today but keeps only the summary rows.

To materialise it, extend that script to dump the per-fact ranking to
${RERANKED_RANKING}, then resubmit this job. That step is CPU-only.
============================================================
EOF
  exit 1
fi

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
mkdir -p "${OUTPUT_DIR}"

# Every reranker setting matches the without-reranking arm so the two metrics.json files are
# directly comparable; only the input ranking differs.
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

echo "============================================================"
echo "Deployed-stage rerank over the candidate-level-reranked ranking"
echo "RERANKED_RANKING=${RERANKED_RANKING}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RERANK_MODEL=${RERANK_MODEL}  LIST_SIZE=${RERANK_LIST_SIZE}"
echo "Compare against runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json"
echo "  (without candidate-level reranking: qwen_reranked.accuracy = 0.2375)"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

# The pipeline reranks whatever sits at OUTPUT_DIR/bm25_candidates.jsonl when
# --reuse-candidates is passed (run_fintagging_grounding_baseline.py:3987, :4045). That is the
# supported way to rerank an externally produced ranking; there is no separate flag for it.
# Staged as a copy so the source ranking is never mutated by the run.
STAGED="${OUTPUT_DIR}/bm25_candidates.jsonl"
if [[ ! -f "${STAGED}" ]]; then
  echo "Staging ${RERANKED_RANKING} -> ${STAGED}"
  cp "${RERANKED_RANKING}" "${STAGED}"
fi

# --query-mode must match the staged file's own query_mode, or validate_candidate_records
# rejects it (run_fintagging_grounding_baseline.py:883). The ranking is frozen-AGS's, reordered
# by candidate-level LLM reranking; the mode tag is unchanged by that reordering.
python run_fintagging_grounding_baseline.py \
  --test-jsonl "${TEST_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --query-mode frozen_ags \
  --reuse-candidates \
  --run-rerank \
  --rerank-model "${RERANK_MODEL}" \
  --rerank-backend "${RERANK_BACKEND}" \
  --rerank-list-size "${RERANK_LIST_SIZE}" \
  ${LIMIT:+--limit "${LIMIT}"}

echo "Done. Then rerun: ./run_ags_verifier_bridge.sh"
