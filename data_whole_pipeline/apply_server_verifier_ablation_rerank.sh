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
#   dump runs at the deployed truncation order, under which the deterministic arm reproduces
#   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags's retrieval stage to 0.000000. So
#   ARM=no_llm must come back at accuracy 0.2375; anything else means this path diverges from
#   the deployed pipeline and the other arms should not be trusted.

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"
# Environment, copied verbatim from apply_server_ags_bridge_deployed_rerank.sh. The B200 needs
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

ARM="${ARM:?set ARM to one of hybrid_full|no_llm|no_determ|llm_only|no_verifier|hybrid_m5}"

# arm -> dump flags. Two flags carry the whole verifier ablation: which verifier supplies the
# rerank term, and whether the term is applied at all (beta). The diagnostic arms below add
# AblationConfig knobs in EXTRA_FLAGS instead; they are all deterministic-core, so MODE is
# deterministic and no verdicts are read.
#
# Each diagnostic arm's beta is the one run_test_rows.py selected for it
# (runs_ags_table5_ablation/qwen32b_fixed/selected_betas.json), NOT a blanket 0.6 -- using 0.6
# everywhere would rerank a ranking that no row of tab:ablation reports.
EXTRA_FLAGS=""
case "${ARM}" in
  hybrid_full) MODE=hybrid;        BETA=0.6; TOP_M_FLAG="" ;;
  no_llm)      MODE=deterministic; BETA=0.6; TOP_M_FLAG="" ;;
  no_determ)   MODE=llm_drop;      BETA=0.6; TOP_M_FLAG="" ;;
  llm_only)    MODE=llm_strict;    BETA=0.6; TOP_M_FLAG="" ;;
  no_verifier) MODE=deterministic; BETA=0.0; TOP_M_FLAG="" ;;
  hybrid_m5)   MODE=hybrid;        BETA=0.6; TOP_M_FLAG="--top-m 5" ;;

  # tab:ablation diagnostic rows. Verified against ablation.csv's pooled retrieval-stage
  # numbers, so each maps to the variant the paper actually prints:
  label_form)      MODE=deterministic; BETA=0.8; TOP_M_FLAG=""; EXTRA_FLAGS="--renderings def" ;;
  definition_form) MODE=deterministic; BETA=0.2; TOP_M_FLAG=""
                   EXTRA_FLAGS="--renderings lab --lab-only-fallback none" ;;
  mean_fusion)     MODE=deterministic; BETA=0.6; TOP_M_FLAG=""; EXTRA_FLAGS="--fusion mean" ;;
  raw_scaling)     MODE=deterministic; BETA=0.6; TOP_M_FLAG=""; EXTRA_FLAGS="--scaling raw" ;;
  oracle_single)   MODE=deterministic; BETA=0.6; TOP_M_FLAG=""; EXTRA_FLAGS="--oracle-best-single" ;;
  # -ensemble is the mean of idx0 and idx1, so it is two runs that are averaged afterwards.
  ensemble_idx0)   MODE=deterministic; BETA=0.6; TOP_M_FLAG=""
                   EXTRA_FLAGS="--n-hypotheses 1 --kept-hypothesis-idx 0" ;;
  ensemble_idx1)   MODE=deterministic; BETA=0.6; TOP_M_FLAG=""
                   EXTRA_FLAGS="--n-hypotheses 1 --kept-hypothesis-idx 1" ;;
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
  # VERDICTS overrides dump_reranked_ranking.py's default verdicts file. That default is the
  # 2026-07-25 generation, whose summary has no window_source field because it predates the
  # --window-source fused fix -- i.e. its top-K_v window was cut from the deterministically
  # reranked final_candidates. Every rerank_* dir produced before 2026-07-28 used it, so the
  # Final Acc. column of the LLM-consuming arms was confounded even after the retrieval-stage
  # numbers were regenerated. Point this at verdicts_k10_fused to get a clean Final column.
  python "${REPO_ROOT}/ags_table5_ablation/dump_reranked_ranking.py" \
    --verifier-mode "${MODE}" \
    --beta "${BETA}" \
    ${TOP_M_FLAG} \
    ${EXTRA_FLAGS} \
    ${VERDICTS:+--verdicts "${VERDICTS}"} \
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
