#!/bin/bash
set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline"

COMPONENT_SCRIPT="${COMPONENT_SCRIPT:-${REPO_ROOT}/apply_server_ags_component_validation.sh}"
REWARD_SCRIPT="${REWARD_SCRIPT:-${REPO_ROOT}/apply_server_ags_reward_diagnostic.sh}"

COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR:-${REPO_ROOT}/runs_ags_component_validation/qwen3_32b}"
REWARD_OUTPUT_DIR="${REWARD_OUTPUT_DIR:-${REPO_ROOT}/runs_ags_reward_diagnostic/qwen3_32b}"

COMPONENT_OVERWRITE="${COMPONENT_OVERWRITE:-1}"
REWARD_OVERWRITE="${REWARD_OVERWRITE:-0}"
COMPONENT_RESUME="${COMPONENT_RESUME:-1}"
REWARD_RESUME="${REWARD_RESUME:-1}"
DRY_RUN_NO_LLM="${DRY_RUN_NO_LLM:-0}"

cd "${REPO_ROOT}"

for script in "${COMPONENT_SCRIPT}" "${REWARD_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "Missing script: ${script}" >&2
    exit 1
  fi
done

component_submit="$(
  sbatch --parsable \
    --export=ALL,OUTPUT_DIR="${COMPONENT_OUTPUT_DIR}",OVERWRITE="${COMPONENT_OVERWRITE}",RESUME="${COMPONENT_RESUME}",DRY_RUN_NO_LLM="${DRY_RUN_NO_LLM}" \
    "${COMPONENT_SCRIPT}"
)"
component_job="${component_submit%%;*}"

reward_submit="$(
  sbatch --parsable \
    --dependency=afterok:"${component_job}" \
    --export=ALL,OUTPUT_DIR="${REWARD_OUTPUT_DIR}",COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR}",OVERWRITE="${REWARD_OVERWRITE}",RESUME="${REWARD_RESUME}",DRY_RUN_NO_LLM="${DRY_RUN_NO_LLM}" \
    "${REWARD_SCRIPT}"
)"
reward_job="${reward_submit%%;*}"

echo "Submitted AGS component validation job: ${component_job}"
echo "Submitted AGS reward diagnostic job  : ${reward_job}"
echo "Reward diagnostic dependency         : afterok:${component_job}"
echo "Component output                     : ${COMPONENT_OUTPUT_DIR}"
echo "Reward output                        : ${REWARD_OUTPUT_DIR}"
