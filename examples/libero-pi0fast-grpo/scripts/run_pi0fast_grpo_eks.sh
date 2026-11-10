#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# run_pi0fast_grpo_eks.sh -- LIBERO-10 pi0-FAST GRPO on EKS (KubeRay RayJob entrypoint)
#
# Mirrors upstream examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast,
# with the artifact paths staged by manifests/model-download.yaml. Runs on the Ray
# head of a 2-node RayCluster (2x p5en.48xlarge = 16 GPUs, upstream's reference
# config); RLinf connects to the running cluster (ray.init(address="auto")).
#
# Environment (set by the RayJob):
#   CONFIG_NAME      upstream config (default libero_10_grpo_pi0_fast)
#   MODEL_PATH       base policy (lerobot/pi0fast-libero)
#   TEXT_TOKENIZER   google/paligemma-3b-pt-224
#   ACTION_TOKENIZER jadechoghari/tokenizer-lib-mean
#   CKPT_PATH / EXPERIMENT_NAME   log + checkpoint dir = CKPT_PATH/EXPERIMENT_NAME
#   NUM_NODES        Ray nodes (default 2)
#   HYDRA_OVERRIDES  extra Hydra overrides (e.g. runner.max_steps=2 for a smoke run)
# =============================================================================
set -euo pipefail
set -x

VENV_NAME="${VENV_NAME:-pi0_fast}"
CONFIG_NAME="${CONFIG_NAME:-libero_10_grpo_pi0_fast}"
MODEL_PATH="${MODEL_PATH:-/fsx/models/pi0fast-libero}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-/fsx/models/paligemma-3b-pt-224}"
ACTION_TOKENIZER="${ACTION_TOKENIZER:-/fsx/models/tokenizer-lib-mean}"
CKPT_PATH="${CKPT_PATH:-/fsx/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero-pi0fast-grpo}"
NUM_NODES="${NUM_NODES:-2}"

export PYTHONPATH="${PYTHONPATH:-}"
UV_PATH="${UV_PATH:-/opt/venv}"
# shellcheck disable=SC1090
source "${UV_PATH}/${VENV_NAME}/bin/activate"
echo "Python: $(command -v python) $(python -c 'import sys,torch; print(sys.version.split()[0], torch.__version__)')"

for p in "${MODEL_PATH}" "${TEXT_TOKENIZER}" "${ACTION_TOKENIZER}"; do
  [ -e "${p}" ] || { echo "ERROR: ${p} not found; run manifests/model-download.yaml first"; exit 1; }
done

# Same environment as upstream run_embodiment.sh (LIBERO renders headless with EGL there;
# MUJOCO_GL can be overridden).
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export REPO_PATH=/workspace/RLinf
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"

LOG_DIR="${CKPT_PATH}/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

HYDRA_ARGS="runner.logger.log_path=${LOG_DIR}"
HYDRA_ARGS="${HYDRA_ARGS} cluster.num_nodes=${NUM_NODES}"
HYDRA_ARGS="${HYDRA_ARGS} actor.model.model_path=${MODEL_PATH}"
HYDRA_ARGS="${HYDRA_ARGS} actor.model.pi0_fast.text_tokenizer_name=${TEXT_TOKENIZER}"
HYDRA_ARGS="${HYDRA_ARGS} actor.model.pi0_fast.action_tokenizer_name=${ACTION_TOKENIZER}"
if [ -n "${HYDRA_OVERRIDES:-}" ]; then
  HYDRA_ARGS="${HYDRA_ARGS} ${HYDRA_OVERRIDES}"
fi

cd "${REPO_PATH}"
echo "=== pi0-FAST GRPO: ${CONFIG_NAME}, ${NUM_NODES} nodes ==="
echo "Hydra args: ${HYDRA_ARGS}"
# shellcheck disable=SC2086
python "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  ${HYDRA_ARGS} 2>&1 | tee -a "${LOG_DIR}/train-$(date -u +%Y%m%dT%H%M%SZ).log"
