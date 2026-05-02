#!/bin/bash
# =============================================================================
# FlowGRPO SD3 multi-node — colocated rollout (SGLang engine) GRPO, colocate mode
# Recipe: conf/experiment/flowgrpo_sglang_sd3_colocate.yaml
#
# Train and rollout actors time-share the same GPUs; offload_train and
# offload_rollout swap them on/off device. No GPU split env vars.
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_multinode.sh                # auto
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_multinode.sh head
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_multinode.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_sglang_sd3_colocate_multinode}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-Flow-GRPO-sglang-colocate-multinode}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/flowgrpo_sglang_sd3_colocate_multinode}"

exec "${SCRIPT_DIR}/../scripts/train.sh" flowgrpo_sglang_sd3_colocate "$@"
