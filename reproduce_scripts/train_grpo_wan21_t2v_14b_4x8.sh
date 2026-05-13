#!/bin/bash
# =============================================================================
# GRPO WAN 2.1 T2V 14B — 4-node x 8-GPU direct-sampling (FSDP engine)
# Recipe: conf/experiment/grpo_wan_t2v_14b_4x8.yaml
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_4x8.sh              # auto
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_4x8.sh head
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_4x8.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_wan21_t2v_14b_4x8}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-WAN2.1-T2V-14B-GRPO-4x8}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_wan_t2v_14b_4x8 "$@"
