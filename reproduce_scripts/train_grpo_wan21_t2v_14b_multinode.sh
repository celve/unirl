#!/bin/bash
# =============================================================================
# GRPO WAN 2.1 T2V 14B multi-node — direct-sampling (FSDP engine)
# Recipe: conf/experiment/grpo_wan_t2v_14b.yaml
#
# Same as the 1.3B variant but with the 14B model, lower resolution
# (480x832 vs 720x1280), and smaller forward batch (1 vs 4).
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_multinode.sh              # auto
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_multinode.sh head
#   bash reproduce_scripts/train_grpo_wan21_t2v_14b_multinode.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_wan21_t2v_14b}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-WAN2.1-T2V-14B-GRPO}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_wan_t2v_14b "$@"
