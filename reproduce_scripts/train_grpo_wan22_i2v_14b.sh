#!/bin/bash
# =============================================================================
# GRPO WAN 2.2 I2V 14B — dual-transformer image-to-video (FSDP engine)
# Recipe: conf/experiment/grpo_wan22_i2v_14b.yaml
#
# Wan2.2-I2V uses two WanTransformer3DModel instances (high-noise + low-noise
# refinement), sharing the same I2V conditioning infrastructure as WAN 2.1.
#
# Dataset format (each line in the JSONL):
#   {"prompt": "A cat jumping off a table", "image": "cat_on_table.png"}
#
# Usage:
#   bash reproduce_scripts/train_grpo_wan22_i2v_14b.sh
#   bash reproduce_scripts/train_grpo_wan22_i2v_14b.sh head
#   bash reproduce_scripts/train_grpo_wan22_i2v_14b.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_wan22_i2v_14b}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/sharegpt4o_image_mini/train.jsonl}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/sharegpt4o_image_mini/test.jsonl}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-WAN2.2-I2V-14B-GRPO}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_wan22_i2v_14b "$@"
