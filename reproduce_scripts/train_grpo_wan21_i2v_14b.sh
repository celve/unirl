#!/bin/bash
# =============================================================================
# GRPO WAN 2.1 I2V 14B multi-node — direct-sampling (FSDP engine)
# Recipe: conf/experiment/grpo_wan_i2v_14b.yaml
#
# Image-to-video variant of the WAN 14B recipe. Requires a JSONL dataset with
# "image" fields pointing to conditioning images (relative to the JSONL dir or
# absolute paths).
#
# Dataset format (each line in the JSONL):
#   {"prompt": "A cat jumping off a table", "image": "cat_on_table.png"}
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_grpo_wan21_i2v_14b.sh              # auto
#   bash reproduce_scripts/train_grpo_wan21_i2v_14b.sh head
#   bash reproduce_scripts/train_grpo_wan21_i2v_14b.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_wan21_i2v_14b}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/sharegpt4o_image_mini/train.jsonl}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/sharegpt4o_image_mini/test.jsonl}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Wan-AI/Wan2.1-I2V-14B-720P-Diffusers}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-WAN2.1-I2V-14B-GRPO}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_wan_i2v_14b "$@"
