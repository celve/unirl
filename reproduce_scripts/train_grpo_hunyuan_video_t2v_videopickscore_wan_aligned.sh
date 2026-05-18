#!/bin/bash
# =============================================================================
# GRPO HunyuanVideo (1.0) T2V — WAN-aligned training params + video reward.
# Recipe: conf/experiment/grpo_hunyuan_video_t2v_videopickscore_wan_aligned.yaml
#
# Based on train_grpo_hunyuan_video_t2v_videopickscore.sh but uses the
# WAN 2.1 14B-aligned experiment config (lr=3e-4, eta=0.7, ema=0.9,
# 480x832, samples_per_prompt=16, video reward with temporal term).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_hunyuan_video_t2v_videopickscore_wan_aligned}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/t2v/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/t2v/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-HunyuanVideo-1.0-T2V-GRPO-videopickscore-wan-aligned}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_hunyuan_video_t2v_videopickscore_wan_aligned "$@"
