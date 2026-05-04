#!/bin/bash
# =============================================================================
# FlowGRPO SD3 multi-node — sglang separate + TransferQueue (Mooncake RDMA).
# Recipe: conf/experiment/flowgrpo_sglang_sd3_separate.yaml   (UNCHANGED)
# TQ overlay: conf/transfer_queue/mooncake_tuned.yaml         (auto-composed)
# Launcher: scripts/train_tq.sh
#
# Mirror of train_grpo_sd3_sglang_colocate_tq_multinode.sh, but routes the
# experiment through the dedicated-rollout split (train+rollout on disjoint
# GPUs per node, default 4+4 on an 8-GPU node).
#
# Usage (same as scripts/train.sh — see scripts/train_tq.sh for TQ knobs):
#   bash reproduce_scripts/train_grpo_sd3_sglang_separate_tq_multinode.sh
#   bash reproduce_scripts/train_grpo_sd3_sglang_separate_tq_multinode.sh head
#   bash reproduce_scripts/train_grpo_sd3_sglang_separate_tq_multinode.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_sglang_sd3_separate_tq_multinode}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-Flow-GRPO-sglang-separate-tq-multinode}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

# SGLang engine GPU split: half train / half rollout per node by default.
# Recipe assumes 8 GPUs/node → 4 train + 4 rollout. Override via env.
GPUS_PER_NODE_FOR_SPLIT="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-$(( GPUS_PER_NODE_FOR_SPLIT / 2 ))}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-$(( GPUS_PER_NODE_FOR_SPLIT - TRAINING_GPUS_PER_NODE ))}"
export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/flowgrpo_sglang_sd3_separate_tq_multinode}"

exec "${SCRIPT_DIR}/../scripts/train_tq.sh" flowgrpo_sglang_sd3_separate "$@"
