#!/bin/bash
# =============================================================================
# FlowGRPO SD3 multi-node — sglang colocate + TransferQueue (Mooncake RDMA).
# Recipe: conf/experiment/flowgrpo_sglang_sd3_colocate.yaml   (UNCHANGED)
# TQ overlay: conf/transfer_queue/mooncake_tuned.yaml         (auto-composed)
# Launcher: scripts/train_tq.sh
#
# Usage (same as scripts/train.sh — see scripts/train_tq.sh for TQ knobs):
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh head
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh train run.num_rollouts=100
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_sglang_sd3_colocate_tq_multinode}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-Flow-GRPO-sglang-colocate-tq-multinode}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/flowgrpo_sglang_sd3_colocate_tq_multinode}"

exec "${SCRIPT_DIR}/../scripts/train_tq.sh" flowgrpo_sglang_sd3_colocate "$@"
