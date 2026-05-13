#!/bin/bash
# =============================================================================
# DiffusionNFT SD3 multi-node — remote RewardService backend
# Recipe: conf/experiment/nft_sd3_reward_service.yaml
#
# Same training setup as train_nft_sd3_multinode.sh, but uses a remote
# RewardService HTTP API instead of local GPU-based scorers. The service
# computes hpsv2, pickscore, unified_reward, imagereward, and hpsv3 in one
# HTTP round trip, freeing all local GPUs for training and rollout.
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_nft_sd3_reward_service_multinode.sh         # auto
#   bash reproduce_scripts/train_nft_sd3_reward_service_multinode.sh head
#   bash reproduce_scripts/train_nft_sd3_reward_service_multinode.sh train
#
#   # Override the reward service URL:
#   REWARD_SERVICE_URL=http://10.0.0.5:8080 \
#       bash reproduce_scripts/train_nft_sd3_reward_service_multinode.sh auto
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/nft_sd3_reward_service}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-DiffusionNFT-RewardService}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/nft_sd3_reward_service}"

exec "${SCRIPT_DIR}/../scripts/train.sh" nft_sd3_reward_service "$@"
