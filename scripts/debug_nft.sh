#!/bin/bash
# =============================================================================
# DiffusionNFT SD3 debug — direct-sampling (FSDP engine) NFT, short-loop
# Recipe: conf/experiment/debug_nft.yaml (inherits nft_sd3 + debug overrides)
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/debug_nft.sh                # auto, 3 rollouts, eval each
#   bash reproduce_scripts/debug_nft.sh head
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug_nft_sd3}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-DiffusionNFT-debug}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/debug_nft_sd3}"

exec "${SCRIPT_DIR}/../scripts/train.sh" debug_nft "$@"
