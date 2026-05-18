#!/bin/bash
# =============================================================================
# GRPO HunyuanVideo-1.5 T2V single-/multi-node — direct-sampling (FSDP engine)
# Reward: standalone VideoPickScore (PickScore on the first frame).
# Recipe: conf/experiment/grpo_hunyuan_veido1p5_t2v_videopickscore.yaml
#
# Usage (same as scripts/train.sh — see that file for role + cluster details):
#   bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh              # auto
#   bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh head
#   bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh train run.num_rollouts=20
#
# Smoke run on the 4-prompt toy set (CPU-light, no real dataset needed):
#   DATA_PATH=$(pwd)/data/samples/video_prompts_toy.txt \
#     bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh \
#     train run.num_rollouts=2 algorithm.prompts_per_rollout=4 \
#     algorithm.samples_per_prompt=2 training.plan.global_batch_size=8 \
#     training.plan.local_batch_size=1 training.plan.local_mini_batch_size=1 \
#     training.topology.actor_count=8
#
# Cluster shape (defaults to single-node 8-GPU; override via env or CLI):
#   NUM_NODES=2 GPUS_PER_NODE=8 \
#     bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh
#
# Optional: point VideoPickScore at locally-staged checkpoints (the YAML
# defaults fall back to the HuggingFace Hub identifiers if these are unset):
#   PICKSCORE_PROCESSOR=/path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
#   PICKSCORE_MODEL=/path/to/PickScore_v1 \
#     bash reproduce_scripts/train_grpo_hunyuan_veido1p5_t2v_videopickscore.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_hunyuan_veido1p5_t2v_videopickscore}"
# Default to the curated T2V prompt corpus under data/datasets/t2v/, which
# is reproducibly split by data/datasets/t2v/preprocess.py (seed=42,
# test_size=2048, train_size=47952). For image-style smoke tests, override
# DATA_PATH=...pickscore/train.txt; for tiny smoke runs use
# data/samples/video_prompts_toy.txt.
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/t2v/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/t2v/test.txt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-HunyuanVideo-1.5-480p-T2V-GRPO-videopickscore}"
export WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

exec "${SCRIPT_DIR}/../scripts/train.sh" grpo_hunyuan_veido1p5_t2v_videopickscore "$@"
