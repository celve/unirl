#!/bin/bash
# =============================================================================
# DanceGRPO training with HunyuanVideo — FastVideo sampling engine (Colocate)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/XueZeyue/DanceGRPO)
#   Paper:   arXiv:2505.07818 "DanceGRPO: Unleashing GRPO on Visual Generation"
#
# STATUS: ⚠️  MIGRATION TARGET — NOT RUNNABLE YET
#   This script uses FastVideo as the sampling engine for future migration.
#   The FSDP-based scripts (train_dancegrpo_hunyuan_separate.sh) are the
#   currently functional versions for DanceGRPO reproduction.
#
# BLOCKING ISSUES (must be resolved before this script can run):
#   See train_dancegrpo_hunyuan_fastvideo_separate.sh for full list.
#   Summary:
#   1. fastvideo engine supports_logprob=False vs GRPO requires_log_prob=True
#   2. log_prob computation is empty (pass placeholder in engine.py:335)
#   3. FastVideoSampler created with model=None (engine.py:188)
#   4. FastVideo upstream defaults eta=0.0 in VideoGenerator path; diffusionrl
#      monkey patch overrides eta but this remains experimental.
#
# MODE: Colocate — inference and training actors share the SAME GPUs
#   - Each GPU runs both FastVideo sampling and FSDP training
#   - Offload pattern: inference offloads → training runs → training offloads
#     → inference onloads → next rollout
#   - Advantage: fewer total GPUs needed (N GPUs instead of 2N)
#   - Disadvantage: sequential execution, offload/onload overhead
#
# ENGINE: FastVideo (MultiprocExecutor) for sampling
#   In colocate mode, the FastVideo generator lives inside the inference actor
#   which shares GPU memory with the training actor via offload/onload cycles.
#   FastVideo workers are spawned during sampling and their models are offloaded
#   to CPU when training takes over the GPU.
#
# GPU LAYOUT (default: 8 GPUs, shared between inference and training):
#   Each GPU hosts both an inference shard (FastVideo) and a training shard (FSDP).
#   Memory budget is split: inference offloads during training, vice versa.
#
#   Single node:  NUM_GPUS=8
#   Multi node:   NUM_NODES=2 GPUS_PER_NODE=8  (16 GPUs)
#
# MEMORY (per GPU, 80GB H800 recommended):
#   Colocate mode is memory-intensive because both engine states need to fit:
#   - FastVideo inference: ~13B/SP_size bf16 model + VAE + text encoder
#   - FSDP training: model shard + optimizer states + gradients
#   - Offload helps: only one is active at a time
#   - 80GB GPUs strongly recommended; 40GB may need aggressive offloading
#
# Usage (after fixing blocking issues):
#   bash train_dancegrpo_hunyuan_fastvideo_colocate.sh
#   NUM_GPUS=16 bash train_dancegrpo_hunyuan_fastvideo_colocate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


echo "================================================================"
echo " WARNING: This script uses FastVideo engine which is NOT YET"
echo " FUNCTIONAL for GRPO training. See script header for details."
echo " Use train_dancegrpo_hunyuan_separate.sh for working version."
echo "================================================================"

# ===== Configurable defaults (override via env vars or CLI "$@") =====
# Keep local checkpoints under models/local by default.
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/hunyuan-video"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_fastvideo_colocate"}
# Hunyuan path in diffusionrl currently expects plain prompts (text mode).
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}

# GPU layout (single-node default, all GPUs shared)
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-${NUM_GPUS}}
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# FastVideo parallelism (within each inference actor)
# NOTE: colocate_rollout_training currently supports only single-GPU inference actors.
FASTVIDEO_NUM_GPUS=${FASTVIDEO_NUM_GPUS:-1}
FASTVIDEO_SP_SIZE=${FASTVIDEO_SP_SIZE:-${FASTVIDEO_NUM_GPUS}}

# Sampling geometry (DanceGRPO-aligned)
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${TOTAL_GPUS}}

# Training geometry
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

# Resolution (DanceGRPO: 480×480, 53 frames)
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

# Reward model
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}
REWARD_PATH=${REWARD_PATH:-"diffusionrl.reward.local.LocalRewardWorker"}
# Experimental/ad-hoc bridge:
# FastVideo rollout old_log_prob is replayed on training actors.
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-"true"}

# Offload configuration
# In colocate mode, inference engine offloads during training and vice versa
OFFLOAD=${OFFLOAD:-"true"}
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-"true"}

# FSDP configuration for training side
FSDP_SHARDING=${FSDP_SHARDING:-"FULL_SHARD"}
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-"false"}

# Note: shell-level path existence check removed. The Python-level fallback in
# arguments.py handles missing local paths by falling back to HuggingFace IDs.

# Fail fast for missing prompt source (avoid silent fallback to default prompts).
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
    echo "Set DATA_PATH to an existing prompt file (recommended: ${REPO_ROOT}/data/samples/video_prompts_toy.txt)."
    exit 1
fi

# Validate batch geometry
TOTAL_SAMPLES=$((PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT))
if [ $((TOTAL_SAMPLES % TOTAL_GPUS)) -ne 0 ]; then
    echo "ERROR: total_samples (${TOTAL_SAMPLES} = ${PROMPTS_PER_BATCH}×${NUM_SAMPLES_PER_PROMPT}) must be divisible by TOTAL_GPUS (${TOTAL_GPUS})"
    exit 1
fi
if [ "${FASTVIDEO_NUM_GPUS}" -gt 1 ]; then
    echo "ERROR: colocate_rollout_training=true does not support multi-GPU inference actors."
    echo "Set FASTVIDEO_NUM_GPUS=1 (current: ${FASTVIDEO_NUM_GPUS})."
    exit 1
fi

echo "======================================================"
echo " DanceGRPO HunyuanVideo — FastVideo Colocate Mode"
echo "======================================================"
echo " Total GPUs (shared):    ${TOTAL_GPUS} (${NUM_NODES} nodes × ${GPUS_PER_NODE})"
echo " FastVideo GPUs/actor:   ${FASTVIDEO_NUM_GPUS}"
echo " FastVideo SP size:      ${FASTVIDEO_SP_SIZE}"
echo " Prompts per batch:      ${PROMPTS_PER_BATCH}"
echo " Samples per prompt (K): ${NUM_SAMPLES_PER_PROMPT}"
echo " Total samples/rollout:  ${TOTAL_SAMPLES}"
echo " Per-rank training:      $((TOTAL_SAMPLES / TOTAL_GPUS))"
echo " Resolution:             ${HEIGHT}×${WIDTH}×${NUM_FRAMES}f"
echo " Reward:                 ${REWARD_MODEL_NAME}"
echo " Replay old log_prob:    ${REPLAY_LOG_PROBS} (experimental)"
echo " Offload inference:      ${OFFLOAD}"
echo " Offload training:       ${OFFLOAD_TRAIN}"
echo " FSDP sharding:          ${FSDP_SHARDING}"
echo "======================================================"

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type hunyuan \
    --sampler-engine-type fastvideo \
    --sampler-path diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path "${REWARD_PATH}" \
    --reward-model-name "${REWARD_MODEL_NAME}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    `# ===== SDE Sampling (aligned with DanceGRPO) =====` \
    --sde-type dance \
    --eta 0.25 \
    --shift 5.0 \
    --num-inference-steps 16 \
    --guidance-scale 0.0 \
    --timestep-fraction 0.6 \
    --init-same-noise true \
    \
    `# ===== GRPO Algorithm =====` \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    --replay-log-probs ${REPLAY_LOG_PROBS} \
    \
    `# ===== Colocate Mode Layout =====` \
    --colocate-rollout-training true \
    --rollout-num-gpus-per-node ${GPUS_PER_NODE} \
    --training-num-nodes ${NUM_NODES} \
    --training-num-gpus-per-node ${GPUS_PER_NODE} \
    --fastvideo-num-gpus ${FASTVIDEO_NUM_GPUS} \
    --sp-size ${FASTVIDEO_SP_SIZE} \
    --offload ${OFFLOAD} \
    --offload-train ${OFFLOAD_TRAIN} \
    \
    `# ===== Training Hyperparams (aligned with DanceGRPO) =====` \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --use-fsdp true \
    --fsdp-sharding-strategy ${FSDP_SHARDING} \
    --fsdp-cpu-offload ${FSDP_CPU_OFFLOAD} \
    --use-gradient-checkpointing true \
    \
    `# ===== Video Resolution =====` \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    `# ===== Weight Sync (checkpoint path for FastVideo) =====` \
    --weight-sync-mode checkpoint_path \
    \
    `# ===== Rollout / Checkpoint =====` \
    --num-rollout 202 \
    --save-steps 50 \
    --logging-steps 1 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
