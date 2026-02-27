#!/bin/bash
# =============================================================================
# DanceGRPO training with HunyuanVideo (Separate mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/XueZeyue/DanceGRPO)
#   Script:  scripts/finetune/finetune_hunyuan_grpo.sh
#   Paper:   arXiv:2505.07818 "DanceGRPO: Unleashing GRPO on Visual Generation"
#
# MODE: Separate — rollout actors and training actors on DIFFERENT GPUs
#   - Rollout actors: run HunyuanSampler (SDE sampling + log_prob)
#   - Training actors: FSDP-wrapped transformer for GRPO loss backprop
#   - Weight sync: training → rollout after each rollout
#   - Advantage: can overlap rollout N+1 with training N (async pipeline)
#
# ENGINE: FSDP (not FastVideo)
#   DanceGRPO GRPO algorithm requires log_prob during sampling.
#   - FSDPRolloutEngine + FSDPHunyuanSampler: supports_logprob=True  ✅
#   - FastVideoRolloutEngine:                 supports_logprob=False ✗
#   Therefore we use FSDP engine for faithful DanceGRPO reproduction.
#
# ALIGNMENT with DanceGRPO finetune_hunyuan_grpo.sh:
#   ┌───────────────────────┬────────────────────┬──────────────────────┐
#   │ Parameter             │ DanceGRPO          │ diffusionrl (this)     │
#   ├───────────────────────┼────────────────────┼──────────────────────┤
#   │ sde_type              │ implicit (flux_step)│ dance                │
#   │ eta                   │ 0.25               │ 0.25                 │
#   │ shift                 │ 5                  │ 5.0                  │
#   │ sampling_steps        │ 16                 │ 16                   │
#   │ timestep_fraction     │ 0.6                │ 0.6                  │
#   │ guidance tensor       │ 6018.0             │ 6018.0               │
#   │ num_generations       │ 24                 │ num_samples_per_prompt│
#   │ use_group             │ ✅                  │ advantage_type=group │
#   │ use_same_noise        │ ✅                  │ init_same_noise=true │
#   │ learning_rate         │ 1e-5               │ 1e-5                 │
#   │ max_grad_norm         │ 1.0                │ 1.0                  │
#   │ weight_decay          │ 0.0001             │ 0.0001               │
#   │ h×w×t                 │ 480×480×53         │ 480×480×53           │
#   │ KL penalty            │ none               │ false                │
#   │ bestofn               │ 8                  │ (not impl, see NOTE) │
#   │ vq_coef / mq_coef    │ 1.0 / 0.0          │ single reward worker │
#   └───────────────────────┴────────────────────┴──────────────────────┘
#
# NOTE on guidance vs cfg:
#   In DanceGRPO Hunyuan GRPO training, model forward uses guidance tensor 6018.0.
#   The CLI flag --cfg=0.0 in the original script controls dataset text-dropout rate,
#   not this guidance tensor value.
#
# NOTE on bestofn:
#   DanceGRPO uses bestofn=8 (from 24 samples, keep top-8 by reward for
#   training). diffusionrl does not implement best-of-N filtering; all K
#   samples are used with group-relative advantages. This is standard GRPO
#   and should produce comparable results. If you need bestofn, extend
#   RolloutManager._compute_advantages() to add top-K filtering.
#
# NOTE on reward model:
#   DanceGRPO uses VideoAlign (VQ score) for HunyuanVideo.
#   diffusionrl does not bundle VideoAlign. Options:
#     1. Use hpsv2 on decoded frames (default in this script, proxy)
#     2. Implement a VideoAlign reward worker (recommended for reproduction)
#     3. Use VideoRewardWorker (frame-based pickscore + temporal)
#   Set REWARD_MODEL_NAME and REWARD_PATH env vars to override.
#
# GPU LAYOUT (default: 16 GPUs = 8 rollout + 8 training):
#   DanceGRPO uses 16-32 GPUs (all doing sequential sample→train).
#   Separate mode splits GPUs: rollout runs HunyuanSampler in parallel
#   while training does FSDP backprop. Total GPU count is higher but
#   allows async overlap.
#
# MEMORY:
#   HunyuanVideo transformer: ~13B params
#   - Rollout: bf16 ≈ 26GB (single GPU, NO_SHARD)
#   - Training: FSDP FULL_SHARD across 8 GPUs ≈ 6.5GB model + grads/optim
#   - Recommend 80GB GPUs (H800/A100-80G)
#   - For 40GB GPUs: reduce resolution, use gradient checkpointing,
#     or enable FSDP CPU offload
#
# Usage:
#   bash train_dancegrpo_hunyuan_separate.sh
#   ROLLOUT_GPUS=4 TRAINING_GPUS=4 bash train_dancegrpo_hunyuan_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# ===== Configurable defaults (override via env vars or CLI "$@") =====
# Keep local checkpoints under models/local by default.
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/hunyuan-video"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_separate"}
# Hunyuan path in diffusionrl currently expects plain prompts (text mode).
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}

# GPU layout
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
TRAINING_GPUS=${TRAINING_GPUS:-8}

# Sampling geometry
# DanceGRPO: num_generations=24 per rank. In separate mode, each rollout
# actor generates NUM_SAMPLES_PER_PROMPT samples per prompt.
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${ROLLOUT_GPUS}}

# Training geometry
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

# Resolution (DanceGRPO: 480x480, 53 frames)
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

# Reward model
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}
REWARD_PATH=${REWARD_PATH:-"diffusionrl.reward.local.LocalRewardWorker"}

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
if [ $((TOTAL_SAMPLES % TRAINING_GPUS)) -ne 0 ]; then
    echo "ERROR: total_samples (${TOTAL_SAMPLES} = ${PROMPTS_PER_BATCH}×${NUM_SAMPLES_PER_PROMPT}) must be divisible by TRAINING_GPUS (${TRAINING_GPUS})"
    exit 1
fi

echo "======================================================"
echo " DanceGRPO HunyuanVideo — Separate Mode"
echo "======================================================"
echo " Rollout GPUs:           ${ROLLOUT_GPUS}"
echo " Training GPUs:          ${TRAINING_GPUS}"
echo " Prompts per batch:      ${PROMPTS_PER_BATCH}"
echo " Samples per prompt (K): ${NUM_SAMPLES_PER_PROMPT}"
echo " Total samples/rollout:  ${TOTAL_SAMPLES}"
echo " Per-rank training:      $((TOTAL_SAMPLES / TRAINING_GPUS))"
echo " Resolution:             ${HEIGHT}×${WIDTH}×${NUM_FRAMES}f"
echo " Reward:                 ${REWARD_MODEL_NAME}"
echo "======================================================"

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type hunyuan \
    --sampler-engine-type fsdp \
    --sampler-path diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler \
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
    --guidance-scale 6018.0 \
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
    \
    `# ===== Separate Mode Layout =====` \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy PACK \
    \
    `# ===== Training Hyperparams (aligned with DanceGRPO) =====` \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --use-gradient-checkpointing true \
    \
    `# ===== Video Resolution =====` \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    `# ===== Rollout / Checkpoint =====` \
    --num-rollout 202 \
    --save-steps 50 \
    --logging-steps 1 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
