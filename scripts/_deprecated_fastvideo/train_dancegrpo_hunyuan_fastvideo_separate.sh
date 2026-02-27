#!/bin/bash
# =============================================================================
# DanceGRPO training with HunyuanVideo — FastVideo sampling engine (Separate mode)
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
#
#   1. CONFIG VALIDATION FAIL: fastvideo engine declares supports_logprob=False
#      in ENGINE_CAPABILITY_REQUIREMENTS (arguments.py:57-61), but loss_type=grpo
#      requires requires_log_prob=True (arguments.py:70-74).
#      → _validate_engine_contract_compatibility() will raise ValueError at startup.
#      FIX: Update FastVideoRolloutEngine.get_capabilities() to supports_logprob=True
#           AND update ENGINE_CAPABILITY_REQUIREMENTS["fastvideo"] accordingly,
#           after implementing log_prob computation in the engine.
#
#   2. LOG_PROB COMPUTATION IS EMPTY: engine.py:335 has a `pass` placeholder
#      where log_prob should be computed. _convert_to_sampler_output() always
#      returns log_probs=None → validate_contract(requires_log_probs=True) fails.
#      FIX: Either (a) call sampler.compute_log_probs_from_trajectory_mixed() in
#           _convert_to_sampler_output(), or (b) wait for FastVideo PR to add
#           native log_prob during sampling.
#
#   3. SAMPLER model=None: engine.py:188 creates FastVideoSampler(model=None).
#      compute_log_probs_from_trajectory_mixed() calls self.model(...) which
#      will crash with NoneType.
#      FIX: Pass a valid model reference to the sampler, or restructure so that
#           log_prob computation happens inside FastVideo workers.
#
#   4. FastVideo upstream generates ForwardBatch with eta hardcoded to 0.0 in
#      VideoGenerator._generate_single_video(), which biases toward deterministic
#      ODE behavior. diffusionrl monkey patch overrides eta from FastVideoArgs, but
#      this path is still experimental and needs end-to-end validation.
#
# MODE: Separate — inference actors (FastVideo) and training actors (FSDP)
#   - Inference actors: FastVideo MultiprocExecutor for fast video sampling
#   - Training actors: FSDP-wrapped transformer for GRPO loss backprop
#   - Weight sync: checkpoint_path mode (training saves → inference reloads)
#   - Advantage: FastVideo's sequence parallelism for faster sampling
#
# ENGINE: FastVideo (MultiprocExecutor)
#   FastVideo uses MultiprocExecutor internally to spawn GPU worker processes.
#   Each worker loads the full model and uses sequence parallelism (SP) for
#   parallel denoising. SP splits the sequence (frames) across GPUs.
#
#   Ray scheduling: Each inference actor requests num_gpus = fastvideo_num_gpus.
#   FastVideo's MultiprocExecutor manages the GPU allocation internally.
#
# GPU LAYOUT (default: 8 inference + 8 training = 16 GPUs):
#   Inference: FASTVIDEO_NUM_GPUS GPUs per actor × ROLLOUT_ACTORS actors
#   Training:  TRAINING_GPUS GPUs with FSDP
#   Total:     ROLLOUT_ACTORS × FASTVIDEO_NUM_GPUS + TRAINING_GPUS
#
# COMPARISON with FSDP-based script:
#   FSDP script:    Each inference actor runs FSDPHunyuanSampler with
#                   hand-written SDE loop + log_prob (DanceGRPO-aligned).
#   This script:    Each inference actor runs FastVideo VideoGenerator
#                   with MultiprocExecutor for potentially faster sampling.
#                   Log_prob computed via trajectory replay after sampling.
#
# Usage (after fixing blocking issues):
#   bash train_dancegrpo_hunyuan_fastvideo_separate.sh
#   FASTVIDEO_NUM_GPUS=4 ROLLOUT_ACTORS=2 TRAINING_GPUS=8 \
#     bash train_dancegrpo_hunyuan_fastvideo_separate.sh
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
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_fastvideo_separate"}
# Hunyuan path in diffusionrl currently expects plain prompts (text mode).
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}

# FastVideo parallelism
# Each FastVideo inference actor uses FASTVIDEO_NUM_GPUS GPUs
# SP (sequence parallelism) splits frames across GPUs within one actor
FASTVIDEO_NUM_GPUS=${FASTVIDEO_NUM_GPUS:-4}
FASTVIDEO_SP_SIZE=${FASTVIDEO_SP_SIZE:-${FASTVIDEO_NUM_GPUS}}
FASTVIDEO_TP_SIZE=${FASTVIDEO_TP_SIZE:-1}
ALLOW_NOSET_MULTI_GPU_INFERENCE=${ALLOW_NOSET_MULTI_GPU_INFERENCE:-"true"}

# Actor layout
ROLLOUT_ACTORS=${ROLLOUT_ACTORS:-2}
TRAINING_GPUS=${TRAINING_GPUS:-8}
TRAINING_NUM_NODES=${TRAINING_NUM_NODES:-1}
TRAINING_GPUS_PER_NODE=${TRAINING_GPUS_PER_NODE:-${TRAINING_GPUS}}

# Inference layout: total inference GPUs = ROLLOUT_ACTORS × FASTVIDEO_NUM_GPUS
ROLLOUT_TOTAL_GPUS=$((ROLLOUT_ACTORS * FASTVIDEO_NUM_GPUS))
TOTAL_GPUS=$((ROLLOUT_TOTAL_GPUS + TRAINING_GPUS))

# Sampling geometry (DanceGRPO-aligned)
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${ROLLOUT_ACTORS}}

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
echo " DanceGRPO HunyuanVideo — FastVideo Separate Mode"
echo "======================================================"
echo " FastVideo GPUs/actor:   ${FASTVIDEO_NUM_GPUS} (SP=${FASTVIDEO_SP_SIZE})"
echo " NOSET multi-GPU layout: ${ALLOW_NOSET_MULTI_GPU_INFERENCE}"
echo " Inference actors:       ${ROLLOUT_ACTORS}"
echo " Inference total GPUs:   ${ROLLOUT_TOTAL_GPUS}"
echo " Training GPUs:          ${TRAINING_GPUS}"
echo " Total GPUs:             ${TOTAL_GPUS}"
echo " Prompts per batch:      ${PROMPTS_PER_BATCH}"
echo " Samples per prompt (K): ${NUM_SAMPLES_PER_PROMPT}"
echo " Total samples/rollout:  ${TOTAL_SAMPLES}"
echo " Per-rank training:      $((TOTAL_SAMPLES / TRAINING_GPUS))"
echo " Resolution:             ${HEIGHT}×${WIDTH}×${NUM_FRAMES}f"
echo " Reward:                 ${REWARD_MODEL_NAME}"
echo " Replay old log_prob:    ${REPLAY_LOG_PROBS} (experimental)"
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
    `# ===== FastVideo Separate Mode Layout =====` \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_TOTAL_GPUS} \
    --training-num-nodes ${TRAINING_NUM_NODES} \
    --training-num-gpus-per-node ${TRAINING_GPUS_PER_NODE} \
    --placement-strategy PACK \
    --fastvideo-num-gpus ${FASTVIDEO_NUM_GPUS} \
    --sp-size ${FASTVIDEO_SP_SIZE} \
    --allow-noset-multi-gpu-inference ${ALLOW_NOSET_MULTI_GPU_INFERENCE} \
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
    `# ===== Weight Sync (checkpoint path for FastVideo) =====` \
    --weight-sync-mode checkpoint_path \
    \
    `# ===== Rollout / Checkpoint =====` \
    --num-rollout 202 \
    --save-steps 50 \
    --logging-steps 1 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
