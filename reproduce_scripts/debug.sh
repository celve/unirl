#!/bin/bash
# =============================================================================
# Debug script: full pipeline with debug_save_intermediates enabled.
#
# Runs the SAME configuration as train_nft_sd3_train_actor_sampling_ocr.sh
# but with --debug.debug-save-intermediates true, which persists every
# rollout payload (training batch + intermediates) to disk.
#
# Saved payloads can later be replayed with:
#   --debug.debug-mode train_only --debug.debug-load-path <saved_payload.pt>
#
# Usage:
#   bash debug.sh                          # default: 3 rollouts, save payloads
#   bash debug.sh --rollout.control.num-rollout 1  # just 1 rollout
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL="stabilityai/stable-diffusion-3.5-medium"
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/datasets/ocr/train.txt"}

NUM_GPUS=${NUM_GPUS:-8}

# Rollout setttings
NUM_INFERENCE_STEPS=10 # denoising steps during rollout (sampling) stage.
NUM_SAMPLES_PER_PROMPT=24 # group size
PROMPTS_PER_BATCH=48 # number of (unique) prompts per epoch
DIRECT_SAMPLING_BATCH_SIZE=192 # Actual peak forward batch size during sampling stage.

# Training settings
LOCAL_MICRO_BATCH_SIZE=12 # Local peak forward/backward batch size during optimization
NUM_UPDATES_PER_LOCAL_BATCH=${NUM_UPDATES_PER_LOCAL_BATCH:-1}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-false}
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${LOCAL_MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}")
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH file not found: ${DATA_PATH}"
    exit 1
fi

REPORT_TO_WANDB=false
WANDB_PROJECT_NAME="diffusionrl-diffusionNFT"
WANDB_RUN_NAME="SD3.5-DiffusionNFT" # change to your own name
WANDB_LOG_MEDIA=true
WANDB_MEDIA_MAX_ITEMS=4 # Max number of image reports per logging step
WANDB_TAGS="reproduce,sd3.5,nft,ocr" # reward=ocr, flow
WANDB_ENTITY=${WANDB_ENTITY:-"diffusionrl-reproduce"} # Set empty to skip: WANDB_ENTITY=""
LOGGING_STEPS=1

WANDB_ENTITY_ARGS=()
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARGS+=(--rollout.logging.wandb-entity "${WANDB_ENTITY}")
fi

REWARD_NAME="pickscore" # pickscore, ocr, clip, hpsv2
REWARD_DEVICE="cuda"
REWARD_LOCATION="sampling_actor" # run reward scorer on sampling actors

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
NFT_ALGO_KWARG_ARGS=(
    --algorithm.kwarg "beta=0.1"
    --algorithm.kwarg "adv_mode=raw"
    --algorithm.kwarg "adv_clip_max=5.0"
    --algorithm.kwarg "use_adaptive_weight=true"
    --algorithm.kwarg "train_timestep_mode=all"
    --algorithm.kwarg "shuffle_train_timesteps=true"
    --algorithm.kwarg "apply_time_shift_in_loss=false"
    --algorithm.kwarg "use_reference_ema=true"
    --algorithm.kwarg "reference_update_timing=rollout_end"
    --algorithm.kwarg "ema_decay=0.001"
    --algorithm.kwarg "decay_type=warmup"
    --algorithm.kwarg "ema_flat_steps=0"
    --algorithm.kwarg "ema_uprate=0.001"
    --algorithm.kwarg "ema_uphold=0.5"
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "kl_coef=0.0001"
    --algorithm.adv-normalization "group"
    --algorithm.use-global-std "true"
    --algorithm.adv-norm-eps "1e-4"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

# Debug settings
DEBUG_SAVE_DIR=${DEBUG_SAVE_DIR:-"${REPO_ROOT}/debug_outputs/debug"}
DEBUG_NUM_ROLLOUTS=${DEBUG_NUM_ROLLOUTS:-1}

    # --debug.debug-save-intermediates true \
    # --debug.debug-save-dir "${DEBUG_SAVE_DIR}" \

    # --debug.debug-mode train_only \
    # --debug.debug-load-path "${DEBUG_SAVE_DIR}/rollout_payload_0.pt" \
    # --debug.debug-num-rollouts 3 \
python -m diffusionrl.train \
    --debug.debug-mode train_only \
    --debug.debug-load-path "${DEBUG_SAVE_DIR}/rollout_payload_0.pt" \
    --debug.debug-num-rollouts 3 \
    \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward.reward-model-name "${REWARD_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.shift 3.0 \
    --sampling.eta 0.0 \
    --sampling.sde-type dpm2 \
    --sampling.timestep-fraction 0.1,0.4 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.guidance-scale 1.0 \
    --sampling.sampling-adapter old \
    "${NFT_ALGO_KWARG_ARGS[@]}" \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --training.num-updates-per-local-batch ${NUM_UPDATES_PER_LOCAL_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.topology.mode direct_rollout \
--sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    --training.use-gradient-checkpointing false \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.control.num-rollout ${DEBUG_NUM_ROLLOUTS} \
    --rollout.artifacts.save-steps 0 \
    --rollout.evaluation.eval-steps 1 \
    --rollout.logging.logging-steps ${LOGGING_STEPS} \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --rollout.logging.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.logging.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.logging.run-name "${WANDB_RUN_NAME}" \
    --rollout.logging.wandb-log-media ${WANDB_LOG_MEDIA} \
    --rollout.logging.wandb-media-max-items ${WANDB_MEDIA_MAX_ITEMS} \
    --rollout.logging.wandb-tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    --sync.protocol disabled \
    "$@"
