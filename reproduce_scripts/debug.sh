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
#   bash debug.sh --rollout.num-rollout 1  # just 1 rollout
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL="stabilityai/stable-diffusion-3.5-medium"
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH="${REPO_ROOT}/data/datasets/pickscore/train.txt"

NUM_GPUS=${NUM_GPUS:-8}

# Rollout setttings
NUM_INFERENCE_STEPS=10 # denoising steps during rollout (sampling) stage.
NUM_SAMPLES_PER_PROMPT=24 # group size
PROMPTS_PER_BATCH=48 # number of (unique) prompts per epoch
DIRECT_SAMPLING_BATCH_SIZE=192 # Actual peak forward batch size during sampling stage.

# Training settings
GRADIENT_ACCUMULATION_BATCH_SIZE=12 # Actually peak forward/backward batch size during optimization
MULTI_UPDATE_BATCH_SIZE=72 # Effective batch size for multi-update. Set `prompts_per_batch * num_samples_per_prompt` // NUM_GPUS // n for n updates per epoch.
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
UPDATE_MODE="single_update" # multi_update or single_update. single_update for NFT, multi_update for GRPO.

NFT_ALGO_KWARGS=${NFT_ALGO_KWARGS:-'{"beta":0.1,"adv_mode":"raw","adv_clip_max":5.0,"use_adaptive_weight":true,"ema_decay":0.001}'}
NFT_LOSS_KWARGS=${NFT_LOSS_KWARGS:-'{"beta":0.1,"adv_mode":"raw","adv_clip_max":5.0,"use_adaptive_weight":true,"nft_timestep_mode":"all","nft_shuffle_timesteps":true,"nft_apply_shift":false,"use_ema":true,"ema_decay":0.001,"decay_type":"warmup","ema_flat_steps":0,"ema_uprate":0.001,"ema_uphold":0.5,"shuffle_seed":42,"shuffle_samples":true}'}
if [ $(( DIRECT_SAMPLING_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
if [ "${DIRECT_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_TOTAL_SAMPLES}" ] && [ $(( ROLLOUT_TOTAL_SAMPLES % DIRECT_SAMPLING_BATCH_SIZE )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must evenly divide rollout_total_samples (${ROLLOUT_TOTAL_SAMPLES})"
    exit 1
fi
GRADIENT_ACCUMULATION_ARGS=()
if [ -n "${GRADIENT_ACCUMULATION_BATCH_SIZE}" ]; then
    GRADIENT_ACCUMULATION_ARGS+=(--training.gradient-accumulation-batch-size "${GRADIENT_ACCUMULATION_BATCH_SIZE}")
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
    WANDB_ENTITY_ARGS+=(--rollout.wandb-entity "${WANDB_ENTITY}")
fi

REWARD_NAME="pickscore" # pickscore, ocr, clip, hpsv2
REWARD_DEVICE="cuda"
REWARD_EXECUTION_MODE="rollout" # compute reward during rollout stage

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-false}

# Debug settings
DEBUG_SAVE_DIR=${DEBUG_SAVE_DIR:-"${REPO_ROOT}/debug_outputs/debug"}
DEBUG_NUM_ROLLOUTS=${DEBUG_NUM_ROLLOUTS:-1}

    # --debug.debug-save-intermediates true \
    # --debug.debug-save-dir "${DEBUG_SAVE_DIR}" \

    # --debug.debug-mode train_only \
    # --debug.debug-load-path "${DEBUG_SAVE_DIR}/rollout_payload_0.pt" \
    # --debug.debug-num-rollouts 3 \
python -m diffusionrl.train \
    --debug.debug-save-intermediates true \
    --debug.debug-save-dir "${DEBUG_SAVE_DIR}" \
    \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name "${REWARD_NAME}" \
    --reward.reward-execution-mode "${REWARD_EXECUTION_MODE}" \
    --reward.local-reward-device "${REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --algorithm.loss-type nft \
    --sampling.shift 3.0 \
    --sampling.eta 0.0 \
    --sampling.sde-type sde \
    --sampling.timestep-fraction 0.1,0.4 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.guidance-scale 1.0 \
    --sampling.sampling-adapter old \
    --algorithm.algorithm-kwargs "${NFT_ALGO_KWARGS}" \
    --algorithm.loss-kwargs "${NFT_LOSS_KWARGS}" \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    "${GRADIENT_ACCUMULATION_ARGS[@]}" \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty true \
    --algorithm.kl-coef 0.0001 \
    --algorithm.advantage-type per_prompt \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --sampling.training-actor-direct-sampling true \
    --sampling.direct-sampling-batch-size ${DIRECT_SAMPLING_BATCH_SIZE} \
    --ray.colocate-rollout-training true \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    --ray.offload false \
    \
    --training.learning-rate 3e-4 \
    --training.update-mode ${UPDATE_MODE} \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    --training.use-gradient-checkpointing false \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout ${DEBUG_NUM_ROLLOUTS} \
    --rollout.save-steps 0 \
    --rollout.eval-steps 1 \
    --rollout.logging-steps ${LOGGING_STEPS} \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    --rollout.wandb-log-media ${WANDB_LOG_MEDIA} \
    --rollout.wandb-media-max-items ${WANDB_MEDIA_MAX_ITEMS} \
    --rollout.wandb-tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    "$@"
