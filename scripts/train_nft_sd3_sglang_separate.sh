#!/bin/bash
# =============================================================================
# DiffusionNFT training with SD3 model - SGLang backend (Separate mode)
# =============================================================================
#
# NOTE:
#   sampling_mode='training_actor' currently requires sampler_engine_type=fsdp.
#   This script is the SGLang equivalent in separate rollout/training mode.
#
# Usage:
#   bash train_nft_sd3_sglang_separate.sh
#   ROLLOUT_GPUS=4 TRAINING_GPUS=4 bash train_nft_sd3_sglang_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables (.env)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# Prefer local sibling sglang checkout when available; otherwise use installed package.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/nft_sd3_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-6}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-24}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}

if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
NFT_ALGO_KWARGS=${NFT_ALGO_KWARGS:-"{\"beta\":0.1,\"adv_mode\":\"raw\",\"adv_clip_max\":5.0,\"use_adaptive_weight\":true,\"train_timestep_mode\":\"all\",\"shuffle_train_timesteps\":true,\"apply_time_shift_in_loss\":false,\"use_reference_ema\":true,\"ema_decay\":0.001,\"decay_type\":\"warmup\",\"ema_flat_steps\":75,\"ema_uprate\":0.0075,\"ema_uphold\":0.999,\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}"}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-engine-type sglang \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --sampling.replay-log-probs "${REPLAY_LOG_PROBS}" \
    --sampling.tp-size ${TP_SIZE} \
    --algorithm.algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.time-shift 3.0 \
    --sampling.sde-type dpm2 \
    --sampling.num-inference-steps 10 \
    --sampling.guidance-scale 1.0 \
    --sampling.sampling-adapter old \
    --algorithm.algorithm-kwargs "${NFT_ALGO_KWARGS}" \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --training.gradient-accumulation-batch-size ${BATCH_SIZE} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.kl-coef 0.0001 \
    --algorithm.adv-normalization group \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --ray.colocate-rollout-training false \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 3e-4 \
    --training.update-mode single_update \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --rollout.eval-steps 60 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
