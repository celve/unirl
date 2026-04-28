#!/bin/bash
# =============================================================================
# MixGRPO training with WAN 2.1 model (training-actor direct sampling)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/_check_wandb.sh"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/wan2.1-t2v"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_wan_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}
NUM_GPUS=${NUM_GPUS:-8}

NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-25}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

HEIGHT=${HEIGHT:-320}
WIDTH=${WIDTH:-576}
NUM_FRAMES=${NUM_FRAMES:-49}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
ETA=${ETA:-0.7}
SHIFT=${SHIFT:-5.0}
SDE_RATIO=${SDE_RATIO:-0.5}

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_wan_train_actor_sampling}
WANDB_LOG_MEDIA=${WANDB_LOG_MEDIA:-true}
WANDB_MEDIA_MAX_ITEMS=${WANDB_MEDIA_MAX_ITEMS:-8}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-pickscore}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cuda}
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.micro-batch-size "${MICRO_BATCH_SIZE}")
fi

if [ "${NUM_SAMPLES_PER_PROMPT}" -lt 2 ]; then
    echo "ERROR: MixGRPO uses group advantages; set NUM_SAMPLES_PER_PROMPT >= 2." >&2
    exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2
    exit 1
fi

check_wandb_auth

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type wan21 \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-components "${REWARD_MODEL_NAME}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type dance \
    --sampling.eta "${ETA}" \
    --sampling.shift "${SHIFT}" \
    --sampling.num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --sampling.max-samples-per-request "${DIRECT_SAMPLING_BATCH_SIZE}" \
    --sampling.guidance-scale "${GUIDANCE_SCALE}" \
    --sampling.height "${HEIGHT}" \
    --sampling.width "${WIDTH}" \
    --sampling.num-frames "${NUM_FRAMES}" \
    --algorithm.rollout-scheduler.timestep-fraction "[${SDE_RATIO}]" \
    \
    --algorithm.kwarg "clip_range=1e-4" \
    --algorithm.kwarg "use_kl_penalty=false" \
    --algorithm.adv-normalization "group" \
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}" \
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}" \
    --algorithm.prompts-per-rollout "${PROMPTS_PER_BATCH}" \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --algorithm.samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}" \
    \
    --rollout.mode direct_sampling \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node "${NUM_GPUS}" \
    \
    --training.learning-rate 1e-5 \
    --training.max-grad-norm 1.0 \
    --training.use-lora false \
    --training.use-gradient-checkpointing true \
    \
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --evaluation.eval-steps 60 \
    --logging.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --logging.report-to-wandb "${REPORT_TO_WANDB}" \
    --logging.project-name "${WANDB_PROJECT_NAME}" \
    --logging.run-name "${WANDB_RUN_NAME}" \
    --logging.log-media "${WANDB_LOG_MEDIA}" \
    --logging.media-max-items "${WANDB_MEDIA_MAX_ITEMS}" \
    --sync.protocol disabled \
    "$@"
