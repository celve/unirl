#!/bin/bash
# =============================================================================
# MixGRPO training with SD3 model - SGLang backend (Separate mode)
# =============================================================================
#
# NOTE:
#   training_actor_direct_sampling=true currently requires sampler_engine_type=fsdp.
#   This script is the SGLang equivalent in separate rollout/training mode.
#
# Usage:
#   bash train_mixgrpo_sd3_sglang_separate.sh
#   ROLLOUT_GPUS=4 TRAINING_GPUS=4 bash train_mixgrpo_sd3_sglang_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/home/aiops/wanghn/mmgrpo/sglang/python}"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_sd3_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
WINDOW_MAX_ITERS_PER_GROUP=${WINDOW_MAX_ITERS_PER_GROUP:-10}
WINDOW_MIN_ITERS_PER_GROUP=${WINDOW_MIN_ITERS_PER_GROUP:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}

if [ "${NUM_SAMPLES_PER_PROMPT}" -lt 2 ]; then
    echo "ERROR: MixGRPO uses group advantages; set NUM_SAMPLES_PER_PROMPT >= 2 to avoid NaN."
    exit 1
fi
if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-engine-type sglang \
    --sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --tp-size ${TP_SIZE} \
    --algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 4.5 \
    \
    --mixed-sampling true \
    --sde-ratio 0.5 \
    --timestep-strategy window \
    --window-strategy progressive \
    --window-group-size 4 \
    --window-iters-per-group 25 \
    --window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP} \
    --window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP} \
    --window-overlap true \
    --window-roll-back true \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    --reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 2 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --num-rollout 300 \
    --save-steps 50 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
