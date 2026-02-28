#!/bin/bash
# =============================================================================
# DanceGRPO training with FLUX model — SGLang backend (Separate mode)
# =============================================================================
#
# This script mirrors train_dancegrpo_flux_separate.sh but uses the SGLang
# diffusion engine instead of the FSDP sampler.
#
# Key differences from the FSDP version:
#   - --sampler-engine-type sglang   (instead of --sampler-path)
#   - SGLANG_PYTHON_PATH must point to your local sglang/python directory
#   - --tp-size controls tensor-parallelism inside the SGLang engine
#   - Weight sync uses checkpoint_path (automatic for sglang engine)
#
# Prerequisites:
#   - SGLang with diffusion patches (branch local/diffusion-rl)
#     export SGLANG_PYTHON_PATH=/path/to/sglang/python
#   - FLUX model weights at PRETRAINED_MODEL path
#
# Usage:
#   bash train_dancegrpo_flux_sglang_separate.sh
#   ROLLOUT_GPUS=2 TRAINING_GPUS=2 bash train_dancegrpo_flux_sglang_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========== SGLang Configuration ==========
# Point to local sglang/python with diffusion-rl patches
export SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/home/aiops/wanghn/mmgrpo/sglang/python}"

# ========== Default values (can be overridden via environment) ==========
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/flux"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}

if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type flux \
    --sampler-engine-type sglang \
    --sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --replay-log-probs "${REPLAY_LOG_PROBS}" \
    --tp-size ${TP_SIZE} \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type flux_dance \
    --eta 0.3 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 3.5 \
    --timestep-fraction 0.6 \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 1 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --use-lora true \
    \
    --height 256 \
    --width 256 \
    \
    --num-rollout 300 \
    --save-steps 40 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
