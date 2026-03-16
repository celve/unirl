#!/bin/bash
# =============================================================================
# DanceGRPO training with FLUX model — SGLang backend (Separate mode)
# =============================================================================
#
# This script mirrors the legacy FSDP-separate layout but uses the SGLang
# diffusion engine instead of the FSDP sampler.
#
# Key differences from the FSDP version:
#   - --sampling.sampler-engine-type sglang   (instead of --sampling.sampler-path)
#   - Optional local debug override via SGLANG_PYTHON_PATH/PYTHONPATH
#   - --sampling.tp-size controls tensor-parallelism inside the SGLang engine
#   - Weight sync uses checkpoint_path (automatic for sglang engine)
#
# Prerequisites:
#   - Install sglang[diffusion] (default runtime path)
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

# Load environment variables (.env)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# ========== SGLang Configuration ==========
# Prefer local sibling sglang checkout when available; otherwise use installed package.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi

# ========== Default values (can be overridden via environment) ==========
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type flux \
    --sampling.sampler-engine-type sglang \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --sampling.replay-log-probs "${REPLAY_LOG_PROBS}" \
    --sampling.tp-size ${TP_SIZE} \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flux_dance \
    --sampling.eta 0.3 \
    --sampling.time-shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 3.5 \
    --sampling.timestep-fraction 0.6 \
    \
    --algorithm.algorithm-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --training.gradient-accumulation-batch-size ${BATCH_SIZE} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.adv-normalization group \
    --algorithm.adv-clip-abs 5.0 \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --ray.colocate-rollout-training false \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 1e-5 \
    --training.update-mode single_update \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank ${LORA_RANK} \
    --training.lora-alpha ${LORA_ALPHA} \
    --training.use-lora true \
    \
    --height 256 \
    --width 256 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 40 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
