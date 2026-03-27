#!/bin/bash
# =============================================================================
# Plugin Demo (algorithm/reward via diffusionrl_plugins)
# =============================================================================
#
# This script demonstrates end-to-end plugin wiring with dotpaths:
# - algorithm: diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm
# - reward: diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer
#
# Note:
# - `wan21` model and `minimal_sampler` are templates only. They are not used here.
# - This demo uses built-in SD3 model+sampler so it can run as a complete example.
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

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/plugin_demo_sd3"}

NUM_GPUS=${NUM_GPUS:-1}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
PLUGIN_ALGO_KWARG_ARGS=(
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    \
    --algorithm.algorithm-path diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm \
    --reward.reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer \
    \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.3 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 8 \
    --sampling.guidance-scale 4.5 \
    \
    "${PLUGIN_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout 1 \
    --training.local-micro-batch-size 1 \
    --algorithm.samples-per-prompt 2 \
    \
    --rollout.topology.mode direct_sampling \
--ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node "${NUM_GPUS}" \
    \
    --training.learning-rate 1e-5 \
    --training.max-grad-norm 1.0 \
    --training.use-lora true \
    --training.lora-rank 8 \
    --training.lora-alpha 16 \
    \
    --height 256 \
    --width 256 \
    \
    --rollout.control.num-rollout 5 \
    --rollout.artifacts.save-steps 1000 \
    --rollout.logging.logging-steps 1 \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --sync.protocol disabled \
    "$@"
