#!/bin/bash
# =============================================================================
# Plugin Demo (algorithm/loss/reward/rollout-pipeline via diffusionrl_plugins)
# =============================================================================
#
# This script demonstrates end-to-end plugin wiring with dotpaths:
# - algorithm: diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm
# - loss: diffusionrl_plugins.losses.minimal_loss.MinimalBackwardLoss
# - reward: diffusionrl_plugins.rewards.minimal_reward.MinimalRewardWorker
# - rollout pipeline: diffusionrl_plugins.rollout_fns.minimal_pipeline.minimal_pipeline
#
# Note:
# - `wan21` model and `minimal_sampler` are templates only. They are not used here.
# - This demo uses built-in SD3 model+sampler so it can run as a complete example.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/plugin_demo_sd3"}

NUM_GPUS=${NUM_GPUS:-1}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --sampling.sampler-engine-type fsdp \
    \
    --algorithm.algorithm-path diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm \
    --algorithm.loss-type custom \
    --algorithm.loss-path diffusionrl_plugins.losses.minimal_loss.MinimalBackwardLoss \
    --algorithm.loss-kwargs '{"scale": 1.0}' \
    --reward.reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardWorker \
    --rollout.rollout-pipeline-path diffusionrl_plugins.rollout_fns.minimal_pipeline.minimal_pipeline \
    \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type sde \
    --sampling.eta 0.3 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 8 \
    --sampling.guidance-scale 4.5 \
    \
    --algorithm.prompts-per-batch 1 \
    --training.gradient-accumulation-batch-size 1 \
    --algorithm.num-samples-per-prompt 2 \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --sampling.training-actor-direct-sampling true \
    --ray.colocate-rollout-training true \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node "${NUM_GPUS}" \
    \
    --training.learning-rate 1e-5 \
    --training.update-mode single_update \
    --training.max-grad-norm 1.0 \
    --training.use-lora true \
    --training.lora-rank 8 \
    --training.lora-alpha 16 \
    \
    --height 256 \
    --width 256 \
    \
    --rollout.num-rollout 5 \
    --rollout.save-steps 1000 \
    --rollout.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
