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

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --sampler-engine-type fsdp \
    \
    --algorithm-path diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm \
    --loss-type custom \
    --loss-path diffusionrl_plugins.losses.minimal_loss.MinimalBackwardLoss \
    --loss-kwargs-json '{"scale": 1.0}' \
    --reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardWorker \
    --rollout-pipeline-path diffusionrl_plugins.rollout_fns.minimal_pipeline.minimal_pipeline \
    \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.3 \
    --shift 3.0 \
    --num-inference-steps 8 \
    --guidance-scale 4.5 \
    \
    --prompts-per-batch 1 \
    --batch-size 1 \
    --num-samples-per-prompt 2 \
    \
    --training-actor-direct-sampling true \
    --colocate-rollout-training true \
    --rollout-num-nodes 0 \
    --rollout-num-gpus-per-node 0 \
    --training-num-gpus-per-node "${NUM_GPUS}" \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 1 \
    --num-inner-epochs 1 \
    --max-grad-norm 1.0 \
    --use-lora true \
    --lora-rank 8 \
    --lora-alpha 16 \
    \
    --height 256 \
    --width 256 \
    \
    --num-rollout 5 \
    --save-steps 1000 \
    --logging-steps 1 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
