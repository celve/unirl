#!/bin/bash
# =============================================================================
# Single GPU Test for Ray Colocate/Separate Modes
# =============================================================================
#
# Purpose: Test Ray actor initialization with minimal resources
# to debug colocate/separate mode issues without OOM interference.
#
# Usage:
#   bash test_ray_single_gpu.sh colocate  # Test colocate mode
#   bash test_ray_single_gpu.sh separate  # Test separate mode
#   bash test_ray_single_gpu.sh training  # Test training-actor sampling mode
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODE=${1:-colocate}
shift 2>/dev/null || true  # Remove mode argument so $@ doesn't pass it to python
echo "Testing Ray mode: ${MODE}"

# Use SD3-medium as it's smaller than FLUX
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR="${REPO_ROOT}/outputs/test_ray_${MODE}"
DATA_PATH="${REPO_ROOT}/data/samples/prompts_toy.json"

if [ "$MODE" == "colocate" ]; then
    # Colocate mode: share 1 GPU for both inference and training
    python -m diffusionrl.train \
        --pretrained-model-path "${PRETRAINED_MODEL}" \
        --model-type sd3 \
        --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
        --algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
        --reward-path diffusionrl.workers.reward.local.LocalRewardWorker \
        --reward-model-name ocr \
        --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
        --data-path "${DATA_PATH}" \
        \
        --loss-type nft \
        --shift 3.0 \
        --num-inference-steps 2 \
        --guidance-scale 1.0 \
        \
        --batch-size 1 \
        --num-samples-per-prompt 1 \
        --clip-range 1e-4 \
        --kl-coef 0.0001 \
        --advantage-type per_prompt \
        \
        --colocate-inference-training true \
        --inference-num-gpus-per-node 1 \
        --training-num-gpus-per-node 1 \
        --offload true \
        \
        --learning-rate 1e-5 \
        --gradient-accumulation-steps 1 \
        --max-grad-norm 1.0 \
        --lora-rank 8 \
        --lora-alpha 16 \
        --use-lora true \
        --use-fsdp false \
        \
        --height 512 \
        --width 512 \
        \
        --num-rollout 3 \
        --save-steps 100 \
        --eval-steps 100 \
        --logging-steps 1 \
        --output-dir "${OUTPUT_DIR}" \
        "$@"
elif [ "$MODE" == "training" ]; then
    # Training-actor sampling mode: use training actors for sampling, no inference actors
    python -m diffusionrl.train \
        --pretrained-model-path "${PRETRAINED_MODEL}" \
        --model-type sd3 \
        --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
        --algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
        --reward-path diffusionrl.workers.reward.local.LocalRewardWorker \
        --reward-model-name ocr \
        --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
        --data-path "${DATA_PATH}" \
        \
        --loss-type nft \
        --shift 3.0 \
        --num-inference-steps 2 \
        --guidance-scale 1.0 \
        \
        --batch-size 1 \
        --num-samples-per-prompt 1 \
        --clip-range 1e-4 \
        --kl-coef 0.0001 \
        --advantage-type per_prompt \
        \
        --sampling-backend training \
        --colocate-inference-training false \
        --inference-num-nodes 0 \
        --inference-num-gpus-per-node 0 \
        --training-num-gpus-per-node 1 \
        --offload false \
        \
        --learning-rate 1e-5 \
        --gradient-accumulation-steps 1 \
        --max-grad-norm 1.0 \
        --lora-rank 8 \
        --lora-alpha 16 \
        --use-lora true \
        --use-fsdp false \
        \
        --height 512 \
        --width 512 \
        \
        --num-rollout 3 \
        --save-steps 100 \
        --eval-steps 100 \
        --logging-steps 1 \
        --output-dir "${OUTPUT_DIR}" \
        "$@"
else
    # Separate mode: 1 GPU for inference, 1 GPU for training
    python -m diffusionrl.train \
        --pretrained-model-path "${PRETRAINED_MODEL}" \
        --model-type sd3 \
        --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
        --algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
        --reward-path diffusionrl.workers.reward.local.LocalRewardWorker \
        --reward-model-name ocr \
        --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
        --data-path "${DATA_PATH}" \
        \
        --loss-type nft \
        --shift 3.0 \
        --num-inference-steps 2 \
        --guidance-scale 1.0 \
        \
        --batch-size 1 \
        --num-samples-per-prompt 1 \
        --clip-range 1e-4 \
        --kl-coef 0.0001 \
        --advantage-type per_prompt \
        \
        --colocate-inference-training false \
        --inference-num-gpus-per-node 1 \
        --training-num-gpus-per-node 1 \
        --placement-strategy SPREAD \
        \
        --learning-rate 1e-5 \
        --gradient-accumulation-steps 1 \
        --max-grad-norm 1.0 \
        --lora-rank 8 \
        --lora-alpha 16 \
        --use-lora true \
        --use-fsdp false \
        \
        --height 512 \
        --width 512 \
        \
        --num-rollout 3 \
        --save-steps 100 \
        --eval-steps 100 \
        --logging-steps 1 \
        --output-dir "${OUTPUT_DIR}" \
        "$@"
fi
