#!/bin/bash
# =============================================================================
# FlowGRPO training with SD3 model (SGLang colocate mode) — OCR reward
# =============================================================================
#
# Mapped from origin/reproduce-refactor's colocate OCR reproduce script onto
# the current refactored CLI namespace.
#
# Expected first-rollout reward: ~0.4
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"stabilityai/stable-diffusion-3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_sglang_colocate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/datasets/ocr/train.txt"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"${REPO_ROOT}/data/datasets/ocr/test.txt"}
NUM_GPUS=${NUM_GPUS:-8}

NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-24}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-48}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-24}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

LOCAL_MICRO_BATCH_SIZE=${LOCAL_MICRO_BATCH_SIZE:-12}
NUM_UPDATES_PER_LOCAL_BATCH=${NUM_UPDATES_PER_LOCAL_BATCH:-2}

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-"diffusionrl-flowgrpo"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"SD3.5-FlowGRPO-SGLang-Col"}
WANDB_LOG_MEDIA=${WANDB_LOG_MEDIA:-true}
WANDB_MEDIA_MAX_ITEMS=${WANDB_MEDIA_MAX_ITEMS:-48}
WANDB_TAGS=${WANDB_TAGS:-"sglang,colocate,sd3.5,flow,ocr"}
WANDB_ENTITY=${WANDB_ENTITY:-"diffusionrl-reproduce"}
LOGGING_STEPS=${LOGGING_STEPS:-1}

WANDB_ENTITY_ARGS=()
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARGS+=(--rollout.logging.wandb-entity "${WANDB_ENTITY}")
fi

SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-false}
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

FLOWGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=true"
    --algorithm.kwarg "kl_coef=0.04"
    --algorithm.adv-normalization "group"
    --algorithm.use-global-std "true"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

echo "==========================================="
echo "FlowGRPO SD3 — SGLang Colocate Mode"
echo "  Total GPUs: ${NUM_GPUS}"
echo "  Micro batch: ${LOCAL_MICRO_BATCH_SIZE}"
echo "  Updates/local batch: ${NUM_UPDATES_PER_LOCAL_BATCH}"
echo "  Samples/prompt: ${NUM_SAMPLES_PER_PROMPT}"
echo "  Prompts/batch: ${PROMPTS_PER_BATCH}"
echo "  Total samples/rollout: ${ROLLOUT_TOTAL_SAMPLES}"
echo "  Rollout batch size: ${ROLLOUT_BATCH_SIZE}"
echo "  Model: ${PRETRAINED_MODEL}"
echo "  Resolution: 512x512"
echo "  Weight sync: tensor_payload"
echo "==========================================="

# Ensure venv-provided NVIDIA libs are visible when runtime depends on them.
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/cublas/lib:${SITE_PKGS}/nvidia/cusolver/lib:${SITE_PKGS}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-model-name ocr \
    --reward.reward-location sampling_actor \
    --reward.local-reward-device cuda \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    --eval-data-path "${EVAL_DATA_PATH}" \
    \
    --rollout.topology.mode colocate_rollout \
    --rollout.topology.service-engine sglang \
    --rollout.topology.service-num-gpus 1 \
    --rollout.topology.rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --sampling.guidance-scale 4.5 \
    --sampling.timestep-fraction 0.99 \
    --sampling.logprob-source native \
    --sampling.replay-log-probs false \
    \
    "${FLOWGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout "${PROMPTS_PER_BATCH}" \
    --training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}" \
    --training.num-updates-per-local-batch "${NUM_UPDATES_PER_LOCAL_BATCH}" \
    --algorithm.samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}" \
    \
    --ray.training-num-gpus-per-node "${NUM_GPUS}" \
    --ray.rollout-num-gpus-per-node "${NUM_GPUS}" \
    --ray.offload-rollout true \
    --ray.offload-train true \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.control.num-rollout "${NUM_ROLLOUT:-10000}" \
    --rollout.artifacts.save-steps 0 \
    --rollout.evaluation.eval-steps 30 \
    --rollout.logging.logging-steps "${LOGGING_STEPS}" \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --rollout.logging.report-to-wandb "${REPORT_TO_WANDB}" \
    --rollout.logging.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.logging.run-name "${WANDB_RUN_NAME}" \
    --rollout.logging.wandb-log-media "${WANDB_LOG_MEDIA}" \
    --rollout.logging.wandb-media-max-items "${WANDB_MEDIA_MAX_ITEMS}" \
    --rollout.logging.wandb-tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    --sync.protocol tensor_payload \
    "$@"
