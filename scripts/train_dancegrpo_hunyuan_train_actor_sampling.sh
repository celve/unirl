#!/bin/bash
# =============================================================================
# DanceGRPO training with HunyuanVideo (Training-actor sampling / Colocate)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/XueZeyue/DanceGRPO)
#   Script:  scripts/finetune/finetune_hunyuan_grpo.sh
#   Paper:   arXiv:2505.07818 "DanceGRPO: Unleashing GRPO on Visual Generation"
#
# MODE: Training-actor sampling (colocate pattern)
#   - NO dedicated rollout actors
#   - Training actors handle BOTH sampling and gradient updates
#   - This is the CLOSEST match to DanceGRPO's original approach where each
#     GPU runs sampling → reward → training sequentially
#   - Advantage: fewer total GPUs needed (N GPUs instead of 2N)
#   - Disadvantage: no async overlap between sampling and training
#
# ENGINE: FSDP (via training actors)
#   When rollout.topology.mode='direct_rollout', the TrainingActor lazy-loads VAE and
#   text encoder, then uses the FSDP-wrapped transformer to sample.
#   Log_prob is computed during sampling (needed for GRPO).
#
# COMPARISON with DanceGRPO original:
#   DanceGRPO:  torchrun --nnodes=4 --nproc_per_node=8  (32 GPUs)
#               Each rank: load data → sample 24 videos → compute reward
#               → compute advantage → GRPO train → sync weights
#
#   This script: Ray actors, each training actor does the same loop.
#               rollout.topology.mode='direct_rollout' → training actors call generate()
#               with the same FSDP model they use for training.
#
# ALIGNMENT with DanceGRPO:
#   All DanceGRPO-specific hyperparameters are preserved:
#   - eta=0.25, shift=5.0, timestep_fraction=0.6
#   - guidance tensor=6018.0, init_same_noise=true
#   - group advantage, clip_range=1e-4, adv_clip_max=5.0
#   - lr=1e-5, max_grad_norm=1.0, weight_decay=0.0001
#   - 480×480×53 frames
#   See the separate script header for full parameter mapping table.
#
# NOTE on guidance vs cfg:
#   In DanceGRPO Hunyuan GRPO training, model forward uses guidance tensor 6018.0.
#   The CLI flag --cfg=0.0 in the original script controls dataset text-dropout rate,
#   not this guidance tensor value.
#
# GPU LAYOUT (default: 8 GPUs, all training):
#   DanceGRPO uses 16-32 H800 GPUs. This script defaults to 8 GPUs for
#   single-node testing. Scale up via NUM_GPUS / multi-node env vars.
#
#   Single node:  NUM_GPUS=8
#   Multi node:   TRAINING_NUM_NODES=4 TRAINING_GPUS_PER_NODE=8  (32 GPUs)
#
# MEMORY (per GPU, 80GB H800 recommended):
#   Training actor holds:
#   - FSDP transformer shard: ~13B/N_gpus params (fp32 master + bf16 forward)
#   - Optimizer states: 2× model shard (Adam momentum + variance)
#   - Sampling: VAE (~300M bf16), text encoder (~7B+300M bf16) — loaded once
#   - Activations: gradient checkpointing enabled to reduce peak memory
#
#   With 8 GPUs FULL_SHARD: ~40-50GB per GPU (tight on 40GB, OK on 80GB)
#   With 16+ GPUs: more comfortable, can increase resolution
#
#   For 40GB GPUs: enable --training.fsdp-cpu-offload true (slower but fits)
#
# NOTE on bestofn:
#   DanceGRPO bestofn=8 (keep top-8 of 24 samples). Not implemented in
#   diffusionrl. All K samples are used with group-relative advantages.
#
# NOTE on reward:
#   DanceGRPO uses VideoAlign (VQ score). Default here is hpsv2 (proxy).
#   For faithful reproduction, implement VideoAlign reward scorer.
#
# Training-actor sampling now reuses the main manager -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_dancegrpo_hunyuan_train_actor_sampling.sh
#   NUM_GPUS=16 bash train_dancegrpo_hunyuan_train_actor_sampling.sh
#   # Multi-node (32 GPUs):
#   TRAINING_NUM_NODES=4 TRAINING_GPUS_PER_NODE=8 \
#     bash train_dancegrpo_hunyuan_train_actor_sampling.sh
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


# ===== Configurable defaults (override via env vars or CLI "$@") =====
# Keep local checkpoints under models/local by default.
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/hunyuan-video"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_colocate"}
# Hunyuan path in diffusionrl currently expects plain prompts (text mode).
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy_8.txt"}

# GPU layout (single-node default)
NUM_GPUS=${NUM_GPUS:-8}
TRAINING_NUM_NODES=${TRAINING_NUM_NODES:-1}
TRAINING_GPUS_PER_NODE=${TRAINING_GPUS_PER_NODE:-${NUM_GPUS}}
TOTAL_GPUS=$((TRAINING_NUM_NODES * TRAINING_GPUS_PER_NODE))

# Sampling geometry
# DanceGRPO: num_generations=24 per rank.
# In train_actor_sampling, each training actor generates K samples per prompt.
# With TOTAL_GPUS actors, total_samples = PROMPTS_PER_BATCH × K.
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${TOTAL_GPUS}}

# Training geometry
LOCAL_MICRO_BATCH_SIZE=${LOCAL_MICRO_BATCH_SIZE-2}

# Resolution (DanceGRPO: 480×480, 53 frames)
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

# Reward model
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}
REWARD_LOCATION=${REWARD_LOCATION:-sampling_actor}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cuda}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-dancegrpo_hunyuan_train_actor_sampling}

# FSDP configuration
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-"false"}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
DANCEGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=false"
    --algorithm.adv-normalization "group"
    --algorithm.adv-clip-abs "5.0"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

# Note: shell-level path existence check removed. The Python-level fallback in
# arguments.py handles missing local paths by falling back to HuggingFace IDs.

# Fail fast for missing prompt source (avoid silent fallback to default prompts).
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
    echo "Set DATA_PATH to an existing prompt file (recommended: ${REPO_ROOT}/data/samples/video_prompts_toy.txt)."
    exit 1
fi
TOTAL_SAMPLES=$((PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT))
LOCAL_BATCH_SIZE=$((TOTAL_SAMPLES / TOTAL_GPUS))
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-${TOTAL_SAMPLES}}
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${LOCAL_MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}")
fi

echo "======================================================"
echo " DanceGRPO HunyuanVideo — Training-Actor Sampling"
echo "======================================================"
echo " Total GPUs:             ${TOTAL_GPUS} (${TRAINING_NUM_NODES} nodes × ${TRAINING_GPUS_PER_NODE})"
echo " Prompts per batch:      ${PROMPTS_PER_BATCH}"
echo " Samples per prompt (K): ${NUM_SAMPLES_PER_PROMPT}"
echo " Total samples/rollout:  ${TOTAL_SAMPLES}"
echo " Per-rank local batch:   ${LOCAL_BATCH_SIZE}"
echo " Gradient accum batch:   ${LOCAL_MICRO_BATCH_SIZE:-disabled}"
echo " Resolution:             ${HEIGHT}×${WIDTH}×${NUM_FRAMES}f"
echo " Reward:                 ${REWARD_MODEL_NAME}"
echo " FSDP CPU offload:       ${FSDP_CPU_OFFLOAD}"
echo "======================================================"

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type hunyuan \
    --sampling.sampler-path diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    `# ===== SDE Sampling (aligned with DanceGRPO) =====` \
    --sampling.sde-type dance \
    --sampling.eta 0.25 \
    --sampling.shift 5.0 \
    --sampling.num-inference-steps 16 \
    --sampling.guidance-scale 6018.0 \
    --sampling.timestep-fraction 0.6 \
    --sampling.init-same-noise true \
    \
    "${DANCEGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.topology.mode direct_rollout \
--sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-nodes ${TRAINING_NUM_NODES} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS_PER_NODE} \
    \
    `# ===== Training Hyperparams (aligned with DanceGRPO) =====` \
    --training.learning-rate 1e-5 \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.fsdp-cpu-offload ${FSDP_CPU_OFFLOAD} \
    --training.use-gradient-checkpointing true \
    \
    `# ===== Video Resolution =====` \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    `# ===== Rollout / Checkpoint =====` \
    --rollout.control.num-rollout 202 \
    --rollout.artifacts.save-steps 50 \
    --rollout.logging.logging-steps 1 \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --rollout.logging.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.logging.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.logging.run-name "${WANDB_RUN_NAME}" \
    --sync.protocol disabled \
    "$@"
