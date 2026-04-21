#!/bin/bash
# =============================================================================
# Debug script for GRPO train-inference consistency analysis — SGLang rollout
#
# Sibling of debug_flowgrpo_sd3_consistency.sh.  Uses the same per-step tensor
# dump hook wired into GRPOAlgorithm.compute_loss, but drives the rollout via
# the SGLang engine in SEPARATE mode (dedicated rollout actors, separate from
# the training actor).
#
# The goal is the same: verify that on the first on-policy step
# (rollout_id=0, before any optimizer update), the training-side
# ``new_log_prob`` reproduces the ``old_log_prob`` received from the rollout
# side — i.e. ratio ≡ 1.  Any ratio drift at this point indicates a
# train-inference math mismatch (SGLang SDE kernel ↔ DiffusionRL replay, or
# precision / autocast / CFG handling).
#
# ── What this script CAN compare (for SGLang) ──
#   • Training-side  ``old_log_prob``  vs  ``new_log_prob``  (primary metric)
#     - ``replay`` mode: both sides compute log_prob via the training-side DiT
#       forward, so this should be identical and exposes
#       replay-vs-loss-forward drift within DiffusionRL.
#     - ``native`` mode: ``old_log_prob`` comes from SGLang's SDE kernel,
#       ``new_log_prob`` from DiffusionRL's, so any mismatch points to
#       SGLang-vs-DiffusionRL SDE math inconsistency.
#
# ── What this script CANNOT directly compare (without SGLang-side hooks) ──
#   • noise_pred / prev_sample_mean per step on the sampling side — SGLang's
#     worker processes are opaque to the training process.  ``sampling/``
#     directory will be empty; the analyzer falls back to its "Training
#     On-Policy Consistency" block, which only needs training-side tensors.
#
# Debug tensors are saved to:
#   ${DEBUG_OUTPUT_DIR}/training/step_XXX/*.pt  (from GRPO loss forward)
#
# After running, analyse with:
#   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR}
#
# Usage:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency_sglang.sh
#
#   # Native SGLang log_probs (stress SGLang's SDE kernel):
#   SGLANG_LOGPROB_MODE=native \
#     bash reproduce_scripts/debug_flowgrpo_sd3_consistency_sglang.sh
#
#   # With LoRA enabled (default: disabled to avoid first-rollout init drift):
#   USE_LORA=true \
#     bash reproduce_scripts/debug_flowgrpo_sd3_consistency_sglang.sh
#
#   # Override any sampling flag via pass-through CLI:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency_sglang.sh \
#       --sampling.eta 0.5
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/_check_wandb.sh"

# ── Ray environment hygiene ──
# Force a private local Ray cluster for this debug run. See the companion
# comment in debug_flowgrpo_sd3_consistency.sh for the full rationale — the
# short version is that a leftover ``ray start --head`` (e.g. from a previous
# multinode run) or an inherited RAY_ADDRESS env var will make
# diffusionrl.train attempt to connect to that cluster and then reject the
# ``num_cpus`` kwarg with:
#   ValueError: When connecting to an existing cluster, num_cpus and
#               num_gpus must not be provided.
unset RAY_ADDRESS
ray stop >/dev/null 2>&1 || true

# ── SGLang source resolution (same pattern as the multinode script) ──
# Prefer the local sibling checkout ../sglang/python when available so the
# scheduler runs the exact code in this repo; otherwise fall back to the
# installed sglang wheel.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi

# ── Defaults ──
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-3.5-medium}"
MODEL_TYPE="${MODEL_TYPE:-sd3}"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-${REPO_ROOT}/debug_output_sglang}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"

# Single-node topology: 1 rollout actor + 1 training actor = 2 GPUs total.
# Keep TP=1 to avoid tensor-parallel numerical differences during diff.
NUM_NODES=1
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-1}"
TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
GPUS_PER_NODE=$(( ROLLOUT_GPUS_PER_NODE + TRAINING_GPUS_PER_NODE ))

# replay is the default SGLang log-prob source in the production multinode
# script; keeping it identical here means ``new_log_prob`` and ``old_log_prob``
# both come from the DiffusionRL training-side DiT → any drift is purely
# within DiffusionRL.  Flip to native to stress-test SGLang's SDE kernel.
SGLANG_LOGPROB_MODE="${SGLANG_LOGPROB_MODE:-replay}"

# LoRA off by default: on the first rollout the SGLang side has not yet
# received any LoRA weights (set_lora_from_tensors is called inside the first
# sync_weights_to_rollout AFTER the first training step), so the two sides'
# effective model is base + (lora_B=0 ⇒ zero contribution).  That equivalence
# is *only* exact while lora_B==0 at init; if PEFT's initialization ever
# changes, the first-rollout on-policy check would silently become invalid.
# Disabling LoRA entirely removes this subtle trap for the debug case.
USE_LORA="${USE_LORA:-false}"

# ── Batch geometry (5 core knobs — see _batch_config.sh for docs) ──
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
PROMPTS_PER_BATCH="${PROMPTS_PER_BATCH:-4}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
# SGLang routes sampling through --rollout.rollout-batch-size, not
# --sampling.max-samples-per-request (which is ignored by the SGLang engine).
SAMPLING_FORWARD_BATCH="${SAMPLING_FORWARD_BATCH:-${NUM_SAMPLES_PER_PROMPT}}"
TRAINING_FORWARD_BATCH="${TRAINING_FORWARD_BATCH:-4}"
NUM_UPDATES="${NUM_UPDATES:-1}"   # must be 1 for on-policy debug
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${SAMPLING_FORWARD_BATCH}}"

# _batch_config.sh picks the GPU count from either (NUM_NODES, GPUS_PER_NODE)
# or NUM_GPUS — in that priority order.  For SGLang separate mode, batch
# geometry should be sized against the *training* data-parallel world-size
# (rollout GPUs never see a TrainingBatch), which equals
# ``NUM_NODES * TRAINING_GPUS_PER_NODE``.  The cleanest way to tell the
# resolver this is to leave NUM_NODES set and pass TRAINING_GPUS_PER_NODE as
# the per-node count.
_BATCH_GPUS_PER_NODE_SAVED="${GPUS_PER_NODE}"
GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE}"
source "${REPO_ROOT}/scripts/_batch_config.sh"
resolve_batch_params
validate_batch_params
print_batch_params
# Restore the physical per-node GPU count (rollout + training) for any
# downstream consumer that inspects it.
GPUS_PER_NODE="${_BATCH_GPUS_PER_NODE_SAVED}"

# ── Sanity: on-policy check requires num_updates == 1 ──
if [ "${NUM_UPDATES_PER_BATCH}" != "1" ]; then
    echo "ERROR: NUM_UPDATES must be 1 for on-policy debug (got ${NUM_UPDATES_PER_BATCH})." >&2
    echo "       Otherwise step_000 training dump already reflects an updated policy." >&2
    exit 1
fi

echo "============================================"
echo " GRPO Train-Inference Consistency Debug (SGLang)"
echo "============================================"
echo " DEBUG_OUTPUT_DIR:       ${DEBUG_OUTPUT_DIR}"
echo " PRETRAINED_MODEL:       ${PRETRAINED_MODEL}"
echo " ROLLOUT_GPUS_PER_NODE:  ${ROLLOUT_GPUS_PER_NODE}"
echo " TRAINING_GPUS_PER_NODE: ${TRAINING_GPUS_PER_NODE}"
echo " TP_SIZE:                ${TP_SIZE}"
echo " SGLANG_LOGPROB_MODE:    ${SGLANG_LOGPROB_MODE}"
echo " USE_LORA:               ${USE_LORA}"
echo " PROMPTS_PER_BATCH:      ${PROMPTS_PER_BATCH}"
echo " NUM_SAMPLES_PER_PROMPT: ${NUM_SAMPLES_PER_PROMPT}"
echo " NUM_UPDATES_PER_BATCH:  ${NUM_UPDATES_PER_BATCH}"
echo "============================================"

# Clean previous debug output so the analyser always sees only this run.
rm -rf "${DEBUG_OUTPUT_DIR}"
mkdir -p "${DEBUG_OUTPUT_DIR}"

# Weight sync dir: local, single-node scratch path.
#
# We use ``nccl_broadcast`` (same as the production multinode script) rather
# than ``tensor_payload``.  ``tensor_payload`` serialises GPU tensors via
# CUDA IPC and the receiver resolves the source device by matching its UUID
# in its own ``CUDA_VISIBLE_DEVICES``.  Ray assigns *disjoint* CVDs to the
# rollout and training actors when ``ROLLOUT_GPUS_PER_NODE + TRAINING_GPUS_PER_NODE == GPUS_PER_NODE``
# so the sender records a UUID the receiver cannot see, producing:
#   RuntimeError: update_weights_from_tensor failed: Invalid device_uuid=<...>
# ``nccl_broadcast`` sidesteps that entirely by opening a NCCL group between
# the two actors, so inter-GPU weight transfer works even with disjoint CVDs.
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-${DEBUG_OUTPUT_DIR}/weight_sync}"
mkdir -p "${WEIGHT_SYNC_DIR}"

REWARD_NAME="pickscore"
REWARD_DEVICE="cuda"

# LoRA CLI args conditionally — avoid passing use-lora=false plus rank/alpha
# on the same line, which the schema still accepts but is noisy.
LORA_ARGS=(--training.use-lora "${USE_LORA}")
if [ "${USE_LORA}" = "true" ]; then
    LORA_ARGS+=(--training.lora-rank 32 --training.lora-alpha 64)
fi

check_wandb_auth

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type "${MODEL_TYPE}" \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-components "${REWARD_NAME}" \
    --reward.local-reward-device "${REWARD_DEVICE}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --rollout.mode separate \
    --rollout.rollout-engine sglang \
    --rollout.num-gpus-per-actor "${TP_SIZE}" \
    --rollout.tp-size "${TP_SIZE}" \
    --rollout.rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --sampling.guidance-scale 1.0 \
    --algorithm.rollout-scheduler.timestep-fraction "[0.0, 0.5]" \
    \
    --algorithm.shuffle-seed 42 \
    --algorithm.shuffle-samples false \
    --algorithm.prompts-per-rollout "${PROMPTS_PER_BATCH}" \
    --training.micro-batch-size "${MICRO_BATCH_SIZE}" \
    --training.num-updates-per-batch "${NUM_UPDATES_PER_BATCH}" \
    --algorithm.samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}" \
    --algorithm.kwarg clip_range=1e-4 \
    --algorithm.kwarg use_kl_penalty=true \
    --algorithm.kwarg kl_coef=0.0 \
    --algorithm.adv-normalization group \
    --algorithm.use-global-std true \
    \
    --sync.protocol nccl_broadcast \
    --sync.dir "${WEIGHT_SYNC_DIR}" \
    --ray.rollout-num-nodes "${NUM_NODES}" \
    --ray.rollout-num-gpus-per-node "${ROLLOUT_GPUS_PER_NODE}" \
    --ray.training-num-gpus-per-node "${TRAINING_GPUS_PER_NODE}" \
    --ray.training-num-nodes "${NUM_NODES}" \
    --ray.offload-train false \
    --ray.offload-rollout false \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    "${LORA_ARGS[@]}" \
    \
    --sampling.height 512 \
    --sampling.width 512 \
    \
    --rollout.num-rollout 3 \
    --rollout.save-steps 0 \
    --evaluation.eval-steps 0 \
    --logging.logging-steps 1 \
    --rollout.output-dir "${DEBUG_OUTPUT_DIR}/train_output" \
    --logging.report-to-wandb false \
    \
    --debug.output-dir "${DEBUG_OUTPUT_DIR}" \
    \
    "$@"

echo ""
echo "============================================"
echo " Debug tensors saved to: ${DEBUG_OUTPUT_DIR}"
echo ""
echo " Analyse with:"
echo "   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR}"
echo ""
echo " Primary metric for this script is the 'Training On-Policy Consistency'"
echo " block in the analyser output — compare old_log_prob vs new_log_prob at"
echo " step_000 (|ratio - 1| should be ~0 modulo fp precision)."
echo "============================================"
