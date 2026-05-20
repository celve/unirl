#!/bin/bash
# =============================================================================
# HunyuanVideo-1.5 T2V GRPO smoke — trainside (direct-sampling) on 1×8.
#
# NEW-DESIGN path: train_new.py + NewTrainActorGroup. No separate rollout
# actors. Default checkpoint is the diffusers-community 480p T2V variant
# (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`) under
# /apdcephfs_zwfy8/.../public_models — VAE is 32-channel, transformer
# in_channels=65=2*32+1, out_channels=32. Driver/stage agree on
# DEFAULT_LATENT_CHANNELS=32 (see HunyuanVideo15DiffusionStage:380).
#
# Smoke-tuning vs the YAML default (aggressive — goal is "see first
# rollout finish < 2 min", not training quality):
#   - prompts × samples: 2 × 4     (8 global = 1 per actor, smallest legal)
#   - num_inference_steps: 6       (vs YAML 50; just enough to confirm loop)
#   - height/width/num_frames: 320/320/5  (1 latent frame, 20×20 spatial)
#   - num_rollouts: 2
#   - load_vision_encoder=false    (frees ~1.6 GB; T2V doesn't need SigLIP)
# At YAML defaults (50 steps × 4 latent frames × 30×30 spatial × 54-layer
# MMDiT × 16 samples on 8 actors) the first rollout takes ~15 min — fine
# for real training but not for "did the wiring work?".
#
# Default DATA_PATH = data/samples/video_prompts_toy.txt (4 toy prompts;
# matches algorithm.prompts_per_rollout=4). Override via env or extra
# Hydra args:
#   bash scripts/run_hunyuan_video15_t2v_smoke.sh
#   bash scripts/run_hunyuan_video15_t2v_smoke.sh run.num_rollouts=10
#   HUNYUAN_VIDEO15_PATH=/path/to/tencent/HunyuanVideo1.5 bash scripts/run_hunyuan_video15_t2v_smoke.sh
#
# Tencent canonical (if/when downloaded) — point HUNYUAN_VIDEO15_PATH at
# it; the script auto-detects vae/config.json latent_channels and
# appends model.latent_channels override when it's not 32.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Activate the project venv ──
VENV_DIR="${VENV_DIR:-/root/diffusionrl/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: venv not found at ${VENV_DIR}; using system python: $(which python)"
fi

# ── Model path (community 480p T2V variant by default) ──
export HUNYUAN_VIDEO15_PATH="${HUNYUAN_VIDEO15_PATH:-/apdcephfs_zwfy8/share_305110755/hunyuan/public_models/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v}"

# ── Prompt data (toy 4-prompt video list under shared mount; matches
#    smoke prompts_per_rollout=4 exactly so the data source doesn't cycle) ──
export DATA_PATH="${DATA_PATH:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/data/samples/video_prompts_toy.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_PATH}}"

# ── Output ──
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hunyuan_video15_t2v_smoke}"
mkdir -p "${OUTPUT_DIR}"

# ── Runtime env ──
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/models/local/.hf_home}"

echo "================================================"
echo " HunyuanVideo-1.5 T2V smoke — trainside 1×8"
echo "================================================"
echo "Model:  ${HUNYUAN_VIDEO15_PATH}"
echo "Data:   ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ── Pre-flight: the bundle calls AutoencoderKLHunyuanVideo15 /
#    HunyuanVideo15Transformer3DModel / Qwen2_5_VLTextModel / ByT5 /
#    FlowMatchEulerDiscreteScheduler from the same root. All five
#    subfolders must exist.
for sub in transformer vae text_encoder text_encoder_2 tokenizer tokenizer_2 scheduler; do
    if [ ! -d "${HUNYUAN_VIDEO15_PATH}/${sub}" ]; then
        echo "ERROR: ${HUNYUAN_VIDEO15_PATH}/${sub} missing — not a HunyuanVideo-1.5 diffusers checkpoint?"
        exit 1
    fi
done

# Surface the actual VAE channel count so a mismatch with the
# driver-side default doesn't crash mid-rollout. The bundle uses this
# at stage init; we cross-check it here as a friendly fail-fast.
VAE_CHANNELS="$(python3 - <<PY
import json, sys
try:
    with open("${HUNYUAN_VIDEO15_PATH}/vae/config.json") as f:
        print(int(json.load(f).get("latent_channels", 0)))
except Exception as e:
    print(0)
PY
)"
echo "Detected VAE latent_channels: ${VAE_CHANNELS}"
if [ "${VAE_CHANNELS}" = "0" ]; then
    echo "WARN: could not read vae/config.json; stage will fall back to DEFAULT_LATENT_CHANNELS=32"
elif [ "${VAE_CHANNELS}" != "32" ]; then
    echo "INFO: non-canonical channel count detected — appending model.latent_channels=${VAE_CHANNELS} override."
    LATENT_CHANNELS_OVERRIDE="model.latent_channels=${VAE_CHANNELS}"
else
    LATENT_CHANNELS_OVERRIDE=""
fi
echo ""

# ── GPU snapshot ──
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Stop any stale Ray cluster ──
ray stop >/dev/null 2>&1 || true
sleep 1

# ── Hydra overrides ──
HYDRA_OVERRIDES=(
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
    "run.num_rollouts=2"

    # Smoke batch geometry: smallest legal (8 actors → ≥8 global; with
    # num_updates=1 the local_batch=1=local_mini geometry works out).
    "algorithm.prompts_per_rollout=2"
    "algorithm.samples_per_prompt=4"
    "training.plan.global_batch_size=8"
    "training.plan.local_batch_size=1"
    "training.plan.local_mini_batch_size=1"
    "training.plan.micro_batch_size=1"
    "training.plan.num_updates_per_batch=1"

    # Shrink the per-call HV1.5 forward to under ~10s (54-layer MMDiT is
    # the bottleneck; cutting steps + spatial tokens drops first-rollout
    # wall time from ~15 min to ~1–2 min).
    "sampling.num_inference_steps=6"
    "sampling.height=320"
    "sampling.width=320"
    "sampling.num_frames=5"
    # ``num_inference_steps=6`` × ``timestep_fraction=[0.0, 0.6]`` (YAML)
    # leaves only 3 timesteps in the SDE pool; the YAML's
    # ``num_sde_steps=8`` would fail validation. Cap to 2 (still gives a
    # non-trivial GRPO replay set without overshooting the small pool).
    "algorithm.scheduler.num_sde_steps=2"

    # Per-call chunk size for trainside generate.
    "rollout.plan.forward_batch_size=1"

    # T2V: skip SigLIP (~1.6 GB).
    "model.load_vision_encoder=false"

    "logging.report_to_wandb=false"
)
if [ -n "${LATENT_CHANNELS_OVERRIDE}" ]; then
    HYDRA_OVERRIDES+=("${LATENT_CHANNELS_OVERRIDE}")
fi

# Append any user-supplied overrides AFTER smoke defaults.
HYDRA_OVERRIDES+=("$@")

CMD=(
    python -m diffusionrl.train_new
    "+experiment=grpo_hunyuan_video15_t2v_videopickscore_new"
    "${HYDRA_OVERRIDES[@]}"
)

echo "Command:"
echo "  ${CMD[*]}"
echo ""

cleanup() {
    echo ""
    echo "Cleaning up..."
    ray stop >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

echo "========== Starting training =========="
cd "${REPO_ROOT}"
exec "${CMD[@]}"
