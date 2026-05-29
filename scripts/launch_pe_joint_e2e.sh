#!/usr/bin/env bash
# Launch the PE-joint multi-track END-TO-END test (LIN-282 final) on a
# single 1x8 H20 colocate pod. Mirrors launch_pe_joint_smoke.sh but for
# the scaled recipe (P=6 N=4 M=4 → 96 SD samples / rollout, offload,
# 100 rollouts). Same model sizes as the smoke (Qwen3-0.6B + SD3.5-medium);
# 8B promotion is a follow-up once the batch-geometry path passes.
# Started at N=8 M=16 (768 samples) but hit CUDA OOM in DenoisingStage
# at rollout=0 (M=16 SD3 forward needed +12 GB activation on top of
# resident sgl engines + ~13 GB Ray/CUDA overhead per actor).
#
# Optional env vars (have sensible defaults below):
#   LOCATION             — gz | bj | zw — Ceph location for log tee
#                          (default: gz). Must match the pod's region.
#   WANDB_API_KEY        — required only when REPORT_TO_WANDB=true
#   WANDB_ENTITY         — wandb entity (default: linyuwus)
#   REPORT_TO_WANDB      — true|false (default: false)
#
# Pre-conditions on pod (handled by the launch sequence in
# /Users/linyu/.claude/plans/make-the-plan-enchanted-gem.md):
#   - `source /etc/bashrc` already executed (proxy vars).
#   - .venv activated; vllm == 0.20.0; sglang on diffusionrl branch.
#   - SD3.5 + Qwen3-0.6B available under /apdcephfs_zwfy8/.../public_models.
#   - pickscore HF cache pre-warmed (handled by the smoke's prior run).
#
# Usage on pod (inside tmux):
#   # Optional:
#   # export REPORT_TO_WANDB=true
#   # export WANDB_API_KEY='...'
#   bash scripts/launch_pe_joint_e2e.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# === Optional env vars w/ defaults ===
export LOCATION="${LOCATION:-zw}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
if [ "${REPORT_TO_WANDB}" = "true" ]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY required when REPORT_TO_WANDB=true}"
    export WANDB_ENTITY="${WANDB_ENTITY:-linyuwus}"
fi

# === Model paths ===
PUBLIC_MODELS_ROOT="/apdcephfs_zwfy8/share_305110755/hunyuan/public_models"
export PRETRAINED_MODEL="${PUBLIC_MODELS_ROOT}/stable-diffusion-3.5-medium"
export LLM_MODEL="${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-0.6B"

# === Proxy (libs that read env vars directly: wandb, transformers, HF) ===
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,.tencent.com}"

# === PyTorch allocator: reduce fragmentation across rollouts ===
# First e2e attempt's OOM trace showed 8 GiB reserved-but-unallocated
# despite the rollout/train cycle freeing. expandable_segments lets the
# allocator reuse fragmented blocks instead of failing on a large alloc.
# NOTE: expandable_segments requires Linux kernel 5.6+ (pidfd_getfd) for
# CUDA IPC used by SGLang weight sync. Disable if kernel doesn't support it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

# === Log destination (Ceph per-region log dir) ===
LOG_DIR="/mnt/${LOCATION}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pe-joint-e2e-$(date +%m%d-%H%M).log"

cd "${REPO_ROOT}"
echo "=== PE-Joint Multi-Track END-TO-END (LIN-282 final) ===" | tee "${LOG_FILE}"
echo "branch:           $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))" | tee -a "${LOG_FILE}"
echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}" | tee -a "${LOG_FILE}"
echo "LLM_MODEL:        ${LLM_MODEL}" | tee -a "${LOG_FILE}"
echo "LOG_FILE:         ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "scale:            P=6 N=4 M=4 → 96 SD samples / 24 LLM seqs per rollout" | tee -a "${LOG_FILE}"
echo "=========================================================" | tee -a "${LOG_FILE}"

# Delegate to the canonical single-node runner with our recipe.
exec bash scripts/run_experiment_single_node.sh pe_joint_multi_track_e2e 2>&1 | tee -a "${LOG_FILE}"
