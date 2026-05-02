#!/bin/bash
# =============================================================================
# Generic multi-node TransferQueue (Mooncake) launcher.
#
# Usage:
#   bash scripts/train_tq.sh <experiment> [role] [hydra overrides...]
#
# Same as scripts/train.sh, plus:
#   - Brings up `mooncake_master` on the head node (HTTP + RPC backend).
#   - Composes +transfer_queue=mooncake_tuned (overlay yaml in the
#     transfer_queue config group; carries SD3-tuned static fields).
#   - Injects transfer_queue.metadata_server / master_server_address Hydra
#     overrides (host-dependent; built from $HEAD_IP and TQ_MC_*_PORT env).
#   - Registers an EXIT trap to tear mooncake_master down on the head.
#
# Workers, stop, status skip the mooncake lifecycle entirely.
#
# To use a different TQ variant, override at the CLI:
#   bash scripts/train_tq.sh <exp> +transfer_queue=mooncake
# (user "$@" is forwarded last so user CLI beats wrapper-injected overrides.)
#
# Knobs:
#   TQ_MC_METADATA_SERVER_PORT  default 50041
#   TQ_MC_RPC_PORT              default 50051
#   TQ_LOGGING_LEVEL            default WARNING
#   MC_YLT_LOG_LEVEL            default warning
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT="${1:-}"
if [ -z "${EXPERIMENT}" ]; then
    echo "ERROR: experiment name required (first positional arg)" >&2
    echo "Usage: bash scripts/train_tq.sh <experiment> [role] [hydra overrides...]" >&2
    exit 1
fi
shift

# Optional role token — mirrors scripts/train.sh:89-97 parsing.
ROLE_TOKEN=""
case "${1:-}" in
    auto|auto_head|auto_worker|head|worker|train|stop|status)
        ROLE_TOKEN="$1"; shift ;;
esac

# Resolve auto -> auto_head / auto_worker — mirrors scripts/train.sh:99-104.
RESOLVED_ROLE="${ROLE_TOKEN:-auto}"
if [ "${RESOLVED_ROLE}" = "auto" ] && [ -n "${INDEX:-}" ]; then
    [ "${INDEX}" = "0" ] && RESOLVED_ROLE="auto_head" || RESOLVED_ROLE="auto_worker"
fi

# TQ knobs + helper.
HEAD_IP="${HEAD_IP:-${CHIEF_IP:-}}"
export TQ_MC_METADATA_SERVER_PORT="${TQ_MC_METADATA_SERVER_PORT:-50041}"
export TQ_MC_RPC_PORT="${TQ_MC_RPC_PORT:-50051}"
export TQ_LOGGING_LEVEL="${TQ_LOGGING_LEVEL:-WARNING}"
export MC_YLT_LOG_LEVEL="${MC_YLT_LOG_LEVEL:-warning}"

# shellcheck source=_mooncake.sh
source "${SCRIPT_DIR}/_mooncake.sh"

# Start mooncake on head/train roles only. Workers, stop, status skip.
case "${RESOLVED_ROLE}" in
    auto_head|head|train|auto)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP (or CHIEF_IP) required" >&2; exit 1; fi
        if ! command -v mooncake_master >/dev/null 2>&1; then
            echo "ERROR: mooncake_master not on PATH; install the mooncake binary" >&2; exit 1
        fi
        mooncake_start
        trap mooncake_kill EXIT
        ;;
esac

# Build TQ Hydra overrides. stop/status skip; they don't drive python.
TQ_OVERRIDES=()
if [ -n "${HEAD_IP}" ] && [ "${RESOLVED_ROLE}" != "stop" ] && [ "${RESOLVED_ROLE}" != "status" ]; then
    TQ_OVERRIDES+=(
        "+transfer_queue=mooncake_tuned"
        "transfer_queue.metadata_server=$(mooncake_metadata_endpoint)"
        "transfer_queue.master_server_address=$(mooncake_master_endpoint)"
    )
fi

# Delegate. Argument order: <experiment> <role-or-empty> <wrapper TQ overrides> "$@"
# scripts/train.sh forwards "$@" verbatim to python -m diffusionrl.train at the
# tail of the Hydra override list, so user-provided overrides in "$@" still
# beat our wrapper-injected ones on conflict.
bash "${SCRIPT_DIR}/train.sh" \
    "${EXPERIMENT}" \
    ${ROLE_TOKEN:+"${ROLE_TOKEN}"} \
    ${TQ_OVERRIDES[@]+"${TQ_OVERRIDES[@]}"} \
    "$@"
