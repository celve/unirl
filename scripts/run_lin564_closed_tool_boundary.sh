#!/usr/bin/env bash
# Future LIN-564 clean react-style behavioral ablation: strict answer reward +
# one closed tool call per turn. Not literal AReaL stop-parameter parity.
# Delegates every operational invariant to the reproducible token-parity launcher.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export LIN564_TOOL_PROTOCOL=closed-one-call
export LIN564_RUN_NAME=${LIN564_RUN_NAME:-lin564_u4_closed_tool_boundary_qwen3_1p7b_s42_20260720}
export LIN564_RUN_DIR=${LIN564_RUN_DIR:-/mnt/gz/logs/$LIN564_RUN_NAME}

exec "$SCRIPT_DIR/run_lin564_token_parity.sh" "$@"
