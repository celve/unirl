#!/bin/bash
# Shared wandb credential check.
# Source this file from any launch script AFTER setting REPORT_TO_WANDB.
#
# Usage:
#   source "${REPO_ROOT}/scripts/_check_wandb.sh"
#   check_wandb_auth          # uses $REPORT_TO_WANDB
#   check_wandb_auth true     # explicit override

check_wandb_auth() {
    local enabled="${1:-${REPORT_TO_WANDB:-false}}"
    if [ "${enabled}" != "true" ]; then
        return 0
    fi

    # 1. WANDB_API_KEY env var
    if [ -n "${WANDB_API_KEY:-}" ]; then
        return 0
    fi

    # 2. wandb login (~/.netrc)
    if [ -f "${HOME}/.netrc" ] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
        return 0
    fi

    echo "ERROR: --logging.report-to-wandb is true but no WandB credentials found." >&2
    echo "  Fix with ONE of:" >&2
    echo "    1) export WANDB_API_KEY=<your-key>" >&2
    echo "    2) wandb login" >&2
    exit 1
}
