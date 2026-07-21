#!/usr/bin/env bash
# LIN-564 U5: U4's closed one-tool-call boundary plus decoder-side answer
# continuation for true NEITHER generations. Run persistently from tmux.

set -euo pipefail

readonly RUN_NAME=lin564_u5b_neither_answer_inject_concatfix_closed_tool_qwen3_1p7b_s42_bj3_20260721
export LIN564_RUN_NAME=$RUN_NAME
export LIN564_RUN_DIR=/mnt/bj/logs/$RUN_NAME
export LIN564_TOOL_PROTOCOL=closed-one-call
export LIN564_ENV_FILE=/root/unirl/.lin564_u5_runtime.env
# U5's real-GPU decoder-prefix and closed-boundary smokes already passed on this
# unchanged rollout code. U5b changes only trainer-side metadata sanitation.
export LIN564_SKIP_PREWARM=1

cd /root/unirl
exec bash scripts/run_lin564_token_parity.sh 10 \
  rollout.config.inject_answer_after_neither=true \
  rollout.config.neither_answer_max_new_tokens=1024
