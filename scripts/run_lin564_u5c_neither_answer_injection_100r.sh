#!/usr/bin/env bash
# LIN-564 U5c: 100-rollout continuation of U5b's closed one-tool-call
# boundary plus decoder-side answer continuation for true NEITHER generations.
# Run persistently from tmux.

set -euo pipefail

readonly RUN_NAME=lin564_u5c_neither_answer_inject_100r_closed_tool_qwen3_1p7b_s42_bj3_20260721
export LIN564_RUN_NAME=$RUN_NAME
export LIN564_RUN_DIR=/mnt/bj/logs/$RUN_NAME
export LIN564_TOOL_PROTOCOL=closed-one-call
export LIN564_ENV_FILE=/root/unirl/.lin564_u5_runtime.env
# The rollout code and exact GPU continuation path passed the U5/U5b smokes.
export LIN564_SKIP_PREWARM=1

cd /root/unirl
exec bash scripts/run_lin564_token_parity.sh 100 \
  rollout.config.inject_answer_after_neither=true \
  rollout.config.neither_answer_max_new_tokens=1024
