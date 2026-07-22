#!/usr/bin/env bash
# LIN-564 U6: user-nudge NEITHER rescue with intervention-aware task credit.
# The policy samples the complete <answer> turn; the triggering NEITHER Part is
# outside downstream task credit and receives only a small rescue-use penalty.

set -euo pipefail

readonly RUN_NAME=lin564_u6_user_nudge_credit_cut_100r_qwen3_1p7b_s42_bj3_20260722
export LIN564_RUN_NAME=$RUN_NAME
export LIN564_RUN_DIR=/mnt/bj/logs/$RUN_NAME
export LIN564_TOOL_PROTOCOL=closed-one-call
export LIN564_ENV_FILE=/root/.config/unirl/lin564_u6_runtime.env
export LIN564_SKIP_PREWARM=1

cd /root/unirl
exec bash scripts/run_lin564_token_parity.sh 100 \
  rollout.config.inject_answer_after_neither=false \
  rollout.config.nudge_answer_after_neither=true \
  rollout.config.neither_answer_max_new_tokens=1024 \
  mask_answer_rescue_trigger_task_credit=true \
  answer_rescue_trigger_penalty=0.05
