#!/usr/bin/env bash
# LIN-564 U7: U6-equivalent policy/training settings with resilient Polaris
# transport and infrastructure-contaminated GRPO-group exclusion.

set -euo pipefail

readonly RUN_NAME=lin564_u7_polaris_reliability_100r_qwen3_1p7b_s42_gz8_20260722
export LIN564_RUN_NAME=$RUN_NAME
export LIN564_RUN_DIR=/mnt/gz/logs/$RUN_NAME
export LIN564_TOOL_PROTOCOL=closed-one-call
export LIN564_ENV_FILE=/root/.config/unirl/lin564_u7_runtime.env
export LIN564_SKIP_PREWARM=1
export LIN564_WEB_PREFLIGHT=1

cd /root/unirl
git grep -q 'class ToolExecutionResult' -- unirl/rollout/loop/tools/tool.py
git grep -q 'transient_exhausted_count' -- unirl/rollout/loop/tools/polaris.py
git grep -q '_infrastructure_group_exclusion' -- unirl/trainer/agentic.py
git grep -q 'reward_schema_version' -- unirl/utils/trajectory_dump.py
exec bash scripts/run_lin564_token_parity.sh 100 \
  rollout.config.inject_answer_after_neither=false \
  rollout.config.nudge_answer_after_neither=true \
  rollout.config.neither_answer_max_new_tokens=1024 \
  mask_answer_rescue_trigger_task_credit=true \
  answer_rescue_trigger_penalty=0.05
