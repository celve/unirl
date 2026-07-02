#!/bin/bash
# Megatron-backend e2e smoke: Qwen3-0.6B GRPO on DAPO-math, DP=8, SGLang colocate.
# Success signal: reward curve trends UP (not flat/fluctuating).
source /etc/bashrc
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,localhost,127.0.0.1,mirrors.tencent.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=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
export QWEN3_PATH=/root/unirl/models/local/Qwen3-0.6B
export DATA_PATH=${DATA_PATH:-/root/unirl/data/dapo_math/train_subset.jsonl}

# Free the keep-warm burn before allocating GPUs.
bash /mnt/bj/diffrl_runtime/burn.sh node-stop 2>/dev/null || true
sleep 5
cd /root/unirl

echo "=== Megatron smoke $(date): Qwen3-0.6B nd=8 ==="
.venv-sglang/bin/python -m unirl.train_ar \
  --config-name=ar/qwen3_grpo_4b_megatron_sglang \
  num_devices=8 num_rollouts=40 \
  batch_size=16 data_source.args.algorithm.prompts_per_rollout=16 \
  sampling.max_new_tokens=1024 rollout.config.max_new_tokens=1024 \
  algorithm.horizon=1024 \
  stack.num_updates_per_batch=1 \
  eval_interval=0 save_interval=0 \
  logging.run_name=megatron_smoke_qwen3-0p6b \
  2>&1 | tee /root/unirl/megatron_smoke.log
echo "=== exited ${PIPESTATUS[0]} $(date) ==="
