#!/bin/bash
# HI3 SP bf16 divergence bisection runner: sp=1 reference then sp=8 compare.
# Env overrides: DTYPE (bf16|fp32) LAYERS SEQLEN MODE.
set -x
cd /root/unirl
source /etc/bashrc 2>/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HI3_MODEL_PATH=${HI3_MODEL_PATH:-/dockerdata/HunyuanImage-3-Instruct}
export DTYPE=${DTYPE:-bf16} LAYERS=${LAYERS:-8} SEQLEN=${SEQLEN:-66} MODE=${MODE:-gen_text}
rm -f /tmp/hi3_bisect_*.pt
echo "=== REF sp=1  DTYPE=$DTYPE LAYERS=$LAYERS SEQLEN=$SEQLEN MODE=$MODE ==="
.venv/bin/torchrun --nproc_per_node=1 --master_port=29580 tests/distributed/parallel/sp_hi3_bisect.py
echo "=== COMPARE sp=8 ==="
.venv/bin/torchrun --nproc_per_node=8 --master_port=29581 tests/distributed/parallel/sp_hi3_bisect.py
echo "=== DONE ==="
