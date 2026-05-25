#!/usr/bin/env bash
# Painless 2-node (2x8 = 16 GPU) launcher for validating TransferQueue on the
# grpo_flux2_klein9b_trainside_2x8 recipe. Thin wrapper over
# run_experiment_multinode.sh that just injects the TQ overrides.
#
# Run the SAME command on BOTH nodes; only NODE_RANK / NODE_IP differ
# (NODE_RANK=0 is the Ray head and runs the training driver; rank!=0 joins).
#
# TQ knobs (env):
#   TQ_BACKEND  simple | mooncake | none   (default: mooncake)
#   PROTOCOL    rdma | tcp                 (mooncake only; default: rdma)
#   MOONCAKE_METADATA_URL   http://<HEAD_IP>:<port>/metadata   (mooncake only)
#   MOONCAKE_MASTER_ADDR    <HEAD_IP>:<rpc_port>               (mooncake only)
# Plus all multinode env (see run_experiment_multinode.sh):
#   NODE_RANK, HEAD_IP, NODE_IP, NUM_NODES(=2), GPUS_PER_NODE(=8),
#   DATA_PATH, EVAL_DATA_PATH, WANDB_*.
#
# Incremental validation ladder (run on the HEAD node; worker mirrors with NODE_RANK=1):
#   # A) multi-node mechanism, zero infra (no mooncake master needed):
#   TQ_BACKEND=simple   NODE_RANK=0 HEAD_IP=$IP NODE_IP=$IP bash scripts/run_tq_2x8.sh
#   # B) mooncake wiring + buffer sizing, no RDMA:
#   TQ_BACKEND=mooncake PROTOCOL=tcp  MOONCAKE_METADATA_URL=http://$IP:8080/metadata \
#     MOONCAKE_MASTER_ADDR=$IP:50051  NODE_RANK=0 HEAD_IP=$IP NODE_IP=$IP bash scripts/run_tq_2x8.sh
#   # C) the real RDMA validation (HCA/PIX affinity, -800):
#   TQ_BACKEND=mooncake PROTOCOL=rdma MOONCAKE_METADATA_URL=http://$IP:8080/metadata \
#     MOONCAKE_MASTER_ADDR=$IP:50051  NODE_RANK=0 HEAD_IP=$IP NODE_IP=$IP bash scripts/run_tq_2x8.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT="${EXPERIMENT:-grpo_flux2_klein9b_trainside_2x8}"
TQ_BACKEND="${TQ_BACKEND:-mooncake}"
PROTOCOL="${PROTOCOL:-rdma}"

TQ_OVERRIDES=()
case "${TQ_BACKEND}" in
  simple)
    TQ_OVERRIDES+=("+transfer_queue=simple")
    ;;
  mooncake)
    : "${MOONCAKE_METADATA_URL:?set MOONCAKE_METADATA_URL=http://<HEAD_IP>:<port>/metadata (external mooncake_master)}"
    : "${MOONCAKE_MASTER_ADDR:?set MOONCAKE_MASTER_ADDR=<HEAD_IP>:<rpc_port> (external mooncake_master)}"
    TQ_OVERRIDES+=(
      "+transfer_queue=mooncake_tuned"
      "transfer_queue.protocol=${PROTOCOL}"
      "transfer_queue.metadata_server=${MOONCAKE_METADATA_URL}"
      "transfer_queue.master_server_address=${MOONCAKE_MASTER_ADDR}"
    )
    ;;
  none|off)
    : # baseline: TQ off — reproduces the driver gather-OOM
    ;;
  *)
    echo "Unknown TQ_BACKEND='${TQ_BACKEND}' (use simple|mooncake|none)" >&2
    exit 2
    ;;
esac

echo "[run_tq_2x8] experiment=${EXPERIMENT} TQ_BACKEND=${TQ_BACKEND} PROTOCOL=${PROTOCOL} NODE_RANK=${NODE_RANK:-0}"
exec "${SCRIPT_DIR}/run_experiment_multinode.sh" "${EXPERIMENT}" "${TQ_OVERRIDES[@]}" "$@"
