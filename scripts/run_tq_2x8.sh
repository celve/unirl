#!/usr/bin/env bash
# Painless N-node launcher for validating TransferQueue on the
# grpo_flux2_klein9b_trainside_2x8 recipe. Thin wrapper over
# run_experiment_multinode.sh that injects the TQ overrides AND auto-derives
# networking from the platform's CHIEF_IP / INDEX.
#
# SAME command on EVERY node — no per-node HEAD_IP / NODE_IP / NODE_RANK. Submit
# it once as the platform multi-node job entrypoint; the platform fans it out to
# all nodes and sets INDEX (node rank) + CHIEF_IP (head addr). Data path defaults
# to the repo's bundled pickscore (run_experiment_multinode.sh default).
#
# TQ knobs (env):
#   TQ_BACKEND  simple | mooncake | none   (default: mooncake)
#   PROTOCOL    rdma | tcp                 (mooncake only; default: rdma)
#   MOONCAKE_METADATA_URL   http://<HEAD_IP>:<port>/metadata   (mooncake only)
#   MOONCAKE_MASTER_ADDR    <HEAD_IP>:<rpc_port>               (mooncake only)
#
# Validation ladder (IDENTICAL line on every node / one platform-job submit):
#   # A) multi-node mechanism, zero infra (no mooncake master needed):
#   TQ_BACKEND=simple bash scripts/run_tq_2x8.sh
#   # B) mooncake wiring + buffer sizing, no RDMA:
#   TQ_BACKEND=mooncake PROTOCOL=tcp  MOONCAKE_METADATA_URL=http://$CHIEF_IP:8080/metadata \
#     MOONCAKE_MASTER_ADDR=$CHIEF_IP:50051 bash scripts/run_tq_2x8.sh
#   # C) the real RDMA validation (HCA/PIX affinity, -800):
#   TQ_BACKEND=mooncake PROTOCOL=rdma MOONCAKE_METADATA_URL=http://$CHIEF_IP:8080/metadata \
#     MOONCAKE_MASTER_ADDR=$CHIEF_IP:50051 bash scripts/run_tq_2x8.sh
#
# (Exec-per-node by hand also works — same line in each node's shell; CHIEF_IP/INDEX
# are already in the taiji env. Explicit NODE_RANK/HEAD_IP/NODE_IP still override.)
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

# Auto-derive networking from the platform's CHIEF_IP so the SAME command works on
# every node — no per-node HEAD_IP / NODE_IP / NODE_RANK. On multi-NIC/container nodes
# `hostname -I` often yields a container-internal IP that peers can't reach; instead
# pick this node's IP on CHIEF_IP's /16 (the chief picks CHIEF_IP itself, workers pick
# their own routable IP). NODE_RANK still comes from INDEX (run_experiment_multinode default).
if [ -n "${CHIEF_IP:-}" ]; then
  : "${HEAD_IP:=$CHIEF_IP}"; export HEAD_IP
  if [ -z "${NODE_IP:-}" ]; then
    _pfx="$(echo "$CHIEF_IP" | cut -d. -f1-2)"
    _ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep "^${_pfx}\." | head -1)"
    [ -n "$_ip" ] && export NODE_IP="$_ip"
  fi
fi

echo "[run_tq_2x8] experiment=${EXPERIMENT} TQ_BACKEND=${TQ_BACKEND} PROTOCOL=${PROTOCOL} NODE_RANK=${NODE_RANK:-${INDEX:-0}} HEAD_IP=${HEAD_IP:-?} NODE_IP=${NODE_IP:-?}"
exec "${SCRIPT_DIR}/run_experiment_multinode.sh" "${EXPERIMENT}" "${TQ_OVERRIDES[@]}" "$@"
