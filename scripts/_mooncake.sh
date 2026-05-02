#!/bin/bash
# scripts/_mooncake.sh
#
# Sourced by scripts/train_tq.sh. Encapsulates the lifecycle of the external
# `mooncake_master` HTTP+RPC server that backs TransferQueue's Mooncake
# storage manager.
#
# Required env (caller sets):
#   HEAD_IP                          IP of the head node (server binds 0.0.0.0)
#   TQ_MC_METADATA_SERVER_PORT       HTTP metadata server port (default 50041)
#   TQ_MC_RPC_PORT                   RPC server port (default 50051)
#
# Exposes:
#   mooncake_metadata_endpoint       echo http://${HEAD_IP}:${PORT}/metadata
#   mooncake_master_endpoint         echo ${HEAD_IP}:${RPC_PORT}
#   mooncake_check_ports             fail-fast if either port is held
#   mooncake_start                   nohup-launch master; populates MOONCAKE_LOG_FILE
#   mooncake_kill                    pkill -x mooncake_master (idempotent)

mooncake_metadata_endpoint() {
    echo "http://${HEAD_IP}:${TQ_MC_METADATA_SERVER_PORT:-50041}/metadata"
}

mooncake_master_endpoint() {
    echo "${HEAD_IP}:${TQ_MC_RPC_PORT:-50051}"
}

mooncake_check_ports() {
    local meta="${TQ_MC_METADATA_SERVER_PORT:-50041}" rpc="${TQ_MC_RPC_PORT:-50051}"
    if lsof -i :"${meta}" >/dev/null 2>&1; then
        echo "ERROR: TQ metadata port ${meta} in use; kill the holder or set TQ_MC_METADATA_SERVER_PORT" >&2
        return 1
    fi
    if lsof -i :"${rpc}" >/dev/null 2>&1; then
        echo "ERROR: TQ RPC port ${rpc} in use; kill the holder or set TQ_MC_RPC_PORT" >&2
        return 1
    fi
}

mooncake_start() {
    mooncake_kill || true
    mooncake_check_ports
    MOONCAKE_LOG_FILE="$(mktemp /tmp/mooncake.log.XXXXXX)"
    echo "[mooncake] starting master; log -> ${MOONCAKE_LOG_FILE}"
    nohup mooncake_master \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port="${TQ_MC_METADATA_SERVER_PORT:-50041}" \
        --rpc_port="${TQ_MC_RPC_PORT:-50051}" \
        --rpc_thread_num=64 \
        --default_kv_lease_ttl=100000000 \
        > "${MOONCAKE_LOG_FILE}" 2>&1 &
}

mooncake_kill() {
    pkill -x mooncake_master || true
}
