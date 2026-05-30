#!/usr/bin/env bash
# run_experiment_multinode_v2.sh — multi-node launcher for the v2 trainer
# (`python -m diffusionrl.train_v2`, flat conf_v2/*.yaml).
#
# SPMD: this same script runs on every node (the bridge launcher fans it out and
# sets INDEX + CHIEF_IP + LOCAL_IP per node). Rank 0 (INDEX=0) starts the Ray
# head and runs the single training driver; every other rank joins Ray and idles
# so the head can place workers across the whole cluster.
#
# Unlike the v1 run_experiment_multinode_taiji.sh this does NOT pass a Hydra
# `+experiment=` group or any transfer-queue / mooncake data-plane: v2 trainside
# colocates rollout + train in-process, and the config is selected with
# `--config-name`. num_devices is derived from HOST_NUM * HOST_GPU_NUM.
#
# Required env (the bridge launcher sets these):
#   EXPERIMENT      conf_v2 config name (no .yaml), e.g. argrpo_qwen_vl_geo3k_mc_4x8
#   HOST_NUM        node count          (alias NUM_NODES)
#   HOST_GPU_NUM    GPUs per node       (alias GPUS_PER_NODE)
#   INDEX           this node's rank    (alias NODE_RANK; 0 = head/driver)
#   CHIEF_IP        head node IP
#   LOCAL_IP        this node's IP
# Optional: RAY_PORT (6379), RAY_CLUSTER_WAIT_S (30), plus any DATA_PATH /
#   EVAL_DATA_PATH / QWEN_VL_PATH the config reads via ${oc.env:...}.
#
# Extra args after the experiment name are forwarded as Hydra overrides.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${EXPERIMENT:-}" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Usage: EXPERIMENT=<conf_v2 name> $0   (or: $0 <conf_v2 name> [hydra overrides...])" >&2
        exit 2
    fi
    EXPERIMENT="$1"
    shift
fi

NUM_NODES="${NUM_NODES:-${HOST_NUM:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
RAY_PORT="${RAY_PORT:-6379}"
NUM_DEVICES=$(( NUM_NODES * GPUS_PER_NODE ))

# This node's routable IP. Prefer LOCAL_IP; else the IP on the chief's /16
# (multi-NIC nodes often have a container-internal IP first); else the first.
all_ips="$(hostname -I 2>/dev/null || true)"
NODE_IP="${NODE_IP:-${LOCAL_IP:-}}"
if [ -z "${NODE_IP}" ] && [ -n "${CHIEF_IP:-}" ]; then
    chief_subnet="$(echo "${CHIEF_IP}" | cut -d. -f1-2)"
    NODE_IP="$(echo "${all_ips}" | tr ' ' '\n' | grep "^${chief_subnet}\." | head -1 || true)"
fi
[ -z "${NODE_IP}" ] && NODE_IP="$(echo "${all_ips}" | awk '{print $1}')"
NODE_IP="${NODE_IP:-127.0.0.1}"
HEAD_IP="${HEAD_IP:-${CHIEF_IP:-${NODE_IP}}}"
export LOCAL_IP="${LOCAL_IP:-${NODE_IP}}"

echo "[v2] rank=${NODE_RANK} ip=${NODE_IP} head=${HEAD_IP} ${NUM_NODES}x${GPUS_PER_NODE} -> num_devices=${NUM_DEVICES} experiment=${EXPERIMENT}"

ray stop >/dev/null 2>&1 || true

if [ "${NODE_RANK}" = "0" ]; then
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --num-gpus="${GPUS_PER_NODE}"
else
    # Worker: join the head's cluster and idle. Only rank 0 runs the driver.
    until ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${GPUS_PER_NODE}"; do
        echo "[v2 worker ${NODE_IP}] Ray head not ready; retry in 5s..."
        sleep 5
    done
    echo "[v2 worker ${NODE_IP}] joined Ray at ${HEAD_IP}:${RAY_PORT}; head owns the driver."
    exec tail -f /dev/null
fi

# rank 0 only: let the cluster assemble, then drive. The driver connects to the
# head we just started via RAY_ADDRESS=auto (the v2 trainer never calls
# ray.init itself — it relies on Ray auto-init against this env).
echo "[v2 head] waiting ${RAY_CLUSTER_WAIT_S:-30}s for ${NUM_NODES} node(s) to join..."
sleep "${RAY_CLUSTER_WAIT_S:-30}"
ray status 2>&1 | grep -E 'GPU|node_' || echo "[v2 head] ray status not ready"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
CMD=( python -m "${TRAIN_MODULE:-diffusionrl.train_v2}" --config-name="${EXPERIMENT}" "num_devices=${NUM_DEVICES}" )
CMD+=( "$@" )
echo "[v2 head] exec: ${CMD[*]}"
exec "${CMD[@]}"
