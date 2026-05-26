#!/usr/bin/env bash
#
# Multi-node experiment launcher for the taiji platform. Two launch modes pick
# how Ray is brought up across nodes (LAUNCH); both end with the head running
# the single training driver and every other node joined to its Ray cluster:
#
#   LAUNCH=spmd (default) — the platform runs this SAME script on every node.
#       rank 0 (INDEX=0) starts the Ray head + driver; every other rank joins
#       Ray and idles. Submit once as the platform's multi-node job entrypoint;
#       taiji fans the script out and sets INDEX + CHIEF_IP per node.
#
#   LAUNCH=ssh — run this script ONCE on the head. It starts the Ray head, then
#       ssh's `ray start` onto every other node in NODE_IP_LIST, then runs the
#       driver. For interactive multi-node sessions where you only have a shell
#       on the head. Prereqs: passwordless ssh head->workers (taiji provides it),
#       the repo at the SAME path on every node (shared mount), and CONDA_ENV set
#       so a non-login ssh shell finds `ray`. taiji sets NODE_IP_LIST (format
#       IP:GPUS,IP:GPUS,...) + CHIEF_IP.
#
# Cluster topology defaults to taiji's job env (see "Cluster topology" below);
# set the explicit vars to run on any other cluster.
#
# Data plane (how rollout samples reach the trainer) is selected with DATA_PLANE:
#   ray         (default) driver gathers rollouts over the Ray object store
#   tq_simple   TransferQueue on Ray-backed host storage (off-driver, zero infra)
#   tq_mooncake TransferQueue over mooncake RDMA (production) — the head auto-starts
#               mooncake_master + http_metadata_server and derives MOONCAKE_* from
#               CHIEF_IP (override MOONCAKE_* to point at external services)
#   keep_local  direct-sampling actors keep rollouts local; only light metadata
#               crosses to the driver (no transfer at all)
#
# Examples:
#   # SPMD batch (taiji lands this same line on every node):
#   bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
#   DATA_PLANE=tq_mooncake PROTOCOL=rdma \
#     bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
#   # ssh fan-out (run once on the head only):
#   LAUNCH=ssh DATA_PLANE=keep_local \
#     bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${EXPERIMENT:-}" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Usage: $0 <experiment> [hydra overrides...]"
        exit 2
    fi
    EXPERIMENT="$1"
    shift
fi

# --- Python env (optional) --------------------------------------------------
if [ -n "${CONDA_ENV:-}" ]; then
    if [ -f "${CONDA_SH:-}" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
    elif [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/data/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}"
elif [ -n "${VENV_DIR:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# --- ssh fan-out worker join (LAUNCH=ssh) -----------------------------------
# In LAUNCH=ssh the head ssh-invokes THIS script with RAY_JOIN_ONLY=1 on each
# worker. The Python env is already activated above; here we just join Ray with
# the head address + this node's IP/GPUs (all passed in over ssh) and return so
# the raylet stays up. Workers do no pip install, no CMD build, no driver.
if [ "${RAY_JOIN_ONLY:-0}" = "1" ]; then
    : "${HEAD_IP:?RAY_JOIN_ONLY requires HEAD_IP (the head passes it over ssh)}"
    : "${NODE_IP:?RAY_JOIN_ONLY requires NODE_IP (the head passes it over ssh)}"
    JOIN_GPUS="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
    JOIN_PORT="${RAY_PORT:-6379}"
    ray stop >/dev/null 2>&1 || true
    until ray start \
        --address="${HEAD_IP}:${JOIN_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${JOIN_GPUS}"; do
        echo "[worker ${NODE_IP}] Ray head not ready; retry in 5s..."
        sleep 5
    done
    echo "[worker ${NODE_IP}] joined Ray at ${HEAD_IP}:${JOIN_PORT}."
    exit 0
fi

# --- Path / logging defaults ------------------------------------------------
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXPERIMENT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

CMD=(
    python -m diffusionrl.train
    "+experiment=${EXPERIMENT}"
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
    "logging.run_name=${WANDB_RUN_NAME}"
    "logging.report_to_wandb=${REPORT_TO_WANDB}"
)
if [ -n "${WANDB_ENTITY:-}" ]; then
    CMD+=("logging.entity=${WANDB_ENTITY}")
fi

# --- Data-plane selection (DATA_PLANE) --------------------------------------
# Append the Hydra overrides that pick how rollout data reaches the trainer.
DATA_PLANE="${DATA_PLANE:-ray}"
case "${DATA_PLANE}" in
    ray)
        : # default driver gather — no override
        ;;
    tq_simple)
        CMD+=("+transfer_queue=simple")
        ;;
    tq_mooncake)
        # Default the mooncake endpoints to the head (CHIEF_IP), where rank 0
        # starts the services below; explicit MOONCAKE_* still win.
        if [ -z "${MOONCAKE_MASTER_ADDR:-}" ] && [ -n "${CHIEF_IP:-}" ]; then
            MOONCAKE_MASTER_ADDR="${CHIEF_IP}:50051"
        fi
        if [ -z "${MOONCAKE_METADATA_URL:-}" ] && [ -n "${CHIEF_IP:-}" ]; then
            MOONCAKE_METADATA_URL="http://${CHIEF_IP}:8080/metadata"
        fi
        if [ -z "${MOONCAKE_METADATA_URL:-}" ] || [ -z "${MOONCAKE_MASTER_ADDR:-}" ]; then
            echo "DATA_PLANE=tq_mooncake needs CHIEF_IP (taiji sets it) to derive the endpoints," >&2
            echo "or explicit MOONCAKE_METADATA_URL + MOONCAKE_MASTER_ADDR." >&2
            exit 2
        fi
        CMD+=(
            "+transfer_queue=mooncake_tuned"
            "transfer_queue.protocol=${PROTOCOL:-rdma}"
            "transfer_queue.metadata_server=${MOONCAKE_METADATA_URL}"
            "transfer_queue.master_server_address=${MOONCAKE_MASTER_ADDR}"
        )
        ;;
    keep_local)
        CMD+=("training.execution.keep_local=true")
        ;;
    *)
        echo "Unknown DATA_PLANE='${DATA_PLANE}' (use ray|tq_simple|tq_mooncake|keep_local)" >&2
        exit 2
        ;;
esac

CMD+=("$@")

echo "Command (DATA_PLANE=${DATA_PLANE}):"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

# --- Cluster topology (taiji platform defaults) -----------------------------
# Defaults come from taiji's multi-node job env (commented per line); the
# explicit vars always win, so this launcher also runs on a non-taiji cluster.
NUM_NODES="${NUM_NODES:-${HOST_NUM:-2}}"               # taiji HOST_NUM:     node count
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"   # taiji HOST_GPU_NUM: GPUs per node
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"                  # taiji INDEX:        this node's rank
RAY_PORT="${RAY_PORT:-6379}"
LAUNCH="${LAUNCH:-spmd}"                               # spmd: platform runs this per node | ssh: head fans out

# This node's IP. Prefer an explicit NODE_IP, else taiji's LOCAL_IP. On multi-NIC
# / container nodes `hostname -I` often returns a container-internal IP that peers
# can't reach, so when CHIEF_IP is known, pick this node's IP on the chief's /16.
all_ips="$(hostname -I 2>/dev/null || true)"
if [ -z "${NODE_IP:-}" ] && [ -n "${LOCAL_IP:-}" ]; then
    NODE_IP="${LOCAL_IP}"
fi
if [ -z "${NODE_IP:-}" ] && [ -n "${CHIEF_IP:-}" ]; then
    chief_subnet="$(echo "${CHIEF_IP}" | cut -d. -f1-2)"
    NODE_IP="$(echo "${all_ips}" | tr ' ' '\n' | grep "^${chief_subnet}\." | head -1 || true)"
fi
if [ -z "${NODE_IP:-}" ]; then
    NODE_IP="$(echo "${all_ips}" | awk '{print $1}')"
fi
NODE_IP="${NODE_IP:-127.0.0.1}"

HEAD_IP="${HEAD_IP:-${CHIEF_IP:-${NODE_IP}}}"          # taiji CHIEF_IP:     head node IP

# Mooncake binds its per-actor client to LOCAL_IP (transfer_queue/runtime.py); when
# the platform did not set it, fall back to this node's routable IP so the client
# does not bind a container-internal interface. Harmless for non-mooncake planes.
export LOCAL_IP="${LOCAL_IP:-${NODE_IP}}"

# Head-only mooncake control plane: bring the services up so a single job
# submission is self-contained. Idempotent (skips if already running); logs to
# /tmp. Override MOONCAKE_* (above) to use external services instead.
start_mooncake_services() {
    if ! command -v mooncake_master >/dev/null 2>&1; then
        echo "WARNING: mooncake_master not on PATH — start the mooncake services manually." >&2
        return
    fi
    if ! pgrep -f mooncake_master >/dev/null 2>&1; then
        echo "Starting mooncake_master (:50051) -> /tmp/mooncake_master.log"
        nohup mooncake_master --port=50051 >/tmp/mooncake_master.log 2>&1 &
    fi
    if ! pgrep -f http_metadata_server >/dev/null 2>&1; then
        echo "Starting mooncake http_metadata_server (:8080) -> /tmp/mooncake_metadata.log"
        nohup python -m mooncake.http_metadata_server --port 8080 >/tmp/mooncake_metadata.log 2>&1 &
    fi
    sleep "${MOONCAKE_STARTUP_WAIT_S:-5}"
}

# Start the Ray head on this node + bring up the mooncake control plane when the
# mooncake data plane is selected. Used by both the SPMD rank-0 path and the ssh
# head path.
start_ray_head() {
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --num-gpus="${GPUS_PER_NODE}"
    if [ "${DATA_PLANE}" = "tq_mooncake" ]; then
        start_mooncake_services
    fi
}

mkdir -p "${OUTPUT_DIR}"
ray stop >/dev/null 2>&1 || true

if [ "${LAUNCH}" = "ssh" ]; then
    # Head-only fan-out: this script runs ONCE on the head. Start the head, then
    # ssh `ray start` onto every other node in NODE_IP_LIST, then fall through to
    # run the driver below. Workers re-enter this script with RAY_JOIN_ONLY=1
    # (early-exit near the top), so they only join Ray and return.
    if [ "${NODE_RANK}" != "0" ]; then
        echo "LAUNCH=ssh must be invoked on the head (rank 0); got NODE_RANK=${NODE_RANK}." >&2
        exit 2
    fi
    if [ -z "${NODE_IP_LIST:-}" ]; then
        echo "LAUNCH=ssh needs NODE_IP_LIST (taiji sets it; format IP:GPUS,IP:GPUS,...)." >&2
        exit 2
    fi
    start_ray_head
    SSH_USER="${SSH_USER:-$(whoami)}"
    # Forward the head's NCCL_* env to each worker's `ray start`. The ssh shell is
    # bare (no .bashrc), so without this the workers miss e.g. NCCL_NET_GDR_LEVEL
    # and crash at multi-node FSDP init. (SPMD mode gets NCCL env per-node from the
    # job env; in ssh mode the head's env is the only source, so propagate it.)
    ssh_nccl_env=""
    while IFS='=' read -r _k _v; do
        [ -n "${_k}" ] && ssh_nccl_env+="${_k}='${_v}' "
    done < <(env | grep '^NCCL_' || true)
    IFS=',' read -ra _NODE_ENTRIES <<< "${NODE_IP_LIST}"
    worker_n=0
    for entry in "${_NODE_ENTRIES[@]}"; do
        w_ip="${entry%%:*}"
        w_gpu="${entry##*:}"
        if [ "${w_gpu}" = "${entry}" ]; then  # entry had no ":GPUS" suffix
            w_gpu="${GPUS_PER_NODE}"
        fi
        if [ -z "${w_ip}" ] || [ "${w_ip}" = "${HEAD_IP}" ] || [ "${w_ip}" = "${NODE_IP}" ]; then
            continue  # the head runs the driver, not a joined worker
        fi
        worker_n=$((worker_n + 1))
        echo "[head] ssh ray start -> ${SSH_USER}@${w_ip} (gpus=${w_gpu})"
        ssh -f -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
            "${SSH_USER}@${w_ip}" \
            "cd '${REPO_ROOT}' && \
             RAY_JOIN_ONLY=1 HEAD_IP='${HEAD_IP}' NODE_IP='${w_ip}' LOCAL_IP='${w_ip}' \
             GPUS_PER_NODE='${w_gpu}' RAY_PORT='${RAY_PORT}' \
             CONDA_ENV='${CONDA_ENV:-}' CONDA_SH='${CONDA_SH:-}' VENV_DIR='${VENV_DIR:-}' \
             ${ssh_nccl_env}\
             nohup bash scripts/run_experiment_multinode_taiji.sh '${EXPERIMENT}' \
             >/tmp/diffusionrl_ray_worker_${worker_n}.log 2>&1 &" \
            || echo "[head] WARNING: ssh to ${w_ip} failed; that node will not join." >&2
    done
    echo "[head] fanned out to ${worker_n} worker(s); they join /tmp/diffusionrl_ray_worker_*.log."
elif [ "${NODE_RANK}" = "0" ]; then
    # SPMD (default): the platform runs this script on every node; rank 0 is head.
    start_ray_head
else
    # SPMD worker: join Ray and idle so the platform keeps this node alive while
    # the head owns the driver.
    until ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${GPUS_PER_NODE}"; do
        echo "Ray head not ready yet; retrying in 5s..."
        sleep 5
    done
    echo "Worker node joined Ray cluster; head node owns the training driver."
    exec tail -f /dev/null  # block forever; only the head runs the driver below
fi

echo "Ray cluster target: ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s)"
sleep "${RAY_CLUSTER_WAIT_S:-30}"
exec "${CMD[@]}"
