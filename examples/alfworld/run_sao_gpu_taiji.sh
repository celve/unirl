#!/usr/bin/env bash
# Launch the LIN-562 SAO/ALFWorld GPU acceptance run on one TaiJi node.

# TaiJi's default shell omits the egress setup.  Keep this inside the launcher
# as well as in the outer taiji_client invocation because Ray/SGLang children
# and libraries such as W&B and Transformers read proxy variables directly.
# shellcheck disable=SC1091
source /etc/bashrc
set -euo pipefail
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export QWEN3_INSTRUCT_PATH="${QWEN3_INSTRUCT_PATH:-${REPO_ROOT}/models/local/Qwen3-4B-Instruct-2507}"
export VALUE_MODEL_PATH="${VALUE_MODEL_PATH:-${QWEN3_INSTRUCT_PATH}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${REPO_ROOT}/data/alfworld}"
export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ALFWORLD_DATA}/base_config.yaml}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/alfworld-sao/train.jsonl}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_PATH}}"

export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-linyuwus}"
export WANDB_PROJECT="${WANDB_PROJECT:-unirl-sao}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-lin562-sao-alfworld-$(date -u +%Y%m%d-%H%M%S)}"
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "${HOME}/.netrc" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "W&B credentials are absent; recording an offline run under ${REPO_ROOT}/wandb."
fi

case "${QWEN3_INSTRUCT_PATH}" in
    /root/unirl/models/local/*) ;;
    *)
        echo "QWEN3_INSTRUCT_PATH must be pod-local under /root/unirl/models/local; got ${QWEN3_INSTRUCT_PATH}" >&2
        exit 2
        ;;
esac

for path in \
    "${QWEN3_INSTRUCT_PATH}/config.json" \
    "${ALFWORLD_CONFIG}" \
    "${DATA_PATH}"; do
    if [ ! -r "${path}" ]; then
        echo "Required local input is missing: ${path}" >&2
        exit 2
    fi
done

export VENV_DIR="${REPO_ROOT}/.venv-sglang"
export ENTRY=train_sao_deep_research
export GPUS_PER_NODE=8
export INSTALL_EDITABLE=0

exec bash examples/run_experiment_single_node.sh \
    alfworld/alfworld_sao_4b_gpu \
    "$@"
