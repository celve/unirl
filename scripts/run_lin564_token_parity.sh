#!/usr/bin/env bash
# LIN-564 U3: AReaL token-normalization/token-loss parity on Qwen3-1.7B.
# Run persistently from tmux. Secrets stay in a pod-local mode-600 env file.

set -euo pipefail

source /etc/bashrc

ROOT=/root/unirl
ENV_FILE=${LIN564_ENV_FILE:-$ROOT/.lin564_deep_research.env}
NUM_ROLLOUTS=${1:-10}
if (($#)); then
  shift
fi

if [[ ! -f $ENV_FILE ]]; then
  echo "missing pod-local secrets file: $ENV_FILE" >&2
  exit 2
fi
if [[ $(stat -c '%a' "$ENV_FILE") != 600 ]]; then
  echo "secrets file must be mode 600: $ENV_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

# Reassert launch invariants after sourcing the historical snapshot: it contains
# stale model/data/judge paths that must never override this pod-local run.
ROOT=/root/unirl
MODEL=$ROOT/models/local/Qwen3-1.7B
DATA=$ROOT/data/asearcher/train.jsonl
PYTHON=$ROOT/.venv-sglang/bin/python
RUN_NAME=${LIN564_RUN_NAME:-lin564_u3_tokadv_tokloss_qwen3_1p7b_s42_20260720}
RUN_DIR=${LIN564_RUN_DIR:-/mnt/gz/logs/$RUN_NAME}
readonly ROOT MODEL DATA PYTHON RUN_NAME RUN_DIR

: "${JUDGE_HOST:?set JUDGE_HOST to the live Qwen2.5-72B service IP}"
: "${SERPER_KEY_ID:?missing SERPER_KEY_ID in $ENV_FILE}"
: "${JINA_API_KEYS:?missing JINA_API_KEYS in $ENV_FILE}"
: "${WANDB_API_KEY:?missing WANDB_API_KEY in $ENV_FILE}"

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
TRAINER_IP=${TRAINER_IP:-$(hostname -I | awk '{print $1}')}
export no_proxy=".woa.com,.oa.com,.polaris,localhost,127.0.0.1,$TRAINER_IP,$JUDGE_HOST"
export NO_PROXY=$no_proxy

export PRETRAINED_MODEL=$MODEL
export QWEN3_INSTRUCT_PATH=$MODEL
export DATA_PATH=$DATA
export EVAL_DATA_PATH=$DATA
export SERPER_URL=${SERPER_URL:-http://trpc-gpt-eval.production.polaris:8080/search}
export SERPER_AUTH=${SERPER_AUTH:-bearer}
export JUDGE_URL=http://$JUDGE_HOST:30000/v1/chat/completions
export JUDGE_MODEL=Qwen2.5-72B-Instruct
export SUMMARY_URL=$JUDGE_URL
export SUMMARY_MODEL=$JUDGE_MODEL
export WANDB_ENTITY=${WANDB_ENTITY:-linyuwus}
export TRAJ_DUMP_DIR=$RUN_DIR/traj
export TRAJ_DUMP_MAX_CHARS=0
export RUN_DATE=2026-07-20
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# These were already disabled in the AReaL-harness baseline. Keep them fixed so
# U3 changes only advantage/loss token weighting relative to that baseline.
export REQUIRE_ANSWER_TAG=0
export HARD_ZERO_EMPTY=0

test -x "$PYTHON"
test -f "$MODEL/config.json"
test -f "$DATA"
test "$(wc -l < "$DATA")" -eq 35054
case $(realpath "$MODEL") in
  /root/unirl/models/local/*) ;;
  *) echo "policy model is not in the pod-local cache: $MODEL" >&2; exit 2 ;;
esac
case $(findmnt -n -o FSTYPE -T "$MODEL") in
  ceph|fuse.ceph) echo "policy model resolves to a Ceph mount: $MODEL" >&2; exit 2 ;;
esac
"$PYTHON" -c \
  'import torch; from importlib.metadata import version; assert torch.__version__ == "2.11.0+cu130"; assert torch.version.cuda == "13.0"; print("sglang", version("sglang"))'
cd "$ROOT"
git grep -q 'adv_normalization_scope == "token-global"' -- unirl/trainer/agentic.py
git grep -q '_GLOBAL_TOKEN_MEAN = "global-token-mean"' -- unirl/train/stack/base.py

mkdir -p "$RUN_DIR" "$TRAJ_DUMP_DIR"
if find "$TRAJ_DUMP_DIR" -maxdepth 1 -name 'rollout_*.jsonl' -print -quit | grep -q .; then
  echo "trajectory directory is not fresh: $TRAJ_DUMP_DIR" >&2
  exit 2
fi
exec > >(tee -a "$RUN_DIR/train.log") 2>&1

# A health endpoint can become ready before graph capture. Require one real
# completion so a dead judge cannot silently turn an entire rollout into zeros.
curl --fail --silent --show-error --max-time 180 \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-72B-Instruct","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0}' \
  "$JUDGE_URL" > "$RUN_DIR/judge_preflight.json"
JUDGE_PREFLIGHT=$RUN_DIR/judge_preflight.json "$PYTHON" -c \
  'import json, os; x=json.load(open(os.environ["JUDGE_PREFLIGHT"])); assert x.get("choices"), x'

if [[ ${LIN564_SKIP_PREWARM:-0} != 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 QWEN3_PATH=$MODEL timeout 1200 \
    "$PYTHON" scripts/sglang_ar_multiturn_smoke.py 2>&1 | tee "$RUN_DIR/prewarm.log"
  grep -q 'SGLANG MULTI-TURN SMOKE PASSED' "$RUN_DIR/prewarm.log"
fi

exec "$PYTHON" -m unirl.train_deep_research \
  --config-name=deep_research/deep_research_search_judge \
  num_devices=8 \
  num_rollouts="$NUM_ROLLOUTS" \
  batch_size=128 \
  data_source.args.algorithm.prompts_per_rollout=128 \
  data_source.args.run.seed=42 \
  rollout.config.episode_sampling.samples_per_prompt=8 \
  sampling.samples_per_prompt=8 \
  adv_normalization_scope=token-global \
  algorithm.loss_agg_mode=global-token-mean \
  logging.run_name="$RUN_NAME" \
  "$@"
