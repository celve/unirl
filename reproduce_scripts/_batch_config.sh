#!/bin/bash
# Shared batch-geometry resolution and validation for reproduce scripts.
#
# Source this file after defining:
#   PROMPTS_PER_BATCH
#   NUM_SAMPLES_PER_PROMPT
#   SAMPLING_FORWARD_BATCH
#   TRAINING_FORWARD_BATCH
#   NUM_UPDATES
#
# And one of:
#   NUM_GPUS
#   NUM_NODES + GPUS_PER_NODE
#
# Derived outputs:
#   TOTAL_GPUS
#   ROLLOUT_TOTAL_SAMPLES
#   LOCAL_BATCH_SIZE
#   LOCAL_UPDATE_BATCH_SIZE
#   DIRECT_SAMPLING_BATCH_SIZE
#   LOCAL_MICRO_BATCH_SIZE
#   NUM_UPDATES_PER_LOCAL_BATCH

resolve_batch_params() {
    if [ -n "${NUM_NODES:-}" ] && [ -n "${GPUS_PER_NODE:-}" ]; then
        TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
    elif [ -n "${NUM_GPUS:-}" ]; then
        TOTAL_GPUS="${NUM_GPUS}"
    else
        echo "ERROR: Either NUM_GPUS or (NUM_NODES + GPUS_PER_NODE) must be set" >&2
        exit 1
    fi

    ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
    LOCAL_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / TOTAL_GPUS ))
    LOCAL_UPDATE_BATCH_SIZE=$(( LOCAL_BATCH_SIZE / NUM_UPDATES ))

    DIRECT_SAMPLING_BATCH_SIZE="${SAMPLING_FORWARD_BATCH}"
    LOCAL_MICRO_BATCH_SIZE="${TRAINING_FORWARD_BATCH}"
    NUM_UPDATES_PER_LOCAL_BATCH="${NUM_UPDATES}"
}

validate_batch_params() {
    local errors=0

    if [ "${TOTAL_GPUS}" -le 0 ]; then
        echo "ERROR: TOTAL_GPUS must be positive, got ${TOTAL_GPUS}" >&2
        errors=$(( errors + 1 ))
    fi

    if [ $(( DIRECT_SAMPLING_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
        echo "ERROR: SAMPLING_FORWARD_BATCH (${DIRECT_SAMPLING_BATCH_SIZE}) must be divisible by NUM_SAMPLES_PER_PROMPT (${NUM_SAMPLES_PER_PROMPT})" >&2
        errors=$(( errors + 1 ))
    fi

    if [ $(( ROLLOUT_TOTAL_SAMPLES % TOTAL_GPUS )) -ne 0 ]; then
        echo "ERROR: ROLLOUT_TOTAL_SAMPLES (${ROLLOUT_TOTAL_SAMPLES}) must be divisible by TOTAL_GPUS (${TOTAL_GPUS})" >&2
        errors=$(( errors + 1 ))
    fi

    if [ $(( LOCAL_BATCH_SIZE % NUM_UPDATES )) -ne 0 ]; then
        echo "ERROR: LOCAL_BATCH_SIZE (${LOCAL_BATCH_SIZE}) must be divisible by NUM_UPDATES (${NUM_UPDATES})" >&2
        errors=$(( errors + 1 ))
    fi

    if [ $(( LOCAL_UPDATE_BATCH_SIZE % LOCAL_MICRO_BATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: LOCAL_UPDATE_BATCH_SIZE (${LOCAL_UPDATE_BATCH_SIZE}) must be divisible by TRAINING_FORWARD_BATCH (${LOCAL_MICRO_BATCH_SIZE})" >&2
        errors=$(( errors + 1 ))
    fi

    if [ "${DIRECT_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_TOTAL_SAMPLES}" ] && \
       [ $(( ROLLOUT_TOTAL_SAMPLES % DIRECT_SAMPLING_BATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: SAMPLING_FORWARD_BATCH (${DIRECT_SAMPLING_BATCH_SIZE}) must evenly divide ROLLOUT_TOTAL_SAMPLES (${ROLLOUT_TOTAL_SAMPLES}) when sub-batching rollout sampling" >&2
        errors=$(( errors + 1 ))
    fi

    if [ "${errors}" -gt 0 ]; then
        echo "Batch geometry validation failed (${errors} error(s))." >&2
        exit 1
    fi
}

print_batch_params() {
    local num_micro=$(( LOCAL_UPDATE_BATCH_SIZE / LOCAL_MICRO_BATCH_SIZE ))
    cat <<EOF
Batch geometry summary:
  prompts_per_batch        = ${PROMPTS_PER_BATCH}
  samples_per_prompt       = ${NUM_SAMPLES_PER_PROMPT}
  rollout_total_samples    = ${ROLLOUT_TOTAL_SAMPLES}
  total_gpus               = ${TOTAL_GPUS}
  local_batch_size         = ${LOCAL_BATCH_SIZE}
  num_updates              = ${NUM_UPDATES_PER_LOCAL_BATCH}
  local_update_batch_size  = ${LOCAL_UPDATE_BATCH_SIZE}
  local_micro_batch_size   = ${LOCAL_MICRO_BATCH_SIZE}
  micro_batches_per_update = ${num_micro}
  sampling_forward_batch   = ${DIRECT_SAMPLING_BATCH_SIZE}
EOF
}
