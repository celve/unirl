#!/bin/bash
# =============================================================================
# Shared batch-size parameter resolution and validation.
#
# Source this file from any reproduce script AFTER setting the 5 core knobs:
#
#   PROMPTS_PER_BATCH=48           # unique prompts per rollout
#   NUM_SAMPLES_PER_PROMPT=24      # samples per prompt (group size)
#   SAMPLING_FORWARD_BATCH=192     # per-device peak forward batch during sampling
#   TRAINING_FORWARD_BATCH=12      # per-device peak forward batch during training
#   NUM_UPDATES=2                  # gradient update steps per local batch
#
# Then call:
#   resolve_batch_params            # derives all secondary params
#   validate_batch_params           # checks divisibility constraints
#   print_batch_params              # prints a summary table
#
# Outputs (available after resolve_batch_params):
#   ROLLOUT_TOTAL_SAMPLES           = PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT
#   TOTAL_GPUS                      = NUM_NODES * GPUS_PER_NODE  (multinode)
#                                   = NUM_GPUS                    (single-node)
#   LOCAL_UPDATE_BATCH_SIZE         = ROLLOUT_TOTAL_SAMPLES / TOTAL_GPUS / NUM_UPDATES
#   DIRECT_SAMPLING_BATCH_SIZE      = SAMPLING_FORWARD_BATCH
#   LOCAL_MICRO_BATCH_SIZE          = TRAINING_FORWARD_BATCH
#   NUM_UPDATES_PER_LOCAL_BATCH     = NUM_UPDATES
# =============================================================================

resolve_batch_params() {
    # ── Total GPU count ──
    # Multinode scripts set NUM_NODES + GPUS_PER_NODE;
    # single-node scripts set NUM_GPUS directly.
    if [ -n "${NUM_NODES:-}" ] && [ -n "${GPUS_PER_NODE:-}" ]; then
        TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
    elif [ -n "${NUM_GPUS:-}" ]; then
        TOTAL_GPUS="${NUM_GPUS}"
    else
        echo "ERROR: Either NUM_GPUS or (NUM_NODES + GPUS_PER_NODE) must be set" >&2
        exit 1
    fi

    # ── Derived params ──
    ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
    LOCAL_UPDATE_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / TOTAL_GPUS / NUM_UPDATES ))

    # ── Map core knobs to CLI param names ──
    DIRECT_SAMPLING_BATCH_SIZE="${SAMPLING_FORWARD_BATCH}"
    LOCAL_MICRO_BATCH_SIZE="${TRAINING_FORWARD_BATCH}"
    NUM_UPDATES_PER_LOCAL_BATCH="${NUM_UPDATES}"
}

validate_batch_params() {
    local errors=0

    # 1. Sampling forward batch must be aligned to group size
    if [ $(( DIRECT_SAMPLING_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
        echo "ERROR: SAMPLING_FORWARD_BATCH (${DIRECT_SAMPLING_BATCH_SIZE}) must be divisible by NUM_SAMPLES_PER_PROMPT (${NUM_SAMPLES_PER_PROMPT})" >&2
        errors=$(( errors + 1 ))
    fi

    # 2. Total samples must split evenly across GPUs and updates
    if [ $(( ROLLOUT_TOTAL_SAMPLES % (TOTAL_GPUS * NUM_UPDATES) )) -ne 0 ]; then
        echo "ERROR: ROLLOUT_TOTAL_SAMPLES (${ROLLOUT_TOTAL_SAMPLES}) must be divisible by TOTAL_GPUS*NUM_UPDATES (${TOTAL_GPUS}*${NUM_UPDATES}=$(( TOTAL_GPUS * NUM_UPDATES )))" >&2
        errors=$(( errors + 1 ))
    fi

    # 3. Update batch must split into whole micro-batches
    if [ $(( LOCAL_UPDATE_BATCH_SIZE % LOCAL_MICRO_BATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: LOCAL_UPDATE_BATCH_SIZE (${LOCAL_UPDATE_BATCH_SIZE}) must be divisible by TRAINING_FORWARD_BATCH (${LOCAL_MICRO_BATCH_SIZE})" >&2
        errors=$(( errors + 1 ))
    fi

    # 4. Sampling batch must evenly divide total samples (if smaller)
    if [ "${DIRECT_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_TOTAL_SAMPLES}" ] && \
       [ $(( ROLLOUT_TOTAL_SAMPLES % DIRECT_SAMPLING_BATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: SAMPLING_FORWARD_BATCH (${DIRECT_SAMPLING_BATCH_SIZE}) must evenly divide ROLLOUT_TOTAL_SAMPLES (${ROLLOUT_TOTAL_SAMPLES})" >&2
        errors=$(( errors + 1 ))
    fi

    if [ "${errors}" -gt 0 ]; then
        echo "Batch geometry validation failed (${errors} error(s))." >&2
        exit 1
    fi
}

print_batch_params() {
    local local_batch=$(( LOCAL_UPDATE_BATCH_SIZE * NUM_UPDATES ))
    local num_micro=$(( LOCAL_UPDATE_BATCH_SIZE / LOCAL_MICRO_BATCH_SIZE ))
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                  Batch Geometry Summary                  │"
    echo "├──────────────────────────────┬──────────────────────────┤"
    printf "│ %-28s │ %24s │\n" "Prompts per batch"        "${PROMPTS_PER_BATCH}"
    printf "│ %-28s │ %24s │\n" "Samples per prompt"       "${NUM_SAMPLES_PER_PROMPT}"
    printf "│ %-28s │ %24s │\n" "Total rollout samples"    "${ROLLOUT_TOTAL_SAMPLES}"
    printf "│ %-28s │ %24s │\n" "Total GPUs"               "${TOTAL_GPUS}"
    echo "├──────────────────────────────┼──────────────────────────┤"
    printf "│ %-28s │ %24s │\n" "Sampling forward batch"   "${DIRECT_SAMPLING_BATCH_SIZE}"
    echo "├──────────────────────────────┼──────────────────────────┤"
    printf "│ %-28s │ %24s │\n" "Local batch (per GPU)"    "${local_batch}"
    printf "│ %-28s │ %24s │\n" "  ├─ Gradient updates"    "${NUM_UPDATES}"
    printf "│ %-28s │ %24s │\n" "  ├─ Update batch size"   "${LOCAL_UPDATE_BATCH_SIZE}"
    printf "│ %-28s │ %24s │\n" "  └─ Micro-batches/update" "${num_micro} × ${LOCAL_MICRO_BATCH_SIZE}"
    echo "└──────────────────────────────┴──────────────────────────┘"
}
