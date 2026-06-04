# Rollout Package

`unirl.rollout` owns the rollout **engine contracts and implementations**. It
does not own request planning (the trainer builds `RolloutReq` via its own
`_build_req`) or train-side loss math; an engine turns a typed request into a
typed response that the training stack consumes.

## Key Files

| File | Purpose |
|---|---|
| `engine/base.py` | Rollout engine interface (`BaseRolloutEngine`) |
| `engine/trainside/` | Direct-sampling engine backed by train-side model execution |
| `engine/sglang/` | Dedicated SGLang rollout engine (diffusion) |
| `engine/sglang_llm/` | Dedicated SGLang rollout engine (autoregressive LLM / VLM) |
| `engine/vllm_omni/` | Dedicated vLLM-Omni rollout engine and HI3 / SD3 adapters |
| `engine/composed/` | `ComposedRolloutEngine` — chains engines for the PE flow |

## Deployment Modes

This package is the **canonical source** for the three deployment shapes —
other READMEs link here rather than re-stating them.

```text
Direct sampling                  Separate                         Colocate
(trainside engine)               (dedicated engine,               (dedicated engine,
                                  disjoint GPU pools)              shared GPU bundles)

┌────────────────┐               ┌──────────────┐                 ┌────────────────────┐
│ train + sample │               │ rollout pool │   ─── sync ───▶ │ rollout & train    │
│  (same actor)  │               └──────────────┘                 │  share GPU bundle, │
└────────────────┘                       ▲                        │  explicit offload  │
        ▲                                │                        │  + sync            │
        └── no sync (importance          │                        └────────────────────┘
            ratio == 1 on               ┌──────────────┐                   ▲
            first update)               │  train pool  │                   │
                                        └──────────────┘                sync required
                                               ▲
                                          sync required
```

| Mode | Config signal | Shape |
|---|---|---|
| Direct sampling | `rollout/engine: trainside` | train actors also sample; no `sync:` section |
| Separate | `rollout/engine: sglang` or `vllm_omni` with non-colocated placement | rollout and train use separate GPU pools; `sync:` required |
| Colocate | dedicated rollout engine plus colocated placement/offload | rollout and train share GPU bundles; offload/onload and sync are explicit |

The direct-vs-dedicated distinction is derived from
`cfg.rollout.engine._target_` by `config.validation.is_direct_sampling`.
For sync backend choices see `unirl/distributed/weight_sync/README.md`.

## Request / Response Boundary

Rollout engines receive `RolloutReq` and return `RolloutResp` — both defined in
`unirl/types/` (`rollout_req.py`, `rollout_resp.py`).

`RolloutReq` carries:

- prompt primitives and optional conditioning media;
- shared sampling parameters and per-request stage params;
- metadata needed for sharding and deterministic noise.

`RolloutResp` carries one or more `tracks` keyed by stage name; each
`tracks[name]` carries:

- typed conditions used by train-side replay;
- a rollout `segment` such as diffusion or AR trajectories;
- rewards, advantages, and metrics;
- optional media previews for logging.

Training should consume `RolloutResp` directly. New rollout engines should
adapt backend-specific wire formats into this typed boundary instead of
leaking backend objects into the trainer or `algorithms/`.

## Engine Guidance

Add a rollout engine by:

1. adding a typed config under `rollout/engine/<engine>/config.py`;
2. registering it in group `rollout/engine`;
3. implementing the engine contract from `engine/base.py`;
4. returning canonical `RolloutResp` data;
5. adding or updating recipes under `examples/<domain>/<model>/`.

If the engine uses a dedicated rollout process, also define how trainer weights
reach that process through `distributed/weight_sync`.
