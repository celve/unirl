# vendor/ — bowen's leo2 recipes, copied verbatim, NOT runnable here

Copied from `personal/bowen/leo2.0/dev` (`examples/diffusion/leo2/`) so the scaling
configuration is visible next to the vertical that runs. Only `_target_` prefixes were
rewritten `unirl.models.leo2.*` -> `experimental.leo2.models.*`. Nothing else was touched, and
**none of these run on this branch today** — each depends on machinery the trainside vertical
does not carry. The inventory below is what a port has to close, measured against `main` and
this package rather than guessed.

| Recipe | Chain | Blocks |
|---|---|---|
| `leo2_t2v_cached` | `<- leo2_t2v_trainside` | 3 bundle fields |
| `leo2_t2v_flowgrpo_cached` | `<- leo2_t2v_cached` | + CP/EP, `hybrid8`, parity key |
| `leo2_t2v_flowgrpo_motion_bilingual` | `<- leo2_t2v_cached` | + audio, `AllSDEScheduler` wiring |
| `leo2_t2v_flowgrpo_separate` | `<- leo2_t2v_flowgrpo_cached` | + rollout engine, weight sync |

## What is missing

**`Leo2PipelineConfig` lacks 8 fields** these set: `preprocessing_cache_dir`,
`preprocessing_cache_mode`, `load_video_vae`, `context_parallel_size`,
`expert_parallel_size`, `enable_deepep`, `inference_cache_method`, `enable_audio`,
`audio_joint_sde`. `vae_on_gpu` and `video_shift` are the only two that exist.

**`fsdp_mode: hybrid8` is not a value `main` accepts** — `_FSDP_MODES` is
`("full", "hybrid", "no_shard")` and `normalize_fsdp_mode` fails closed on anything else.
`sp_size`, `ep_size` and `checkpoint_format` are real `FSDPConfig` fields, so only the mode
name is wrong; whether `hybrid` + `sp_size`/`ep_size` reproduces `hybrid8`'s geometry is
unverified.

**`unirl.distributed.weight_sync.lora.RemoteLoraWeightSync` does not exist on `main`**, and
neither does `experimental.leo2.models.rollout.Leo2RolloutEngine` — this package has no
rollout engine at all. `separate` needs both, which is why it is the furthest from portable.

**`rollout_replay_parity_action: raise`** is the woa-only `FlowGRPO` key that already cost
one 40-minute GPU failure here; see the package README.

**`frame_selection: uniform` is rejected** — `VideoPickScoreScorer` accepts only `"first"` or
`"middle"`, and there is no `num_score_frames` field on `VideoPickScoreSpec`.

`AllSDEScheduler` does exist and its `(num_timesteps, timestep_fraction, num_sde_steps)`
signature matches what `motion_bilingual` passes; whether `sampling.scheduler` is a key the
sampling params read is not established.
