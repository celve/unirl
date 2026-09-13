# Leo2 (HunyuanVideo 2.0, MoE-A12B) × UniRL trainside FlowGRPO

Text-to-video GRPO on the 75B three-stream MMDiT with UniRL's in-process (trainside)
rollout engine. Model-side contracts, precision trade-offs and upstream patches are in
[`unirl/models/leo2/README.md`](../../../unirl/models/leo2/README.md).

## Launch

```bash
export LEO2_CKPT_DIR=/path/to/iter_0063300_torch/weights   # torch-DCP, ~150 GB
export LEO2_GENERATION_CONFIG=/path/to/leo2_genconfig_rl.json
export ASSETS_BASE=/path/to/hymm_ar_assets
export PYTHONPATH=$HYMM:$HYMM/deps/hy_parallelism:$HYMM/deps/IndexKits:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash examples/run_experiment_single_node.sh diffusion/leo2/leo2_t2v_trainside
```

`$HYMM` is a `hunyuan_multimodal_gen_ar` checkout. It goes on `PYTHONPATH` for the driver
and reward actors too, not only the workers — rollout Samples carry hymm-class pickles.
Set `HYMM_REPO_PATH` instead if you would rather the bundle do the `sys.path` insert.

Prompts default to `datasets/leo2_t2v/{train,test}.txt` (32 train + 8 held out).

## The shipped recipe

FSDP2 full-shard over `Leo{,Dual,Triple}Layer` with `root_wrap`, LoRA r64/a256 on both
attention streams plus the shared and text MLPs (MoE experts, router and the audio branch
stay frozen), `FlowSDEStrategy`, PickScore over video, FlowGRPO with `old_logp_source:
replay`. 10 inference steps, eta 0.7, SDE on transitions `[0, 3, 6, 9]`, 4 samples per
prompt, 192×336×49, seed 42.

Measured on 8×H20: **195–203 s/step** (rollout 97–99 s + train 95–103 s), reward
0.727–0.733, peak 51.1–52.4 GB/GPU at 73–77 % util.

## The configuration that actually learns

The shipped recipe is a correctness smoke, not a training run — at 10 steps and eta 0.7 it
produces structureless video while PickScore still reports ~0.73. To train, override:

```bash
bash examples/run_experiment_single_node.sh diffusion/leo2/leo2_t2v_trainside \
  sampling.num_inference_steps=30 sampling.eta=0.5 "sampling.sde_indices=[0,1,2,3,4]" \
  sampling.samples_per_prompt=8 sampling.init_same_noise=true \
  stack.num_updates_per_batch=2 backend.optimizer_cfg.learning_rate=2.5e-5 \
  adv_use_global_std=false \
  +reward.backend.config.frame_selection=uniform +reward.backend.config.num_score_frames=4
```

SDE noise only on the five highest-sigma transitions, the rest ODE; one shared `x_T` per
prompt group; PickScore averaged over 4 uniformly spaced frames. ~630 s/step (rollout
400 + train 225), or ~495 s/step with `algorithm.old_logp_source=rollout` and
`bundle.config.text_encoder_gpu_transient=false`.

**Always eyeball rollout media next to the reward curve.** LoRA lr 1e-4 without KL
reward-hacks into stripe textures by ~rollout 50; 2.5e-5 stays clean, and
`algorithm.beta>0` (KL against the adapter-disabled base) is the intended fix.
