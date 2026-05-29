# HI3-Instruct think_recaption RL — Quick Start

## How to run the experiment

Prerequisites: taiji 2-node pod with vllm-omni + venv already set up.

```bash
# 1. Apply pod-local patches (on EVERY pod: launcher + worker)
DIFFRL=/path/to/diffusionrl

cd /root/vllm-omni
git apply $DIFFRL/patches/vllm_omni/0001-omni-py-lora_request-passthrough.patch

# 2. Verify patches
grep -c 'lora_request' /root/vllm-omni/vllm_omni/entrypoints/omni.py  # expect >= 7

# 3. Launch training (from launcher pod)
cd $DIFFRL
python -m diffusionrl.train +experiment=hi3_think_recaption_colocate
```

Key config knobs (override via CLI):
- `run.num_rollouts=10` — number of RL iterations
- `run.weight_sync_interval=1` — sync LoRA to rollout every N iterations
- `training.policies.0.rank=8` — LoRA rank
- `algorithm.prompts_per_rollout=8` — prompts per rollout batch
- `rollout.engine.model_path=/root/HunyuanImage-3-Instruct` — model checkpoint

---

# Third-party patches required for HI3-Instruct RL

One pod-local patch is currently required for HI3 / vLLM-Omni rollout
workers. The in-repo `vllm_patches.py:VLLMOmniHijack.hijack()` handles
runtime tensor-bag LoRA loading (and now the fp32 `from_layer` skip too,
see "Previously required patches" below), but this upstream change still
needs to be applied in the external vLLM-Omni checkout used by the pod.

## Patch list

| dir | patch | applies to | base commit | status |
|---|---|---|---|---|
| `vllm_omni/` | `0001-omni-py-lora_request-passthrough.patch` | `/root/vllm-omni/vllm_omni/entrypoints/omni.py` | `4a24a51` | **REQUIRED** pod-local; cannot be monkey-patched (function signature change) |

## Why this patch is pod-local (not in-repo monkey-patch)

`Omni.generate()` needs a new `lora_request` parameter in its signature so
callers (our `engine.py`) can pass it. Monkey-patching a method to add a kwarg
is fragile (must copy the full method body, breaks on any upstream refactor).
A clean `git apply` is more maintainable until vllm-omni upstreams this.

## How to apply

On each pod (launcher AND every worker) that runs `train`:

```bash
PATCHES=/apdcephfs_fsgm3/share_305110755/hunyuan/gxhe/project/diffusionrl-main-unified-base-hi3-instruct/diffusionrl/patches

# vllm-omni: lora_request passthrough
cd /root/vllm-omni
git apply $PATCHES/vllm_omni/0001-omni-py-lora_request-passthrough.patch
# verify:
grep -c 'lora_request' vllm_omni/entrypoints/omni.py  # expect >= 7
```

## How to verify before each training run

Add to your launch script:

```bash
EXPECTED_HITS=7
ACTUAL_HITS=$(grep -c 'lora_request' /root/vllm-omni/vllm_omni/entrypoints/omni.py || echo 0)
if [ "$ACTUAL_HITS" -lt "$EXPECTED_HITS" ]; then
    echo "[FATAL] vllm-omni omni.py not patched (got $ACTUAL_HITS hits, expected >=$EXPECTED_HITS)"
    echo "        run: git apply patches/vllm_omni/0001-omni-py-lora_request-passthrough.patch"
    exit 1
fi
```

## Previously required patches (now obsolete)

| patch | reason obsolete |
|---|---|
| `vllm_omni/0002-hunyuan_image3-fp32-gate-bypass-lora.patch` | Redundant once `from_layer` skips fp32 layers (gate stays raw `ReplicatedLinear`) |
| `vllm/0001-utils-skip-fp32-from-layer.patch` | Superseded by `vllm_patches.py:patch_fp32_skip()`. The DiT stage loads inside a `StageDiffusionProc` spawn subprocess, but `wrap_mp_process_for_children()` now propagates the hijack into those children so the in-repo monkey-patch reaches `from_layer()` before model init — no site-packages patch needed. |
