# HI3-Instruct think_recaption RL — Quick Start

## How to run the experiment

Prerequisites: taiji 2-node pod with vllm-omni + venv already set up.

```bash
# 1. Apply pod-local patches (on EVERY pod: launcher + worker)
DIFFRL=/path/to/diffusionrl

cd /root/vllm-omni
git apply $DIFFRL/patches/vllm_omni/0001-omni-py-lora_request-passthrough.patch

VLLM_SITE=$(python -c "import vllm; print(vllm.__path__[0])")
cd $(dirname $VLLM_SITE)
patch -p1 < $DIFFRL/patches/vllm/0001-utils-skip-fp32-from-layer.patch

# 2. Verify patches
grep -c 'lora_request' /root/vllm-omni/vllm_omni/entrypoints/omni.py  # expect >= 7
grep -c 'v44e' $VLLM_SITE/lora/utils.py                                # expect >= 1

# 3. Launch training (from launcher pod)
cd $DIFFRL
python -m diffusionrl.train_new +experiment=hi3_think_recaption_colocate
```

Key config knobs (override via CLI):
- `run.num_rollouts=10` — number of RL iterations
- `run.weight_sync_interval=1` — sync LoRA to rollout every N iterations
- `training.policies.0.rank=8` — LoRA rank
- `algorithm.prompts_per_rollout=8` — prompts per rollout batch
- `rollout.engine.model_path=/root/HunyuanImage-3-Instruct` — model checkpoint

---

# Third-party patches required for HI3-Instruct RL

Only **one** pod-local patch remains. The fp32 LoRA skip logic (formerly
`vllm/0001` + `vllm_omni/0002`) has been migrated into the in-repo
`lora_hijack.py:VLLMOmniHijack.hijack()` monkey-patch — it follows
`git pull` automatically and requires no pod-local intervention.

## Patch list

| dir | patch | applies to | base commit | status |
|---|---|---|---|---|
| `vllm_omni/` | `0001-omni-py-lora_request-passthrough.patch` | `/root/vllm-omni/vllm_omni/entrypoints/omni.py` | `4a24a51` | **REQUIRED** pod-local; cannot be monkey-patched (function signature change) |
| `vllm/` | `0001-utils-skip-fp32-from-layer.patch` | `vllm/lora/utils.py` (site-packages) | vllm 0.10+ | **REQUIRED** pod-local; DiT worker subprocess loads model before extension hijack runs |

## Why this patch is pod-local (not in-repo monkey-patch)

`Omni.generate()` needs a new `lora_request` parameter in its signature so
callers (our `engine.py`) can pass it. Monkey-patching a method to add a kwarg
is fragile (must copy the full method body, breaks on any upstream refactor).
A clean `git apply` is more maintainable until vllm-omni upstreams this.

## How to apply

On each pod (launcher AND every worker) that runs `train_new`:

```bash
PATCHES=/apdcephfs_fsgm3/share_305110755/hunyuan/gxhe/project/diffusionrl-main-unified-base-hi3-instruct/diffusionrl/patches

# 1. vllm-omni: lora_request passthrough
cd /root/vllm-omni
git apply $PATCHES/vllm_omni/0001-omni-py-lora_request-passthrough.patch
# verify:
grep -c 'lora_request' vllm_omni/entrypoints/omni.py  # expect >= 7

# 2. vllm site-packages: fp32 from_layer skip
cd /root/diffusionrl/.venv/lib/python3.12/site-packages
patch -p1 < $PATCHES/vllm/0001-utils-skip-fp32-from-layer.patch
# verify:
python3 -c "import vllm.lora.utils, inspect; print('OK' if 'bfloat16' in inspect.getsource(vllm.lora.utils.from_layer) else 'FAIL')"
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

## Why vllm/0001 is still pod-local (not just in-repo monkey-patch)

The in-repo `lora_hijack.py` monkey-patch works for the AR worker (extension
`__new__` runs before model load). But the DiT stage is loaded inside a
`StageDiffusionProc` subprocess that calls `from_layer()` during model init
**before** the extension's `__new__` is invoked. The monkey-patch is therefore
ineffective for DiT. A pod-local patch to site-packages is still required.
