"""Single-GPU bitwise parity probe: engine Wan2.2 DiT vs trainer diffusers DiT.

The fast iteration loop for the wan22 shared-kernel port (gate G4): build the
vLLM-Omni wan2_2 transformer (parity patches installed) and the diffusers
transformer (parity processor installed) from the SAME checkpoint expert, run
one forward on identical ``(x_t, σ·1000, UMT5-shaped embeds)``, and assert the
outputs are bitwise equal; then run one armed SDE step through the engine
scheduler vs the trainer strategy and assert bitwise logp. Orders faster than
an RL run for isolating a residual seam; on mismatch, per-module capture
hooks report the first divergent stage (patch_embedding / condition_embedder
/ blocks.N / norm_out / proj_out).

Run per expert (same code path, two checkpoints):

    python scripts/wan22_parity_probe.py --model /path/Wan2.2-T2V-A14B-Diffusers --expert high
    python scripts/wan22_parity_probe.py --model /path/Wan2.2-T2V-A14B-Diffusers --expert low

Requires the vllm venv (vllm/vllm_omni 0.20.0 + torch cu129); 14B bf16 x2
copies ≈ 56 GB — fits one H20.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from types import SimpleNamespace

# Parity env MUST be set before any vllm_omni import chain runs.
os.environ["UNIRL_VLLM_OMNI_PARITY"] = "1"
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

import torch  # noqa: E402


def _log(msg: str) -> None:
    print(f"[wan22-probe] {msg}", flush=True)


def _init_vllm_single_process() -> None:
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29511"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def _load_engine_model(model_path: str, subfolder: str) -> torch.nn.Module:
    """Build the vLLM-Omni wan transformer exactly like the worker does:
    modules created under the compute dtype, checkpoint tensors copied in by
    ``load_weights`` (AutoWeightsLoader — fuses to_qkv, remaps to_out.0)."""
    from safetensors import safe_open
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
        create_transformer_from_config,
        load_transformer_config,
    )

    cfg = load_transformer_config(model_path, subfolder, local_files_only=True)
    if not cfg:
        raise FileNotFoundError(f"no transformer config under {model_path}/{subfolder}")

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = create_transformer_from_config(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)

    def _weights():
        shards = sorted(glob.glob(os.path.join(model_path, subfolder, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {model_path}/{subfolder}")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

    model.load_weights(_weights())
    return model.eval()


def _load_trainer_model(model_path: str, subfolder: str) -> torch.nn.Module:
    from diffusers import WanTransformer3DModel

    from unirl.models.wan22.parity import install_shared_kernels

    model = (
        WanTransformer3DModel.from_pretrained(model_path, subfolder=subfolder, torch_dtype=torch.bfloat16)
        .to("cuda", dtype=torch.bfloat16)
        .eval()
    )
    install_shared_kernels(model)
    return model


def _capture_hooks(model: torch.nn.Module, store: list):
    from unirl.kernels.sd3 import parity_debug_sha

    def _mk(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                store.append((name, parity_debug_sha(t)))

        return hook

    handles = [
        model.patch_embedding.register_forward_hook(_mk("patch_embedding")),
        model.condition_embedder.register_forward_hook(_mk("condition_embedder")),
        model.proj_out.register_forward_hook(_mk("proj_out")),
    ]
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(_mk(f"blocks.{i}")))
    return handles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Wan2.2-T2V-A14B-Diffusers checkpoint root")
    ap.add_argument("--expert", choices=["high", "low"], default="high")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--sigma", type=float, default=0.9375, help="probe σ (default: first shift-5 20-step value)")
    ap.add_argument("--bisect", action="store_true", help="per-module SHA capture on both sides")
    args = ap.parse_args()

    # vLLM's distributed init, CustomOp construction, and layer forwards all
    # require an active vllm config (the worker runs inside one) — hold it
    # for the probe's entire lifetime.
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        return _run(args)


def _run(args) -> int:
    subfolder = "transformer" if args.expert == "high" else "transformer_2"
    _log(f"expert={args.expert} subfolder={subfolder}")

    _init_vllm_single_process()

    # Install the engine parity patches in-process (the worker gets them via
    # VLLMOmniHijack; the probe calls them directly).
    from unirl.rollout.engine.vllm_omni.patches.runtime import (
        patch_sd3_shared_kernels,
        patch_wan22_shared_kernels,
    )

    patch_sd3_shared_kernels()
    patch_wan22_shared_kernels()

    engine_model = _load_engine_model(args.model, subfolder)
    _log("engine model loaded")
    trainer_model = _load_trainer_model(args.model, subfolder)
    _log("trainer model loaded (parity kernels installed)")

    # Identical inputs. Random UMT5-shaped embeds are fine for KERNEL parity —
    # both condition embedders run the same diffusers modules on them.
    torch.manual_seed(7)
    lat_h, lat_w = args.height // 8, args.width // 8
    x = torch.randn(1, 16, 1, lat_h, lat_w, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    t = torch.tensor([args.sigma * 1000.0], dtype=torch.float32, device="cuda")
    text_dim = int(getattr(trainer_model.config, "text_dim", 4096))
    enc = torch.randn(1, 512, text_dim, dtype=torch.float32, device="cuda").to(torch.bfloat16)

    from vllm_omni.diffusion.forward_context import set_forward_context

    fake_od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            sequence_parallel_size=1, ulysses_degree=1, ring_degree=1, tensor_parallel_size=1, cfg_parallel_size=1
        )
    )

    eng_trace: list = []
    trn_trace: list = []
    if args.bisect:
        _capture_hooks(engine_model, eng_trace)
        _capture_hooks(trainer_model, trn_trace)

    with torch.no_grad():
        with set_forward_context(omni_diffusion_config=fake_od_config):
            eng_out = engine_model(hidden_states=x, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]
        trn_out = trainer_model(hidden_states=x, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]

    if args.bisect:
        for (en, es), (tn, ts) in zip(eng_trace, trn_trace):
            marker = "OK " if es == ts else ">>> DIVERGES"
            _log(f"{marker} {en:<22} engine={es} trainer={ts}")
            if es != ts:
                break

    if torch.equal(eng_out, trn_out):
        _log(f"FORWARD BITWISE OK  shape={tuple(eng_out.shape)} dtype={eng_out.dtype}")
    else:
        d = (eng_out.float() - trn_out.float()).abs()
        _log(
            f"FORWARD MISMATCH  max|Δ|={d.max().item():.3e} mean|Δ|={d.mean().item():.3e} "
            f"nonzero={int((d > 0).sum())}/{d.numel()} (rerun with --bisect)"
        )
        return 1

    # --- one armed SDE step: engine scheduler vs trainer strategy ---
    from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
        FlowMatchSDEDiscreteScheduler,
    )
    from unirl.sde.kernels import FlowSDEStrategy

    shift, steps = 5.0, 20
    tt = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    sigmas = ((shift * tt) / (1.0 + (shift - 1.0) * tt)).tolist()[:-1]
    sched = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=shift, eta=0.7)
    sched.set_timesteps(sigmas=sigmas, device="cuda")
    sched.arm(eta=0.7, sde_indices=[0])

    master = x.to(torch.float32)  # wan carries fp32 master latents
    gen = torch.Generator(device="cuda").manual_seed(11)
    prev = sched.step(eng_out.to(master.dtype), sched.timesteps[0], master, generator=gen)[0]
    engine_logp = sched._traj_log_probs[0]

    strategy = FlowSDEStrategy()
    _, logp, _ = strategy.denoise(
        noise_pred=eng_out.to(master.dtype),
        sample=master,
        sigma=sched.sigmas[0],
        sigma_next=sched.sigmas[1],
        eta=0.7,
        prev_sample=prev,
        sigma_max=sched.sigmas[1],
        step_index=0,
    )
    if torch.equal(logp.float(), engine_logp.float()):
        _log(f"SDE LOGP BITWISE OK  logp={logp.float().item():.6f}")
    else:
        _log(
            f"SDE LOGP MISMATCH  |Δ|={(logp.float() - engine_logp.float()).abs().max().item():.3e}"
        )
        return 1

    _log("PROBE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
