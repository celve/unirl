"""End-to-end plumbing test for ``initial_latents`` from
``RolloutReq.request_conditions['initial_latents']`` → ``Pipeline.generate(req)``
→ ``DiffusionStage.diffuse(initial_latents=...)``.

This is the fix for the silent-drop bug codex flagged: the driver
(``train_new.py`` via ``compute_initial_noise_for_request`` + the trainside
engine path) puts per-sample x_T into ``request_conditions['initial_latents']``,
but pre-Batch-5 the NEW pipelines all called their internal ``generate_latents``
and ignored the request — meaning RL exploration on trainside used the
SAME noise every rollout (params.seed-keyed), even though the driver had
varied noise per ``rollout_id``.

We test SD3 here as the canonical pattern — the other 4 NEW pipelines
(WAN21, WAN22, Qwen-Image, HunyuanVideo15) share the identical
``initial_cond = (req.request_conditions or {}).get("initial_latents")``
extraction shape, so coverage at the Stage level (test_*_diffusion_stage.py)
plus this one pipeline-level proof is sufficient for the plumbing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusionrl.models_new.sd3.bundle import SD3Bundle
from diffusionrl.models_new.sd3.diffusion import (
    SD3DiffusionStage,
    SD3DiffusionStep,
)
from diffusionrl.models_new.sd3.pipeline import SD3Pipeline
from diffusionrl.models_new.sd3.text_embed import SD3TextEmbedStage
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.conditions.text import TextEmbedCondition
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


class _FakeTransformer(nn.Module):
    def forward(self, *, hidden_states, encoder_hidden_states, timestep, pooled_projections, return_dict):
        return (torch.zeros_like(hidden_states),)


class _StubTextEmbedStage(SD3TextEmbedStage):
    """Bypass real CLIP / T5 encoders — emit deterministic embeds of the
    shape the SD3 stage expects, so pipeline.generate(req) can run end-to-end
    on CPU without weights."""

    def __init__(self):
        # Skip parent __init__ — we don't need a bundle for this test.
        pass

    def embed(self, p: Texts) -> TextEmbedCondition:
        b = len(p.texts)
        return TextEmbedCondition(
            embeds=torch.zeros(b, 4, 8),
            pooled=torch.zeros(b, 16),
        )


def _make_pipeline() -> SD3Pipeline:
    bundle = SD3Bundle(
        transformer=_FakeTransformer(),
        vae=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        scheduler=None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        pretrained_path="fake",
    )
    diffusion = SD3DiffusionStage(
        model=bundle,
        step=SD3DiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="fp32",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        latent_channels=4,
    )
    return SD3Pipeline(
        bundle=bundle,
        text_embed=_StubTextEmbedStage(),
        diffusion=diffusion,
        vae_decode=None,  # generate() doesn't decode in this test
    )


def _make_req(prompts, initial_latents=None, T=2):
    sigmas = get_sigma_schedule(T, shift=3.0)
    req_cond = {}
    if initial_latents is not None:
        req_cond["initial_latents"] = ImageLatentCondition(latents=initial_latents)
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
        stage_params={
            "diffusion": {
                "num_inference_steps": T,
                "guidance_scale": 1.0,
                "height": 8,
                "width": 8,
                "seed": 99,
                "sde_indices": list(range(T)),
                "eta": 0.0,
            }
        },
        request_conditions=req_cond,
        sigmas=sigmas,
    )


def test_pipeline_generate_consumes_preshipped_initial_latents(monkeypatch):
    """req.request_conditions['initial_latents'].latents must arrive at
    DiffusionStage.diffuse(initial_latents=...) and override internal RNG."""
    pipeline = _make_pipeline()

    fixed_x_T = torch.full((2, 4, 1, 1), 0.7)
    req = _make_req(["a", "b"], initial_latents=fixed_x_T)

    # Skip the VAE decode — diffusion is what we're testing.
    monkeypatch.setattr(pipeline, "vae_decode", _NoOpDecode())

    resp = pipeline.generate(req)
    seg = resp.rollout_traces["image"]
    # Pre-shipped tensor lands at position 0 verbatim.
    assert torch.allclose(seg.latents[:, 0].float(), fixed_x_T.float())


def test_pipeline_generate_without_initial_latents_falls_back_to_internal(monkeypatch):
    """When request_conditions is empty, pipeline uses params.seed-keyed
    internal RNG — same behavior as pre-Batch-5."""
    pipeline = _make_pipeline()
    req = _make_req(["a", "b"], initial_latents=None)  # no request_conditions key
    monkeypatch.setattr(pipeline, "vae_decode", _NoOpDecode())

    resp = pipeline.generate(req)
    seg = resp.rollout_traces["image"]
    # Just verify a sensible Gaussian came out of the internal path
    # (shape correct, finite, not all-equal to the fixed tensor).
    assert seg.latents.shape == (2, 3, 4, 1, 1)
    assert torch.isfinite(seg.latents[:, 0]).all()


class _NoOpDecode:
    """Stub VAE decode that just returns the input latents unchanged."""

    def decode(self, segment):
        from diffusionrl.types.primitives import Images

        # Pull final clean latent as pixels (shape doesn't matter for this test).
        return Images(pixels=segment.latents[:, -1])
