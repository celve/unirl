"""Plumbing smoke for the new-protocol SGLang rollout engine.

Runs the full ``SGLangRolloutEngine.generate(req)`` path in-process against a
stub SGLang runtime (no live scheduler, no model, no GPU required). Validates:

1. The engine forwards ``req.request_conditions['initial_latents'].latents``
   into SGLang's ``initial_noise`` kwarg verbatim (caller-supplied noise
   pass-through is the user-requested feature).
2. The engine's response translator emits a fully-populated ``RolloutResp``
   with ``rollout_traces['image']`` (latents/sigmas/indices/sample_indices),
   ``decoded['image']``, and ``conditions['text']`` + ``conditions['negative_text']``.
3. ``SD3Conditions.from_dict(resp.conditions)`` reconstructs the typed
   container the trainer-side replay path will consume.
4. The ``noise_group_ids`` forwarded to SGLang match ``req.group_ids``.
5. ODE-mode rollout (no ``sde_indices``) leaves ``LatentSegment.sde_logp``
   ``None``; SDE-mode native logprob lands on ``sde_logp``.

Run as::

    python scripts/smoke_sd3_t2i_replay_sglang.py

No external dependencies, no Ray, no SGLang server. The full live-SGLang
smoke (real scheduler + model checkpoint) will land as a follow-up once a
matching SD3 / Flux / Mochi checkpoint is staged.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

from diffusionrl.models_new.sd3.conditions import SD3Conditions
from diffusionrl.models_new.sd3.config import SD3PipelineConfig
from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.engine import SGLangRolloutEngine
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import SamplingParams, SDEConfig

# ---------------------------------------------------------------------------
# Stub SGLang runtime
# ---------------------------------------------------------------------------


def _make_fake_result(
    *,
    num_steps: int,
    shift: float,
    batch_per_result: int,
    channels: int,
    h: int,
    w: int,
    sample_hw: tuple[int, int],
    text_tokens: int,
    text_hidden: int,
    with_log_probs: bool,
    with_neg_embeds: bool,
    initial_noise: torch.Tensor | None,
) -> SimpleNamespace:
    """Build one mock ``GenerationResult`` matching SGLang's shape."""
    sigmas = get_sigma_schedule(num_steps, shift=shift)
    if initial_noise is not None:
        # When the caller pre-shipped initial latents, the trajectory's first
        # column reflects them (SGLang would copy ``Req.latents`` into the
        # initial trajectory slot).
        traj_first = initial_noise[:batch_per_result, :channels, :h, :w].clone()
        rest = torch.zeros(batch_per_result, num_steps, channels, h, w)
        traj_latents = torch.cat([traj_first.unsqueeze(1), rest], dim=1)
    else:
        traj_latents = torch.zeros(batch_per_result, num_steps + 1, channels, h, w)
    traj_log_probs = torch.full((batch_per_result, num_steps), -1.5) if with_log_probs else None
    sample_h, sample_w = sample_hw
    samples_t = torch.rand(3, sample_h, sample_w)
    if batch_per_result > 1:
        samples_t = [torch.rand(3, sample_h, sample_w) for _ in range(batch_per_result)]
    prompt_embeds = torch.randn(batch_per_result, text_tokens, text_hidden)
    pooled = torch.randn(batch_per_result, text_hidden)
    encoder_attention_mask = torch.ones(batch_per_result, text_tokens)
    neg_embeds = torch.randn(batch_per_result, text_tokens, text_hidden) if with_neg_embeds else None
    neg_pooled = torch.randn(batch_per_result, text_hidden) if with_neg_embeds else None
    return SimpleNamespace(
        trajectory_latents=traj_latents,
        trajectory_log_probs=traj_log_probs,
        trajectory_timesteps=sigmas,
        samples=samples_t,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        encoder_attention_mask=encoder_attention_mask,
        negative_prompt_embeds=neg_embeds,
        neg_pooled_prompt_embeds=neg_pooled,
    )


class _StubEngine(SGLangRolloutEngine):
    """Bypasses the SGLang ctor; installs in-process stubs for ``_runtime`` /
    ``_generator`` / ``_server_args`` so the parent's ``generate`` path can
    run without a live scheduler or model.
    """

    def __init__(
        self,
        *,
        cfg: SGLangEngineConfig,
        model_config: SD3PipelineConfig,
        result_kwargs: Dict[str, Any],
        num_outputs_per_prompt: int = 1,
        strategy: Any = None,
    ) -> None:
        # Mimic the relevant parts of SGLangRolloutEngine.__init__ without
        # the DiffGenerator boot.
        self.cfg = cfg
        self.model_config = model_config
        self.strategy = strategy
        self.rank = None
        self._device = torch.device("cpu")
        self._sde_label = SGLangRolloutEngine._resolve_sde_label(strategy)
        self._target_modules: List[str] = ["transformer"]
        self._is_offloaded = False
        self._runtime: Dict[str, Any] = {}
        self._result_kwargs = result_kwargs
        self._num_outputs_per_prompt = int(num_outputs_per_prompt)
        self.last_sgl_kwargs: Dict[str, Any] | None = None

        # _server_args is consulted by _resolve_initial_noise via
        # _server_args.pipeline_config.prepare_latent_shape. For this smoke
        # we don't enter that path (we always pass initial_latents), so we
        # stub a permissive surface.
        self._server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                prepare_latent_shape=lambda _b, _n, _f: (1, 4, 8, 8),
            ),
        )

        self._generator = SimpleNamespace(
            generate=self._fake_generate,
            shutdown=lambda: None,
        )

    def _fake_generate(self, *, sampling_params_kwargs: Dict[str, Any]):
        self.last_sgl_kwargs = sampling_params_kwargs
        # One result per unique prompt (post de-expansion) shaped at
        # batch_per_result = num_outputs_per_prompt (or 1 if absent).
        n_results = (
            1
            if isinstance(sampling_params_kwargs.get("prompt"), str)
            else len(sampling_params_kwargs.get("prompt") or [])
        )
        batch_per_result = int(sampling_params_kwargs.get("num_outputs_per_prompt", 1) or 1)
        initial_noise = sampling_params_kwargs.get("initial_noise")
        results: List[SimpleNamespace] = []
        for i in range(n_results):
            sub_noise = None
            if initial_noise is not None:
                start = i * batch_per_result
                end = start + batch_per_result
                sub_noise = initial_noise[start:end]
            results.append(
                _make_fake_result(
                    **self._result_kwargs,
                    batch_per_result=batch_per_result,
                    initial_noise=sub_noise,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_NUM_STEPS = 4
_SHIFT = 3.0
_CHANNELS = 4
_LATENT_HW = (8, 8)
_SAMPLE_HW = (64, 64)
_TEXT_TOKENS = 7
_TEXT_HIDDEN = 12


def _make_cfg() -> SGLangEngineConfig:
    sampling = SamplingParams(
        num_inference_steps=_NUM_STEPS,
        guidance_scale=4.5,
        height=_SAMPLE_HW[0],
        width=_SAMPLE_HW[1],
        num_frames=1,
        seed=0,
        num_samples_per_prompt=1,
        sde_config=SDEConfig(eta=0.7),
        sde_indices=None,
        sampler_kwargs={},
    )
    return SGLangEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=True,
        init_same_noise=False,
        logprob_source="replay",
    )


def _make_model_config() -> SD3PipelineConfig:
    return SD3PipelineConfig(pretrained_model_ckpt_path="/dev/null/fake-sd3")


def _make_req(
    *,
    batch: int,
    sde_indices: List[int] | None = None,
    initial_latents: torch.Tensor | None = None,
) -> RolloutReq:
    diffusion = {
        "height": _SAMPLE_HW[0],
        "width": _SAMPLE_HW[1],
        "num_inference_steps": _NUM_STEPS,
        "guidance_scale": 4.5,
        "eta": 0.7,
        "seed": 12345,
        "sde_indices": sde_indices,
        "num_samples_per_prompt": 1,
    }
    rc: Dict[str, Any] = {}
    if initial_latents is not None:
        rc["initial_latents"] = ImageLatentCondition(latents=initial_latents)
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(batch)],
        group_ids=[f"g{i}" for i in range(batch)],
        primitives={"text": Texts(texts=[f"prompt_{i}" for i in range(batch)])},
        request_conditions=rc,
        stage_params={"diffusion": diffusion},
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_basic_response_shape() -> None:
    """Engine returns a populated RolloutResp with rollout_traces + decoded + conditions."""
    cfg = _make_cfg()
    engine = _StubEngine(
        cfg=cfg,
        model_config=_make_model_config(),
        result_kwargs=dict(
            num_steps=_NUM_STEPS,
            shift=_SHIFT,
            channels=_CHANNELS,
            h=_LATENT_HW[0],
            w=_LATENT_HW[1],
            sample_hw=_SAMPLE_HW,
            text_tokens=_TEXT_TOKENS,
            text_hidden=_TEXT_HIDDEN,
            with_log_probs=False,
            with_neg_embeds=True,
        ),
    )
    req = _make_req(batch=2)
    resp = engine.generate(req)

    assert resp.sample_ids == req.sample_ids
    assert resp.group_ids == req.group_ids
    seg = resp.rollout_traces["image"]
    assert seg.latents.shape == (2, _NUM_STEPS + 1, _CHANNELS, *_LATENT_HW), (
        f"latents shape: got {tuple(seg.latents.shape)}"
    )
    assert seg.sde_logp is None  # replay mode
    decoded = resp.decoded["image"]
    assert decoded.pixels.shape == (2, 3, *_SAMPLE_HW)
    assert decoded.pixels.dtype == torch.float32
    assert "text" in resp.conditions
    assert "negative_text" in resp.conditions
    assert resp.conditions["text"].embeds.shape == (2, _TEXT_TOKENS, _TEXT_HIDDEN)
    print("[PASS] scenario_basic_response_shape")


def scenario_initial_latents_passthrough() -> None:
    """Caller-supplied initial latents land in SGLang's initial_noise kwarg verbatim."""
    cfg = _make_cfg()
    engine = _StubEngine(
        cfg=cfg,
        model_config=_make_model_config(),
        result_kwargs=dict(
            num_steps=_NUM_STEPS,
            shift=_SHIFT,
            channels=_CHANNELS,
            h=_LATENT_HW[0],
            w=_LATENT_HW[1],
            sample_hw=_SAMPLE_HW,
            text_tokens=_TEXT_TOKENS,
            text_hidden=_TEXT_HIDDEN,
            with_log_probs=False,
            with_neg_embeds=False,
        ),
    )
    fixed_x_T = torch.arange(2 * _CHANNELS * _LATENT_HW[0] * _LATENT_HW[1], dtype=torch.float32).reshape(
        2, _CHANNELS, *_LATENT_HW
    )
    req = _make_req(batch=2, initial_latents=fixed_x_T)
    resp = engine.generate(req)

    assert engine.last_sgl_kwargs is not None
    forwarded = engine.last_sgl_kwargs.get("initial_noise")
    assert forwarded is not None, "engine must forward initial_noise to SGLang"
    assert torch.equal(forwarded, fixed_x_T), "initial_noise must be forwarded verbatim"
    # noise_group_ids should mirror req.group_ids.
    assert engine.last_sgl_kwargs.get("noise_group_ids") == req.group_ids
    # Trajectory column 0 should reflect the pre-shipped x_T (stub copies it through).
    seg = resp.rollout_traces["image"]
    assert torch.equal(seg.latents[:, 0], fixed_x_T)
    print("[PASS] scenario_initial_latents_passthrough")


def scenario_typed_conditions_roundtrip() -> None:
    """resp.conditions round-trips through SD3Conditions.from_dict for trainer-side replay."""
    cfg = _make_cfg()
    engine = _StubEngine(
        cfg=cfg,
        model_config=_make_model_config(),
        result_kwargs=dict(
            num_steps=_NUM_STEPS,
            shift=_SHIFT,
            channels=_CHANNELS,
            h=_LATENT_HW[0],
            w=_LATENT_HW[1],
            sample_hw=_SAMPLE_HW,
            text_tokens=_TEXT_TOKENS,
            text_hidden=_TEXT_HIDDEN,
            with_log_probs=False,
            with_neg_embeds=True,  # CFG on → negative_text present
        ),
    )
    req = _make_req(batch=2)
    resp = engine.generate(req)

    typed = SD3Conditions.from_dict(resp.conditions)
    assert typed.text is not None
    assert typed.text.embeds.shape == (2, _TEXT_TOKENS, _TEXT_HIDDEN)
    assert typed.negative_text is not None
    assert typed.negative_text.embeds.shape == (2, _TEXT_TOKENS, _TEXT_HIDDEN)
    print("[PASS] scenario_typed_conditions_roundtrip")


def scenario_sde_mode_native_logprob() -> None:
    """SDE-mode + cfg.logprob_source='native' lands SGLang's log_probs on LatentSegment.sde_logp."""
    sampling = SamplingParams(
        num_inference_steps=_NUM_STEPS,
        guidance_scale=4.5,
        height=_SAMPLE_HW[0],
        width=_SAMPLE_HW[1],
        num_frames=1,
        seed=0,
        num_samples_per_prompt=1,
        sde_config=SDEConfig(eta=0.7),
        sampler_kwargs={},
    )
    cfg = SGLangEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=True,
        init_same_noise=False,
        logprob_source="native",
    )

    class _FlowStrategy:
        canonical_name = "flow"

    engine = _StubEngine(
        cfg=cfg,
        model_config=_make_model_config(),
        result_kwargs=dict(
            num_steps=_NUM_STEPS,
            shift=_SHIFT,
            channels=_CHANNELS,
            h=_LATENT_HW[0],
            w=_LATENT_HW[1],
            sample_hw=_SAMPLE_HW,
            text_tokens=_TEXT_TOKENS,
            text_hidden=_TEXT_HIDDEN,
            with_log_probs=True,
            with_neg_embeds=False,
        ),
        strategy=_FlowStrategy(),
    )
    req = _make_req(batch=2, sde_indices=list(range(_NUM_STEPS)))
    resp = engine.generate(req)

    seg = resp.rollout_traces["image"]
    assert seg.sde_logp is not None, "native mode must populate sde_logp"
    assert seg.sde_logp.shape == (2, _NUM_STEPS)
    assert seg.sde_indices is not None
    assert torch.equal(seg.sde_indices, torch.arange(_NUM_STEPS))
    # SDE kernel kwargs land in SGLang's kwargs.
    assert engine.last_sgl_kwargs["rollout"] is True
    assert engine.last_sgl_kwargs["rollout_sde_type"] == "sde"
    print("[PASS] scenario_sde_mode_native_logprob")


def main() -> int:
    scenario_basic_response_shape()
    scenario_initial_latents_passthrough()
    scenario_typed_conditions_roundtrip()
    scenario_sde_mode_native_logprob()
    print("\nALL SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
