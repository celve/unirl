from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import torch

from diffusionrl.config.require import require
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.request import RolloutRequest


def _deexpand_prompts(
    prompts: List[str],
    num_samples_per_prompt: int,
) -> Tuple[List[str], int]:
    """Collapse prompt-major repeated prompts back to unique prompts when possible."""
    k = int(num_samples_per_prompt)
    if k <= 1:
        return list(prompts), 1

    n = len(prompts)
    if n == 0 or n % k != 0:
        return list(prompts), 1

    num_unique = n // k
    unique_prompts: List[str] = []
    for i in range(num_unique):
        group_start = i * k
        base = prompts[group_start]
        for j in range(1, k):
            if prompts[group_start + j] != base:
                return list(prompts), 1
        unique_prompts.append(base)

    return unique_prompts, k


@dataclass
class SGLangRolloutRequest:
    """Typed per-request kwargs for SGLang's GRPO rollout path."""

    # Schedule / geometry
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    sigmas: List[float]

    # Prompts & K-expansion
    prompt: Union[str, List[str]]
    # ``None`` selects ODE/non-SDE mode (eval, NFT-train) — SGLang runs
    # deterministic sampling and we omit the ``rollout`` / ``rollout_sde_*``
    # kwargs entirely. Non-empty list selects SDE mode (GRPO).
    rollout_sde_indices: Optional[List[int]]
    rollout_noise_level: float

    num_outputs_per_prompt: Optional[int] = None
    # Raw escape-hatch overrides forwarded to SGLang's ``DiffGenerator.generate``
    # ``sampling_params_kwargs``. Merged at the bottom of ``to_kwargs()``;
    # engine-pinned and typed fields below override anything here.
    # Use this for SGLang-side knobs DiffusionRL doesn't model directly
    # (``negative_prompt``, ``return_negative_prompt_embeds``, ``fps``, ...).
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Engine-populated before to_kwargs()
    initial_noise: Optional[torch.Tensor] = None
    rollout_sde_type: Optional[str] = None  # SGLang label ("sde" / "cps"); only required in SDE mode

    @classmethod
    def from_rollout_request(cls, rollout_request: RolloutRequest) -> Self:
        sp = rollout_request.sampling_params
        prompts = list(rollout_request.prompts.prompts)
        require(bool(prompts), "SGLang engine requires non-empty prompts")

        unique_prompts, validated_k = _deexpand_prompts(prompts, int(sp.num_samples_per_prompt))

        sigmas = get_sigma_schedule(
            int(sp.num_inference_steps),
            shift=float(sp.sde_config.shift),
        )[:-1].tolist()

        rollout_sde_indices = sorted(int(v) for v in sp.sde_indices) if sp.sde_indices is not None else None

        return cls(
            num_inference_steps=int(sp.num_inference_steps),
            guidance_scale=float(sp.guidance_scale),
            height=int(sp.height),
            width=int(sp.width),
            num_frames=int(sp.num_frames),
            sigmas=sigmas,
            prompt=unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            rollout_sde_indices=rollout_sde_indices,
            rollout_noise_level=float(sp.sde_config.eta),
            num_outputs_per_prompt=validated_k if validated_k > 1 else None,
            sampler_kwargs=dict(sp.sampler_kwargs or {}),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """Serialize to the dict SGLang's ``DiffGenerator.generate`` consumes.

        Layering (lowest priority → highest):

        1. ``self.sampler_kwargs`` — raw SGLang escape-hatch overrides
           (``negative_prompt``, ``return_negative_prompt_embeds``, ``fps``, ...).
        2. Typed/computed fields — schedule, geometry, prompts.
        3. Engine pins — trajectory return, noise sharing, RNG policy.
        4. SDE-mode kwargs — present only when ``rollout_sde_indices is not None``.

        Engine-state fields (``initial_noise``, and in SDE mode
        ``rollout_sde_type``) must be populated by the engine before calling.
        ``return_trajectory_latents=True`` is pinned in both modes — downstream
        consumers (NFT loss, eval reward) read ``trajectory_latents[:, -1]``
        as the final clean latent and SGLang exposes no separate
        ``final_latents`` field.
        """
        require(self.initial_noise is not None, "initial_noise must be set before to_kwargs() — engine populates it")

        # Negative-prompt invariant: SGLang gates CFG on guidance_scale>1
        # (schedule_batch.py:273-274) and consumes negative_prompt under that
        # flag independently of return_negative_prompt_embeds
        # (text_encoding.py:87-100). Default return_negative_prompt_embeds=False
        # (sampling_params.py:185). If we let a recipe pass negative_prompt
        # without also pinning return_negative_prompt_embeds=True, rollout
        # would condition on the negative prompt while training-side replay
        # falls back to torch.zeros_like(prompt_embeds) (sd3_sampler.py:179)
        # — silent GRPO ratio mismatch. Fail fast at the boundary.
        neg_prompt = self.sampler_kwargs.get("negative_prompt")
        return_neg_embeds = bool(self.sampler_kwargs.get("return_negative_prompt_embeds", False))
        require(
            neg_prompt is None or return_neg_embeds,
            "sampler_kwargs.negative_prompt is set but return_negative_prompt_embeds is not True. "
            "SGLang gates CFG on guidance_scale>1 independently of return_negative_prompt_embeds, "
            "so the rollout would condition on the negative prompt while training-side replay falls "
            "back to zero negative embeds (sd3_sampler.py:179) — silent GRPO ratio mismatch. "
            "Set sampler_kwargs.return_negative_prompt_embeds=True to keep rollout and replay aligned.",
        )

        # Layer 1: user escape-hatch (lowest priority).
        kwargs: Dict[str, Any] = dict(self.sampler_kwargs)

        # Layers 2 + 3: typed/computed fields and engine pins (override layer 1).
        kwargs.update(
            {
                # Schedule / geometry
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "height": self.height,
                "width": self.width,
                "num_frames": self.num_frames,
                "sigmas": self.sigmas,
                # Prompts
                "prompt": self.prompt,
                # Initial noise + RNG: precomputed x_T from DiffusionRL; SGLang uses
                # global RNG for per-step SDE noise (matches FSDP generator=None policy).
                "initial_noise": self.initial_noise,
                "seed": None,
                # Return-shape policy (engine contract — downstream needs latents
                # + prompt embeds; decoded media comes from sampler output, not
                # from per-step decode)
                "save_output": False,
                "return_file_paths_only": False,
                "return_trajectory_latents": True,
                "return_trajectory_decoded": False,
                "return_prompt_embeds": True,
                # SGLang-side noise sharing must be off — initial_noise already
                # encodes the (optional) group sharing pattern.
                "init_same_noise": False,
            }
        )

        # Layer 4: SDE-kernel kwargs only apply when the algorithm requested
        # per-step SDE noise (GRPO). For ODE/non-SDE mode (eval, NFT-train) we
        # omit them entirely so SGLang runs deterministic sampling.
        if self.rollout_sde_indices is not None:
            require(
                self.rollout_sde_type is not None,
                "rollout_sde_type must be set before to_kwargs() in SDE mode — engine populates it",
            )
            kwargs["rollout"] = True
            kwargs["rollout_sde_type"] = self.rollout_sde_type
            kwargs["rollout_noise_level"] = self.rollout_noise_level
            kwargs["rollout_sde_indices"] = self.rollout_sde_indices
        if self.num_outputs_per_prompt is not None:
            kwargs["num_outputs_per_prompt"] = self.num_outputs_per_prompt
        return kwargs
