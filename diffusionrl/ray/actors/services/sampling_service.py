"""Sampling service facade for TrainingActor RPC methods."""

from __future__ import annotations

from typing import List, Optional, Set

import torch

from diffusionrl.ray.actors.internal import ActorSamplingExecutor
from diffusionrl.types.sampling import InferenceRequest, SamplerOutput


class TrainingActorSamplingService:
    """Delegates training-actor sampling RPCs to ActorSamplingExecutor."""

    def __init__(self, executor: Optional[ActorSamplingExecutor] = None) -> None:
        self._executor = executor or ActorSamplingExecutor()

    @property
    def executor(self) -> ActorSamplingExecutor:
        return self._executor

    def generate(
        self,
        actor,
        *,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
        decode_for_reward: bool = False,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> SamplerOutput:
        return self._executor.generate(
            actor,
            prompts=prompts,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            text_ids=text_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            decode_for_reward=decode_for_reward,
            sde_indices=sde_indices,
            **kwargs,
        )

    def generate_batch(self, actor, requests: List[InferenceRequest]) -> List[SamplerOutput]:
        return self._executor.generate_batch(actor, requests)

    def sample_batch(self, actor, prompts: Optional[List[str]] = None, **kwargs) -> SamplerOutput:
        return self._executor.sample_batch(actor, prompts=prompts, **kwargs)
