"""Training-actor sampling execution helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.types.sampling import RolloutRequest, RolloutOutput
from diffusionrl.utils import load_function
from diffusionrl.utils.adapter_utils import switch_adapter

logger = logging.getLogger(__name__)


@contextmanager
def sampling_eval_context(modules: List[nn.Module]):
    """Temporarily switch modules to eval/no-grad and restore all training flags."""
    original_modes = [(module, bool(module.training)) for module in modules if isinstance(module, nn.Module)]
    param_states: List[Tuple[torch.nn.Parameter, bool]] = []
    seen_params: Set[int] = set()
    for module, _ in original_modes:
        for param in module.parameters(recurse=True):
            pid = id(param)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            param_states.append((param, bool(param.requires_grad)))
            if param.requires_grad:
                param.requires_grad_(False)
    for module, _ in original_modes:
        module.eval()

    try:
        # Use no_grad instead of inference_mode to avoid FSDP grad_fn/AccumulateGrad
        # assertion failures when returning to training.
        with torch.no_grad():
            yield
    finally:
        for module, was_training in original_modes:
            module.train(was_training)
        for param, requires_grad in param_states:
            param.requires_grad_(requires_grad)


def _tensor_to_pil(images: torch.Tensor) -> List[Any]:
    """Convert tensor to PIL images; video tensors use middle frame."""
    from PIL import Image
    import numpy as np

    pil_images = []
    images = images.cpu()

    if images.dim() == 5:
        frame_count = images.shape[2]
        images = images[:, :, frame_count // 2]

    for img in images:
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))

    return pil_images


class ActorSamplingExecutor:
    """Sampling executor used by TrainingActor RPC boundary."""

    def iter_reflection_modules(
        self,
        obj: Any,
        *,
        include_transformer: bool,
    ) -> List[Tuple[str, nn.Module]]:
        """Collect likely offloadable submodules from arbitrary objects via reflection."""
        if obj is None:
            return []
        known_names = {
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
            "image_encoder",
        }
        results: List[Tuple[str, nn.Module]] = []
        for name, value in obj.__dict__.items():
            if not isinstance(value, nn.Module):
                continue
            base_name = name.lstrip("_").lower()
            if not include_transformer and "transformer" in base_name:
                continue
            if base_name in known_names or any(
                token in base_name for token in ("encoder", "vae", "transformer")
            ):
                results.append((name, value))
        return results

    def ensure_sampling_components(self, actor: Any) -> None:
        if actor._sampling_ready:
            return
        if actor.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")

        try:
            actor.model_bundle.load_aux_components()
        except Exception as e:
            logger.warning("Failed to load auxiliary components: %s", e)
            raise

        actor.text_encoder = getattr(actor.model_bundle, "text_encoder", None)
        actor.vae = getattr(actor.model_bundle, "vae", None)
        actor.scheduler = getattr(actor.model_bundle, "scheduler", None)

        sampler_path = actor._sampling_config.get("sampler_path")
        if not sampler_path:
            raise ValueError("sampling_config must provide sampler_path for training-actor sampling")

        sampler_cls = load_function(sampler_path)
        sampler_kwargs = dict(actor._sampling_config.get("sampler_kwargs", {}))
        extra_kwargs = {}
        if hasattr(actor.model_bundle, "get_sampler_extra_kwargs"):
            extra_kwargs = actor.model_bundle.get_sampler_extra_kwargs() or {}
        for key, value in extra_kwargs.items():
            sampler_kwargs.setdefault(key, value)

        actor._sampler = sampler_cls(
            model=actor.model,
            text_encoder=actor.text_encoder,
            vae=actor.vae,
            eta=actor._sampling_config.get("eta", 1.0),
            sde_type=actor._sampling_config.get("sde_type", "sde"),
            shift=actor._sampling_config.get("shift", 3.0),
            **sampler_kwargs,
        )

        actor._sampling_ready = True

    def _encode_prompt(self, actor: Any, prompts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        if actor.model_bundle is None:
            raise RuntimeError("Model bundle not loaded")
        if not hasattr(actor.model_bundle, "encode_prompt_for_inference"):
            raise RuntimeError("Model bundle does not support inference prompt encoding")
        return actor.model_bundle.encode_prompt_for_inference(prompts, **kwargs)

    def _decode_latents(self, actor: Any, latents: torch.Tensor) -> torch.Tensor:
        if actor.vae is None:
            raise RuntimeError("VAE not available for decoding")

        with torch.no_grad():
            if hasattr(actor.vae, "config") and hasattr(actor.vae.config, "scaling_factor"):
                scaling_factor = actor.vae.config.scaling_factor
            else:
                scaling_factor = 0.18215

            latents_float = latents.to(dtype=torch.float32)
            decoded = actor.vae.to(torch.float32).decode(latents_float / scaling_factor).sample
            return (decoded + 1) / 2

    def _iter_sampling_mode_modules(self, actor: Any) -> List[nn.Module]:
        modules: List[nn.Module] = []
        seen: Set[int] = set()
        for component in (actor.model, actor.text_encoder, actor.vae):
            if isinstance(component, nn.Module):
                ident = id(component)
                if ident not in seen:
                    modules.append(component)
                    seen.add(ident)

        if actor.model_bundle is not None and hasattr(actor.model_bundle, "iter_offloadable_modules"):
            for _name, component in actor.model_bundle.iter_offloadable_modules(include_transformer=True):
                if isinstance(component, nn.Module):
                    ident = id(component)
                    if ident not in seen:
                        modules.append(component)
                        seen.add(ident)
        return modules

    @contextmanager
    def _sampling_eval_context(self, actor: Any):
        modules = self._iter_sampling_mode_modules(actor)
        with sampling_eval_context(modules):
            yield

    def generate(
        self,
        actor: Any,
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
    ) -> RolloutOutput:
        if not actor._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        if actor._is_offloaded:
            actor.onload()

        self.ensure_sampling_components(actor)

        num_inference_steps = num_inference_steps or actor._sampling_config.get("num_inference_steps", 50)
        if guidance_scale is None:
            guidance_scale = actor._sampling_config.get("guidance_scale", 7.5)
        height = height or actor._sampling_config.get("height", 256)
        width = width or actor._sampling_config.get("width", 256)
        num_frames = num_frames or actor._sampling_config.get("num_frames", 16)

        sampling_adapter = kwargs.pop("sampling_adapter", None)
        if sampling_adapter is None:
            sampling_adapter = actor._sampling_config.get("sampling_adapter")

        init_same_noise = kwargs.pop("init_same_noise", actor._sampling_config.get("init_same_noise", False))
        num_samples_per_prompt = kwargs.pop(
            "num_samples_per_prompt",
            actor._sampling_config.get("num_samples_per_prompt", 1),
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=actor._device)
            generator.manual_seed(seed)

        with self._sampling_eval_context(actor):
            if prompts is not None and prompt_embeds is None:
                encoded = self._encode_prompt(actor, prompts)
                prompt_embeds = encoded.get("prompt_embeds")
                pooled_prompt_embeds = encoded.get("pooled_prompt_embeds", pooled_prompt_embeds)
                negative_prompt_embeds = encoded.get("negative_prompt_embeds")
                negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")
                if text_ids is None:
                    text_ids = encoded.get("text_ids")
            else:
                negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
                negative_pooled_prompt_embeds = kwargs.pop("negative_pooled_prompt_embeds", None)

            if sampling_adapter and actor.model is not None:
                with switch_adapter(actor.model, sampling_adapter):
                    output = actor._sampler.sample(
                        prompts=prompts,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        encoder_attention_mask=encoder_attention_mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        generator=generator,
                        sde_indices=sde_indices,
                        text_ids=text_ids,
                        init_same_noise=init_same_noise,
                        num_samples_per_prompt=num_samples_per_prompt,
                        **kwargs,
                    )
            else:
                output = actor._sampler.sample(
                    prompts=prompts,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    encoder_attention_mask=encoder_attention_mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=generator,
                    sde_indices=sde_indices,
                    text_ids=text_ids,
                    init_same_noise=init_same_noise,
                    num_samples_per_prompt=num_samples_per_prompt,
                    **kwargs,
                )

        if decode_for_reward:
            try:
                decoded = self._decode_latents(actor, output.latents)
                decoded_images = _tensor_to_pil(decoded)
                output = RolloutOutput(
                    latents=output.latents,
                    timesteps=output.timesteps,
                    trajectories=output.trajectories,
                    log_probs=output.log_probs,
                    embeddings=output.embeddings,
                    decoded_images=decoded_images,
                    metadata=output.metadata,
                    step_indices=output.step_indices,
                )
            except Exception as e:
                logger.warning("Failed to decode latents: %s", e)

        return output.to_device("cpu")

    def generate_batch(self, actor: Any, requests: List[RolloutRequest]) -> List[RolloutOutput]:
        outputs = []
        for request in requests:
            output = self.generate(
                actor,
                prompts=request.prompts,
                prompt_embeds=request.prompt_embeds,
                pooled_prompt_embeds=request.pooled_prompt_embeds,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )
            outputs.append(output)
        return outputs

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
    ) -> RolloutOutput:
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

    def generate_batch(self, actor, requests: List[RolloutRequest]) -> List[RolloutOutput]:
        return self._executor.generate_batch(actor, requests)
