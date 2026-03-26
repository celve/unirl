"""Training-actor sampling runtime helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, List, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.samplers.fsdp import sampler_runner
from diffusionrl.types.sde import SDEConfig
from diffusionrl.types.sampling import RolloutSamples, RolloutRequest

logger = logging.getLogger(__name__)


@contextmanager
def sampling_eval_context(modules: List[nn.Module]):
    """Temporarily switch modules to eval/no-grad and restore training flags."""
    original_modes = [
        (module, bool(module.training))
        for module in modules
        if isinstance(module, nn.Module)
    ]
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


class ActorSamplingExecutor:
    """Training-actor sampling runtime owned by the TrainingActor RPC boundary."""

    def iter_reflection_modules(
        self,
        obj: Any,
        *,
        include_transformer: bool,
    ) -> List[Tuple[str, nn.Module]]:
        """Collect likely offloadable submodules from arbitrary objects via reflection."""
        return sampler_runner.iter_offloadable_modules(
            obj, include_transformer=include_transformer
        )

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
            raise ValueError(
                "sampling_config must provide sampler_path for training-actor sampling"
            )

        sampler_kwargs = dict(actor._sampling_config.get("sampler_kwargs", {}))
        for _reserved in ("autocast_precision", "trajectory_precision", "logprob_precision"):
            sampler_kwargs.pop(_reserved, None)
        sde_config = SDEConfig.from_mapping(actor._sampling_config.get("sde_config"))
        actor._sampler = sampler_runner.create_sampler(
            sampler_path=sampler_path,
            model=actor.model,
            text_encoder=actor.text_encoder,
            vae=actor.vae,
            eta=sde_config.eta,
            sde_type=sde_config.sde_type,
            shift=sde_config.shift,
            model_bundle=actor.model_bundle,
            autocast_precision=actor._sampling_config.get("autocast_precision", "bf16"),
            trajectory_precision=actor._sampling_config.get("trajectory_precision", "fp16"),
            logprob_precision=actor._sampling_config.get("logprob_precision", "fp32"),
            **sampler_kwargs,
        )

        actor._sampling_ready = True

    def decode_latents(self, actor: Any, latents: torch.Tensor) -> torch.Tensor:
        self.ensure_sampling_components(actor)
        return sampler_runner.decode_latents(actor.vae, latents)

    def _iter_sampling_mode_modules(self, actor: Any) -> List[nn.Module]:
        modules: List[nn.Module] = []
        seen: Set[int] = set()
        for component in (actor.model, actor.text_encoder, actor.vae):
            if isinstance(component, nn.Module):
                ident = id(component)
                if ident not in seen:
                    modules.append(component)
                    seen.add(ident)

        if actor.model_bundle is not None and hasattr(
            actor.model_bundle, "iter_offloadable_modules"
        ):
            for _name, component in actor.model_bundle.iter_offloadable_modules(
                include_transformer=True
            ):
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

    def generate_raw(
        self,
        actor: Any,
        request: RolloutRequest,
    ) -> RolloutSamples:
        """Run the sampler only; output finalization stays in TrainingActor."""
        if not actor._is_initialized:
            raise RuntimeError("Actor not initialized. Call init() first.")

        if actor._is_offloaded:
            actor.onload()

        self.ensure_sampling_components(actor)

        with self._sampling_eval_context(actor):
            return sampler_runner.generate_prompt_only_rollout(
                host_label="Training-actor sampling",
                request=request,
                model=actor.model,
                sampler=actor._sampler,
                model_bundle=actor.model_bundle,
                device=actor._device,
            )


__all__ = [
    "sampling_eval_context",
    "ActorSamplingExecutor",
]
