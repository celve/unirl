"""In-process FSDP-native sampling engine.

Wraps ``FSDPBaseSampler`` with the ``BaseRolloutEngine`` interface so that
``TrainActor`` can use the same engine abstraction as ``RolloutActor``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples
from diffusionrl.samplers.engine import BaseRolloutEngine
from diffusionrl.samplers.fsdp.base_sampler import FSDPBaseSampler

logger = logging.getLogger(__name__)


class FSDPSamplingEngine(BaseRolloutEngine):
    """In-process FSDP-native sampling engine.

    Unlike ``SGLangRolloutEngine`` which manages an external inference
    service, this engine borrows the training model and delegates to an
    ``FSDPBaseSampler`` for the actual denoising math.  It is designed
    for the *direct-sampling* path where ``TrainActor`` generates
    rollouts on the same GPU that trains.

    Lifecycle::

        engine = FSDPSamplingEngine(sampling_params)
        engine.initialize(device)
        engine.bind_model(model=model, model_bundle=model_bundle)
        # ... later ...
        samples = engine.generate(request)
    """

    def __init__(self, sampling_params: Any) -> None:
        super().__init__(config=sampling_params)
        self._sampling_params = sampling_params
        self._sampler: Optional[FSDPBaseSampler] = None
        self._model: Optional[nn.Module] = None
        self._model_bundle: Optional[Any] = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, device: torch.device) -> None:
        """Store the target device.

        Real initialisation happens in :meth:`bind_model` once the
        training model and model-bundle are available.
        """
        self._device = device

    def bind_model(
        self,
        *,
        model: nn.Module,
        model_bundle: Any,
    ) -> None:
        """Bind to the training actor's model and create the sampler.

        Must be called once after :meth:`initialize` and before the
        first :meth:`generate` call.
        """
        self._model = model
        self._model_bundle = model_bundle

        sp = self._sampling_params

        try:
            model_bundle.load_aux_components()
        except Exception as exc:
            logger.warning("Failed to load auxiliary components: %s", exc)
            raise

        sampler_dotpath = model_bundle.default_sampler_dotpath()
        if not sampler_dotpath:
            raise ValueError(
                "model_bundle.default_sampler_dotpath() must return a "
                "non-empty dotpath for direct sampling"
            )

        text_encoder = getattr(model_bundle, "text_encoder", None)
        vae = getattr(model_bundle, "vae", None)

        sde_config = sp.sde_config
        self._sampler = FSDPBaseSampler.from_config(
            sampler_dotpath=sampler_dotpath,
            model=model,
            text_encoder=text_encoder,
            vae=vae,
            eta=sde_config.eta,
            sde_type=sde_config.sde_type,
            shift=sde_config.shift,
            model_bundle=model_bundle,
            autocast_precision=sp.autocast_precision,
            trajectory_precision=sp.trajectory_precision,
            logprob_precision=sp.logprob_precision,
            **dict(sp.sampler_kwargs),
        )
        self._is_initialized = True
        logger.info("FSDPSamplingEngine bound (sampler=%s)", sampler_dotpath)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: RolloutRequest) -> RolloutSamples:
        """Generate samples by delegating to the FSDP sampler.

        Wraps the call in an eval context (model.eval + no_grad) and
        restores training mode afterwards.
        """
        self._require_ready()
        with self._eval_context():
            return self._sampler.generate(
                request,
                host_label="FSDPSamplingEngine",
            )

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        self._require_ready()
        return self._sampler.encode_prompt(prompts, **kwargs)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_ready()
        return self._sampler.decode_latents(latents)

    # ------------------------------------------------------------------
    # Weight sync (no-op for in-process FSDP)
    # ------------------------------------------------------------------

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """No-op: the sampler shares the training model reference."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        if self._sampler is None:
            raise RuntimeError(
                "FSDPSamplingEngine not ready. "
                "Call initialize() then bind_model() first."
            )

    @contextmanager
    def _eval_context(self):
        """Temporarily switch sampling-relevant modules to eval / no-grad."""
        modules: List[nn.Module] = []
        seen: Set[int] = set()

        if self._model_bundle is not None and hasattr(
            self._model_bundle, "iter_offloadable_modules"
        ):
            for _name, m in self._model_bundle.iter_offloadable_modules(
                include_transformer=True,
            ):
                modules.append(m)
                seen.add(id(m))

        if isinstance(self._model, nn.Module) and id(self._model) not in seen:
            modules.append(self._model)

        was_training = [m.training for m in modules]
        for m in modules:
            m.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            for m, was in zip(modules, was_training):
                m.train(was)


__all__ = ["FSDPSamplingEngine"]
