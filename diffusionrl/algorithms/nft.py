"""
NFT (Negative Fine-Tuning) Algorithm Implementation — algorithm-owned loss.

DiffusionNFT forward process diffusion RL. The algorithm file owns rollout
requirements, advantage processing, and the forward-process loss entrypoint.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

from diffusionrl.algorithms.base import (
    BaseAlgorithm,
    BaseAlgorithmConfig,
    EMASpec,
    SamplingRequirements,
)
from diffusionrl.algorithms.registry import register_algorithm
from diffusionrl.types.forward_context import ForwardContext
from diffusionrl.types.training_batch import TrainingBatch as _TrainingBatch
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NFTAlgorithmConfig(BaseAlgorithmConfig):
    beta: float = 0.1
    adv_clip_max: float = 5.0
    adv_mode: str = "raw"
    use_adaptive_weight: bool = True
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    use_reference_ema: bool = True
    ema_decay: float = 0.001
    ema_decay_type: str = "constant"
    ema_flat_steps: int = 0
    ema_uprate: float = 0.001
    ema_uphold: float = 0.5
    reference_update_timing: str = "optimizer_step"
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"
    train_timestep_mode: str = "random"
    shuffle_train_timesteps: bool = True
    apply_time_shift_in_loss: bool = False
    training_scheduler_config: Dict[str, Any] = field(default_factory=dict)


@register_algorithm(component_name="nft", component_cfg=NFTAlgorithmConfig)
class NFTAlgorithm(BaseAlgorithm):
    """
    NFT (Negative Fine-Tuning) Algorithm — algorithm owns loss and advantages.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation and post-processing
    3. Loss computation (compute_loss / _compute_loss_core)
    4. Gradient computation (compute_loss_and_backward)
    5. NFT-specific old/reference prediction handling

    Forward process diffusion RL that optimizes directly on the forward
    diffusion process instead of reverse sampling trajectories.

    Key differences from GRPO:
    - No trajectory storage needed (only clean latents)
    - No log probabilities needed (no importance sampling)
    - Uses dual adapter mechanism (new/old) with EMA update
    - Forward diffusion happens in loss computation

    Dual Adapter Mechanism:
    - new_adapter (default): Current policy being trained
    - old_adapter: Reference policy (updated via EMA after each step)

    Loss Formula:
    - positive = beta * new + (1-beta) * old  (move toward new for high advantage)
    - negative = (1+beta) * old - beta * new  (move away from new for low advantage)
    - loss = r * L_positive + (1-r) * L_negative
    - where r = normalized_advantage in [0, 1]
    """

    def __init__(
        self,
        *,
        config: NFTAlgorithmConfig,
        **kwargs,
    ):
        if not isinstance(config, NFTAlgorithmConfig):
            raise TypeError(f"{type(self).__name__} expects NFTAlgorithmConfig, got {type(config).__name__}.")
        super().__init__(
            kl_coef=config.kl_coef,
            component_mix_stage=config.component_mix_stage,
            adv_normalization_scope=config.adv_normalization_scope,
            samples_per_prompt=config.samples_per_prompt,
            num_inference_steps=config.num_inference_steps,
            eval_ema_decay=config.eval_ema_decay,
            eval_ema_update_interval=config.eval_ema_update_interval,
            epsilon=config.epsilon,
            clip_max=config.clip_max,
            use_global_std=config.use_global_std,
            trim_outliers_ratio=config.trim_outliers_ratio,
            **kwargs,
        )
        self.config = config
        self.beta = config.beta
        self.adv_clip_max = config.adv_clip_max
        self.adv_mode = config.adv_mode
        self.use_adaptive_weight = config.use_adaptive_weight
        self._eta = config.eta
        self._sde_type = config.sde_type
        self._shift = config.shift
        self.use_reference_ema = bool(config.use_reference_ema)
        self.ema_decay = config.ema_decay
        self.ema_decay_type = str(config.ema_decay_type)
        self.ema_flat_steps = int(config.ema_flat_steps)
        self.ema_uprate = float(config.ema_uprate)
        self.ema_uphold = float(config.ema_uphold)
        self.reference_update_timing = str(config.reference_update_timing).strip().lower()
        self.old_adapter_name = config.old_adapter_name
        self.new_adapter_name = config.new_adapter_name
        self.train_timestep_mode = str(config.train_timestep_mode)
        self.shuffle_train_timesteps = bool(config.shuffle_train_timesteps)
        self.apply_time_shift_in_loss = bool(config.apply_time_shift_in_loss)
        self.training_scheduler_config = dict(config.training_scheduler_config)
        self.training_timestep_fraction = self.training_scheduler_config.get(
            "timestep_fraction",
            1.0,
        )

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def sde_type(self) -> str:
        return self._sde_type

    @property
    def time_shift(self) -> float:
        return self._shift

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return NFT sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=False,
            requires_log_prob=False,
            requires_embeddings=True,
            requires_clean_latents=True,
        )

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(
            enable_eval_ema=True,
            eval_ema_decay=self.eval_ema_decay,
            eval_ema_update_interval=self.eval_ema_update_interval,
            reference_mode=("nft_old_policy" if self.use_reference_ema else "none"),
            reference_decay=self.ema_decay,
            reference_decay_type=self.ema_decay_type,
            reference_flat_steps=self.ema_flat_steps,
            reference_uprate=self.ema_uprate,
            reference_uphold=self.ema_uphold,
            reference_update_timing=self.reference_update_timing,
            old_adapter_name=self.old_adapter_name,
            new_adapter_name=self.new_adapter_name,
        )

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Optional[Set[int]]:
        del current_step
        return None

    def get_sampler_validation_config(self, *, allow_replay: bool) -> Dict[str, Any]:
        del allow_replay
        return {
            "allow_replay": False,
            "assert_step_alignment": False,
            "mode_label": "forward",
        }

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        del num_steps
        return set(sde_indices)

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(batch, _TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")
        del current_step

        timestep_mode = kwargs.get("timestep_mode", self.train_timestep_mode)
        shuffle_timesteps = kwargs.get("shuffle_timesteps", self.shuffle_train_timesteps)
        timestep_fraction = kwargs.get(
            "timestep_fraction",
            self.training_timestep_fraction,
        )

        if timestep_mode == "all" and batch.timesteps is not None:
            timesteps = batch.timesteps.detach().flatten()
        else:
            timesteps = torch.rand(batch.batch_size, device=batch.advantages.device)

        if (
            timesteps.numel() > 1
            and torch.isclose(
                timesteps[-1],
                torch.zeros((), device=timesteps.device, dtype=timesteps.dtype),
                atol=1e-8,
            ).item()
        ):
            timesteps = timesteps[:-1]

        if timesteps.numel() > 0 and timestep_fraction is not None and timestep_fraction != 1.0:
            from diffusionrl.utils.scheduler_utils import normalize_timestep_fraction

            frac_start, frac_end = normalize_timestep_fraction(timestep_fraction)
            n = timesteps.numel()
            effective_start = int(n * frac_start)
            effective_end = min(int(n * frac_end), n)
            if effective_start < effective_end:
                timesteps = timesteps[effective_start:effective_end]
            else:
                timesteps = timesteps[:0]

        if timesteps.numel() == 0:
            timesteps = torch.rand(batch.batch_size, device=batch.advantages.device)

        if shuffle_timesteps:
            perm = torch.randperm(timesteps.numel(), device=timesteps.device)
            timesteps = timesteps[perm]

        return timesteps

    # ------------------------------------------------------------------
    # Rollout geometry / request planning
    # ------------------------------------------------------------------

    # ==================================================================
    # Objective computation
    # ==================================================================

    @property
    def name(self) -> str:
        return "NFT"

    def prepare_loss_advantages(
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform rollout advantages into NFT loss-side weighting signal.

        Args:
            advantages: Rollout advantages [B]

        Returns:
            Loss-side transformed advantages [B]
        """
        if self.adv_mode == "raw":
            return advantages
        elif self.adv_mode == "sign":
            return torch.sign(advantages)
        elif self.adv_mode == "binary":
            return (advantages > 0).float()
        elif self.adv_mode == "one_only":
            return torch.where(
                advantages > 0,
                torch.ones_like(advantages),
                torch.zeros_like(advantages),
            )
        elif self.adv_mode == "all":
            mean = advantages.mean()
            std = advantages.std() + self.epsilon
            return (advantages - mean) / std
        elif self.adv_mode == "per_timestep":
            return advantages
        else:
            return advantages

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion to clean samples.

        Flow matching forward process: xt = (1-t)*x0 + t*noise
        """
        if noise is None:
            noise = randn_tensor(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)

        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = (1 - t_expanded) * x0 + t_expanded * noise

        return xt, noise

    @torch.no_grad()
    def get_old_prediction(
        self,
        model: nn.Module,
        ctx: ForwardContext,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        old_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Get prediction from the NFT old policy.

        Accepted paths:
        1. Explicit old_model
        2. Full-parameter EMA swap
        3. LoRA old-adapter switch

        Reproduce mode must fail fast if none of these are available or if
        adapter switching fails. Falling back to the base model or current
        model would silently change the NFT objective semantics.
        """
        plugin = self._get_forward_plugin(model)
        ctx_kwargs = ctx.to_dict()
        if old_model is not None:
            return plugin.forward(model=old_model, latents=latents, sigma=sigma, **ctx_kwargs)

        adapter_model = model.module if hasattr(model, "module") else model
        if hasattr(adapter_model, "set_adapter"):
            try:
                with switch_adapter(adapter_model, self.old_adapter_name):
                    return plugin.forward(model=model, latents=latents, sigma=sigma, **ctx_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "NFT old-policy prediction failed while switching adapters. "
                    f"Expected adapter={self.old_adapter_name!r}. "
                    "Refusing to fall back to base/current model because that would "
                    "change the training objective."
                ) from exc

        raise RuntimeError(
            "NFT old-policy prediction requires one of: explicit old_model, "
            "full-parameter EMA, or adapter switching support via set_adapter(). "
            f"Model type={type(adapter_model).__name__} does not expose a valid old-policy path."
        )

    @torch.no_grad()
    def get_ref_prediction(
        self,
        model: nn.Module,
        ctx: ForwardContext,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        ref_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Get reference prediction for KL regularization (base model)."""
        plugin = self._get_forward_plugin(model)
        ctx_kwargs = ctx.to_dict()
        adapter_model = model.module if hasattr(model, "module") else model
        if hasattr(adapter_model, "disable_adapter"):
            try:
                with adapter_model.disable_adapter():
                    return plugin.forward(model=model, latents=latents, sigma=sigma, **ctx_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "NFT reference prediction failed while disabling adapters. "
                    "Refusing to fall back to current model because that would collapse "
                    "the KL term toward zero."
                ) from exc
        if ref_model is not None:
            return plugin.forward(model=ref_model, latents=latents, sigma=sigma, **ctx_kwargs)
        raise RuntimeError(
            "NFT reference prediction requires either disable_adapter() support "
            "or an explicit ref_model. No valid base-model reference path was available."
        )

    # ------------------------------------------------------------------
    # Algorithm-owned training step (Phase 2)
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        timesteps: Any = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """NFT loss + backward for a single micro-batch.

        Returns:
            ``(loss, metrics, num_timesteps, has_backward)``
        """
        from diffusionrl.types.training_batch import TrainingBatch

        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")

        if timesteps is None:
            timesteps = self.resolve_training_timesteps(
                batch=batch,
                current_step=0,
                **kwargs,
            )

        ctx = batch.forward_context
        total_loss = 0.0
        has_backward = False
        timestep_metrics: List[Dict[str, Any]] = []

        num_timesteps = timesteps.numel()
        for t in timesteps:
            loss, metrics = self.compute_loss(
                model=model,
                batch=batch,
                ctx=ctx,
                timestep_values=t,
            )
            scaled_loss = loss * loss_scale / num_timesteps
            scaled_loss.backward()
            has_backward = True
            total_loss += scaled_loss.detach().item()
            timestep_metrics.append(metrics)

        all_metrics = aggregate_numeric_metrics(timestep_metrics)

        return total_loss, all_metrics, num_timesteps, has_backward

    # ------------------------------------------------------------------
    # Forward plugin
    # ------------------------------------------------------------------

    def _compute_loss_core(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        advantages: torch.Tensor,
        ctx: ForwardContext,
        *,
        ref_model: Optional[nn.Module] = None,
        old_model: Optional[nn.Module] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = advantages.device
        batch_size = advantages.shape[0]

        t = kwargs.pop("timestep_values")
        assert t is not None, "timestep_values must be provided"
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.repeat(batch_size)

        xt, _ = self.forward_diffusion(
            x0,
            t,
            generator=generator,
        )

        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))

        adapter_model = model.module if hasattr(model, "module") else model
        assert hasattr(adapter_model, "set_adapter"), "adapter_model must have set_adapter method"
        adapter_model.set_adapter(self.new_adapter_name)

        plugin = self._get_forward_plugin(model)
        ctx_kwargs = ctx.to_dict()
        grad_context = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()

        with grad_context:
            forward_prediction = plugin.forward(model=model, latents=xt, sigma=t, **ctx_kwargs)

        old_prediction = self.get_old_prediction(
            model,
            ctx,
            latents=xt,
            sigma=t,
            old_model=old_model,
        )

        if hasattr(adapter_model, "set_adapter"):
            adapter_model.set_adapter(self.new_adapter_name)

        adv_processed = self.prepare_loss_advantages(advantages)
        adv_clipped = torch.clamp(adv_processed, -self.adv_clip_max, self.adv_clip_max)
        r = (adv_clipped / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0, 1)

        positive_prediction = self.beta * forward_prediction + (1 - self.beta) * old_prediction.detach()
        negative_prediction = (1 + self.beta) * old_prediction.detach() - self.beta * forward_prediction

        x0_positive = xt - t_expanded * positive_prediction
        x0_negative = xt - t_expanded * negative_prediction

        if self.use_adaptive_weight:
            with torch.no_grad():
                weight_positive = (
                    torch.abs(x0_positive.double() - x0.double())
                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                    .clip(min=1e-5)
                )
                weight_negative = (
                    torch.abs(x0_negative.double() - x0.double())
                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                    .clip(min=1e-5)
                )
            positive_loss = ((x0_positive - x0) ** 2 / weight_positive).mean(dim=tuple(range(1, x0.ndim)))
            negative_loss = ((x0_negative - x0) ** 2 / weight_negative).mean(dim=tuple(range(1, x0.ndim)))
        else:
            positive_loss = ((x0_positive - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)))
            negative_loss = ((x0_negative - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)))

        r_expanded = r.view(-1, *([1] * (positive_loss.ndim - 1)))
        policy_loss = (r_expanded * positive_loss / self.beta + (1 - r_expanded) * negative_loss / self.beta).mean()

        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "positive_loss": positive_loss.mean().detach(),
            "negative_loss": negative_loss.mean().detach(),
            "x0_norm": torch.mean(x0**2).detach(),
            "prediction_deviation": torch.mean((forward_prediction - old_prediction) ** 2).detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantages.std().detach(),
            "r_mean": r.mean().detach(),
        }

        total_loss = policy_loss * self.adv_clip_max

        if self.kl_coef > 0:
            ref_prediction = self.get_ref_prediction(
                model,
                ctx,
                latents=xt,
                sigma=t,
                ref_model=ref_model,
            )
            kl_div = ((forward_prediction - ref_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
            kl_div = torch.mean(kl_div)
            total_loss = total_loss + self.kl_coef * kl_div
            loss_terms["kl_div"] = kl_div.detach()

        loss_terms["total_loss"] = total_loss.detach()
        return total_loss, loss_terms

    def compute_loss(
        self,
        model: nn.Module,
        batch: _TrainingBatch,
        *,
        ctx: ForwardContext,
        ref_model: Optional[nn.Module] = None,
        old_model: Optional[nn.Module] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single NFT loss entrypoint for debugging and training."""
        return self._compute_loss_core(
            model=model,
            x0=batch.clean_latents.float(),
            advantages=batch.advantages,
            ctx=ctx,
            ref_model=ref_model,
            old_model=old_model,
            generator=generator,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Configuration export
    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration as dictionary."""
        config = super().get_config()
        config.update(
            {
                "name": self.name,
                "beta": self.beta,
                "adv_clip_max": self.adv_clip_max,
                "adv_mode": self.adv_mode,
                "use_adaptive_weight": self.use_adaptive_weight,
                "sde_config": {"eta": self._eta, "sde_type": self._sde_type, "shift": self._shift},
                "time_shift": self.time_shift,
                "ema_decay": self.ema_decay,
                "ema_decay_type": self.ema_decay_type,
                "ema_flat_steps": self.ema_flat_steps,
                "ema_uprate": self.ema_uprate,
                "ema_uphold": self.ema_uphold,
                "reference_update_timing": self.reference_update_timing,
                "use_reference_ema": self.use_reference_ema,
                "old_adapter_name": self.old_adapter_name,
                "new_adapter_name": self.new_adapter_name,
                "train_timestep_mode": self.train_timestep_mode,
                "shuffle_train_timesteps": self.shuffle_train_timesteps,
                "apply_time_shift_in_loss": self.apply_time_shift_in_loss,
            }
        )
        return config
