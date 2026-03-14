"""
NFT (Negative Fine-Tuning) Algorithm Implementation — unified algorithm + loss.

DiffusionNFT forward process diffusion RL.  This class is the single source
of truth for both rollout-side requirements (sampling, advantages) and
training-side gradient computation (the former ``NFTLoss``).

Key Features:
- Forward process optimization (no trajectory storage needed)
- Dual LoRA adapter support (old/new predictions)
- Advantage-weighted positive/negative predictions
- Adaptive weighting for stable training
- Sequence Parallel (SP) compatible
- Typed interface via compute_batch() for type-safe training

Based on: DiffusionNFT
"""

import logging
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

from diffusionrl.types import ForwardTrainingBatch
from diffusionrl.utils.adapter_utils import switch_adapter
from .base import BaseAlgorithm, SamplingRequirements

logger = logging.getLogger(__name__)


class NFTAlgorithm(BaseAlgorithm):
    """
    NFT (Negative Fine-Tuning) Algorithm — unified algorithm + loss.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation (compute_advantages, inherited)
    3. Loss / gradient computation (compute_batch)
    4. Static requirements declaration (declared_requirements)
    5. Dual adapter EMA update (post_optimizer_step_hook)

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

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        """Declare data requirements for the contracts / validation pipeline."""
        return {
            "requires_trajectory": False,
            "requires_log_prob": False,
            "requires_embeddings": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> "NFTAlgorithm":
        """Create NFTAlgorithm from a loss_config dictionary.

        Reads constructor parameters from config top-level keys,
        with ``loss_kwargs`` sub-dict taking precedence for overrides.
        """
        extra = config.get("loss_kwargs") or {}
        known_keys = {
            "beta",
            "adv_clip_max",
            "adv_mode",
            "use_adaptive_weight",
            "shift",
            "old_adapter_name",
            "new_adapter_name",
            "kl_coef",
        }
        runtime_only_keys = {
            "use_ema",
            "ema_decay",
            "eval_ema_decay",
            "eval_ema_update_interval",
            "nft_timestep_mode",
            "nft_shuffle_timesteps",
            "nft_apply_shift",
            "decay_type",
            "ema_flat_steps",
            "ema_uprate",
            "ema_uphold",
            "shuffle_samples",
            "shuffle_seed",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys and key not in runtime_only_keys)
        if unknown:
            warnings.warn(
                f"NFTAlgorithm.from_config received unknown loss_kwargs keys: {unknown}. "
                "These keys are ignored by NFT algorithm constructor.",
                stacklevel=3,
            )

        def _get(key, default):
            return extra.get(key, config.get(key, default))

        return cls(
            beta=float(_get("beta", 0.1)),
            adv_clip_max=float(_get("adv_clip_max", 5.0)),
            adv_mode=str(_get("adv_mode", "raw")),
            use_adaptive_weight=bool(_get("use_adaptive_weight", True)),
            shift=float(_get("shift", 3.0)),
            old_adapter_name=str(_get("old_adapter_name", "old")),
            new_adapter_name=str(_get("new_adapter_name", "default")),
            kl_coef=float(_get("kl_coef", 0.0)),
            ema_decay=float(_get("ema_decay", 0.001)),
        )

    def __init__(
        self,
        beta: float = 0.1,
        adv_clip_max: float = 5.0,
        adv_mode: str = "raw",
        use_adaptive_weight: bool = True,
        shift: float = 3.0,
        ema_decay: float = 0.001,
        kl_coef: float = 0.0,
        old_adapter_name: str = "old",
        new_adapter_name: str = "default",
        # BaseAlgorithm params
        advantage_type: str = "group",
        epsilon: float = 1e-4,
        clip_max: float = 5.0,
        use_per_prompt_tracker: bool = False,
        per_prompt_buffer_size: int = 16,
        per_prompt_min_count: int = 2,
        use_global_std: bool = False,
        **kwargs,
    ):
        # Remove clip_range from kwargs if present, NFT doesn't use it
        kwargs.pop('clip_range', None)
        super().__init__(
            clip_range=0.0,  # Not used by NFT
            kl_coef=kl_coef,
            advantage_type=advantage_type,
            epsilon=epsilon,
            clip_max=clip_max,
            use_per_prompt_tracker=use_per_prompt_tracker,
            per_prompt_buffer_size=per_prompt_buffer_size,
            per_prompt_min_count=per_prompt_min_count,
            use_global_std=use_global_std,
            **kwargs,
        )
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.adv_mode = adv_mode
        self.use_adaptive_weight = use_adaptive_weight
        self.shift = shift
        self.ema_decay = ema_decay
        self.old_adapter_name = old_adapter_name
        self.new_adapter_name = new_adapter_name
        self.model_type = "default"
        self._forward_plugin = None

    @classmethod
    def from_args(cls, args: Any) -> "NFTAlgorithm":
        """Construct NFT algorithm from runtime args."""
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "beta": 0.1,
                "adv_clip_max": 5.0,
                "adv_mode": "raw",
                "use_adaptive_weight": True,
                "shift": getattr(args.sampling, "shift", 3.0),
                "ema_decay": 0.001,
                "use_per_prompt_tracker": getattr(args.algorithm, "use_per_prompt_stat_tracker", False),
                "per_prompt_buffer_size": getattr(args.algorithm, "per_prompt_buffer_size", 16),
                "per_prompt_min_count": getattr(args.algorithm, "per_prompt_min_count", 2),
                "use_global_std": getattr(args.algorithm, "use_global_std", False),
            }
        )
        kwargs.update(cls._algorithm_kwargs_from_args(args))
        return cls(**kwargs)

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return NFT sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=False,
            requires_log_prob=False,
            extras={
                "sde_ratio": 0.0,
                "requires_clean_latents": True,
                "forward_diffusion_in_loss": True,
            },
        )

    # ==================================================================
    # Loss computation (formerly in NFTLoss)
    # ==================================================================

    @property
    def requires_trajectory(self) -> bool:
        """NFT doesn't need trajectories - only clean images."""
        return False

    @property
    def requires_log_prob(self) -> bool:
        """NFT doesn't use importance sampling."""
        return False

    @property
    def name(self) -> str:
        return "NFT"

    def process_advantages(
        self,
        advantages: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Process advantages according to the specified mode.

        Args:
            advantages: Raw advantages [B]
            epsilon: Small value for numerical stability

        Returns:
            Processed advantages [B]
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
            std = advantages.std() + epsilon
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
        apply_shift: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion to clean samples.

        Flow matching forward process: xt = (1-t)*x0 + t*noise
        """
        if noise is None:
            noise = randn_tensor(
                x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
            )

        if apply_shift:
            t_shifted = (self.shift * t) / (1 + (self.shift - 1) * t)
        else:
            t_shifted = t

        t_expanded = t_shifted.view(-1, *([1] * (x0.ndim - 1)))
        xt = (1 - t_expanded) * x0 + t_expanded * noise

        return xt, noise

    def get_old_prediction(
        self,
        model: nn.Module,
        model_kwargs: Dict[str, Any],
        old_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Get prediction from old model/adapter.

        Tries in order:
        1. Separate old_model if provided
        2. Full-param EMA mode (swap EMA weights, compute, restore)
        3. Dual adapter mode (switch to old adapter)
        4. Disable adapter mode (LoRA base model)
        5. Detached current prediction (fallback)
        """
        with torch.no_grad():
            if old_model is not None:
                return old_model(**model_kwargs)[0]

            old_params_ema = getattr(self, "_old_params_ema", None)
            if old_params_ema is not None:
                trainable = [p for p in model.parameters() if p.requires_grad]
                old_params_ema.copy_ema_to(trainable, store_temp=True, grad=True)
                try:
                    return model(**model_kwargs)[0]
                finally:
                    old_params_ema.copy_temp_to(trainable)

            adapter_model = model.module if hasattr(model, "module") else model
            if hasattr(adapter_model, "set_adapter"):
                try:
                    with switch_adapter(adapter_model, self.old_adapter_name):
                        return model(**model_kwargs)[0]
                except Exception:
                    pass

            if hasattr(adapter_model, "disable_adapter"):
                try:
                    with adapter_model.disable_adapter():
                        return model(**model_kwargs)[0]
                except Exception:
                    pass

            return model(**model_kwargs)[0]

    def get_ref_prediction(
        self,
        model: nn.Module,
        model_kwargs: Dict[str, Any],
        old_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Get reference prediction for KL regularization (base model)."""
        with torch.no_grad():
            adapter_model = model.module if hasattr(model, "module") else model
            if hasattr(adapter_model, "disable_adapter"):
                try:
                    with adapter_model.disable_adapter():
                        return model(**model_kwargs)[0]
                except Exception:
                    pass
            if old_model is not None:
                return old_model(**model_kwargs)[0]
            return model(**model_kwargs)[0]

    def _compute_core(
        self,
        model: nn.Module,
        samples: Dict[str, Any],
        timestep_idx: int,
        advantages: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        old_model: Optional[nn.Module] = None,
        generator: Optional[torch.Generator] = None,
        attn_metadata: Optional[Any] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Internal: compute NFT forward process loss.

        Called by ``compute_batch()`` which is the public interface.
        """
        device = advantages.device
        batch_size = advantages.shape[0]

        x0 = samples["clean_latents"]

        if prompt_embeds is None:
            prompt_embeds = samples.get("prompt_embeds")
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = samples.get("pooled_prompt_embeds")
        if text_ids is None:
            text_ids = samples.get("text_ids")
        if image_ids is None:
            image_ids = samples.get("image_ids")
        if encoder_attention_mask is None:
            encoder_attention_mask = samples.get("encoder_attention_mask")

        provided_t = kwargs.get("timestep_values")
        assert provided_t is not None, "timestep_values must be provided"
        if provided_t is None:
            t = torch.rand(batch_size, device=device)
            apply_shift = True
        else:
            t = provided_t.to(device)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            apply_shift = bool(kwargs.get("apply_shift", False))

        xt, noise = self.forward_diffusion(
            x0,
            t,
            generator=generator,
            apply_shift=apply_shift,
        )

        if apply_shift:
            t_shifted = (self.shift * t) / (1 + (self.shift - 1) * t)
        else:
            t_shifted = t
        t_expanded = t_shifted.view(-1, *([1] * (x0.ndim - 1)))

        if config is not None and hasattr(config, "get_torch_dtype"):
            model_dtype = config.get_torch_dtype()
        else:
            model_dtype = prompt_embeds.dtype if prompt_embeds is not None else xt.dtype
        autocast_enabled = model_dtype in (torch.float16, torch.bfloat16)

        xt_cast = xt.to(model_dtype)
        prompt_embeds_cast = prompt_embeds.to(model_dtype) if prompt_embeds is not None else None
        pooled_embeds_cast = pooled_prompt_embeds.to(model_dtype) if pooled_prompt_embeds is not None else None
        guidance_scale = getattr(config, "guidance_scale", 3.5) if config is not None else 3.5
        plugin = self._get_forward_plugin(model)
        model_kwargs = plugin.prepare_model_kwargs(
            latents=xt_cast,
            sigma=t_shifted,
            prompt_embeds=prompt_embeds_cast,
            pooled_prompt_embeds=pooled_embeds_cast,
            guidance_scale=guidance_scale,
            text_ids=text_ids,
            image_ids=image_ids,
            encoder_attention_mask=encoder_attention_mask,
        )

        adapter_model = model.module if hasattr(model, "module") else model
        assert hasattr(adapter_model, "set_adapter"), "adapter_model must have set_adapter method"
        adapter_model.set_adapter(self.new_adapter_name)

        autocast_ctx_fn = lambda: torch.autocast("cuda", model_dtype) if autocast_enabled else nullcontext[None]()
        grad_context = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()

        with grad_context, autocast_ctx_fn():
            forward_prediction = model(**model_kwargs)[0]

        adapter_model.set_adapter(self.old_adapter_name)
        with torch.no_grad(), autocast_ctx_fn():
            old_prediction = model(**model_kwargs)[0]

        adapter_model.set_adapter(self.new_adapter_name)

        adv_processed = self.process_advantages(advantages)
        adv_clipped = torch.clamp(adv_processed, -self.adv_clip_max, self.adv_clip_max)

        r = (adv_clipped / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0, 1)

        positive_prediction = (
            self.beta * forward_prediction + (1 - self.beta) * old_prediction.detach()
        )
        negative_prediction = (
            (1 + self.beta) * old_prediction.detach() - self.beta * forward_prediction
        )

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

            positive_loss = ((x0_positive - x0) ** 2 / weight_positive).mean(
                dim=tuple(range(1, x0.ndim))
            )
            negative_loss = ((x0_negative - x0) ** 2 / weight_negative).mean(
                dim=tuple(range(1, x0.ndim))
            )
        else:
            positive_loss = ((x0_positive - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)))
            negative_loss = ((x0_negative - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)))

        r_expanded = r.view(-1, *([1] * (positive_loss.ndim - 1)))
        policy_loss = (
            r_expanded * positive_loss / self.beta
            + (1 - r_expanded) * negative_loss / self.beta
        ).mean()

        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "positive_loss": positive_loss.mean().detach(),
            "negative_loss": negative_loss.mean().detach(),
            "x0_norm": torch.mean(x0**2).detach(),
            "prediction_deviation": torch.mean(
                (forward_prediction - old_prediction) ** 2
            ).detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantages.std().detach(),
            "r_mean": r.mean().detach(),
        }

        total_loss = policy_loss * self.adv_clip_max

        if self.kl_coef > 0:
            with torch.no_grad(), autocast_ctx_fn(), adapter_model.disable_adapter():
                ref_prediction = model(**model_kwargs)[0]
            kl_div = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_div = torch.mean(kl_div)
            total_loss = total_loss + self.kl_coef * kl_div
            loss_terms["kl_div"] = kl_div.detach()

        loss_terms["total_loss"] = total_loss.detach()

        return total_loss, loss_terms

    def _get_forward_plugin(self, model: nn.Module):
        """Get the forward plugin for this model."""
        if self._forward_plugin is None:
            if self.model_type != "default":
                raise RuntimeError(
                    "No forward_plugin set on NFTAlgorithm for model_type="
                    f"{self.model_type!r}. TrainingActor must inject plugin via "
                    "model_bundle.forward_plugin()."
                )
            from diffusionrl.models.forward_plugins import DefaultForwardPlugin
            self._forward_plugin = DefaultForwardPlugin()
            logger.warning(
                "No forward_plugin set on NFTAlgorithm (model_type=%s). "
                "Using DefaultForwardPlugin. Set algorithm._forward_plugin from "
                "model_bundle.forward_plugin() for model-specific behavior.",
                self.model_type,
            )
        return self._forward_plugin

    def compute_batch(
        self,
        model: nn.Module,
        batch: ForwardTrainingBatch,
        ref_model: Optional[nn.Module] = None,
        old_model: Optional[nn.Module] = None,
        generator: Optional[torch.Generator] = None,
        attn_metadata: Optional[Any] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute NFT loss using typed ForwardTrainingBatch.

        This is the typed interface for NFT loss computation.

        Args:
            model: The diffusion model being trained
            batch: Typed ForwardTrainingBatch containing clean_latents and embeddings
            ref_model: Reference model for KL penalty
            old_model: Old model for positive/negative predictions
            generator: Random number generator
            attn_metadata: Attention metadata for FastVideo
            config: Training configuration

        Returns:
            Tuple of (loss, metrics_dict)
        """
        samples = {
            "clean_latents": batch.clean_latents,
        }

        return self._compute_core(
            model=model,
            samples=samples,
            timestep_idx=0,
            advantages=batch.advantages,
            prompt_embeds=batch.embeddings.prompt_embeds,
            pooled_prompt_embeds=batch.embeddings.pooled_prompt_embeds,
            text_ids=batch.embeddings.text_ids,
            image_ids=batch.embeddings.image_ids,
            encoder_attention_mask=batch.embeddings.encoder_attention_mask,
            ref_model=ref_model,
            old_model=old_model,
            generator=generator,
            attn_metadata=attn_metadata,
            config=config,
            **kwargs,
        )

    # ==================================================================
    # Dual adapter EMA update (inlined, no longer delegates to NFTLoss)
    # ==================================================================

    def update_old_adapter(self, model: nn.Module) -> bool:
        """
        Update old adapter weights using EMA from new adapter.

        Should be called after each optimizer step.

        Args:
            model: Model with dual LoRA adapters

        Returns:
            True if update was successful
        """
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            return False

        try:
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                name: param.data.clone()
                for name, param in adapter_model.named_parameters()
                if self.new_adapter_name.lower() in name.lower()
            }

            adapter_model.set_adapter(self.old_adapter_name)
            for name, param in adapter_model.named_parameters():
                if self.old_adapter_name.lower() in name.lower() and name in new_params:
                    param.data = self.ema_decay * param.data + (1 - self.ema_decay) * new_params[name]

            adapter_model.set_adapter(self.new_adapter_name)
            return True

        except Exception:
            return False

    # ========== NFT-specific hooks ==========

    def requires_ema_update(self) -> bool:
        """NFT requires EMA updates for dual adapter mechanism."""
        return True

    def get_ema_decay(self) -> float:
        """Get EMA decay rate."""
        return self.ema_decay

    def post_optimizer_step_hook(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update old adapter via EMA after optimizer step."""
        success = self.update_old_adapter(model)
        return {"ema_updated": success}

    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration as dictionary."""
        config = super().get_config()
        config.update({
            "name": self.name,
            "beta": self.beta,
            "adv_clip_max": self.adv_clip_max,
            "adv_mode": self.adv_mode,
            "use_adaptive_weight": self.use_adaptive_weight,
            "shift": self.shift,
            "ema_decay": self.ema_decay,
            "old_adapter_name": self.old_adapter_name,
            "new_adapter_name": self.new_adapter_name,
            "requires_trajectory": self.requires_trajectory,
            "requires_log_prob": self.requires_log_prob,
        })
        return config
