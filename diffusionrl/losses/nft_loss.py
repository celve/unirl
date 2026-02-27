"""
NFT Loss for forward process diffusion RL.

This loss implements DiffusionNFT's forward process optimization,
which directly optimizes on the forward diffusion process instead of
the reverse sampling trajectory.

Key Features:
- Forward process optimization (no trajectory storage needed)
- Dual LoRA adapter support (old/new predictions)
- Advantage-weighted positive/negative predictions
- Adaptive weighting for stable training
- Sequence Parallel (SP) compatible
- Typed interface via compute_batch() for type-safe training

Based on: DiffusionNFT
"""

from typing import Dict, Any, Tuple, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

from diffusionrl.types import ForwardTrainingBatch, PromptEmbeddings
from diffusionrl.utils.adapter_utils import switch_adapter


class NFTLoss:
    """
    NFT (DiffusionNFT) forward process loss.

    This loss optimizes the diffusion model using the forward process:
    1. Take a clean image x0
    2. Add noise to get xt = (1-t)*x0 + t*noise
    3. Compute velocity prediction v(xt, t)
    4. Create positive/negative predictions based on advantage
    5. Optimize using weighted flow matching loss

    Key insight: Only needs clean images, not sampling trajectories.

    Dual Adapter Mechanism:
    - new_adapter (default): Current policy being trained
    - old_adapter: Reference policy (updated via EMA after each step)

    Loss Formula:
    - positive = beta * new + (1-beta) * old  (move toward new for high advantage)
    - negative = (1+beta) * old - beta * new  (move away from new for low advantage)
    - loss = r * L_positive + (1-r) * L_negative
    - where r = normalized_advantage in [0, 1]

    Args:
        beta: Interpolation weight for positive/negative predictions (default 0.1)
        adv_clip_max: Maximum advantage clipping value (default 5.0)
        adv_mode: Advantage processing mode ("raw", "sign", "binary", "one_only")
        use_adaptive_weight: Whether to use adaptive weighting (default True)
        shift: Time shift parameter for sigma schedule (default 3.0)
        old_adapter_name: Name of old adapter (default "old")
        new_adapter_name: Name of new adapter (default "default")
        kl_coef: KL regularization coefficient (default 0.0)

    Example:
        loss_fn = NFTLoss(beta=0.1, adv_clip_max=5.0)
        loss, metrics = loss_fn.compute(
            model=model,
            samples={"clean_latents": x0, "prompt_embeds": embeds},
            timestep_idx=0,
            advantages=advantages,
        )
    """

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": False,
            "requires_log_prob": False,
            "requires_embeddings": True,
        }

    def __init__(
        self,
        beta: float = 0.1,
        adv_clip_max: float = 5.0,
        adv_mode: str = "raw",
        use_adaptive_weight: bool = True,
        shift: float = 3.0,
        old_adapter_name: str = "old",
        new_adapter_name: str = "default",
        kl_coef: float = 0.0,
    ):
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.adv_mode = adv_mode
        self.use_adaptive_weight = use_adaptive_weight
        self.shift = shift
        self.old_adapter_name = old_adapter_name
        self.new_adapter_name = new_adapter_name
        self.kl_coef = kl_coef
        self.model_type = "default"
        self._forward_plugin = None

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

        Modes:
            - "raw": Use advantages as-is
            - "sign": Use sign of advantages
            - "binary": Binary (positive=1, negative=0)
            - "one_only": Only positive samples contribute
            - "all": Normalize advantages globally (DiffusionNFT style)
            - "per_timestep": Normalize per timestep (requires timestep info)
        """
        if self.adv_mode == "raw":
            return advantages
        elif self.adv_mode == "sign":
            return torch.sign(advantages)
        elif self.adv_mode == "binary":
            return (advantages > 0).float()
        elif self.adv_mode == "one_only":
            # Only positive samples contribute
            return torch.where(
                advantages > 0,
                torch.ones_like(advantages),
                torch.zeros_like(advantages),
            )
        elif self.adv_mode == "all":
            # Global normalization across all samples (DiffusionNFT style)
            # This normalizes advantages to have zero mean and unit variance
            mean = advantages.mean()
            std = advantages.std() + epsilon
            return (advantages - mean) / std
        elif self.adv_mode == "per_timestep":
            # For per_timestep mode, normalization should happen at the algorithm level
            # where timestep information is available. Here we just pass through.
            # The actual per-timestep normalization would require timestep indices
            # which are not available in the loss function.
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

        Args:
            x0: Clean samples [B, C, H, W] or [B, C, T, H, W]
            t: Timesteps [B] in [0, 1]
            noise: Optional pre-generated noise
            generator: Random number generator
            apply_shift: Whether to apply time shift

        Returns:
            Tuple of (noisy_samples xt, noise)
        """
        if noise is None:
            noise = randn_tensor(
                x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
            )

        # Apply time shift if requested
        if apply_shift:
            t_shifted = (self.shift * t) / (1 + (self.shift - 1) * t)
        else:
            t_shifted = t

        # Forward diffusion: xt = (1-t)*x0 + t*noise (flow matching formulation)
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
        2. Dual adapter mode (switch to old adapter)
        3. Disable adapter mode (LoRA base model)
        4. Detached current prediction (fallback)

        Args:
            model: Current model
            model_kwargs: Model forward kwargs
            old_model: Optional separate old model

        Returns:
            Old model prediction (detached)
        """
        with torch.no_grad():
            # Option 1: Separate old model
            if old_model is not None:
                return old_model(**model_kwargs)[0]

            # Option 2: Dual adapter mode
            adapter_model = model.module if hasattr(model, "module") else model
            if hasattr(adapter_model, "set_adapter"):
                try:
                    with switch_adapter(adapter_model, self.old_adapter_name):
                        return model(**model_kwargs)[0]
                except Exception:
                    pass

            # Option 3: Disable adapter mode
            if hasattr(adapter_model, "disable_adapter"):
                try:
                    with adapter_model.disable_adapter():
                        return model(**model_kwargs)[0]
                except Exception:
                    pass

            # Option 4: Fallback - use current model prediction (detached)
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

    def compute(
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
        Compute NFT forward process loss.

        Args:
            model: The diffusion model being trained (current policy)
            samples: Dictionary containing:
                - 'clean_latents': [B, C, H, W] clean image latents (x0)
            timestep_idx: Not used (we sample random timesteps)
            advantages: Per-sample advantages [B]
            prompt_embeds: Text encoder hidden states
            pooled_prompt_embeds: Pooled text embeddings
            text_ids: Text position IDs (for Flux)
            image_ids: Image position IDs (for Flux)
            ref_model: Reference model for KL penalty
            old_model: Old model for positive/negative predictions
            generator: Random number generator
            attn_metadata: Attention metadata for FastVideo
            config: Training configuration

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = advantages.device
        batch_size = advantages.shape[0]

        # Get clean latents
        x0 = samples["clean_latents"]  # [B, C, H, W] or [B, C, T, H, W]

        # Get embeddings from samples dict if not provided as args
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

        # Sample random timesteps unless explicitly provided
        provided_t = kwargs.get("timestep_values")
        if provided_t is None:
            t = torch.rand(batch_size, device=device)
            apply_shift = True
        else:
            t = provided_t.to(device)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            apply_shift = bool(kwargs.get("apply_shift", False))

        # Forward diffusion: xt = (1-t)*x0 + t*noise
        xt, noise = self.forward_diffusion(
            x0,
            t,
            generator=generator,
            apply_shift=apply_shift,
        )

        # Apply time shift for model input
        if apply_shift:
            t_shifted = (self.shift * t) / (1 + (self.shift - 1) * t)
        else:
            t_shifted = t
        t_expanded = t_shifted.view(-1, *([1] * (x0.ndim - 1)))

        # Prepare model inputs
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

        # Get predictions from current model
        try:
            from fastvideo.forward_context import set_forward_context

            if attn_metadata is not None:
                with set_forward_context(current_timestep=timestep_idx, attn_metadata=attn_metadata):
                    with torch.autocast("cuda", model_dtype) if autocast_enabled else nullcontext():
                        forward_prediction = model(**model_kwargs)[0]
            else:
                with torch.autocast("cuda", model_dtype) if autocast_enabled else nullcontext():
                    forward_prediction = model(**model_kwargs)[0]
        except ImportError:
            with torch.autocast("cuda", model_dtype) if autocast_enabled else nullcontext():
                forward_prediction = model(**model_kwargs)[0]

        # Get predictions from old model/adapter
        old_prediction = self.get_old_prediction(model, model_kwargs, old_model)

        # Process advantages
        adv_processed = self.process_advantages(advantages)
        adv_clipped = torch.clamp(adv_processed, -self.adv_clip_max, self.adv_clip_max)

        # Normalize to [0, 1] range for weighting
        # High advantage (r -> 1): use positive prediction
        # Low advantage (r -> 0): use negative prediction
        r = (adv_clipped / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0, 1)

        # Create positive and negative predictions
        # positive = beta * new + (1-beta) * old  (move toward new)
        # negative = (1+beta) * old - beta * new  (move away from new)
        positive_prediction = (
            self.beta * forward_prediction + (1 - self.beta) * old_prediction.detach()
        )
        negative_prediction = (
            (1 + self.beta) * old_prediction.detach() - self.beta * forward_prediction
        )

        # Compute x0 predictions from velocity
        # For flow matching: v = noise - x_0, so x_0 = xt - t * v
        x0_positive = xt - t_expanded * positive_prediction
        x0_negative = xt - t_expanded * negative_prediction

        # Compute losses
        if self.use_adaptive_weight:
            # Adaptive weighting based on prediction error
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

        # Weighted combination based on advantage
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

        # KL regularization (optional) - DiffusionNFT style
        if self.kl_coef > 0:
            ref_prediction = self.get_ref_prediction(model, model_kwargs, old_model=old_model)
            kl_div = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_div = torch.mean(kl_div)
            total_loss = total_loss + self.kl_coef * kl_div
            loss_terms["kl_div"] = kl_div.detach()

        loss_terms["total_loss"] = total_loss.detach()

        return total_loss, loss_terms

    def _get_forward_plugin(self, model: nn.Module):
        """Get or create forward plugin for current NFT model type."""
        if self._forward_plugin is None:
            from diffusionrl.models.forward_plugins import detect_model_type, get_forward_plugin

            if self.model_type == "default":
                detected_type = detect_model_type(model)
                self._forward_plugin = get_forward_plugin(detected_type)
            else:
                self._forward_plugin = get_forward_plugin(self.model_type)
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

        This is the typed interface for NFT loss computation. It uses
        ForwardTrainingBatch instead of dictionaries.

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
        # Convert typed batch to samples dict for internal compute
        samples = {
            "clean_latents": batch.clean_latents,
        }

        return self.compute(
            model=model,
            samples=samples,
            timestep_idx=0,  # Not used for NFT
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

    def compute_aggregated(
        self,
        model: nn.Module,
        samples: Dict[str, Any],
        advantages: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute NFT loss.

        For NFT, we only compute once (forward process samples random timestep).
        """
        return self.compute(
            model=model,
            samples=samples,
            timestep_idx=0,  # Not used for NFT
            advantages=advantages,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **kwargs,
        )

    def update_old_adapter(
        self,
        model: nn.Module,
        ema_decay: float = 0.001,
    ) -> bool:
        """
        Update old adapter weights using EMA from new adapter.

        This implements the dual adapter EMA update from DiffusionNFT.

        Args:
            model: Model with dual LoRA adapters
            ema_decay: EMA decay rate (smaller = slower update)

        Returns:
            True if update was successful, False otherwise
        """
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            return False

        try:
            # Get new adapter parameters
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                name: param.data.clone()
                for name, param in adapter_model.named_parameters()
                if "lora" in name.lower()
            }

            # Update old adapter with EMA (match DiffusionNFT: old = old*decay + new*(1-decay))
            adapter_model.set_adapter(self.old_adapter_name)
            for name, param in adapter_model.named_parameters():
                if "lora" in name.lower() and name in new_params:
                    param.data = ema_decay * param.data + (1 - ema_decay) * new_params[name]

            # Switch back to new adapter
            adapter_model.set_adapter(self.new_adapter_name)
            return True

        except Exception:
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for logging/checkpointing."""
        return {
            "name": self.name,
            "beta": self.beta,
            "adv_clip_max": self.adv_clip_max,
            "adv_mode": self.adv_mode,
            "use_adaptive_weight": self.use_adaptive_weight,
            "shift": self.shift,
            "old_adapter_name": self.old_adapter_name,
            "new_adapter_name": self.new_adapter_name,
            "kl_coef": self.kl_coef,
            "requires_trajectory": self.requires_trajectory,
            "requires_log_prob": self.requires_log_prob,
        }
