"""
GRPO Loss for diffusion model policy optimization.

This loss implements Group Relative Policy Optimization (GRPO) using
importance sampling ratios computed from log probabilities.

Key Features:
- PPO-style clipped policy gradient
- KL regularization (flow_grpo style with LoRA disable_adapter)
- Support for both SDE and Mixed sampling modes
- Per-timestep or aggregated loss
- Typed interface via compute_timestep() for type-safe training

Formula:
    ratio = exp(new_log_prob - old_log_prob)
    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]

Based on: unified_grpo, flow_grpo, DanceGRPO, MixGRPO
"""

import math
import logging
from typing import Dict, Any, Tuple, Optional, List, Set, Union
import torch
import torch.nn as nn

from diffusionrl.types import TimestepData, PromptEmbeddings

logger = logging.getLogger(__name__)


class GRPOLoss:
    """
    GRPO (Group Relative Policy Optimization) loss.

    This loss uses importance sampling to optimize diffusion model policies:
        L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]

    where:
        ratio = exp(log_prob_new - log_prob_old)
        A = advantage (normalized reward)

    Supports:
    - Full SDE sampling (DanceGRPO/flow_grpo style)
    - Mixed ODE-SDE sampling (MixGRPO style) - only computes loss on SDE steps
    - KL penalty with LoRA disable_adapter or separate ref model
    - Multiple SDE formulations (sde, cps, dance)

    Example:
        loss_fn = GRPOLoss(
            clip_range=1e-4,
            use_kl_penalty=True,
            kl_coef=0.01,
            eta=0.7,
            sde_type="sde",
        )

        loss, metrics = loss_fn.compute(
            model=model,
            samples=sampling_output,
            timestep_idx=t,
            advantages=advantages,
            prompt_embeds=prompt_embeds,
        )
    """

    def __init__(
        self,
        clip_range: float = 1e-4,
        clip_range_mode: str = "constant",
        use_kl_penalty: bool = True,
        kl_coef: float = 0.01,
        ratio_reg_coef: float = 0.0,
        eta: float = 0.7,
        sde_type: str = "sde",
        ignore_last: bool = False,
        frozen_init_timesteps: int = 0,
        model_type: str = "default",
    ):
        """
        Initialize GRPO loss.

        Args:
            clip_range: PPO clipping range (epsilon)
            clip_range_mode: Clip range schedule ("constant", "linear_decay", "cosine_decay")
            use_kl_penalty: Whether to add KL penalty
            kl_coef: KL penalty coefficient
            ratio_reg_coef: Coefficient for ratio regularization
            eta: Noise level for SDE
            sde_type: SDE formulation ("sde", "cps", "dance")
            ignore_last: Skip the last timestep (t->0) in loss computation (MixGRPO).
                The last step has very low noise level, causing unstable log_prob.
            frozen_init_timesteps: Skip the first N timesteps in loss computation (MixGRPO).
                Early timesteps may have high variance.
            model_type: Model type for forward plugin selection ("flux", "sd3", "hunyuan", "default").
                If not specified, model type will be auto-detected from the model class name.
        """
        self.clip_range = clip_range
        self.clip_range_mode = clip_range_mode
        self.use_kl_penalty = use_kl_penalty
        self.kl_coef = kl_coef
        self.ratio_reg_coef = ratio_reg_coef
        self.eta = eta
        self.sde_type = sde_type
        self.ignore_last = ignore_last
        self.frozen_init_timesteps = frozen_init_timesteps
        self.model_type = model_type
        self._forward_plugin = None  # Lazy loaded

    def get_clip_range(self, progress: float = 0.0) -> float:
        """
        Get the current clip range based on training progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Current clip range
        """
        if self.clip_range_mode == "constant":
            return self.clip_range
        elif self.clip_range_mode == "linear_decay":
            return self.clip_range * (1 - 0.5 * progress)
        elif self.clip_range_mode == "cosine_decay":
            return self.clip_range * (0.5 * (1 + math.cos(math.pi * progress)))
        else:
            return self.clip_range

    def compute_log_prob(
        self,
        pred: torch.Tensor,
        sample: torch.Tensor,
        next_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability for SDE step.

        Delegates to the canonical implementation in samplers/log_prob.py
        to ensure consistency between sampling and training paths.

        Formula (SDE type):
            std_dev_t = sqrt(sigma / (1 - sigma)) * eta
            prev_sample_mean = sample * (1 + std_dev_t²/(2σ)*dt) + pred * (...)
            log_prob = -||next_sample - mean||² / (2σ²) - log(σ) - 0.5*log(2π)

        Args:
            pred: Model velocity prediction [B, C, H, W] or [B, C, T, H, W]
            sample: Current sample (x_t)
            next_sample: Next sample (x_{t-1})
            sigma: Current sigma
            sigma_next: Next sigma
            sigma_max: Maximum sigma value

        Returns:
            Tuple of (log_prob [B], prev_sample_mean [B, C, ...])
        """
        from diffusionrl.samplers.log_prob import compute_sde_log_prob

        return compute_sde_log_prob(
            noise_pred=pred,
            sample=sample,
            prev_sample=next_sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            sde_type=self.sde_type,
            sigma_max=sigma_max,
        )

    def compute(
        self,
        model: nn.Module,
        samples: Dict[str, Any],
        timestep_idx: int,
        advantages: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        training_progress: float = 0.0,
        model_forward_fn: Optional[callable] = None,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GRPO loss for a single timestep.

        Args:
            model: The diffusion model being trained
            samples: Dictionary containing:
                - 'latents' or 'trajectories': [B, num_steps+1, C, H, W]
                - 'log_probs' or 'log_probs_dict': old log probabilities
                - 'timesteps' or 'sigmas': sigma schedule
            timestep_idx: Which timestep to compute loss for
            advantages: Per-sample advantages [B]
            prompt_embeds: Text encoder hidden states [B, seq, hidden]
            pooled_prompt_embeds: Pooled text embeddings [B, hidden]
            sigmas: Sigma schedule (if not in samples)
            ref_model: Reference model for KL penalty
            training_progress: Training progress in [0, 1]
            model_forward_fn: Custom forward function
            guidance_scale: CFG scale
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = advantages.device
        batch_size = advantages.shape[0]

        # Get latents (support both key names)
        latents_key = "trajectories" if "trajectories" in samples else "latents"
        all_latents = samples[latents_key]

        # Check if this is a mixed sampler with sparse log_probs
        log_probs_dict = samples.get("log_probs_dict")
        sde_indices = samples.get("sde_indices")

        # Validate log_probs_dict is not empty when using mixed sampling
        if log_probs_dict is not None and len(log_probs_dict) == 0:
            logger.warning(
                "log_probs_dict is empty - all timesteps appear to be ODE steps. "
                "Check that sde_ratio > 0 and sde_indices is properly computed. "
                "Falling back to full tensor log_probs if available."
            )
            # Try fallback to tensor log_probs
            log_probs_tensor = samples.get("log_probs")
            if log_probs_tensor is not None and isinstance(log_probs_tensor, torch.Tensor):
                log_probs_dict = None  # Will use tensor path below

        if log_probs_dict is not None:
            # Mixed sampler mode - check if this timestep has log_prob
            if timestep_idx not in log_probs_dict:
                # ODE step - no loss for this timestep
                logger.warning(
                    f"Timestep {timestep_idx} missing from log_probs_dict (available: {list(log_probs_dict.keys())}). "
                    f"This may indicate ODE steps not computing log_prob. Consider enabling log_prob for all steps."
                )
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    "skip_reason": "ode_step",
                    "timestep_idx": timestep_idx,
                }
            old_log_probs = log_probs_dict[timestep_idx]
        else:
            # Full SDE mode
            log_probs = samples.get("log_probs", samples.get("old_log_probs"))
            if isinstance(log_probs, torch.Tensor):
                old_log_probs = log_probs[:, timestep_idx]
            else:
                old_log_probs = log_probs[timestep_idx]

        # Get current and next latents from trajectory
        latents = all_latents[:, timestep_idx]
        next_latents = all_latents[:, timestep_idx + 1]

        # Get sigma schedule
        if sigmas is None:
            sigmas = samples.get("sigmas", samples.get("timesteps"))
        sigma = sigmas[timestep_idx]
        sigma_next = sigmas[timestep_idx + 1]
        sigma_max = sigmas[1].item() if sigmas[1].dim() == 0 else sigmas[1][0].item()

        # Convert sigma to tensor if needed
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device)
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(sigma_next, device=device)

        # Forward pass through model
        if model_forward_fn is not None:
            pred = model_forward_fn(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                **kwargs,
            )
        else:
            pred = self._default_forward(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                text_ids=kwargs.get('text_ids'),
                image_ids=kwargs.get('image_ids'),
                negative_prompt_embeds=samples.get("negative_prompt_embeds"),
                negative_pooled_prompt_embeds=samples.get("negative_pooled_prompt_embeds"),
            )

        # Compute new log probability
        pred_f = pred.float()
        latents_f = latents.float()
        next_latents_f = next_latents.float()

        new_log_prob, prev_sample_mean = self.compute_log_prob(
            pred=pred_f,
            sample=latents_f,
            next_sample=next_latents_f,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
        )

        # Compute importance sampling ratio with clamping for numerical stability
        log_prob_diff = torch.clamp(new_log_prob - old_log_probs, -20.0, 20.0)
        ratio = torch.exp(log_prob_diff)

        # Get clip range
        clip_range = self.get_clip_range(training_progress)

        # PPO-style clipped loss
        adv = advantages.detach()
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # Compute metrics
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
        clipfrac_gt_one = (ratio - 1.0 > clip_range).float().mean()
        clipfrac_lt_one = (1.0 - ratio > clip_range).float().mean()
        approx_kl = 0.5 * torch.mean(log_prob_diff ** 2)

        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std().detach(),
            "ratio_max": ratio.max().detach(),
            "ratio_min": ratio.min().detach(),
            "clip_fraction": clip_fraction.detach(),
            "clipfrac_gt_one": clipfrac_gt_one.detach(),
            "clipfrac_lt_one": clipfrac_lt_one.detach(),
            "approx_kl": approx_kl.detach(),
            "new_log_prob_mean": new_log_prob.mean().detach(),
            "old_log_prob_mean": old_log_probs.mean().detach(),
        }

        total_loss = policy_loss

        # KL penalty (optional)
        if self.use_kl_penalty and self.kl_coef > 0:
            kl_loss = self._compute_kl_penalty(
                model=model,
                latents=latents,
                latents_f=latents_f,
                next_latents_f=next_latents_f,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=samples.get("negative_prompt_embeds"),
                negative_pooled_prompt_embeds=samples.get("negative_pooled_prompt_embeds"),
                prev_sample_mean=prev_sample_mean,
                ref_model=ref_model,
                guidance_scale=guidance_scale,
                model_forward_fn=model_forward_fn,
                **kwargs,
            )

            if kl_loss is not None:
                total_loss = total_loss + self.kl_coef * kl_loss
                loss_terms["kl_loss"] = kl_loss.detach()

        # Ratio regularization (prevents ratio from exploding)
        if self.ratio_reg_coef > 0:
            ratio_reg = torch.mean((new_log_prob - old_log_probs) ** 2)
            total_loss = total_loss + self.ratio_reg_coef * ratio_reg
            loss_terms["ratio_reg"] = ratio_reg.detach()

        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["timestep_idx"] = timestep_idx

        return total_loss, loss_terms

    def compute_timestep(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        sigmas: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        training_progress: float = 0.0,
        model_forward_fn: Optional[callable] = None,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GRPO loss for a single timestep using typed data structures.

        This is the typed interface for GRPO loss computation. It uses
        TimestepData and PromptEmbeddings instead of dictionaries.

        Args:
            model: The diffusion model being trained
            timestep_data: Typed container with latents, log_prob, sigmas for this step
            advantages: Per-sample advantages [B]
            embeddings: Typed container with prompt embeddings
            sigmas: Full sigma schedule (optional, uses timestep_data if not provided)
            ref_model: Reference model for KL penalty
            training_progress: Training progress in [0, 1]
            model_forward_fn: Custom forward function
            guidance_scale: CFG scale
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = advantages.device
        batch_size = advantages.shape[0]

        # Extract data from typed structures
        latents = timestep_data.latents
        next_latents = timestep_data.next_latents
        old_log_probs = timestep_data.log_prob
        sigma = timestep_data.sigma
        sigma_next = timestep_data.sigma_next
        timestep_idx = timestep_data.timestep_idx

        # Handle ODE step (no log_prob)
        if old_log_probs is None:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skip_reason": "ode_step",
                "timestep_idx": timestep_idx,
            }

        # Get sigma_max for log_prob computation
        # sigma_max is used to handle the sigma=1.0 boundary case
        if sigmas is not None:
            sigma_max = sigmas[1].item() if sigmas[1].dim() == 0 else sigmas[1][0].item()
        else:
            # Fallback: use sigma_next if available, otherwise use a safe default
            # The key is that sigma_max must be < 1.0 to avoid division by zero
            sigma_val = sigma.item() if sigma.dim() == 0 else sigma[0].item()
            sigma_next_val = sigma_next.item() if isinstance(sigma_next, torch.Tensor) and sigma_next.dim() == 0 else (sigma_next[0].item() if isinstance(sigma_next, torch.Tensor) else sigma_next)
            # Use sigma_next if sigma is at the boundary (1.0), otherwise use sigma
            sigma_max = sigma_next_val if sigma_val >= 0.9999 else sigma_val

        # Convert sigma to tensor if needed
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device)
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(sigma_next, device=device)

        # Extract embeddings
        prompt_embeds = embeddings.prompt_embeds
        pooled_prompt_embeds = embeddings.pooled_prompt_embeds
        negative_prompt_embeds = getattr(embeddings, "negative_prompt_embeds", None)
        negative_pooled_prompt_embeds = getattr(embeddings, "negative_pooled_prompt_embeds", None)

        # Forward pass through model
        if model_forward_fn is not None:
            pred = model_forward_fn(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                text_ids=embeddings.text_ids,
                image_ids=embeddings.image_ids,
                **kwargs,
            )
        else:
            pred = self._default_forward(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                text_ids=embeddings.text_ids,
                image_ids=embeddings.image_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

        # Compute new log probability
        pred_f = pred.float()
        latents_f = latents.float()
        next_latents_f = next_latents.float()

        new_log_prob, prev_sample_mean = self.compute_log_prob(
            pred=pred_f,
            sample=latents_f,
            next_sample=next_latents_f,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
        )

        # Compute importance sampling ratio with clamping for numerical stability
        log_prob_diff = torch.clamp(new_log_prob - old_log_probs, -20.0, 20.0)
        ratio = torch.exp(log_prob_diff)

        # Get clip range
        clip_range = self.get_clip_range(training_progress)

        # PPO-style clipped loss
        adv = advantages.detach()
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # Compute metrics
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
        clipfrac_gt_one = (ratio - 1.0 > clip_range).float().mean()
        clipfrac_lt_one = (1.0 - ratio > clip_range).float().mean()
        approx_kl = 0.5 * torch.mean(log_prob_diff ** 2)

        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std().detach(),
            "ratio_max": ratio.max().detach(),
            "ratio_min": ratio.min().detach(),
            "clip_fraction": clip_fraction.detach(),
            "clipfrac_gt_one": clipfrac_gt_one.detach(),
            "clipfrac_lt_one": clipfrac_lt_one.detach(),
            "approx_kl": approx_kl.detach(),
            "new_log_prob_mean": new_log_prob.mean().detach(),
            "old_log_prob_mean": old_log_probs.mean().detach(),
        }

        total_loss = policy_loss

        # KL penalty (optional)
        if self.use_kl_penalty and self.kl_coef > 0:
            kl_loss = self._compute_kl_penalty(
                model=model,
                latents=latents,
                latents_f=latents_f,
                next_latents_f=next_latents_f,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prev_sample_mean=prev_sample_mean,
                ref_model=ref_model,
                guidance_scale=guidance_scale,
                model_forward_fn=model_forward_fn,
                **kwargs,
            )

            if kl_loss is not None:
                total_loss = total_loss + self.kl_coef * kl_loss
                loss_terms["kl_loss"] = kl_loss.detach()

        # Ratio regularization
        if self.ratio_reg_coef > 0:
            ratio_reg = torch.mean((new_log_prob - old_log_probs) ** 2)
            total_loss = total_loss + self.ratio_reg_coef * ratio_reg
            loss_terms["ratio_reg"] = ratio_reg.detach()

        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["timestep_idx"] = timestep_idx

        return total_loss, loss_terms

    def _get_forward_plugin(self, model: nn.Module):
        """Get or create the forward plugin for this model.

        Lazy loads the plugin on first use, using model_type from config
        or auto-detecting from the model class name.
        """
        if self._forward_plugin is None:
            from diffusionrl.models.forward_plugins import get_forward_plugin, detect_model_type

            if self.model_type == "default":
                # Auto-detect model type from model class
                detected_type = detect_model_type(model)
                self._forward_plugin = get_forward_plugin(detected_type)
                logger.debug(f"Auto-detected model_type={detected_type} for forward plugin")
            else:
                self._forward_plugin = get_forward_plugin(self.model_type)

        return self._forward_plugin

    def _default_forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Default forward pass using model-specific plugin.

        Delegates to the appropriate forward plugin based on model_type,
        ensuring model-specific logic is encapsulated in plugins rather
        than scattered throughout the loss code.

        Args:
            model: The diffusion model
            latents: Input latents [B, C, H, W] or [B, C, T, H, W]
            sigma: Current sigma/timestep value
            prompt_embeds: Text encoder hidden states [B, seq, hidden]
            pooled_prompt_embeds: Pooled text embeddings [B, hidden]
            guidance_scale: Classifier-free guidance scale
            text_ids: Position IDs for text tokens (FLUX)
            image_ids: Position IDs for image patches (FLUX)
            negative_prompt_embeds: Negative prompt embeddings (for CFG)
            negative_pooled_prompt_embeds: Pooled negative embeddings
            **kwargs: Additional model-specific arguments

        Returns:
            Model prediction (velocity/noise) [B, C, H, W] or [B, C, T, H, W]
        """
        plugin = self._get_forward_plugin(model)

        return plugin.forward(
            model=model,
            latents=latents,
            sigma=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            text_ids=text_ids,
            image_ids=image_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **kwargs,
        )

    def _compute_kl_penalty(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        latents_f: torch.Tensor,
        next_latents_f: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_pooled_prompt_embeds: Optional[torch.Tensor],
        prev_sample_mean: torch.Tensor,
        ref_model: Optional[nn.Module],
        guidance_scale: float,
        model_forward_fn: Optional[callable],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Compute KL penalty between policy and reference."""
        ref_prev_sample_mean = None

        # Try LoRA disable_adapter first (flow_grpo style)
        adapter_model = model.module if hasattr(model, 'module') else model
        if hasattr(adapter_model, 'disable_adapter'):
            try:
                with torch.no_grad():
                    with adapter_model.disable_adapter():
                        if model_forward_fn is not None:
                            ref_pred = model_forward_fn(
                                model=model,
                                latents=latents,
                                sigma=sigma,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                guidance_scale=guidance_scale,
                                **kwargs,
                            )
                        else:
                            ref_pred = self._default_forward(
                                model=model,
                                latents=latents,
                                sigma=sigma,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                guidance_scale=guidance_scale,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                **kwargs,
                            )

                _, ref_prev_sample_mean = self.compute_log_prob(
                    pred=ref_pred.float(),
                    sample=latents_f,
                    next_sample=next_latents_f,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    sigma_max=sigma_max,
                )
            except Exception:
                pass

        # Fallback to ref_model if provided
        if ref_prev_sample_mean is None and ref_model is not None:
            with torch.no_grad():
                if model_forward_fn is not None:
                    ref_pred = model_forward_fn(
                        model=ref_model,
                        latents=latents,
                        sigma=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        **kwargs,
                    )
                else:
                    ref_pred = self._default_forward(
                        model=ref_model,
                        latents=latents,
                        sigma=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        **kwargs,
                    )

            _, ref_prev_sample_mean = self.compute_log_prob(
                pred=ref_pred.float(),
                sample=latents_f,
                next_sample=next_latents_f,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
            )

        if ref_prev_sample_mean is not None:
            # KL loss calculation matching flow_grpo
            dt = sigma_next - sigma
            std_dev_t = torch.sqrt(
                sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
            ) * self.eta

            kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(
                dim=tuple(range(1, prev_sample_mean.ndim))
            ) / (2 * std_dev_t ** 2 + 1e-12)
            kl_loss = kl_loss.mean()

            return kl_loss

        return None

    def compute_aggregated(
        self,
        model: nn.Module,
        samples: Dict[str, Any],
        advantages: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ignore_last: Optional[bool] = None,
        frozen_init_timesteps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GRPO loss aggregated over all timesteps.

        This is a convenience method that computes loss for all timesteps
        and aggregates them.

        Args:
            model: The diffusion model being trained
            samples: Sampling trajectory and log_probs
            advantages: Per-sample advantages [B]
            prompt_embeds: Text encoder hidden states
            pooled_prompt_embeds: Pooled text embeddings
            ignore_last: Override instance ignore_last setting (MixGRPO).
                Skip the last timestep (t->0) which has unstable log_prob.
            frozen_init_timesteps: Override instance setting.
                Skip the first N timesteps in loss computation.
            **kwargs: Additional arguments

        Returns:
            Tuple of (total_loss, aggregated_metrics)
        """
        # Use instance settings if not overridden
        if ignore_last is None:
            ignore_last = self.ignore_last
        if frozen_init_timesteps is None:
            frozen_init_timesteps = self.frozen_init_timesteps

        # Determine which timesteps to compute loss for
        sde_indices = samples.get("sde_indices")
        latents_key = "trajectories" if "trajectories" in samples else "latents"

        if sde_indices is not None:
            # Mixed sampler - only compute for SDE steps
            timestep_indices = list(sde_indices)
        else:
            # Full SDE sampler - compute for all steps
            num_steps = samples[latents_key].shape[1] - 1
            timestep_indices = list(range(num_steps))

        # Apply ignore_last: skip the last timestep (MixGRPO)
        # The last step (t->0) has very low noise level, causing unstable log_prob
        if ignore_last and len(timestep_indices) > 0:
            max_idx = max(timestep_indices)
            timestep_indices = [t for t in timestep_indices if t != max_idx]
            logger.debug(f"ignore_last: skipping timestep {max_idx}")

        # Apply frozen_init_timesteps: skip the first N timesteps (MixGRPO)
        if frozen_init_timesteps > 0:
            timestep_indices = [t for t in timestep_indices if t >= frozen_init_timesteps]
            logger.debug(f"frozen_init_timesteps: skipping timesteps 0-{frozen_init_timesteps-1}")

        device = advantages.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        aggregated_terms = {
            "policy_loss": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "num_steps": len(timestep_indices),
        }
        num_computed = 0

        for t in timestep_indices:
            loss, terms = self.compute(
                model=model,
                samples=samples,
                timestep_idx=t,
                advantages=advantages,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                **kwargs,
            )

            if "skip_reason" not in terms:
                total_loss = total_loss + loss
                num_computed += 1
                for key in ["policy_loss", "ratio_mean", "clip_fraction", "approx_kl"]:
                    if key in terms:
                        val = terms[key]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        aggregated_terms[key] += val

        # Average over timesteps
        if num_computed > 0:
            total_loss = total_loss / num_computed
            for key in ["policy_loss", "ratio_mean", "clip_fraction", "approx_kl"]:
                aggregated_terms[key] /= num_computed

        aggregated_terms["total_loss"] = total_loss.detach()

        return total_loss, aggregated_terms
