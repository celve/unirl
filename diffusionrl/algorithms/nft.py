"""
NFT (Negative Fine-Tuning) Algorithm Implementation — algorithm-owned loss.

DiffusionNFT forward process diffusion RL. The algorithm file owns rollout
requirements, advantage processing, and the forward-process loss entrypoint.
"""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from diffusionrl.types.sampling import RolloutRequest

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

from diffusionrl.config.build_domain_args import resolve_sde_config
from diffusionrl.types import ForwardTrainingBatch, SDEConfig
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.misc import aggregate_numeric_metrics

from .base import BaseAlgorithm, EMASpec, SamplingRequirements
from .forward_context import ForwardContext

logger = logging.getLogger(__name__)


def _resolve_algorithm_sde_config(config: Dict[str, Any]) -> SDEConfig:
    return resolve_sde_config(config)


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

    @classmethod
    def from_config(cls, config: dict) -> "NFTAlgorithm":
        """Create NFTAlgorithm from an algorithm_config dictionary.

        Reads NFT-specific extension keys from ``algorithm_kwargs`` and shared
        framework-owned fields from the top-level algorithm_config surface.
        """
        extra = cls.resolve_config_kwargs(config)
        sde_config = _resolve_algorithm_sde_config(config)
        training_scheduler_config = dict(config.get("training_scheduler") or {})
        logger.info(
            "%s uses training_scheduler.timestep_fraction for training timestep "
            "filtering and ignores rollout_scheduler configuration.",
            cls.__name__,
        )
        known_keys = {
            "beta",
            "adv_clip_max",
            "adv_mode",
            "use_adaptive_weight",
            "clip_range",
            "skip_last_timestep",
            "skip_initial_timesteps",
            "old_adapter_name",
            "new_adapter_name",
            "kl_coef",
            "use_reference_ema",
            "ema_decay",
            "decay_type",
            "ema_flat_steps",
            "ema_uprate",
            "ema_uphold",
            "reference_update_timing",
            "train_timestep_mode",
            "shuffle_train_timesteps",
            "apply_time_shift_in_loss",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys)
        if unknown:
            raise ValueError(
                "algorithm.algorithm_kwargs contains unsupported keys for algorithm_type='nft': "
                f"{unknown}."
            )

        return cls(
            beta=float(extra.get("beta", 0.1)),
            adv_clip_max=float(extra.get("adv_clip_max", 5.0)),
            adv_mode=str(extra.get("adv_mode", "raw")),
            use_adaptive_weight=bool(extra.get("use_adaptive_weight", True)),
            component_mix_stage=str(config.get("component_mix_stage", "reward")),
            sde_config=sde_config,
            use_reference_ema=bool(extra.get("use_reference_ema", True)),
            old_adapter_name=str(extra.get("old_adapter_name", "old")),
            new_adapter_name=str(extra.get("new_adapter_name", "default")),
            ema_decay=float(extra.get("ema_decay", 0.001)),
            ema_decay_type=str(extra.get("decay_type", "constant")),
            ema_flat_steps=int(extra.get("ema_flat_steps", 0)),
            ema_uprate=float(extra.get("ema_uprate", 0.001)),
            ema_uphold=float(extra.get("ema_uphold", 0.5)),
            reference_update_timing=str(extra.get("reference_update_timing", "optimizer_step")),
            train_timestep_mode=str(extra.get("train_timestep_mode", "random")),
            shuffle_train_timesteps=bool(extra.get("shuffle_train_timesteps", True)),
            apply_time_shift_in_loss=bool(extra.get("apply_time_shift_in_loss", False)),
            training_scheduler_config=training_scheduler_config,
            samples_per_prompt=int(config.get("samples_per_prompt", 1)),
            num_inference_steps=int(config.get("num_inference_steps", 0)),
            eval_ema_decay=float(config.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(config.get("eval_ema_update_interval", 1)),
            kl_coef=float(extra.get("kl_coef", 0.0)),
            adv_normalization=str(config.get("adv_normalization", "group")),
            epsilon=float(config.get("adv_norm_eps", 1e-8)),
            clip_max=config.get("adv_clip_abs", 5.0),
            use_global_std=bool(config.get("use_global_std", False)),
            trimmed_ratio=float(config.get("trimmed_ratio", 0.0)),
        )

    def __init__(
        self,
        beta: float = 0.1,
        adv_clip_max: float = 5.0,
        adv_mode: str = "raw",
        use_adaptive_weight: bool = True,
        component_mix_stage: str = "reward",
        sde_config: Optional[SDEConfig] = None,
        use_reference_ema: bool = True,
        ema_decay: float = 0.001,
        ema_decay_type: str = "constant",
        ema_flat_steps: int = 0,
        ema_uprate: float = 0.001,
        ema_uphold: float = 0.5,
        reference_update_timing: str = "optimizer_step",
        kl_coef: float = 0.0,
        old_adapter_name: str = "old",
        new_adapter_name: str = "default",
        train_timestep_mode: str = "random",
        shuffle_train_timesteps: bool = True,
        apply_time_shift_in_loss: bool = False,
        training_scheduler_config: Optional[Dict[str, Any]] = None,
        # BaseAlgorithm params
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
        num_inference_steps: int = 0,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: float = 5.0,
        use_global_std: bool = False,
        **kwargs,
    ):
        super().__init__(
            kl_coef=kl_coef,
            component_mix_stage=component_mix_stage,
            adv_normalization=adv_normalization,
            samples_per_prompt=samples_per_prompt,
            num_inference_steps=num_inference_steps,
            eval_ema_decay=eval_ema_decay,
            eval_ema_update_interval=eval_ema_update_interval,
            epsilon=epsilon,
            clip_max=clip_max,
            use_global_std=use_global_std,
            **kwargs,
        )
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.adv_mode = adv_mode
        self.use_adaptive_weight = use_adaptive_weight
        self.sde_config = sde_config or SDEConfig()
        self.use_reference_ema = bool(use_reference_ema)
        self.ema_decay = ema_decay
        self.ema_decay_type = str(ema_decay_type)
        self.ema_flat_steps = int(ema_flat_steps)
        self.ema_uprate = float(ema_uprate)
        self.ema_uphold = float(ema_uphold)
        self.reference_update_timing = str(reference_update_timing).strip().lower()
        self.old_adapter_name = old_adapter_name
        self.new_adapter_name = new_adapter_name
        self.train_timestep_mode = str(train_timestep_mode)
        self.shuffle_train_timesteps = bool(shuffle_train_timesteps)
        self.apply_time_shift_in_loss = bool(apply_time_shift_in_loss)
        self.training_scheduler_config = dict(training_scheduler_config or {})
        self.training_timestep_fraction = self.training_scheduler_config.get(
            "timestep_fraction",
            1.0,
        )

    @property
    def eta(self) -> float:
        return self.sde_config.eta

    @property
    def sde_type(self) -> str:
        return self.sde_config.sde_type

    @property
    def time_shift(self) -> float:
        return self.sde_config.shift

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return NFT sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=False,
            requires_log_prob=False,
            requires_embeddings=True,
            extras={
                "sde_ratio": 0.0,
                "requires_clean_latents": True,
                "forward_diffusion_in_loss": True,
            },
        )

    def compute_advantages_with_components(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        reward_components: Optional[Dict[str, List[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if self.component_mix_stage != "advantage" or not reward_components:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)

        default_weights = {name: 1.0 for name in reward_components}
        if reward_component_weights:
            for name, weight in reward_component_weights.items():
                if name in default_weights:
                    default_weights[name] = float(weight)

        weighted_advantages = torch.zeros_like(rewards)
        total_weight = 0.0
        for component_name, component_rewards in reward_components.items():
            component_tensor = torch.tensor(
                component_rewards,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            if component_tensor.shape != rewards.shape:
                logger.warning(
                    "Skipping reward component %s due to shape mismatch: expected=%s got=%s",
                    component_name,
                    tuple(rewards.shape),
                    tuple(component_tensor.shape),
                )
                continue
            component_advantages = self.compute_advantages(
                rewards=component_tensor,
                group_ids=group_ids,
            )
            weight = float(default_weights.get(component_name, 1.0))
            weighted_advantages += component_advantages * weight
            total_weight += weight

        if total_weight <= 0:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)
        return weighted_advantages / total_weight

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(
            enable_eval_ema=True,
            eval_decay=self.eval_ema_decay,
            eval_update_interval=self.eval_ema_update_interval,
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
    ) -> Set[int]:
        del current_step
        if self.sde_config is None:
            return set()

        if self.num_inference_steps < 1:
            raise ValueError(
                "Algorithm.num_inference_steps must be set to a positive integer to resolve SDE indices for rollout."
            )

        return set(range(self.num_inference_steps))

    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        del args
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
        if not isinstance(batch, ForwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects ForwardTrainingBatch, got {type(batch).__name__}"
            )
        del current_step

        timestep_mode = kwargs.get("timestep_mode", self.train_timestep_mode)
        shuffle_timesteps = kwargs.get(
            "shuffle_timesteps", self.shuffle_train_timesteps
        )
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

        if (
            timesteps.numel() > 0
            and timestep_fraction is not None
            and timestep_fraction != 1.0
        ):
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

    def assemble_training_batch(
        self,
        *,
        request: "RolloutRequest",
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        return self._assemble_forward_batch(
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=request.prompts,
        )

    def _assemble_forward_batch(
        self,
        *,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
    ) -> Any:
        from diffusionrl.types.sampling import PromptEmbeddings, RolloutSamples
        from diffusionrl.types.training_batch import ForwardTrainingBatch

        clean_latents = []
        all_prompt_embeds = []
        all_pooled_prompt_embeds = []
        all_encoder_attention_mask = []
        all_negative_prompt_embeds = []
        all_negative_pooled_prompt_embeds = []
        all_text_ids = []
        all_image_ids = []
        timesteps: Optional[torch.Tensor] = None

        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, RolloutSamples):
                raise TypeError(
                    f"Assemble stage expects RolloutSamples, got {type(output).__name__} at index={idx}."
                )
            _embeddings = output.aux.get("embeddings")
            if _embeddings is None:
                raise ValueError(f"RolloutSamples at index={idx} missing embeddings in forward path.")

            clean_latents.append(output.latents)
            emb = _embeddings
            all_prompt_embeds.append(emb.prompt_embeds)
            if emb.pooled_prompt_embeds is not None:
                all_pooled_prompt_embeds.append(emb.pooled_prompt_embeds)
            if emb.encoder_attention_mask is not None:
                all_encoder_attention_mask.append(emb.encoder_attention_mask)
            if emb.negative_prompt_embeds is not None:
                all_negative_prompt_embeds.append(emb.negative_prompt_embeds)
            if emb.negative_pooled_prompt_embeds is not None:
                all_negative_pooled_prompt_embeds.append(emb.negative_pooled_prompt_embeds)
            if emb.text_ids is not None:
                all_text_ids.append(emb.text_ids)
            if emb.image_ids is not None:
                all_image_ids.append(emb.image_ids)
            if timesteps is None:
                timesteps = output.timesteps
            elif output.timesteps is None:
                raise ValueError(
                    f"RolloutSamples at index={idx} missing timesteps while earlier outputs provided them."
                )
            elif not torch.equal(timesteps.to(output.timesteps.device), output.timesteps):
                raise ValueError("Mismatched timesteps across sampler outputs")

        if not clean_latents:
            raise ValueError("No clean latents found in sampler outputs")
        if timesteps is None:
            raise ValueError("No timesteps found in sampler outputs")
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
        if prompt_embeds is None:
            raise ValueError("No prompt embeddings found in sampler outputs")

        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None,
            encoder_attention_mask=torch.cat(all_encoder_attention_mask, dim=0) if all_encoder_attention_mask else None,
            negative_prompt_embeds=torch.cat(all_negative_prompt_embeds, dim=0) if all_negative_prompt_embeds else None,
            negative_pooled_prompt_embeds=torch.cat(all_negative_pooled_prompt_embeds, dim=0) if all_negative_pooled_prompt_embeds else None,
            text_ids=torch.cat(all_text_ids, dim=0) if all_text_ids else None,
            image_ids=all_image_ids[0] if all_image_ids else None,
        )

        batch = ForwardTrainingBatch(
            clean_latents=torch.cat(clean_latents, dim=0),
            advantages=advantages,
            embeddings=embeddings,
            rewards=rewards,
            prompts=prompts,
            timesteps=timesteps,
        )
        batch.validate()
        return batch

    # ==================================================================
    # Objective computation
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
            noise = randn_tensor(
                x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
            )

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
        if old_model is not None:
            return ctx.forward(old_model, latents=latents, sigma=sigma)

        adapter_model = model.module if hasattr(model, "module") else model
        if hasattr(adapter_model, "set_adapter"):
            try:
                with switch_adapter(adapter_model, self.old_adapter_name):
                    return ctx.forward(model, latents=latents, sigma=sigma)
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
        adapter_model = model.module if hasattr(model, "module") else model
        if hasattr(adapter_model, "disable_adapter"):
            try:
                with adapter_model.disable_adapter():
                    return ctx.forward(model, latents=latents, sigma=sigma)
            except Exception as exc:
                raise RuntimeError(
                    "NFT reference prediction failed while disabling adapters. "
                    "Refusing to fall back to current model because that would collapse "
                    "the KL term toward zero."
                ) from exc
        if ref_model is not None:
            return ctx.forward(ref_model, latents=latents, sigma=sigma)
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
        timesteps: Optional[Any],
        guidance_scale: float = 3.5,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """NFT training step over one micro-batch.

        Here ``timesteps`` means the forward diffusion times to apply for
        this update chunk. The executor resolves them once and reuses the
        same sequence across all accumulation micro-batches so NFT keeps the
        original update-level timestep semantics.

        Returns:
            ``(scaled_loss, metrics_dict, num_timesteps, has_backward)``
        """
        if not isinstance(batch, ForwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects ForwardTrainingBatch, got {type(batch).__name__}"
            )

        timestep_mode = kwargs.get("timestep_mode", self.train_timestep_mode)
        shuffle_timesteps = kwargs.get("shuffle_timesteps", self.shuffle_train_timesteps)
        apply_shift = kwargs.get("apply_shift", self.apply_time_shift_in_loss)
        timestep_fraction = kwargs.get(
            "timestep_fraction",
            self.training_timestep_fraction,
        )
        if timesteps is None:
            timesteps = self.resolve_training_timesteps(
                batch=batch,
                timestep_mode=timestep_mode,
                shuffle_timesteps=shuffle_timesteps,
                timestep_fraction=timestep_fraction,
            )

        ctx = self._make_forward_context(model, batch.embeddings, guidance_scale)
        total_loss_accum = 0.0
        has_backward = False
        timestep_metrics: List[Dict[str, Any]] = []

        per_timestep_scale = loss_scale / timesteps.numel()
        for t in timesteps:
            loss, metrics = self.compute_loss(
                model=model,
                batch=batch,
                ctx=ctx,
                timestep_values=t,
            )
            scaled_loss = loss * per_timestep_scale
            scaled_loss.backward()
            has_backward = True
            total_loss_accum += scaled_loss.detach().item()
            timestep_metrics.append(metrics)

        all_metrics = aggregate_numeric_metrics(timestep_metrics)

        return total_loss_accum, all_metrics, timesteps.numel(), has_backward

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

        grad_context = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()

        with grad_context:
            forward_prediction = ctx.forward(model, latents=xt, sigma=t)

        old_prediction = self.get_old_prediction(
            model, ctx, latents=xt, sigma=t, old_model=old_model,
        )

        if hasattr(adapter_model, "set_adapter"):
            adapter_model.set_adapter(self.new_adapter_name)

        adv_processed = self.prepare_loss_advantages(advantages)
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
            ref_prediction = self.get_ref_prediction(
                model, ctx, latents=xt, sigma=t, ref_model=ref_model,
            )
            kl_div = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_div = torch.mean(kl_div)
            total_loss = total_loss + self.kl_coef * kl_div
            loss_terms["kl_div"] = kl_div.detach()

        loss_terms["total_loss"] = total_loss.detach()
        return total_loss, loss_terms

    def compute_loss(
        self,
        model: nn.Module,
        batch: ForwardTrainingBatch,
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
        config.update({
            "name": self.name,
            "beta": self.beta,
            "adv_clip_max": self.adv_clip_max,
            "adv_mode": self.adv_mode,
            "use_adaptive_weight": self.use_adaptive_weight,
            "sde_config": self.sde_config.to_dict(),
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
            "requires_trajectory": self.requires_trajectory,
            "requires_log_prob": self.requires_log_prob,
        })
        return config
