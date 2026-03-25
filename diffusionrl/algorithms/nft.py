"""
NFT (Negative Fine-Tuning) Algorithm Implementation — algorithm-owned loss.

DiffusionNFT forward process diffusion RL. The algorithm file owns rollout
requirements, advantage processing, and the forward-process loss entrypoint.
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

from diffusionrl.types import ForwardTrainingBatch, SDEConfig
from diffusionrl.utils.adapter_utils import switch_adapter
from diffusionrl.utils.dtypes import parse_torch_dtype
from .base import BaseAlgorithm, EMASpec, SamplingRequirements

logger = logging.getLogger(__name__)


def _resolve_algorithm_sde_config(config: Dict[str, Any]) -> SDEConfig:
    return SDEConfig.from_mapping(config.get("sde_config"))


class _NFTLoss:
    """Private NFT objective owned and created by NFTAlgorithm."""

    def __init__(self, algorithm: "NFTAlgorithm") -> None:
        self.algorithm = algorithm

    def _compute_core(
        self,
        model: nn.Module,
        samples: Dict[str, Any],
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
        algorithm = self.algorithm
        del attn_metadata

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
        t = provided_t.to(device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        apply_shift = bool(kwargs.get("apply_shift", False))

        xt, _ = algorithm.forward_diffusion(
            x0,
            t,
            generator=generator,
            apply_shift=apply_shift,
        )

        if apply_shift:
            t_shifted = (algorithm.time_shift * t) / (1 + (algorithm.time_shift - 1) * t)
        else:
            t_shifted = t
        t_expanded = t_shifted.view(-1, *([1] * (x0.ndim - 1)))

        algorithm_autocast_dtype = getattr(algorithm, "autocast_dtype", None)
        if algorithm_autocast_dtype in (torch.float16, torch.bfloat16):
            model_dtype = algorithm_autocast_dtype
        elif config is not None and hasattr(config, "get_torch_dtype"):
            model_dtype = config.get_torch_dtype()
        else:
            model_dtype = prompt_embeds.dtype if prompt_embeds is not None else xt.dtype
        autocast_enabled = model_dtype in (torch.float16, torch.bfloat16)

        xt_cast = xt.to(model_dtype)
        prompt_embeds_cast = prompt_embeds.to(model_dtype) if prompt_embeds is not None else None
        pooled_embeds_cast = pooled_prompt_embeds.to(model_dtype) if pooled_prompt_embeds is not None else None
        guidance_scale = getattr(config, "guidance_scale", 3.5) if config is not None else 3.5
        plugin = algorithm._get_forward_plugin(model)
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
        adapter_model.set_adapter(algorithm.new_adapter_name)

        autocast_ctx_fn = lambda: torch.autocast("cuda", model_dtype) if autocast_enabled else nullcontext[None]()
        grad_context = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()

        with grad_context, autocast_ctx_fn():
            forward_prediction = model(**model_kwargs)[0]

        with autocast_ctx_fn():
            old_prediction = algorithm.get_old_prediction(
                model,
                model_kwargs,
                old_model=old_model,
            )

        if hasattr(adapter_model, "set_adapter"):
            adapter_model.set_adapter(algorithm.new_adapter_name)

        adv_processed = algorithm.prepare_loss_advantages(advantages)
        adv_clipped = torch.clamp(adv_processed, -algorithm.adv_clip_max, algorithm.adv_clip_max)
        r = (adv_clipped / algorithm.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0, 1)

        positive_prediction = (
            algorithm.beta * forward_prediction + (1 - algorithm.beta) * old_prediction.detach()
        )
        negative_prediction = (
            (1 + algorithm.beta) * old_prediction.detach() - algorithm.beta * forward_prediction
        )

        x0_positive = xt - t_expanded * positive_prediction
        x0_negative = xt - t_expanded * negative_prediction

        if algorithm.use_adaptive_weight:
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
            r_expanded * positive_loss / algorithm.beta
            + (1 - r_expanded) * negative_loss / algorithm.beta
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

        total_loss = policy_loss * algorithm.adv_clip_max

        if algorithm.kl_coef > 0:
            with autocast_ctx_fn():
                ref_prediction = algorithm.get_ref_prediction(
                    model,
                    model_kwargs,
                    ref_model=ref_model,
                )
            kl_div = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_div = torch.mean(kl_div)
            total_loss = total_loss + algorithm.kl_coef * kl_div
            loss_terms["kl_div"] = kl_div.detach()

        loss_terms["total_loss"] = total_loss.detach()
        return total_loss, loss_terms

    def compute_loss(
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
        samples = {"clean_latents": batch.clean_latents}
        return self._compute_core(
            model=model,
            samples=samples,
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


class NFTAlgorithm(BaseAlgorithm):
    """
    NFT (Negative Fine-Tuning) Algorithm — algorithm owns loss and advantages.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation and post-processing
    3. Loss ownership / debug entry (compute_loss -> self.loss_fn)
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
    _loss_cls = _NFTLoss

    @classmethod
    def from_config(cls, config: dict) -> "NFTAlgorithm":
        """Create NFTAlgorithm from an algorithm_config dictionary.

        Reads NFT-specific extension keys from ``algorithm_kwargs`` and shared
        framework-owned fields from the top-level algorithm_config surface.
        """
        extra = cls.resolve_config_kwargs(config)
        sde_config = _resolve_algorithm_sde_config(config)
        known_keys = {
            "beta",
            "adv_clip_max",
            "adv_mode",
            "use_adaptive_weight",
            # Backward-compatibility shim for legacy NFT launch scripts.
            # NFT never consumed PPO-style ratio clipping, but older bash
            # entrypoints still pass clip_range through algorithm_kwargs.
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
            samples_per_prompt=int(config.get("samples_per_prompt", 1)),
            eval_ema_decay=float(config.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(config.get("eval_ema_update_interval", 1)),
            kl_coef=float(extra.get("kl_coef", 0.0)),
            adv_normalization=str(config.get("adv_normalization", "group")),
            epsilon=float(config.get("adv_norm_eps", 1e-8)),
            clip_max=config.get("adv_clip_abs", 5.0),
            use_global_std=bool(config.get("use_global_std", False)),
            trimmed_ratio=float(config.get("trimmed_ratio", 0.0)),
            autocast_precision=config.get("training_autocast_precision"),
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
        # BaseAlgorithm params
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: float = 5.0,
        use_global_std: bool = False,
        trimmed_ratio: float = 0.0,
        autocast_precision: Any = None,
        **kwargs,
    ):
        super().__init__(
            kl_coef=kl_coef,
            component_mix_stage=component_mix_stage,
            adv_normalization=adv_normalization,
            samples_per_prompt=samples_per_prompt,
            eval_ema_decay=eval_ema_decay,
            eval_ema_update_interval=eval_ema_update_interval,
            epsilon=epsilon,
            clip_max=clip_max,
            use_global_std=use_global_std,
            trimmed_ratio=trimmed_ratio,
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
        self.model_type = "default"
        self.autocast_dtype = parse_torch_dtype(
            autocast_precision,
            field_name="training_autocast_precision",
            allow_none=True,
        )
        self._forward_plugin = None

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
        timestep_scheduler: Optional[Any],
        current_step: int,
    ) -> Optional[Set[int]]:
        del timestep_scheduler, current_step
        return None

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

    # ------------------------------------------------------------------
    # Rollout geometry / request planning
    # ------------------------------------------------------------------

    def assemble_training_batch(
        self,
        *,
        request: Any,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        del sde_indices
        return self._assemble_forward_batch(
            request=request,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
        )

    def _assemble_forward_batch(
        self,
        *,
        request: Any,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Any:
        from diffusionrl.types.sampling import RolloutSamples, PromptEmbeddings
        from diffusionrl.types.training_batch import ForwardTrainingBatch, build_rollout_extras

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
            embeddings_item = output.aux.get("embeddings")
            if embeddings_item is None:
                raise ValueError(f"RolloutSamples at index={idx} missing embeddings in forward path.")

            clean_latents.append(output.latents)
            emb = embeddings_item
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
            prompts=list(request.prompts),
            prompt_ids=(
                list(request.meta.get("prompt_ids")) if request.meta.get("prompt_ids") is not None else None
            ),
            sample_ids=(
                list(request.meta.get("sample_ids")) if request.meta.get("sample_ids") is not None else None
            ),
            group_ids=(
                list(request.meta.get("group_ids")) if request.meta.get("group_ids") is not None else None
            ),
            timesteps=timesteps,
            extras=build_rollout_extras(
                request=request,
                sampler_outputs=sampler_outputs,
            ),
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
            t_shifted = (self.time_shift * t) / (1 + (self.time_shift - 1) * t)
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
        Get prediction from the NFT old policy.

        Accepted paths:
        1. Explicit old_model
        2. Full-parameter EMA swap
        3. LoRA old-adapter switch

        Reproduce mode must fail fast if none of these are available or if
        adapter switching fails. Falling back to the base model or current
        model would silently change the NFT objective semantics.
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

    def get_ref_prediction(
        self,
        model: nn.Module,
        model_kwargs: Dict[str, Any],
        ref_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Get reference prediction for KL regularization (base model)."""
        with torch.no_grad():
            adapter_model = model.module if hasattr(model, "module") else model
            if hasattr(adapter_model, "disable_adapter"):
                try:
                    with adapter_model.disable_adapter():
                        return model(**model_kwargs)[0]
                except Exception as exc:
                    raise RuntimeError(
                        "NFT reference prediction failed while disabling adapters. "
                        "Refusing to fall back to current model because that would collapse "
                        "the KL term toward zero."
                    ) from exc
            if ref_model is not None:
                return ref_model(**model_kwargs)[0]
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
        mini_batch_slices: Tuple[Tuple[int, int], ...],
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> tuple:
        """NFT training step: forward diffusion at sampled/scheduled timesteps.

        The algorithm decides timestep mode, shuffling, and accumulation.

        Returns:
            ``(avg_loss, metrics_dict, num_timesteps, actual_mini_batches, has_backward)``
        """
        if not isinstance(batch, ForwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects ForwardTrainingBatch, got {type(batch).__name__}"
            )

        timestep_mode = kwargs.get("timestep_mode", self.train_timestep_mode)
        shuffle_timesteps = kwargs.get("shuffle_timesteps", self.shuffle_train_timesteps)
        apply_shift = kwargs.get("apply_shift", self.apply_time_shift_in_loss)
        timestep_fraction = kwargs.get("timestep_fraction", 1.0)

        mini_batches = tuple((int(start), int(end)) for start, end in mini_batch_slices)
        if not mini_batches:
            raise ValueError(f"{type(self).__name__} requires non-empty mini_batch_slices.")
        actual_mini_batches = len(mini_batches)
        num_mini_batches = len(mini_batches)

        if timestep_mode == "all" and batch.timesteps is not None:
            timesteps = batch.timesteps.detach().flatten()
        else:
            timesteps = torch.rand(batch.batch_size, device=batch.advantages.device)

        # Filter trailing zero
        if timesteps.numel() > 1 and torch.isclose(
            timesteps[-1],
            torch.zeros((), device=timesteps.device, dtype=timesteps.dtype),
            atol=1e-8,
        ).item():
            timesteps = timesteps[:-1]

        # Apply timestep_fraction
        if timesteps.numel() > 0 and timestep_fraction is not None and timestep_fraction != 1.0:
            from diffusionrl.samplers.schedulers.timestep_window import _normalize_timestep_fraction
            frac_start, frac_end = _normalize_timestep_fraction(timestep_fraction)
            n = timesteps.numel()
            effective_start = int(n * frac_start)
            effective_end = min(int(n * frac_end), n)
            if effective_start < effective_end:
                timesteps = timesteps[effective_start:effective_end]
            else:
                timesteps = timesteps[:0]

        if timesteps.numel() == 0:
            # Fall back to random
            timesteps = torch.rand(batch.batch_size, device=batch.advantages.device)

        if shuffle_timesteps:
            perm = torch.randperm(timesteps.numel(), device=timesteps.device)
            timesteps = timesteps[perm]

        effective_mini_batches = actual_mini_batches * timesteps.numel()
        total_loss_accum = 0.0
        has_backward = False
        mini_batch_metrics_list = []

        for start, end in mini_batches:
            mini_batch = batch.slice(start, end)
            mini_loss_sum = 0.0
            metric_sums: Dict[str, float] = {}
            metric_counts: Dict[str, int] = {}
            mini_metrics: Dict[str, Any] = {}

            for t in timesteps:
                loss, metrics = self.compute_loss(
                    model=model,
                    batch=mini_batch,
                    timestep_values=t,
                    apply_shift=apply_shift,
                )
                scaled_loss = loss / effective_mini_batches
                scaled_loss.backward()
                has_backward = True
                mini_loss_sum += scaled_loss.detach().item()

                for key, value in metrics.items():
                    metric_val = value.item() if isinstance(value, torch.Tensor) else float(value)
                    metric_sums[key] = metric_sums.get(key, 0.0) + metric_val
                    metric_counts[key] = metric_counts.get(key, 0) + 1
                    if key not in mini_metrics:
                        mini_metrics[key] = metric_val

            for key, total in metric_sums.items():
                count = metric_counts.get(key, 0)
                if count > 0:
                    mini_metrics[key] = total / count

            mini_batch_metrics_list.append(mini_metrics)
            total_loss_accum += mini_loss_sum

        all_metrics: Dict[str, Any] = {}
        if mini_batch_metrics_list:
            keys = mini_batch_metrics_list[0].keys()
            for key in keys:
                values = [m.get(key) for m in mini_batch_metrics_list if m.get(key) is not None]
                if values and isinstance(values[0], (int, float)):
                    all_metrics[key] = sum(values) / len(values)
                else:
                    all_metrics[key] = mini_batch_metrics_list[-1].get(key)

        return (
            total_loss_accum,
            all_metrics,
            timesteps.numel(),
            effective_mini_batches,
            has_backward,
        )

    # ------------------------------------------------------------------
    # Forward plugin
    # ------------------------------------------------------------------

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

    def compute_loss(
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
        """Single NFT loss entrypoint for debugging and training."""
        if self.loss_fn is None:
            raise RuntimeError(f"{type(self).__name__} loss_fn is not initialized.")
        return self.loss_fn.compute_loss(
            model=model,
            batch=batch,
            ref_model=ref_model,
            old_model=old_model,
            generator=generator,
            attn_metadata=attn_metadata,
            config=config,
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
