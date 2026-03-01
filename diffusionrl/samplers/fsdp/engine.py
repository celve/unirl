"""
FSDP Inference Engine.

Native PyTorch inference engine compatible with FSDP.
Supports both image and video models with DanceGRPO-aligned sampling.

Multi-GPU Support:
- Single GPU (default): One actor per GPU, data parallel across actors
- Multi-GPU per actor: Uses FSDP wrapper for model sharding

Architecture (multi-GPU mode):
    FSDPRolloutEngine (num_gpus=4)
    ├── _init_distributed()
    │   └── torch.distributed.init_process_group(backend="nccl")
    ├── Model with FSDP (inference mode, NO_SHARD)
    └── generate() → unified inference across GPUs
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Union
import torch
import torch.nn as nn

from ..engine import BaseRolloutEngine, EngineConfig, EngineCapabilities, register_engine
from diffusionrl.types import RolloutOutput, RolloutRequest
from diffusionrl.utils import load_function
from . import sampler_runner

logger = logging.getLogger(__name__)


@register_engine("fsdp")
class FSDPRolloutEngine(BaseRolloutEngine):
    """
    FSDP Inference Engine.

    Uses native PyTorch for inference, compatible with FSDP weight synchronization.
    Supports:
    - Image models: FLUX, SD3
    - Video models: HunyuanVideo (DanceGRPO-aligned)

    Ray Scheduling:
    - Single GPU per actor (num_gpus=1, default)
    - Multi-GPU per actor (num_gpus>1, requires FSDP wrapper)
    - Supports offload/onload for memory management
    - Weight sync via state_dict transfer

    Multi-GPU Mode:
        When num_gpus > 1, the engine uses FSDP to wrap the model for
        multi-GPU inference. This is useful for large models that don't
        fit on a single GPU or for faster inference with model parallelism.
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize FSDP engine.

        Args:
            config: Engine configuration
        """
        super().__init__(config)
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.sampler = None
        self.model_bundle = None
        self._device = None

        # Multi-GPU configuration
        self._num_gpus = config.engine_kwargs.get("num_gpus", 1)
        self._use_fsdp = self._num_gpus > 1
        self._fsdp_cpu_offload = bool(config.engine_kwargs.get("cpu_offload", False))
        self._fsdp_sharding_strategy = config.engine_kwargs.get(
            "fsdp_sharding_strategy", "NO_SHARD"
        )
        self._distributed_initialized = False

    @classmethod
    def declared_capabilities(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    def _init_distributed(self) -> None:
        """
        Initialize distributed for multi-GPU inference.

        Sets up torch.distributed process group using NCCL backend.
        Environment variables (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK)
        should be set by the Ray actor before calling this method.
        """
        if self._num_gpus <= 1:
            return

        if self._distributed_initialized:
            return

        import torch.distributed as dist

        # Set environment variables (fallback defaults for single-node)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", str(self._num_gpus))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")

        if not dist.is_initialized():
            logger.info(
                f"Initializing distributed: MASTER_ADDR={os.environ['MASTER_ADDR']}, "
                f"MASTER_PORT={os.environ['MASTER_PORT']}, "
                f"WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}"
            )
            dist.init_process_group(backend="nccl")
            logger.info("Distributed process group initialized")

        self._distributed_initialized = True

    def _wrap_model_fsdp(self, model: nn.Module) -> nn.Module:
        """
        Wrap model with FSDP for multi-GPU inference.

        Args:
            model: The model to wrap

        Returns:
            FSDP-wrapped model (or original model if not using multi-GPU)
        """
        if not self._use_fsdp or model is None:
            return model

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload

        # Map string sharding strategy to enum
        sharding_map = {
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        }
        sharding_strategy = sharding_map.get(
            self._fsdp_sharding_strategy, ShardingStrategy.NO_SHARD
        )

        # Mixed precision policy for inference
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        cpu_offload = CPUOffload(offload_params=True) if self._fsdp_cpu_offload else None

        logger.info(
            "Wrapping model with FSDP (sharding=%s, cpu_offload=%s)",
            self._fsdp_sharding_strategy,
            self._fsdp_cpu_offload,
        )

        wrapped_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

        return wrapped_model

    def initialize(self, device: torch.device) -> None:
        """
        Initialize models and sampler.

        Args:
            device: Target device for inference
        """
        self._device = device
        logger.info(f"Initializing FSDP engine on {device} (num_gpus={self._num_gpus})")

        # Initialize distributed if multi-GPU
        if self._use_fsdp:
            self._init_distributed()

        # Load model bundle if configured
        if self.config.model_path and self.config.pretrained_model_saved_path:
            model_cls = load_function(self.config.model_path)
            use_lora = bool(self.config.engine_kwargs.get("use_lora", False))
            model_kwargs = {
                "pretrained_path": self.config.pretrained_model_saved_path,
                "vae_saved_path": self.config.engine_kwargs.get("vae_saved_path"),
                "text_encoder_path": self.config.engine_kwargs.get("text_encoder_path"),
                "device": device,
                "use_lora": use_lora,
            }
            if use_lora:
                if "lora_rank" in self.config.engine_kwargs:
                    model_kwargs["lora_rank"] = self.config.engine_kwargs.get("lora_rank")
                if "lora_alpha" in self.config.engine_kwargs:
                    model_kwargs["lora_alpha"] = self.config.engine_kwargs.get("lora_alpha")
                if "lora_target_modules" in self.config.engine_kwargs:
                    model_kwargs["lora_target_modules"] = self.config.engine_kwargs.get("lora_target_modules")
            if self._fsdp_cpu_offload:
                try:
                    import inspect

                    if "skip_device_move" in inspect.signature(model_cls.__init__).parameters:
                        model_kwargs["skip_device_move"] = True
                except Exception:
                    # Best-effort: some bundles may not expose the signature cleanly.
                    pass

            self.model_bundle = model_cls(**model_kwargs)
            self.model = getattr(self.model_bundle, 'transformer', None)
            self.text_encoder = getattr(self.model_bundle, 'text_encoder', None)
            self.vae = getattr(self.model_bundle, 'vae', None)
            self.scheduler = getattr(self.model_bundle, 'scheduler', None)

            # Wrap model with FSDP if multi-GPU
            if self._use_fsdp and self.model is not None:
                self.model = self._wrap_model_fsdp(self.model)
                # Update model_bundle's transformer reference
                if hasattr(self.model_bundle, 'transformer'):
                    self.model_bundle.transformer = self.model

        # Create sampler via shared runner
        sampler_path = self.config.engine_kwargs.get("sampler_path")
        if sampler_path:
            sampler_kwargs = dict(self.config.engine_kwargs.get("sampler_kwargs", {}))
            self.sampler = sampler_runner.create_sampler(
                sampler_path=sampler_path,
                model=self.model,
                text_encoder=self.text_encoder,
                vae=self.vae,
                eta=self.config.eta,
                sde_type=self.config.sde_type,
                shift=self.config.shift,
                model_bundle=self.model_bundle,
                **sampler_kwargs,
            )

        self._is_initialized = True
        logger.info("FSDP engine initialized")

    def generate(self, request: RolloutRequest) -> RolloutOutput:
        """
        Generate samples with log probabilities.

        Args:
            request: RolloutRequest with prompts and generation parameters.

        Returns:
            RolloutOutput with trajectories and log_probs
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized")

        if self.sampler is None:
            raise RuntimeError("Sampler not loaded")

        self.wake_up()

        # Set seed
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(request.seed)

        # Use defaults if not specified
        num_inference_steps = request.num_inference_steps or self.config.num_inference_steps
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else self.config.guidance_scale
        height = request.height or self.config.height
        width = request.width or self.config.width
        num_frames = request.num_frames or self.config.num_frames

        # Also check config for sampling_adapter
        sampling_adapter = request.sampling_adapter
        if sampling_adapter is None:
            sampling_adapter = self.config.engine_kwargs.get("sampling_adapter")

        return sampler_runner.run_sample(
            model=self.model,
            sampler=self.sampler,
            sampling_adapter=sampling_adapter,
            prompts=request.prompts,
            prompt_embeds=request.prompt_embeds,
            pooled_prompt_embeds=request.pooled_prompt_embeds,
            encoder_attention_mask=request.encoder_attention_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            latents=request.latents,
            generator=generator,
            sde_indices=request.sde_indices,
            **request.kwargs,
        )

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts to embeddings."""
        return sampler_runner.encode_prompt(self.model_bundle, prompts, **kwargs)

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights from training.

        Args:
            state_dict: New model state dict
        """
        if self.model is None:
            logger.warning("No model to update weights")
            return

        if self._use_fsdp:
            # For FSDP-wrapped models, use FSDP's state dict handling
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            # Set up full state dict config for loading
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=False,
                rank0_only=False,
            )

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)

        logger.info("FSDP engine weights updated")

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        """Load a serialized state_dict from path and update model weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Invalid checkpoint for update_weights_from_path: {checkpoint_path}")
        self.update_weights(state_dict)

    def _safe_to_device(self, component: nn.Module, device: Union[str, torch.device], name: str) -> None:
        if component is None or not hasattr(component, "to"):
            return
        try:
            component.to(device)
        except Exception as e:
            logger.warning("Could not move %s to %s: %s", name, device, e)

    def _move_aux_components(self, device: Union[str, torch.device], include_transformer: bool) -> None:
        # Engine-level components
        if include_transformer and self.model is not None:
            self._safe_to_device(self.model, device, "model")
        if self.text_encoder is not None:
            self._safe_to_device(self.text_encoder, device, "text_encoder")
        if self.vae is not None:
            self._safe_to_device(self.vae, device, "vae")

        # Model bundle components
        if self.model_bundle is not None:
            for attr_name, component in self.model_bundle.iter_offloadable_modules(
                include_transformer=include_transformer
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"model_bundle.{attr_name}")

        # Sampler components (may hold references to text encoders)
        if self.sampler is not None:
            for attr_name, component in sampler_runner.iter_offloadable_modules(
                self.sampler, include_transformer=include_transformer
            ):
                if not include_transformer and component is self.model:
                    continue
                self._safe_to_device(component, device, f"sampler.{attr_name}")

    def sleep(self) -> None:
        """Offload models to CPU.

        Offloads all model components including:
        - Transformer model
        - All text encoders (SD3 has 3: CLIP1, CLIP2, T5)
        - VAE
        - Any components in model_bundle
        - Any components in sampler
        """
        if self._is_offloaded:
            logger.debug("FSDP engine already sleeping; skipping repeated sleep()")
            return

        if self._use_fsdp:
            # For FSDP models, avoid moving the FSDP-wrapped transformer.
            # Offload auxiliary components (text encoders/vae) only.
            self._move_aux_components("cpu", include_transformer=False)
            torch.cuda.empty_cache()
            self._is_offloaded = True
            logger.info(
                "FSDP multi-GPU mode: sleep complete (cpu_offload=%s, aux components moved to CPU)",
                self._fsdp_cpu_offload,
            )
            return

        # Non-FSDP: offload all components
        self._move_aux_components("cpu", include_transformer=True)
        torch.cuda.empty_cache()
        self._is_offloaded = True
        logger.info("FSDP engine entered sleep state")

    def wake_up(self) -> None:
        """Load models back to GPU.

        Loads all model components including:
        - Transformer model
        - All text encoders (SD3 has 3: CLIP1, CLIP2, T5)
        - VAE
        - Any components in model_bundle
        - Any components in sampler
        """
        if not self._is_offloaded:
            logger.debug("FSDP engine already awake; skipping repeated wake_up()")
            return

        if self._use_fsdp:
            # For FSDP models, device placement is managed by FSDP.
            if self._device is not None:
                self._move_aux_components(self._device, include_transformer=False)
            self._is_offloaded = False
            logger.info(
                "FSDP engine: wake_up complete (cpu_offload=%s, aux components moved to GPU)",
                self._fsdp_cpu_offload,
            )
            return

        self._move_aux_components(self._device, include_transformer=True)
        self._is_offloaded = False
        logger.info("FSDP engine exited sleep state")

    def get_capabilities(self) -> EngineCapabilities:
        model_type = getattr(self.model_bundle, "model_type", None)
        return EngineCapabilities(
            supports_logprob=True,
            supports_trajectory=True,
            supports_prompt_embeddings=True,
            supports_guidance_scale=(model_type != "hunyuan"),
            weight_load_mode="state_dict",
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents using VAE."""
        return sampler_runner.decode_latents(self.vae, latents)

    @property
    def supports_distributed(self) -> bool:
        """FSDP engine supports multi-GPU when configured with num_gpus > 1."""
        return self._use_fsdp

    @property
    def requires_external_service(self) -> bool:
        """FSDP engine does not require external service."""
        return False

    def cleanup(self) -> None:
        """
        Clean up resources including distributed process group.

        Should be called when the engine is no longer needed.
        """
        if self._distributed_initialized:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("Destroyed distributed process group")
            self._distributed_initialized = False

        # Clear model references
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.sampler = None
        self.model_bundle = None

        torch.cuda.empty_cache()
        logger.info("FSDP engine cleaned up")
