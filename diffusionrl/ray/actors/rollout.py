"""diffusionrl Rollout actor implementation (generation side)."""
import logging
from typing import Any, Dict, List, Optional

import ray
import torch

from diffusionrl.types.sampling import RolloutRequest, RolloutOutput
from diffusionrl.samplers.engine import (
    BaseRolloutEngine,
    EngineConfig,
    get_engine,
)
from diffusionrl.utils.weight_sync_checkpoint import wait_for_published_checkpoint

from .base import log_gpu_state, log_resource_ids, tensor_to_pil

logger = logging.getLogger(__name__)


@ray.remote
class RolloutActor:
    """
    Rollout Actor - Manages sampling and generation via Engine interface.

    This actor provides a unified interface for different rollout backends:
    - FSDP: Native PyTorch, DanceGRPO-aligned (single or multi-GPU)
    - FastVideo: Efficient video generation (supports multi-GPU SP/TP)
    - SGLang: Distributed rollout inference (future)

    All engines implement the same interface, making Ray scheduling consistent.

    GPU Allocation:
        GPU count is configured at actor creation via .options(num_gpus=N).
        - FSDP: num_gpus=1 (single GPU per actor, default)
        - FSDP multi-GPU: num_gpus>1 (uses FSDP wrapper for model parallelism)
        - FastVideo: num_gpus=sp_size (SP requires multiple GPUs per actor)

        FastVideo spawns MultiprocExecutor internally, which creates
        worker processes for each GPU. The Ray actor acts as the coordinator.

    Example:
        # Single GPU (FSDP)
        actor = RolloutActor.options(num_gpus=1).remote(rank=0, world_size=1)

        # Multi-GPU FSDP (4 GPUs)
        actor = RolloutActor.options(num_gpus=4).remote(
            rank=0, world_size=1, num_gpus_allocated=4
        )

        # Multi-GPU (FastVideo SP=4)
        actor = RolloutActor.options(num_gpus=4).remote(
            rank=0, world_size=1, num_gpus_allocated=4
        )
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        config: Optional[dict] = None,
        num_gpus_allocated: int = 1,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        base_gpu_id: int = 0,
        force_set_cuda_visible_devices: bool = False,
    ):
        """
        Initialize rollout actor.

        Args:
            rank: This actor's rank in the rollout group
            world_size: Total number of rollout actors
            config: Optional initial configuration
            num_gpus_allocated: Number of GPUs allocated to this actor
                               (must match Ray's num_gpus option)
            master_addr: Master node address for distributed (multi-GPU)
            master_port: Master node port for distributed (multi-GPU)
            base_gpu_id: Starting physical GPU ID (for Slime NOSET pattern).
                        When > 0, CUDA_VISIBLE_DEVICES is set manually to
                        [base_gpu_id, base_gpu_id+1, ..., base_gpu_id+num_gpus-1].
            force_set_cuda_visible_devices: Force manual CUDA_VISIBLE_DEVICES setup
                        even when base_gpu_id is 0 (needed for NOSET mode).
        """
        self.rank = rank
        self.world_size = world_size
        self.config = config or {}
        self.num_gpus_allocated = num_gpus_allocated
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_gpu_id = base_gpu_id
        self.force_set_cuda_visible_devices = bool(force_set_cuda_visible_devices)
        self.engine: Optional[BaseRolloutEngine] = None
        self._device = None
        self._warned_ignored_prompt_embedding_input = False

    def _log_resource_ids(self, tag: str) -> None:
        log_resource_ids(tag, self.rank)

    def _log_gpu_state(self, tag: str) -> None:
        offloaded = None
        if self.engine is not None:
            try:
                offloaded = self.engine.is_offloaded
            except Exception:
                offloaded = None
        log_gpu_state(tag, self.rank, device=self._device, offloaded=offloaded)

    def _setup_distributed_env(self) -> None:
        """
        Setup environment variables for multi-GPU distributed rollout.

        This is called before engine initialization when num_gpus_allocated > 1.
        When using the Slime NOSET pattern (base_gpu_id > 0), also sets
        CUDA_VISIBLE_DEVICES manually since Ray won't do it.
        """
        import os
        import socket

        if self.num_gpus_allocated <= 1:
            return

        # When using NOSET mode, manually set CUDA_VISIBLE_DEVICES.
        # base_gpu_id can be 0 when actor is assigned the first physical GPU group.
        if self.force_set_cuda_visible_devices or self.base_gpu_id > 0:
            gpu_range = ",".join(
                str(self.base_gpu_id + i) for i in range(self.num_gpus_allocated)
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_range
            logger.info(f"Rank {self.rank}: Set CUDA_VISIBLE_DEVICES={gpu_range}")

        # Get master address
        master_addr = self.master_addr
        if master_addr is None:
            master_addr = socket.gethostbyname(socket.gethostname())

        # Get master port
        master_port = self.master_port
        if master_port is None:
            # Find a free port
            import socket as sock
            with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
                s.bind(('', 0))
                master_port = s.getsockname()[1]

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(self.num_gpus_allocated)
        os.environ["RANK"] = "0"  # Single actor manages all GPUs
        os.environ["LOCAL_RANK"] = "0"

        logger.info(
            f"Rank {self.rank}: Distributed env setup - "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, "
            f"WORLD_SIZE={self.num_gpus_allocated}"
        )

    def _ensure_engine_ready_for_generate(self) -> None:
        """Ensure generation path always starts from an active engine state."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not self.engine.is_initialized:
            raise RuntimeError("Engine initialization incomplete.")
        self.engine.wake_up()

    def _prepare_engine_for_weight_update(self) -> None:
        """Ensure engine is active before updating weights."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        self.engine.wake_up()

    def _configure_sglang_ports(self, engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure each SGLang rollout actor gets a distinct port tuple.

        Without explicit ports, upstream ServerArgs may derive identical random
        ports across forked actor processes, which causes EADDRINUSE in colocate
        mode when multiple local schedulers start concurrently.
        """
        import os

        resolved = dict(engine_kwargs)
        raw_server_kwargs = resolved.get("server_kwargs")
        server_kwargs: Dict[str, Any] = (
            dict(raw_server_kwargs)
            if isinstance(raw_server_kwargs, dict)
            else {}
        )

        # Respect explicit top-level overrides if provided.
        for key in ("port", "scheduler_port", "master_port"):
            raw_value = resolved.get(key)
            if raw_value is None:
                continue
            try:
                server_kwargs.setdefault(key, int(raw_value))
            except (TypeError, ValueError):
                logger.warning(
                    "Rank %s: invalid sglang %s=%r, ignoring explicit value.",
                    self.rank,
                    key,
                    raw_value,
                )

        # Fill missing port fields deterministically from actor rank.
        try:
            base_port = int(
                resolved.get(
                    "sglang_port_base",
                    os.getenv("DIFFUSIONRL_SGLANG_PORT_BASE", 33000),
                )
            )
        except (TypeError, ValueError):
            base_port = 33000
        try:
            port_stride = int(
                resolved.get(
                    "sglang_port_stride",
                    os.getenv("DIFFUSIONRL_SGLANG_PORT_STRIDE", 100),
                )
            )
        except (TypeError, ValueError):
            port_stride = 100
        if port_stride < 32:
            port_stride = 32

        actor_index = int(self.rank)
        actor_base_port = base_port + actor_index * port_stride
        # Keep enough spacing for broker (port+1) and two extra dedicated ports.
        if actor_base_port > 65000:
            raise ValueError(
                "Configured SGLang actor port base/stride exceeds valid port range. "
                f"base={base_port}, stride={port_stride}, actor_rank={self.rank}"
            )

        server_kwargs.setdefault("port", actor_base_port)
        server_kwargs.setdefault("scheduler_port", actor_base_port + 11)
        server_kwargs.setdefault("master_port", actor_base_port + 23)

        resolved["server_kwargs"] = server_kwargs
        logger.info(
            "Rank %s: SGLang ports configured as port=%s scheduler_port=%s master_port=%s",
            self.rank,
            server_kwargs.get("port"),
            server_kwargs.get("scheduler_port"),
            server_kwargs.get("master_port"),
        )
        return resolved

    def init(self, engine_config: dict) -> None:
        """
        Initialize the rollout engine.

        Args:
            engine_config: Configuration for the engine including:
                - sampler_engine_type: "fsdp" or "sglang" (required)
                - sampler_path: Path to sampler class (required)
                - model_path: Path to model bundle class
                - pretrained_model_saved_path: Path to pretrained weights
                - num_inference_steps: Number of denoising steps
                - eta: SDE noise coefficient
                - sde_type: Type of SDE ("sde", "cps", "dance")
                - shift: Time shift for sigma schedule
                - engine_kwargs: Additional engine-specific arguments

        Raises:
            ValueError: If sampler_engine_type or sampler_path is not provided
        """
        logger.info(f"Rank {self.rank}: Initializing rollout actor (num_gpus={self.num_gpus_allocated})...")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed environment for multi-GPU
        self._setup_distributed_env()

        # Get sampler_engine_type (must be provided by caller, validated in arguments.py)
        sampler_engine_type = engine_config.get("sampler_engine_type")
        if sampler_engine_type is None:
            raise ValueError(
                "sampler_engine_type must be provided in engine_config. "
                "This should be set automatically via --model-type or explicitly via --sampler-engine-type"
            )

        # Get sampler_path (must be provided by caller, validated in arguments.py)
        sampler_path = engine_config.get("sampler_path")
        if sampler_path is None:
            raise ValueError(
                "sampler_path must be provided in engine_config. "
                "This should be set automatically via --model-type or explicitly via --sampler-path"
            )

        # Build EngineConfig
        engine_kwargs = dict(engine_config.get("engine_kwargs", {}))

        # Add sampler_path to engine_kwargs
        engine_kwargs["sampler_path"] = sampler_path

        # Pass base_gpu_id to engine for NOSET pattern
        engine_kwargs["base_gpu_id"] = self.base_gpu_id
        engine_kwargs["force_set_cuda_visible_devices"] = self.force_set_cuda_visible_devices

        # For multi-GPU FSDP, ensure num_gpus is set in engine_kwargs
        if sampler_engine_type == "fsdp" and self.num_gpus_allocated > 1:
            if "num_gpus" not in engine_kwargs:
                engine_kwargs["num_gpus"] = self.num_gpus_allocated
                logger.info(f"Rank {self.rank}: Setting FSDP num_gpus={self.num_gpus_allocated}")
        elif sampler_engine_type == "sglang":
            engine_kwargs = self._configure_sglang_ports(engine_kwargs)

        config = EngineConfig(
            model_path=engine_config.get("model_path", ""),
            pretrained_model_saved_path=engine_config.get("pretrained_model_saved_path", ""),
            num_inference_steps=engine_config.get("num_inference_steps", 50),
            eta=engine_config.get("eta", 1.0),
            sde_type=engine_config.get("sde_type", "sde"),
            shift=engine_config.get("shift", 3.0),
            guidance_scale=engine_config.get("guidance_scale", 7.5),
            height=engine_config.get("height", 256),
            width=engine_config.get("width", 256),
            num_frames=engine_config.get("num_frames", 16),
            engine_kwargs=engine_kwargs,
        )

        # Create engine
        try:
            engine_cls = get_engine(sampler_engine_type)
        except ValueError as exc:
            raise ValueError(f"Unknown sampler_engine_type: {sampler_engine_type}. {exc}") from exc
        self.engine = engine_cls(config)

        # Initialize engine
        self.engine.initialize(self._device)

        logger.info(f"Rank {self.rank}: Rollout actor initialized with {sampler_engine_type} engine")
        self._log_resource_ids("rollout_init")
        self._log_gpu_state("rollout_init")

    def generate(
        self,
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
        **kwargs,
    ) -> RolloutOutput:
        """
        Generate samples with log probabilities.

        Args:
            prompts: List of text prompts
            prompt_embeds: Deprecated (ignored)
            pooled_prompt_embeds: Deprecated (ignored)
            encoder_attention_mask: Attention mask
            text_ids: Text position IDs (for FLUX)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Output height
            width: Output width
            num_frames: Number of frames (for video)
            seed: Random seed for reproducibility
            decode_for_reward: If True, decode latents for reward computation
            **kwargs: Additional engine arguments

        Returns:
            RolloutOutput containing trajectories, log_probs, etc.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "RolloutActor.generate requires non-empty text prompts. "
                "Prompt-embedding-only input is no longer supported."
            )

        ignored_embedding_input = (
            prompt_embeds is not None
            or pooled_prompt_embeds is not None
            or encoder_attention_mask is not None
            or text_ids is not None
            or kwargs.get("negative_prompt_embeds") is not None
            or kwargs.get("negative_pooled_prompt_embeds") is not None
        )
        if ignored_embedding_input:
            if not self._warned_ignored_prompt_embedding_input:
                logger.warning(
                    "RolloutActor now uses prompt-only input; external embedding tensors are ignored. "
                    "Engines are responsible for per-request prompt encoding."
                )
                self._warned_ignored_prompt_embedding_input = True
            prompt_embeds = None
            pooled_prompt_embeds = None
            encoder_attention_mask = None
            text_ids = None
            kwargs.pop("negative_prompt_embeds", None)
            kwargs.pop("negative_pooled_prompt_embeds", None)

        self._ensure_engine_ready_for_generate()
        engine_caps = self.engine.get_capabilities_dict()
        if (
            guidance_scale is not None
            and not engine_caps.get("supports_guidance_scale", True)
            and float(guidance_scale) != float(getattr(self.engine.config, "guidance_scale", guidance_scale))
        ):
            raise ValueError(
                f"Engine {type(self.engine).__name__} does not support custom guidance_scale, "
                f"but guidance_scale={guidance_scale} was provided."
            )

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        self._log_gpu_state("inference_generate_start")
        # Generate
        output = self.engine.generate(
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
            seed=seed,
            text_ids=text_ids,
            return_decoded_for_reward=decode_for_reward,
            **kwargs,
        )

        # Attach capability snapshot for control-plane decisions/debugging.
        meta = dict(output.metadata or {})
        meta.setdefault("engine_capabilities", self.engine.get_capabilities_dict())
        output.metadata = meta

        # Decode latents for reward if requested
        if decode_for_reward:
            if not output.has_decoded_images:
                try:
                    decoded = self.engine.decode_latents(output.latents)
                    decoded_images = self._tensor_to_pil(decoded)
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
                    logger.warning(f"Failed to decode latents: {e}")

        # Move tensors to CPU for Ray serialization (RolloutManager has no GPU)
        output = output.to_device("cpu")
        self._log_gpu_state("inference_generate_end")
        return output

    def _tensor_to_pil(self, images: torch.Tensor) -> List[Any]:
        return tensor_to_pil(images)

    def generate_batch(
        self,
        requests: List[RolloutRequest],
    ) -> List[RolloutOutput]:
        """
        Generate samples for multiple requests.

        Args:
            requests: List of inference requests

        Returns:
            List of RolloutOutput for each request
        """
        outputs = []
        for request in requests:
            output = self.generate(
                prompts=request.prompts,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )
            outputs.append(output)
        return outputs

    def update_weights(self, state_dict_or_ref) -> None:
        """
        Update model weights from training actor.

        Args:
            state_dict_or_ref: Either ObjectRef containing state dict, or state dict directly.
                              Ray may auto-dereference ObjectRef when passing between actors.
        """
        if self.engine is None:
            logger.warning("No engine to update weights")
            return

        # Support path-based sync for direct checkpoint transfer.
        if isinstance(state_dict_or_ref, str):
            self.update_weights_from_path(state_dict_or_ref)
            return

        # Handle both ObjectRef and direct dict (Ray auto-dereferences when passing between actors)
        if isinstance(state_dict_or_ref, ray.ObjectRef):
            state_dict = ray.get(state_dict_or_ref)
        else:
            state_dict = state_dict_or_ref

        self._prepare_engine_for_weight_update()
        self.engine.update_weights(state_dict)
        logger.info("Rank %s: Weights updated", self.rank)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Update model weights from a shared checkpoint path."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return {"rank": int(self.rank), "checksum": None}

        wait_for_published_checkpoint(checkpoint_path)
        self._prepare_engine_for_weight_update()
        if hasattr(self.engine, "update_weights_from_path"):
            self.engine.update_weights_from_path(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.engine.update_weights(state_dict)
        logger.info("Rank %s: Weights updated from path %s", self.rank, checkpoint_path)
        checksum = None
        get_checksum_fn = getattr(self.engine, "get_last_weight_checksum", None)
        if callable(get_checksum_fn):
            try:
                raw_checksum = get_checksum_fn()
                if isinstance(raw_checksum, dict) and raw_checksum:
                    checksum = {str(k): str(v) for k, v in raw_checksum.items()}
            except Exception as exc:
                logger.warning(
                    "Rank %s: failed to query engine checksum after update: %s",
                    self.rank,
                    exc,
                )
        return {"rank": int(self.rank), "checksum": checksum}

    def sleep(self) -> None:
        """Put engine into sleep mode to release runtime resources."""
        if self.engine is not None:
            self.engine.sleep()
        logger.info(f"Rank {self.rank}: Engine entered sleep mode")
        self._log_gpu_state("inference_sleep")

    def wake_up(self) -> None:
        """Wake engine up for generation or weight update."""
        if self.engine is not None:
            self.engine.wake_up()
        logger.info(f"Rank {self.rank}: Engine wake_up complete")
        self._log_gpu_state("inference_wake_up")

    def health_check(self) -> bool:
        """Check if actor is healthy."""
        if self.engine is None:
            return False
        return self.engine.health_check()

    def is_offloaded(self) -> bool:
        """Check if actor is currently offloaded to CPU."""
        if self.engine is None:
            return False
        return self.engine.is_offloaded

    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.engine is not None:
            return self.engine.get_memory_info()
        return {}


__all__ = ["RolloutActor"]
