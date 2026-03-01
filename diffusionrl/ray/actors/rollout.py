"""diffusionrl Rollout actor implementation (generation side)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch

from diffusionrl.types.sampling import PromptEmbeddings, RolloutRequest, RolloutOutput
from diffusionrl.samplers.engine import (
    BaseRolloutEngine,
    DistributedWeightSyncCapable,
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
        self._transport_dtype: Optional[torch.dtype] = None
        self._transport_drop_decoded_videos: bool = False
        self._transport_log_payload_bytes: bool = False
        self._scheduler_endpoint: Optional[str] = None
        self._weight_update_target: str = f"actor_rank:{self.rank}"

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

    @staticmethod
    def _parse_transport_dtype(value: Any) -> Optional[torch.dtype]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "off", "disable", "disabled", "fp32", "float32"}:
            return None
        if text in {"fp16", "float16", "half"}:
            return torch.float16
        if text in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(
            f"Unsupported rollout transport dtype: {value!r}. "
            "Expected one of: fp32, fp16, bf16."
        )

    @staticmethod
    def _to_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return default

    @staticmethod
    def _normalize_scheduler_host(value: Any) -> str:
        host = str(value or "").strip()
        if not host:
            return "127.0.0.1"
        if host.startswith("tcp://"):
            host = host[len("tcp://") :]
        elif host.startswith("http://"):
            host = host[len("http://") :]
        elif host.startswith("https://"):
            host = host[len("https://") :]
        host = host.split("/", 1)[0].strip()
        if host == "localhost":
            return "127.0.0.1"
        return host

    @staticmethod
    def _format_scheduler_endpoint(host: str, port: int) -> str:
        host_text = str(host).strip()
        if ":" in host_text and not host_text.startswith("["):
            host_text = f"[{host_text}]"
        return f"tcp://{host_text}:{int(port)}"

    @classmethod
    def _parse_scheduler_endpoint(
        cls,
        value: Any,
    ) -> Optional[Tuple[str, int]]:
        if value is None:
            return None

        if isinstance(value, dict):
            endpoint_value = (
                value.get("scheduler_endpoint")
                or value.get("endpoint")
                or value.get("scheduler")
            )
            if endpoint_value is not None:
                return cls._parse_scheduler_endpoint(endpoint_value)

            host = value.get("host", value.get("scheduler_host"))
            port = value.get("scheduler_port", value.get("port"))
            if host is None or port is None:
                return None
            return cls._normalize_scheduler_host(host), int(port)

        text = str(value).strip()
        if not text:
            return None
        if text.startswith("tcp://"):
            text = text[len("tcp://") :]
        elif text.startswith("http://"):
            text = text[len("http://") :]
        elif text.startswith("https://"):
            text = text[len("https://") :]
        text = text.split("/", 1)[0].strip()

        if text.startswith("["):
            end = text.find("]")
            if end <= 0 or end + 1 >= len(text) or text[end + 1] != ":":
                raise ValueError(
                    f"Invalid scheduler endpoint {value!r}; expected tcp://[host]:port."
                )
            host = text[1:end]
            port_text = text[end + 2 :]
        else:
            if ":" not in text:
                raise ValueError(
                    f"Invalid scheduler endpoint {value!r}; expected host:port."
                )
            host, port_text = text.rsplit(":", 1)

        return cls._normalize_scheduler_host(host), int(port_text)

    @classmethod
    def _parse_scheduler_endpoint_pool(
        cls,
        value: Any,
    ) -> List[Tuple[str, int]]:
        if value is None:
            return []

        raw_items: Any = value
        if isinstance(raw_items, str):
            text = raw_items.strip()
            if not text:
                return []
            if text.startswith("["):
                raw_items = json.loads(text)
            else:
                raw_items = [part.strip() for part in text.split(",") if part.strip()]

        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, (list, tuple)):
            raise TypeError(
                "Scheduler endpoint pool must be list/tuple/string/dict, "
                f"got: {type(raw_items).__name__}"
            )

        parsed: List[Tuple[str, int]] = []
        for item in raw_items:
            endpoint = cls._parse_scheduler_endpoint(item)
            if endpoint is None:
                continue
            parsed.append(endpoint)
        return parsed

    def _extract_scheduler_endpoint_from_engine_kwargs(
        self,
        engine_kwargs: Dict[str, Any],
    ) -> Optional[str]:
        server_kwargs = engine_kwargs.get("server_kwargs")
        if not isinstance(server_kwargs, dict):
            return None
        host = server_kwargs.get("host")
        port = server_kwargs.get("scheduler_port")
        if host is None or port is None:
            return None
        try:
            host_text = self._normalize_scheduler_host(host)
            return self._format_scheduler_endpoint(host_text, int(port))
        except Exception:
            return None

    def _record_weight_update_target(
        self,
        *,
        sampler_engine_type: str,
        engine_kwargs: Dict[str, Any],
    ) -> None:
        if str(sampler_engine_type).lower() != "sglang":
            self._scheduler_endpoint = None
            self._weight_update_target = f"actor_rank:{self.rank}"
            return

        endpoint = self._extract_scheduler_endpoint_from_engine_kwargs(engine_kwargs)
        if self.engine is not None:
            server_args = getattr(self.engine, "_server_args", None)
            resolved = getattr(server_args, "scheduler_endpoint", None)
            if isinstance(resolved, str) and resolved.strip():
                endpoint = resolved.strip()

        self._scheduler_endpoint = endpoint
        if endpoint:
            self._weight_update_target = f"sglang_scheduler:{endpoint}"
        else:
            self._weight_update_target = f"sglang_rank:{self.rank}"

        logger.info(
            "Rank %s: resolved rollout weight-update target=%s",
            self.rank,
            self._weight_update_target,
        )

    def _configure_transport_options(self, engine_kwargs: Dict[str, Any]) -> None:
        raw_dtype = engine_kwargs.get(
            "rollout_transport_dtype",
            engine_kwargs.get("transport_dtype"),
        )
        self._transport_dtype = self._parse_transport_dtype(raw_dtype)
        self._transport_drop_decoded_videos = self._to_bool(
            engine_kwargs.get(
                "rollout_transport_drop_decoded_videos",
                engine_kwargs.get("transport_drop_decoded_videos"),
            ),
            default=False,
        )
        self._transport_log_payload_bytes = self._to_bool(
            engine_kwargs.get(
                "rollout_transport_log_payload_bytes",
                engine_kwargs.get("transport_log_payload_bytes"),
            ),
            default=False,
        )
        if (
            self._transport_dtype is not None
            or self._transport_drop_decoded_videos
            or self._transport_log_payload_bytes
        ):
            logger.info(
                "Rank %s: rollout transport optimization enabled "
                "(dtype=%s, drop_decoded_videos=%s, log_payload_bytes=%s)",
                self.rank,
                self._transport_dtype,
                self._transport_drop_decoded_videos,
                self._transport_log_payload_bytes,
            )

    @staticmethod
    def _estimate_tree_tensor_bytes(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(RolloutActor._estimate_tree_tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(RolloutActor._estimate_tree_tensor_bytes(v) for v in value)
        return 0

    def _estimate_rollout_output_bytes(self, output: RolloutOutput) -> int:
        total = 0
        for tensor in (output.latents, output.timesteps, output.trajectories, output.step_indices):
            if torch.is_tensor(tensor):
                total += int(tensor.numel() * tensor.element_size())
        if output.log_probs is not None:
            for value in output.log_probs.data.values():
                if torch.is_tensor(value):
                    total += int(value.numel() * value.element_size())
        if output.embeddings is not None:
            for value in output.embeddings.to_dict().values():
                if torch.is_tensor(value):
                    total += int(value.numel() * value.element_size())
        total += self._estimate_tree_tensor_bytes(output.metadata)
        return total

    def _maybe_cast_tensor_for_transport(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        target_dtype = self._transport_dtype
        if target_dtype is None:
            return value
        if not torch.is_tensor(value):
            return value
        if value.is_floating_point() and value.dtype != target_dtype:
            return value.to(dtype=target_dtype)
        return value

    def _cast_metadata_tensors_for_transport(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return self._maybe_cast_tensor_for_transport(value)
        if isinstance(value, dict):
            return {k: self._cast_metadata_tensors_for_transport(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_metadata_tensors_for_transport(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._cast_metadata_tensors_for_transport(v) for v in value)
        return value

    def _optimize_output_for_transport(self, output: RolloutOutput) -> RolloutOutput:
        if (
            self._transport_dtype is None
            and not self._transport_drop_decoded_videos
            and not self._transport_log_payload_bytes
        ):
            return output

        bytes_before = self._estimate_rollout_output_bytes(output) if self._transport_log_payload_bytes else 0

        metadata = dict(output.metadata or {})
        if self._transport_drop_decoded_videos and isinstance(metadata.get("decoded_videos"), torch.Tensor):
            decoded = metadata.pop("decoded_videos")
            metadata["decoded_videos_dropped"] = True
            metadata["decoded_videos_shape"] = tuple(int(v) for v in decoded.shape)
            metadata["decoded_videos_dtype"] = str(decoded.dtype)

        log_probs = output.log_probs
        if log_probs is not None and self._transport_dtype is not None:
            log_probs = type(log_probs).from_dict(
                {
                    int(step): self._maybe_cast_tensor_for_transport(value)
                    for step, value in log_probs.data.items()
                }
            )

        embeddings = output.embeddings
        if embeddings is not None and self._transport_dtype is not None:
            embeddings = PromptEmbeddings(
                prompt_embeds=self._maybe_cast_tensor_for_transport(embeddings.prompt_embeds),
                pooled_prompt_embeds=self._maybe_cast_tensor_for_transport(
                    embeddings.pooled_prompt_embeds
                ),
                encoder_attention_mask=self._maybe_cast_tensor_for_transport(
                    embeddings.encoder_attention_mask
                ),
                negative_prompt_embeds=self._maybe_cast_tensor_for_transport(
                    embeddings.negative_prompt_embeds
                ),
                negative_pooled_prompt_embeds=self._maybe_cast_tensor_for_transport(
                    embeddings.negative_pooled_prompt_embeds
                ),
                text_ids=embeddings.text_ids,
                image_ids=embeddings.image_ids,
            )

        if self._transport_dtype is not None:
            metadata = self._cast_metadata_tensors_for_transport(metadata)

        optimized = RolloutOutput(
            latents=self._maybe_cast_tensor_for_transport(output.latents),
            timesteps=output.timesteps,
            trajectories=self._maybe_cast_tensor_for_transport(output.trajectories),
            log_probs=log_probs,
            embeddings=embeddings,
            decoded_images=output.decoded_images,
            metadata=metadata,
            step_indices=output.step_indices,
        )

        if self._transport_log_payload_bytes:
            bytes_after = self._estimate_rollout_output_bytes(optimized)
            delta = bytes_before - bytes_after
            ratio = (float(bytes_after) / float(bytes_before)) if bytes_before > 0 else 1.0
            logger.info(
                "Rank %s: rollout transport bytes before=%d after=%d saved=%d ratio=%.4f",
                self.rank,
                bytes_before,
                bytes_after,
                delta,
                ratio,
            )

        return optimized

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
        for key in ("host", "port", "scheduler_port", "master_port"):
            raw_value = resolved.get(key)
            if raw_value is None:
                continue
            try:
                if key == "host":
                    server_kwargs.setdefault(key, str(raw_value))
                else:
                    server_kwargs.setdefault(key, int(raw_value))
            except (TypeError, ValueError):
                logger.warning(
                    "Rank %s: invalid sglang %s=%r, ignoring explicit value.",
                    self.rank,
                    key,
                    raw_value,
                )

        local_mode = self._to_bool(resolved.get("local_mode", True), default=True)

        raw_endpoint_pool = resolved.get(
            "remote_scheduler_endpoints",
            resolved.get(
                "scheduler_endpoints",
                resolved.get("sglang_scheduler_endpoints"),
            ),
        )
        endpoint_pool = self._parse_scheduler_endpoint_pool(raw_endpoint_pool)
        if endpoint_pool and local_mode:
            logger.warning(
                "Rank %s: remote scheduler endpoints were provided while local_mode=True; "
                "forcing local_mode=False.",
                self.rank,
            )
            local_mode = False
            resolved["local_mode"] = False

        if not local_mode:
            selected_endpoint: Optional[Tuple[str, int]] = None
            if endpoint_pool:
                selected_endpoint = endpoint_pool[int(self.rank) % len(endpoint_pool)]
            else:
                for key in (
                    "remote_scheduler_endpoint",
                    "scheduler_endpoint",
                    "sglang_scheduler_endpoint",
                ):
                    if resolved.get(key) is None:
                        continue
                    selected_endpoint = self._parse_scheduler_endpoint(resolved.get(key))
                    break
                if selected_endpoint is None:
                    selected_endpoint = self._parse_scheduler_endpoint(server_kwargs)

            if selected_endpoint is None:
                raise ValueError(
                    "SGLang remote scheduler mode (local_mode=false) requires explicit "
                    "scheduler host/port. Provide one of: "
                    "remote_scheduler_endpoints / scheduler_endpoints / "
                    "remote_scheduler_endpoint / server_kwargs{host,scheduler_port}."
                )

            host, scheduler_port = selected_endpoint
            server_kwargs["host"] = str(host)
            server_kwargs["scheduler_port"] = int(scheduler_port)
            resolved["server_kwargs"] = server_kwargs
            logger.info(
                "Rank %s: SGLang remote scheduler configured as %s",
                self.rank,
                self._format_scheduler_endpoint(host, scheduler_port),
            )
            return resolved

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
        self._configure_transport_options(engine_kwargs)

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
        self._record_weight_update_target(
            sampler_engine_type=str(sampler_engine_type),
            engine_kwargs=engine_kwargs,
        )

        logger.info(f"Rank {self.rank}: Rollout actor initialized with {sampler_engine_type} engine")
        self._log_resource_ids("rollout_init")
        self._log_gpu_state("rollout_init")

    def generate(self, request: RolloutRequest) -> RolloutOutput:
        """
        Generate samples with log probabilities.

        Args:
            request: RolloutRequest with prompts and generation parameters.

        Returns:
            RolloutOutput containing trajectories, log_probs, etc.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not isinstance(request.prompts, list) or len(request.prompts) == 0:
            raise ValueError(
                "RolloutActor.generate requires non-empty text prompts. "
                "Prompt-embedding-only input is no longer supported."
            )

        ignored_embedding_input = (
            request.prompt_embeds is not None
            or request.pooled_prompt_embeds is not None
            or request.encoder_attention_mask is not None
            or request.text_ids is not None
            or request.kwargs.get("negative_prompt_embeds") is not None
            or request.kwargs.get("negative_pooled_prompt_embeds") is not None
        )
        if ignored_embedding_input:
            if not self._warned_ignored_prompt_embedding_input:
                logger.warning(
                    "RolloutActor now uses prompt-only input; external embedding tensors are ignored. "
                    "Engines are responsible for per-request prompt encoding."
                )
                self._warned_ignored_prompt_embedding_input = True
            # Clear embedding fields before passing to engine
            request = RolloutRequest(
                prompts=request.prompts,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                eta=request.eta,
                sde_type=request.sde_type,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                seed=request.seed,
                latents=request.latents,
                sde_indices=request.sde_indices,
                decode_for_reward=request.decode_for_reward,
                sampling_adapter=request.sampling_adapter,
                return_trajectories=request.return_trajectories,
                return_log_probs=request.return_log_probs,
                kwargs={k: v for k, v in request.kwargs.items()
                        if k not in ("negative_prompt_embeds", "negative_pooled_prompt_embeds")},
            )

        self._ensure_engine_ready_for_generate()
        engine_caps = self.engine.get_capabilities_dict()
        if (
            request.guidance_scale is not None
            and not engine_caps.get("supports_guidance_scale", True)
            and float(request.guidance_scale) != float(getattr(self.engine.config, "guidance_scale", request.guidance_scale))
        ):
            raise ValueError(
                f"Engine {type(self.engine).__name__} does not support custom guidance_scale, "
                f"but guidance_scale={request.guidance_scale} was provided."
            )

        self._log_gpu_state("inference_generate_start")
        # Generate
        output = self.engine.generate(request)

        # Attach capability snapshot for control-plane decisions/debugging.
        meta = dict(output.metadata or {})
        meta.setdefault("engine_capabilities", self.engine.get_capabilities_dict())
        output.metadata = meta

        # Decode latents for reward if requested
        if request.decode_for_reward:
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

        output = self._optimize_output_for_transport(output)

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
        return [self.generate(req) for req in requests]

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts via engine-side prompt encoder for rollout fallback wiring."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("encode_prompt requires non-empty prompts list.")

        fn = getattr(self.engine, "encode_prompt", None)
        if not callable(fn):
            raise NotImplementedError(
                f"Engine {type(self.engine).__name__} does not support encode_prompt()."
            )

        self._ensure_engine_ready_for_generate()
        encoded = fn(list(prompts), **kwargs)
        if not isinstance(encoded, dict):
            raise TypeError(
                f"Engine encode_prompt() must return dict, got {type(encoded).__name__}."
            )

        result: Dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                result[str(key)] = value.detach().cpu()
        return result

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

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Update weights using serialized tensor payload."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        if not isinstance(self.engine, DistributedWeightSyncCapable):
            raise NotImplementedError(
                f"{type(self.engine).__name__} does not support update_weights_from_tensor. "
                "Only engines implementing DistributedWeightSyncCapable support tensor/distributed weight sync."
            )
        self._prepare_engine_for_weight_update()
        self.engine.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        """Initialize custom distributed weight-update group in engine workers."""
        if self.engine is None:
            logger.warning("No engine to initialize weight update group")
            return
        if not isinstance(self.engine, DistributedWeightSyncCapable):
            raise NotImplementedError(
                f"{type(self.engine).__name__} does not support init_weights_update_group. "
                "Only engines implementing DistributedWeightSyncCapable support tensor/distributed weight sync."
            )
        self._prepare_engine_for_weight_update()
        self.engine.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        """Destroy custom distributed weight-update group in engine workers."""
        if self.engine is None:
            logger.warning("No engine to destroy weight update group")
            return
        if not isinstance(self.engine, DistributedWeightSyncCapable):
            raise NotImplementedError(
                f"{type(self.engine).__name__} does not support destroy_weights_update_group. "
                "Only engines implementing DistributedWeightSyncCapable support tensor/distributed weight sync."
            )
        self.engine.destroy_weights_update_group(group_name=group_name)

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        """Receive weights from custom distributed broadcast group."""
        if self.engine is None:
            logger.warning("No engine to update weights")
            return
        if not isinstance(self.engine, DistributedWeightSyncCapable):
            raise NotImplementedError(
                f"{type(self.engine).__name__} does not support update_weights_from_distributed. "
                "Only engines implementing DistributedWeightSyncCapable support tensor/distributed weight sync."
            )
        self._prepare_engine_for_weight_update()
        self.engine.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            target_modules=target_modules,
            flush_cache=flush_cache,
        )

    def get_num_gpus_allocated(self) -> int:
        """Return physical GPU count allocated to this rollout actor."""
        return int(self.num_gpus_allocated)

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

    def get_weight_update_target(self) -> Dict[str, Any]:
        """Return logical rollout weight-update target for dedup routing."""
        return {
            "rank": int(self.rank),
            "target": str(self._weight_update_target),
            "scheduler_endpoint": self._scheduler_endpoint,
        }

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
