"""diffusionrl Rollout actor implementation (generation side)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
from diffusionrl.reward.actor_local import ActorLocalRewardPrecompute
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.types.batch_ops import concat_columnar_values
from diffusionrl.types.engine import (
    EngineConfig,
    normalize_engine_type,
    uses_dedicated_rollout_engine,
)
from diffusionrl.types.sampling import (
    LogProbData,
    PromptEmbeddings,
    RolloutRequest,
    RolloutSamples,
)
from diffusionrl.samplers.engine import (
    BaseRolloutEngine,
    DistributedWeightSyncCapable,
    get_engine,
)
from diffusionrl.distributed.weight_sync_checkpoint import wait_for_published_checkpoint

from .actor_base import log_gpu_state, log_resource_ids
from .sampling_runtime import finalize_sampling_output

logger = logging.getLogger(__name__)


@ray.remote
class RolloutActor:
    """
    Rollout Actor - Manages sampling and generation via Engine interface.

    This actor hosts dedicated rollout-side services only:
    - SGLang: Distributed rollout inference service

    Direct-sampling engines (for example the default FSDP sampler path) run on
    TrainingActor and should never be instantiated here.

    GPU Allocation:
        GPU count is configured at actor creation via .options(num_gpus=N).
        - FSDP: num_gpus=1 (single GPU per actor, default)
        - FSDP multi-GPU: num_gpus>1 (uses FSDP wrapper for model parallelism)
    Example:
        actor = RolloutActor.options(num_gpus=1).remote(
            rank=0, world_size=1, num_gpus_allocated=1
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
        self._rollout_batch_size: Optional[int] = None
        self.num_gpus_allocated = num_gpus_allocated
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_gpu_id = base_gpu_id
        self.force_set_cuda_visible_devices = bool(force_set_cuda_visible_devices)
        self.engine: Optional[BaseRolloutEngine] = None
        self._device = None
        self._transport_dtype: Optional[torch.dtype] = None
        self._transport_drop_decoded_videos: bool = False
        self._transport_log_payload_bytes: bool = False
        self._scheduler_endpoint: Optional[str] = None
        self._weight_update_target: str = f"actor_rank:{self.rank}"
        self._reward_schema: Optional[RewardSchema] = None
        self._local_reward_runtime: Optional[ActorLocalRewardPrecompute] = None

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

    def _uses_rollout_local_reward(self) -> bool:
        return bool(self._reward_schema is not None and self._reward_schema.uses_sampling_actor_execution)

    def _ensure_local_reward_runtime(self) -> ActorLocalRewardPrecompute:
        if not self._uses_rollout_local_reward():
            raise RuntimeError("Local reward runtime requested but reward_location!='sampling_actor'.")
        if self._local_reward_runtime is None:
            self._local_reward_runtime = ActorLocalRewardPrecompute(self._reward_schema)
        return self._local_reward_runtime

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

    def _estimate_rollout_output_bytes(self, output: RolloutSamples) -> int:
        total = 0
        trajectories = output.aux.get("trajectories")
        step_indices = output.aux.get("step_indices")
        log_probs = output.aux.get("log_probs")
        embeddings = output.aux.get("embeddings")
        metadata = output.aux.get("metadata")
        for tensor in (output.latents, output.timesteps, trajectories, step_indices):
            if torch.is_tensor(tensor):
                total += int(tensor.numel() * tensor.element_size())
        if log_probs is not None:
            for value in log_probs.data.values():
                if torch.is_tensor(value):
                    total += int(value.numel() * value.element_size())
        if embeddings is not None:
            for value in embeddings.to_dict().values():
                if torch.is_tensor(value):
                    total += int(value.numel() * value.element_size())
        total += self._estimate_tree_tensor_bytes(metadata)
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

    def _optimize_output_for_transport(self, output: RolloutSamples) -> RolloutSamples:
        if (
            self._transport_dtype is None
            and not self._transport_drop_decoded_videos
            and not self._transport_log_payload_bytes
        ):
            return output

        bytes_before = self._estimate_rollout_output_bytes(output) if self._transport_log_payload_bytes else 0

        raw_metadata = output.aux.get("metadata")
        metadata = dict(raw_metadata or {})
        if self._transport_drop_decoded_videos and isinstance(metadata.get("decoded_videos"), torch.Tensor):
            decoded = metadata.pop("decoded_videos")
            metadata["decoded_videos_dropped"] = True
            metadata["decoded_videos_shape"] = tuple(int(v) for v in decoded.shape)
            metadata["decoded_videos_dtype"] = str(decoded.dtype)

        log_probs = output.aux.get("log_probs")
        if log_probs is not None and self._transport_dtype is not None:
            log_probs = type(log_probs).from_dict(
                {
                    int(step): self._maybe_cast_tensor_for_transport(value)
                    for step, value in log_probs.data.items()
                }
            )

        embeddings = output.aux.get("embeddings")
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

        optimized = RolloutSamples(
            latents=self._maybe_cast_tensor_for_transport(output.latents),
            timesteps=output.timesteps,
            aux={
                **dict(output.aux),
                "trajectories": self._maybe_cast_tensor_for_transport(output.aux.get("trajectories")),
                "log_probs": log_probs,
                "embeddings": embeddings,
                "decoded_images": output.aux.get("decoded_images"),
                "metadata": metadata,
                "step_indices": output.aux.get("step_indices"),
            },
            meta=output.meta,
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

    def init(self, config: dict) -> None:
        """
        Initialize the rollout actor and underlying engine.

        Args:
            config: Rollout actor init config including:
                - engine_runtime_config: final rollout-engine runtime payload
                - reward_config: rollout-side reward execution config

        Raises:
            ValueError: If required sections or fields are not provided
        """
        logger.info(f"Rank {self.rank}: Initializing rollout actor (num_gpus={self.num_gpus_allocated})...")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed environment for multi-GPU
        self._setup_distributed_env()

        if not isinstance(config, dict):
            raise ValueError(f"rollout actor init config must be a dict, got: {type(config).__name__}")
        engine_runtime_config = config.get("engine_runtime_config")
        if not isinstance(engine_runtime_config, dict):
            raise ValueError(
                "rollout actor init config must provide engine_runtime_config as dict. "
                f"Got: {type(engine_runtime_config).__name__}"
            )

        # Get sampler_engine_type (must be provided by caller, validated in arguments.py)
        sampler_engine_type = normalize_engine_type(engine_runtime_config.get("sampler_engine_type"))
        if not sampler_engine_type:
            raise ValueError(
                "sampler_engine_type must be provided in engine_runtime_config. "
                "This should be resolved from rollout.topology.service_engine before actor init."
            )
        if not uses_dedicated_rollout_engine(sampler_engine_type):
            raise ValueError(
                f"sampler_engine_type={sampler_engine_type!r} does not use rollout actors. "
                "Instantiate this sampler on TrainingActor instead."
            )

        sampler_path = engine_runtime_config.get("sampler_path")
        if sampler_path is None:
            raise ValueError(
                "sampler_path must be provided in engine_runtime_config. "
                "This should be resolved from sampling.sampler_path before actor init."
            )

        reward_config = config.get("reward_config", {})
        if not isinstance(reward_config, dict):
            raise ValueError(
                "reward_config must be provided in rollout actor init config as dict. "
                f"Got: {type(reward_config).__name__}"
            )
        self._reward_schema = RewardSchema(**reward_config)

        # Build EngineConfig
        engine_kwargs = dict(engine_runtime_config.get("engine_kwargs", {}))
        self._configure_transport_options(engine_kwargs)

        # Add sampler_path to engine_kwargs
        engine_kwargs["sampler_path"] = sampler_path

        # Pass base_gpu_id to engine for NOSET pattern
        engine_kwargs["base_gpu_id"] = self.base_gpu_id
        engine_kwargs["force_set_cuda_visible_devices"] = self.force_set_cuda_visible_devices

        if sampler_engine_type == "sglang":
            engine_kwargs = self._configure_sglang_ports(engine_kwargs)

        resolved_engine_config = EngineConfig(
            model_path=engine_runtime_config.get("model_path", ""),
            pretrained_model_saved_path=engine_runtime_config.get("pretrained_model_saved_path", ""),
            num_inference_steps=int(engine_runtime_config.get("num_inference_steps", 50)),
            eta=float(engine_runtime_config.get("eta", 1.0)),
            sde_type=str(engine_runtime_config.get("sde_type", "flow")),
            shift=float(engine_runtime_config.get("shift", 1.0)),
            guidance_scale=float(engine_runtime_config.get("guidance_scale", 7.5)),
            height=int(engine_runtime_config.get("height", 256)),
            width=int(engine_runtime_config.get("width", 256)),
            num_frames=int(engine_runtime_config.get("num_frames", 16)),
            engine_kwargs=engine_kwargs,
        )
        rollout_batch_size = engine_runtime_config.get("rollout_batch_size")
        self._rollout_batch_size = int(rollout_batch_size) if rollout_batch_size is not None else None

        # Create engine
        try:
            engine_cls = get_engine(sampler_engine_type)
        except ValueError as exc:
            raise ValueError(f"Unknown sampler_engine_type: {sampler_engine_type}. {exc}") from exc
        self.engine = engine_cls(resolved_engine_config)

        # Initialize engine
        self.engine.initialize(self._device)
        self._record_weight_update_target(
            sampler_engine_type=str(sampler_engine_type),
            engine_kwargs=engine_kwargs,
        )

        logger.info(
            "Rank %s: Rollout actor initialized with %s engine%s",
            self.rank,
            sampler_engine_type,
            f" (rollout_batch_size={self._rollout_batch_size})" if self._rollout_batch_size else "",
        )
        self._log_resource_ids("rollout_init")
        self._log_gpu_state("rollout_init")

    def generate(self, request: RolloutRequest) -> RolloutSamples:
        """
        Generate samples with log probabilities.

        Args:
            request: RolloutRequest with prompts and generation parameters.

        Returns:
            RolloutSamples containing trajectories, log_probs, etc.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        if not isinstance(request.prompts, list) or len(request.prompts) == 0:
            raise ValueError(
                "RolloutActor.generate requires non-empty text prompts. "
                "Prompt-embedding-only input is no longer supported."
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
        batch_size = self._rollout_batch_size
        n_prompts = len(request.prompts)
        if batch_size and n_prompts > batch_size:
            outputs = []
            for i in range(0, n_prompts, batch_size):
                sub_request = request.slice_prompts(i, min(i + batch_size, n_prompts))
                outputs.append(self.engine.generate(sub_request))
            output = self._merge_rollout_samples(outputs)
        else:
            output = self.engine.generate(request)
        output = finalize_sampling_output(
            output=output,
            request=request,
            host_label="rollout engine",
            decode_latents_fn=self.engine.decode_latents,
            metadata_defaults={
                "engine_capabilities": engine_caps,
            },
            local_reward_attach_fn=(
                lambda current_output: self._ensure_local_reward_runtime().attach_to_output(
                    output=current_output,
                    prompts=list(request.prompts),
                    prompt_ids=request.meta.get("prompt_ids"),
                    sample_ids=request.meta.get("sample_ids"),
                    group_ids=request.meta.get("group_ids"),
                    prompt_metadata=request.meta.get("prompt_metadata"),
                    keep_reward_media_for_driver=bool(
                        request.sampling.get("keep_reward_media_for_driver", False)
                    ),
                    samples_per_prompt=max(
                        1, int(request.sampling.get("samples_per_prompt", 1))
                    ),
                )
            ) if self._uses_rollout_local_reward() else None,
            transport_optimize_fn=self._optimize_output_for_transport,
            move_output_to_cpu=True,
        )
        self._log_gpu_state("inference_generate_end")
        return output

    @staticmethod
    def _merge_rollout_samples(outputs: List[RolloutSamples]) -> RolloutSamples:
        """Merge multiple rollout sub-batches into a single batch-shaped payload."""
        if not outputs:
            raise ValueError("Cannot merge empty rollout outputs.")
        if len(outputs) == 1:
            return outputs[0]

        batch_sizes = [output.batch_size for output in outputs]
        latents = torch.cat([output.latents for output in outputs], dim=0)
        timesteps = outputs[0].timesteps

        aux: Dict[str, Any] = {}

        trajectories = [output.aux.get("trajectories") for output in outputs]
        if all(value is not None for value in trajectories):
            aux["trajectories"] = torch.cat(trajectories, dim=0)

        log_probs = [output.aux.get("log_probs") for output in outputs]
        if all(value is not None for value in log_probs):
            merged_log_probs: Dict[int, torch.Tensor] = {}
            all_indices = sorted(set().union(*(value.sde_indices for value in log_probs)))
            for step_idx in all_indices:
                merged_log_probs[step_idx] = torch.cat(
                    [value[step_idx] for value in log_probs],
                    dim=0,
                )
            aux["log_probs"] = LogProbData.from_dict(merged_log_probs)

        embeddings = [output.aux.get("embeddings") for output in outputs]
        if all(value is not None for value in embeddings):
            def _cat_embed(attr: str):
                values = [getattr(value, attr) for value in embeddings]
                if all(v is not None for v in values):
                    return torch.cat(values, dim=0)
                return None

            aux["embeddings"] = PromptEmbeddings(
                prompt_embeds=_cat_embed("prompt_embeds"),
                pooled_prompt_embeds=_cat_embed("pooled_prompt_embeds"),
                encoder_attention_mask=_cat_embed("encoder_attention_mask"),
                negative_prompt_embeds=_cat_embed("negative_prompt_embeds"),
                negative_pooled_prompt_embeds=_cat_embed("negative_pooled_prompt_embeds"),
                text_ids=_cat_embed("text_ids"),
                image_ids=_cat_embed("image_ids"),
            )

        decoded_images = [output.aux.get("decoded_images") for output in outputs]
        if any(value is not None for value in decoded_images):
            merged_decoded_images = []
            for value in decoded_images:
                if value:
                    merged_decoded_images.extend(list(value))
            if merged_decoded_images:
                aux["decoded_images"] = merged_decoded_images

        metadata_values = [
            dict(output.aux.get("metadata") or {})
            for output in outputs
            if isinstance(output.aux.get("metadata"), dict)
        ]
        if metadata_values:
            merged_metadata = dict(metadata_values[0])
            decoded_videos = [value.get("decoded_videos") for value in metadata_values]
            if all(torch.is_tensor(value) for value in decoded_videos):
                merged_metadata["decoded_videos"] = torch.cat(decoded_videos, dim=0)
            aux["metadata"] = merged_metadata

        step_indices_values = [output.aux.get("step_indices") for output in outputs]
        if all(value is not None for value in step_indices_values):
            aux["step_indices"] = step_indices_values[0]

        handled_keys = {
            "trajectories",
            "log_probs",
            "embeddings",
            "decoded_images",
            "metadata",
            "step_indices",
        }
        other_keys = sorted(set().union(*(output.aux.keys() for output in outputs)) - handled_keys)
        for key in other_keys:
            merged_value = concat_columnar_values(
                [output.aux.get(key) for output in outputs],
                batch_sizes=batch_sizes,
            )
            if merged_value is not None:
                aux[key] = merged_value

        merged_meta = concat_columnar_values([output.meta for output in outputs], batch_sizes=batch_sizes) or {}
        return RolloutSamples(
            latents=latents,
            timesteps=timesteps,
            aux=aux,
            meta=merged_meta,
        )

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

    def set_lora_from_tensors(self, adapter_name: str, lora_tensors: dict) -> None:
        """Set LoRA adapter weights on rollout engines from serialized tensors."""
        if self.engine is None:
            logger.warning("No engine to set LoRA tensors")
            return
        if not isinstance(self.engine, DistributedWeightSyncCapable):
            raise NotImplementedError(
                f"{type(self.engine).__name__} does not support set_lora_from_tensors. "
                "Only engines implementing DistributedWeightSyncCapable support tensor/distributed weight sync."
            )
        self.engine.set_lora_from_tensors(adapter_name, lora_tensors)

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
        if self._local_reward_runtime is not None:
            self._local_reward_runtime.offload()
        logger.info(f"Rank {self.rank}: Engine entered sleep mode")
        self._log_gpu_state("inference_sleep")

    def wake_up(self) -> None:
        """Wake engine up for generation or weight update."""
        if self.engine is not None:
            self.engine.wake_up()
        if self._local_reward_runtime is not None:
            self._local_reward_runtime.onload()
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
