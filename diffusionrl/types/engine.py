"""Shared engine contracts and rollout-engine type helpers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from diffusionrl.sde.rules import normalize_sde_type

ROLLOUT_ENGINE_TYPES: Set[str] = {
    "sglang",
}


def normalize_engine_type(name: Any) -> str:
    """Normalize rollout engine type text."""

    return str(name or "").strip().lower()


def uses_dedicated_rollout_engine(name: Any) -> bool:
    """Return whether the engine runs as a dedicated rollout-side service."""

    return normalize_engine_type(name) in ROLLOUT_ENGINE_TYPES


@dataclass
class EngineConfig:
    """Configuration for rollout-side inference engines.

    All necessary settings are explicit typed fields.  The ``engine_kwargs``
    dict is retained only as an escape hatch for rare / advanced ServerArgs
    overrides that are not worth promoting to first-class fields.
    """

    # --- Model ---
    model_dotpath: str = ""
    pretrained_model_ckpt_path: str = ""
    sampler_dotpath: Optional[str] = None

    # --- Sampling ---
    num_inference_steps: int = 50
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    guidance_scale: float = 7.5
    fps: int = 24

    height: int = 256
    width: int = 256
    num_frames: int = 16

    # --- Parallelism & GPU ---
    num_gpus: int = 1
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None

    # --- SGLang engine behaviour ---
    local_mode: bool = True
    logprob_source: str = "replay"
    verify_weight_checksum: bool = True
    require_memory_api: bool = False
    disable_autocast: bool = False

    # --- Weight sync ---
    target_modules: Optional[Tuple[str, ...]] = None
    weight_sync_dir: Optional[str] = None

    # --- LoRA ---
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: Optional[Tuple[str, ...]] = None
    lora_merge_mode: Optional[str] = None

    # --- Auxiliary models ---
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    prompt_encoder_dtype: Optional[str] = None

    # --- SGLang network (usually auto-configured by RolloutActor) ---
    host: Optional[str] = None
    port: Optional[int] = None
    scheduler_port: Optional[int] = None
    sglang_master_port: Optional[int] = None
    sglang_port_base: int = 33000
    sglang_port_stride: int = 100

    # --- Escape hatch for rare / advanced ServerArgs overrides ---
    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.sde_type = normalize_sde_type(self.sde_type)
        self.logprob_source = str(self.logprob_source or "replay").strip().lower()
        if self.logprob_source not in {"replay", "native"}:
            self.logprob_source = "replay"

    # -----------------------------------------------------------------
    # SGLang ServerArgs construction
    # -----------------------------------------------------------------

    def build_server_kwargs(self, server_args_cls: Any) -> Dict[str, Any]:
        """Build a kwargs dict suitable for ``ServerArgs.from_kwargs()``.

        Priority (highest → lowest):
        1. Typed fields on this config (when set / non-None).
        2. ``engine_kwargs`` entries whose key is a valid ServerArgs field
           (escape hatch for rare overrides not covered by typed fields).
        """
        allowed_keys = {
            f.name for f in dataclasses.fields(server_args_cls)
        }
        result: Dict[str, Any] = {}

        # Escape-hatch entries (lowest priority) --------------------------
        for key, value in (self.engine_kwargs or {}).items():
            if key in allowed_keys:
                result[key] = value

        # Typed fields (override escape hatch) ----------------------------
        model_path = self.pretrained_model_ckpt_path or self.model_dotpath
        if model_path:
            result["model_path"] = model_path

        result["num_gpus"] = self.num_gpus

        if self.tp_size is not None:
            result["tp_size"] = int(self.tp_size)
        if self.sp_degree is not None and "sp_degree" in allowed_keys:
            result["sp_degree"] = int(self.sp_degree)

        if "disable_autocast" in allowed_keys:
            result["disable_autocast"] = bool(self.disable_autocast)

        if self.lora_merge_mode is not None and "lora_merge_mode" in allowed_keys:
            result["lora_merge_mode"] = self.lora_merge_mode
        elif self.use_lora and "lora_merge_mode" in allowed_keys:
            result.setdefault("lora_merge_mode", "online")

        if self.host is not None:
            result["host"] = str(self.host)
        if self.port is not None:
            result["port"] = int(self.port)
        if self.scheduler_port is not None:
            result["scheduler_port"] = int(self.scheduler_port)
        if self.sglang_master_port is not None:
            result["master_port"] = int(self.sglang_master_port)

        return result

    def with_sglang_ports(self, rank: int) -> "EngineConfig":
        """Return a new config with SGLang ports filled from *rank*.

        In remote mode (``local_mode=False``), validates that host and
        scheduler_port are already set and returns unchanged.

        In local mode, assigns deterministic ports based on rank so that
        co-located actors on the same node don't collide.
        """
        if not self.local_mode:
            if self.host is None or self.scheduler_port is None:
                raise ValueError("SGLang remote mode requires host + scheduler_port on EngineConfig.")
            return self

        stride = max(32, self.sglang_port_stride)
        actor_base = self.sglang_port_base + int(rank) * stride
        if actor_base > 65000:
            raise ValueError(
                f"SGLang port range exceeded: base={self.sglang_port_base}, "
                f"stride={stride}, rank={rank}"
            )

        return dataclasses.replace(
            self,
            port=self.port if self.port is not None else actor_base,
            scheduler_port=self.scheduler_port if self.scheduler_port is not None else actor_base + 11,
            sglang_master_port=self.sglang_master_port if self.sglang_master_port is not None else actor_base + 23,
        )



__all__ = [
    "ROLLOUT_ENGINE_TYPES",
    "normalize_engine_type",
    "uses_dedicated_rollout_engine",
    "EngineConfig",
]
