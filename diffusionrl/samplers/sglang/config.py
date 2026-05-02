"""SGLang rollout-engine configuration.

Registered under ``rollout/engine: sglang`` with ``_target_`` pointing at
``SGLangRolloutEngine`` so ``build(cfg.rollout.engine, rank=...)`` materializes
the runtime engine directly from its composed cfg section.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from omegaconf import SI

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.types.sampling import SamplingParams

# SGLang port layout for co-located actors on a single node. Each actor
# reserves a stride-sized slice starting at ``base + rank * stride``:
# slot 0 is port, slot 11 is scheduler_port, slot 23 is master_port.
# Not user-tunable — callers who need a different layout override the
# individual port fields (or the ``master_port`` entry in ``engine_kwargs``)
# directly.
_SGLANG_PORT_BASE = 33000
_SGLANG_PORT_STRIDE = 100


@register_config(
    group="rollout/engine",
    name="sglang",
    target="diffusionrl.samplers.sglang.engine.SGLangRolloutEngine",
)
@dataclass
class SGLangEngineConfig:
    """Configuration for the SGLang rollout-side inference engine.

    All SGLang-specific settings are explicit typed fields. The
    ``engine_kwargs`` dict is retained only as an escape hatch for rare
    / advanced ServerArgs overrides not worth promoting to first-class
    fields.
    """

    # --- Sampling (nested — same shape as FSDPEngineConfig.sampling) ---
    # Default to a live interpolation back to top-level cfg.sampling so the
    # engine's sampling block tracks the canonical recipe spec without each
    # recipe having to override every field. Mirrors the pattern on
    # BaseAlgorithmConfig.sampling (algorithms/base.py). Without this, the
    # engine falls through to a fresh SamplingParams() with SDEConfig defaults
    # (eta=1.0, shift=3.0), so the recipe's sampling.sde_config.eta=0.7 never
    # propagates into rollout-side log_prob math.
    sampling: SamplingParams = field(default_factory=lambda: SI("${sampling}"))

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

    # --- LoRA (ServerArgs-only knob; the bundle-side lora flags live on
    # ModelBundleConfig and are injected via the engine's model_config dep) ---
    lora_merge_mode: Optional[str] = None

    # --- SGLang network ---
    # ``host`` + ``scheduler_port`` are required for remote mode; in local
    # mode all three are auto-derived from rank by ``with_sglang_ports``.
    # ``master_port`` is not exposed here — it's always derived and flows
    # through ``engine_kwargs`` so only advanced users override it.
    host: Optional[str] = None
    port: Optional[int] = None
    scheduler_port: Optional[int] = None

    # --- Escape hatch for rare / advanced ServerArgs overrides ---
    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.logprob_source = str(self.logprob_source or "replay").strip().lower()
        if self.logprob_source not in {"replay", "native"}:
            self.logprob_source = "native"
        require(self.num_gpus >= 1, f"SGLangEngineConfig.num_gpus must be >= 1; got {self.num_gpus!r}")
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"SGLangEngineConfig.tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.sp_degree is None or self.sp_degree >= 1,
            f"SGLangEngineConfig.sp_degree must be >= 1 when set; got {self.sp_degree!r}",
        )
        require(
            self.local_mode or (self.host is not None and self.scheduler_port is not None),
            f"SGLangEngineConfig: remote mode (local_mode=False) requires host and scheduler_port; got host={self.host!r}, scheduler_port={self.scheduler_port!r}",
        )

    # -----------------------------------------------------------------
    # SGLang ServerArgs construction
    # -----------------------------------------------------------------

    def build_server_kwargs(
        self,
        server_args_cls: Any,
        *,
        model_config: ModelBundleConfig,
    ) -> Dict[str, Any]:
        """Build a kwargs dict suitable for ``ServerArgs.from_kwargs()``.

        Priority (highest → lowest):
        1. Typed fields on this config (when set / non-None).
        2. ``engine_kwargs`` entries whose key is a valid ServerArgs field
           (escape hatch for rare overrides not covered by typed fields).

        ``model_config`` supplies the model checkpoint path and the
        bundle-side LoRA flag; both live on :class:`ModelBundleConfig` so
        the engine and the training model bundle cannot drift.
        """
        allowed_keys = {f.name for f in dataclasses.fields(server_args_cls)}
        result: Dict[str, Any] = {}

        # Escape-hatch entries (lowest priority) --------------------------
        for key, value in (self.engine_kwargs or {}).items():
            if key in allowed_keys:
                result[key] = value

        # Typed fields (override escape hatch) ----------------------------
        if model_config.pretrained_model_ckpt_path:
            result["model_path"] = model_config.pretrained_model_ckpt_path

        result["num_gpus"] = self.num_gpus

        if self.tp_size is not None:
            result["tp_size"] = int(self.tp_size)
        if self.sp_degree is not None and "sp_degree" in allowed_keys:
            result["sp_degree"] = int(self.sp_degree)

        if "disable_autocast" in allowed_keys:
            result["disable_autocast"] = bool(self.disable_autocast)

        if self.lora_merge_mode is not None and "lora_merge_mode" in allowed_keys:
            result["lora_merge_mode"] = self.lora_merge_mode
        elif model_config.use_lora and "lora_merge_mode" in allowed_keys:
            result.setdefault("lora_merge_mode", "online")

        # LoRA layout must match the training-side PEFT config.  If the
        # rollout engine keeps ``lora_target_modules=None`` while training only
        # injects LoRA into a subset of linears (e.g. SD3 joint-attention),
        # SGLang wraps every linear layer and emits a wall of ``LoRA adapter
        # None does not contain the weights for layer '...'`` warnings, then
        # silently disables LoRA on the unmatched layers.  Forward the
        # materialised list explicitly so both sides agree.
        if (
            model_config.use_lora
            and model_config.lora_target_modules is not None
            and "lora_target_modules" in allowed_keys
        ):
            result["lora_target_modules"] = list(model_config.lora_target_modules)

        if self.host is not None:
            result["host"] = str(self.host)
        if self.port is not None:
            result["port"] = int(self.port)
        if self.scheduler_port is not None:
            result["scheduler_port"] = int(self.scheduler_port)

        return result

    def with_sglang_ports(self, rank: int) -> "SGLangEngineConfig":
        """Return a new config with SGLang ports filled from *rank*.

        Remote-mode (``local_mode=False``) requires host + scheduler_port;
        that's already enforced by ``__post_init__``, so here we just return
        the config unchanged. In local mode, assigns deterministic ports
        based on rank so co-located actors on the same node don't collide.
        The derived master_port is threaded through ``engine_kwargs`` so
        ``build_server_kwargs`` picks it up via the ServerArgs escape hatch.
        """
        if not self.local_mode:
            return self

        actor_base = _SGLANG_PORT_BASE + int(rank) * _SGLANG_PORT_STRIDE
        require(
            actor_base <= 65000,
            f"SGLang port range exceeded: base={_SGLANG_PORT_BASE}, stride={_SGLANG_PORT_STRIDE}, rank={rank}",
        )

        new_engine_kwargs = dict(self.engine_kwargs or {})
        new_engine_kwargs.setdefault("master_port", actor_base + 23)

        return dataclasses.replace(
            self,
            port=self.port if self.port is not None else actor_base,
            scheduler_port=self.scheduler_port if self.scheduler_port is not None else actor_base + 11,
            engine_kwargs=new_engine_kwargs,
        )


__all__ = ["SGLangEngineConfig"]
