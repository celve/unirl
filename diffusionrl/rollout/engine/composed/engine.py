"""ComposedRolloutEngine — two-child engine orchestrating PE serial flow.

Holds an ``ar`` child and a ``diffusion`` child. ``generate(req)`` runs
them sequentially:

1. AR child rewrites each of P prompts into N PE candidates  → ``[P*N]`` AR track
2. Diffusion child generates M images per PE                  → ``[P*N*M]`` diffusion track
3. Returns a 2-track ``RolloutResp`` with explicit lineage
   (``parent_track="ar"`` on diffusion, ``parent_ids`` chains from
   prompt → PE → image).

Children are single polymorphic fields on the config, resolved against the
``rollout/engine`` ConfigStore group. After ``expand_polymorphic_fields``
runs, each child's config is a typed dataclass — no runtime introspection
needed.

Weight sync uses prefix-based tensor routing: the training side prepends
``"{track_name}."`` to tensor keys; this engine demuxes by prefix and
forwards each subset to the matching child with the prefix stripped.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.config.instantiate import build
from diffusionrl.config.require import require
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, _track_with_field
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params

logger = logging.getLogger(__name__)


class ComposedRolloutEngine(BaseRolloutEngine):
    """Two-child rollout engine for prompt-enhancement (PE) serial flow."""

    _component_name = "composed"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: ComposedRolloutEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, ComposedRolloutEngineConfig),
            f"ComposedRolloutEngine requires ComposedRolloutEngineConfig; got {type(config).__name__}",
        )
        self.cfg = config
        self._device = device
        self._strategy = strategy
        self._rank = rank
        self._model_config = model_config

        deps = dict(device=device, rank=rank, model_config=model_config)

        self._ar: BaseRolloutEngine = build(
            config.ar,
            strategy=None,
            **deps,
        )
        self._diffusion: BaseRolloutEngine = build(
            config.diffusion,
            strategy=strategy,
            **deps,
        )

        self._child_by_name: Dict[str, BaseRolloutEngine] = {
            "ar": self._ar,
            "diffusion": self._diffusion,
        }

        if config.sleep_diffusion_on_start:
            self._diffusion.sleep()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        for name, child in self._child_by_name.items():
            try:
                child.shutdown()
            except Exception as exc:
                logger.warning("Child %r shutdown raised: %s", name, exc)

    def sleep(self) -> None:
        for child in self._child_by_name.values():
            child.sleep()

    def wake_up(self) -> None:
        for child in self._child_by_name.values():
            child.wake_up()

    @property
    def is_offloaded(self) -> bool:
        return all(child.is_offloaded for child in self._child_by_name.values())

    def health_check(self) -> bool:
        return all(child.health_check() for child in self._child_by_name.values())

    def get_memory_info(self) -> Dict[str, float]:
        out: Dict[str, float] = {"allocated_gb": 0.0, "cached_gb": 0.0}
        for child in self._child_by_name.values():
            m = child.get_memory_info() or {}
            out["allocated_gb"] += float(m.get("allocated_gb", 0.0))
            out["cached_gb"] += float(m.get("cached_gb", 0.0))
        return out

    # ------------------------------------------------------------------
    # Generation — PE serial flow
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run PE serial flow: AR expansion → diffusion sampling → 2-track resp."""
        require(int(req.batch_size) > 0, "ComposedRolloutEngine.generate: empty req")
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None,
            "ComposedRolloutEngine.generate: req.primitives['text'] required",
        )
        P = int(req.batch_size)

        ar_params = get_ar_params(req.sampling_params)
        diff_params = get_diffusion_params(req.sampling_params)
        N = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        M = int(diff_params.samples_per_prompt) if diff_params is not None else 1
        require(N >= 1, f"ComposedRolloutEngine.generate: ar.n={N} must be >= 1")
        require(M >= 1, f"ComposedRolloutEngine.generate: diffusion.samples_per_prompt={M} must be >= 1")

        # ── Stage 1: AR child ────────────────────────────────────────
        self._ar.wake_up()
        self._diffusion.sleep()

        ar_shell = req.make_root_track(
            track_name="ar",
            branch=N,
            decode_to_condition=None,
        )

        ar_sub_req = RolloutReq(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            primitives={"text": text_primitive},
            request_conditions=dict(req.request_conditions),
            sampling_params=ar_params,
            stage_config={k: v for k, v in req.stage_config.items() if k in ("chat",)},
            sigmas=None,
        )
        ar_resp = self._ar.generate(ar_sub_req)
        require(
            len(ar_resp.tracks) == 1,
            f"ComposedRolloutEngine: AR child must return single-track resp; got {sorted(ar_resp.tracks.keys())}",
        )
        ar_inner = next(iter(ar_resp.tracks.values()))
        require(
            len(ar_inner.sample_ids) == P * N,
            f"ComposedRolloutEngine: AR child returned {len(ar_inner.sample_ids)} samples; expected {P}*{N}={P * N}",
        )

        ar_track = _track_with_field(ar_shell, "segment", ar_inner.segment)
        ar_track = _track_with_field(ar_track, "decoded", ar_inner.decoded)
        ar_track = _track_with_field(ar_track, "conditions", dict(ar_inner.conditions))

        # ── Stage 1 → Stage 2 transition ────────────────────────────
        self._ar.sleep()
        self._diffusion.wake_up()

        # ── Stage 2: diffusion track shell via fork_track ───────────
        diff_shell = ar_track.fork_track(
            parent_name="ar",
            child_name="diffusion",
            branch=M,
            decode_to_condition=None,
        )

        pe_texts = ar_track.decoded
        require(
            pe_texts is not None and len(pe_texts.texts) == P * N,
            f"ComposedRolloutEngine: AR track decoded missing or wrong "
            f"length (expected {P * N}, got "
            f"{len(pe_texts.texts) if pe_texts is not None else 'None'})",
        )
        expanded_texts = [t for t in pe_texts.texts for _ in range(M)]
        require(
            len(expanded_texts) == P * N * M,
            f"ComposedRolloutEngine: expanded_texts wrong length ({len(expanded_texts)} vs expected {P * N * M})",
        )

        diff_sub_req = RolloutReq(
            sample_ids=list(diff_shell.sample_ids),
            group_ids=list(diff_shell.parent_ids or diff_shell.sample_ids),
            primitives={"text": Texts(texts=expanded_texts)},
            request_conditions=dict(req.request_conditions),
            sampling_params=diff_params,
            sigmas=req.sigmas,
        )
        diff_resp = self._diffusion.generate(diff_sub_req)
        require(
            len(diff_resp.tracks) == 1,
            f"ComposedRolloutEngine: diffusion child must return single-track "
            f"resp; got {sorted(diff_resp.tracks.keys())}",
        )
        diff_inner = next(iter(diff_resp.tracks.values()))
        require(
            len(diff_inner.sample_ids) == P * N * M,
            f"ComposedRolloutEngine: diffusion child returned "
            f"{len(diff_inner.sample_ids)} samples; expected "
            f"{P}*{N}*{M}={P * N * M}",
        )

        diff_track = _track_with_field(diff_shell, "segment", diff_inner.segment)
        diff_track = _track_with_field(diff_track, "decoded", diff_inner.decoded)
        diff_track = _track_with_field(diff_track, "conditions", dict(diff_inner.conditions))
        diff_track = _track_with_field(diff_track, "media_preview", diff_inner.media_preview)

        return RolloutResp(tracks={"ar": ar_track, "diffusion": diff_track})

    # ------------------------------------------------------------------
    # Prefix demux helper
    # ------------------------------------------------------------------

    def _demux_by_prefix(
        self,
        keys: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Split a dict by child-name prefix, stripping the prefix."""
        result: Dict[str, Dict[str, Any]] = {}
        for child_name in self._child_by_name:
            prefix = f"{child_name}."
            subset = {k[len(prefix) :]: v for k, v in keys.items() if k.startswith(prefix)}
            if subset:
                result[child_name] = subset
        return result

    # ------------------------------------------------------------------
    # Weight sync — prefix-based demux
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
    ) -> None:
        for child in self._child_by_name.values():
            child.update_weights_from_ipc(
                peft_config=peft_config,
                base_sync_done=base_sync_done,
                use_shm=use_shm,
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
        for child in self._child_by_name.values():
            child.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )

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
        has_prefix = any(n.startswith(f"{cn}.") for n in names for cn in self._child_by_name)
        if has_prefix:
            for child_name, child in self._child_by_name.items():
                prefix = f"{child_name}."
                indices = [i for i, n in enumerate(names) if n.startswith(prefix)]
                if indices:
                    child.update_weights_from_distributed(
                        names=[names[i][len(prefix) :] for i in indices],
                        dtypes=[dtypes[i] for i in indices],
                        shapes=[shapes[i] for i in indices],
                        group_name=group_name,
                        target_modules=target_modules,
                        flush_cache=flush_cache,
                    )
        else:
            for child in self._child_by_name.values():
                child.update_weights_from_distributed(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=group_name,
                    target_modules=target_modules,
                    flush_cache=flush_cache,
                )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        for child in self._child_by_name.values():
            child.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        demuxed = self._demux_by_prefix(lora_tensors)
        if demuxed:
            for child_name, child_tensors in demuxed.items():
                child = self._child_by_name[child_name]
                child.set_lora_from_tensors(
                    adapter_name,
                    child_tensors,
                    peft_config=peft_config,
                )
        else:
            for child in self._child_by_name.values():
                child.set_lora_from_tensors(
                    adapter_name,
                    lora_tensors,
                    peft_config=peft_config,
                )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        for child in self._child_by_name.values():
            child.update_weights_from_tensor(
                serialized_named_tensors=serialized_named_tensors,
                target_modules=target_modules,
                load_format=load_format,
                flush_cache=flush_cache,
            )


__all__ = ["ComposedRolloutEngine"]
