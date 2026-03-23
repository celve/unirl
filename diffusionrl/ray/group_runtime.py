"""External runtime facades layered on top of actor groups."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ray

from .group_base import ActorGroupHandle

logger = logging.getLogger(__name__)


class TrainingGroupRuntime:
    """Training-side runtime control layered above a thin actor group."""

    def __init__(self, handle: ActorGroupHandle):
        self._handle = handle.snapshot()
        self.num_actors = int(self._handle.num_actors)
        self._expected_global_batch_size_cache: Optional[int] = None
        self._train_backend_info_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def from_group(cls, group: Any) -> "TrainingGroupRuntime":
        return cls(group.snapshot())

    def update_weights(self) -> None:
        self._handle.call_all("update_weights")

    def get_rank0_ip_and_free_port(self, start_port: int = 26000) -> Dict[str, Any]:
        payload = self._handle.call_rank0(
            "get_node_ip_and_free_port",
            start_port=int(start_port),
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid rank0 network payload: {payload!r}")
        return {
            "master_address": str(payload["master_address"]),
            "master_port": int(payload["master_port"]),
        }

    def async_init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> ray.ObjectRef:
        return self._handle.call_rank0_async(
            "init_weights_update_group",
            master_address=master_address,
            master_port=int(master_port),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
        )

    def destroy_weights_update_group(self, group_name: str) -> None:
        self._handle.call_rank0(
            "destroy_weights_update_group",
            group_name=str(group_name),
        )

    def sync_weights_to_rollout_ipc(
        self,
        *,
        rollout_runtime: "RolloutGroupRuntime",
        target_modules: Optional[List[str]] = None,
        bucket_size_mb: int = 256,
        flush_cache: bool = True,
        tp_payload_count: int = 1,
    ) -> Dict[str, int]:
        results = self._handle.call_all(
            "sync_weights_to_rollout_ipc",
            rollout_weight_sink=rollout_runtime,
            target_modules=target_modules,
            bucket_size_mb=int(bucket_size_mb),
            flush_cache=bool(flush_cache),
            tp_payload_count=max(1, int(tp_payload_count)),
        )
        rank0 = results[0] if results else {}
        if not isinstance(rank0, dict):
            return {"buckets": 0, "payloads": 0}
        return {
            "buckets": int(rank0.get("buckets", 0)),
            "payloads": int(rank0.get("payloads", 0)),
        }

    def sync_weights_to_rollout_nccl(
        self,
        *,
        rollout_runtime: "RolloutGroupRuntime",
        group_name: str,
        target_modules: Optional[List[str]] = None,
        bucket_size_mb: int = 256,
        flush_cache: bool = True,
    ) -> Dict[str, int]:
        results = self._handle.call_all(
            "sync_weights_to_rollout_nccl",
            rollout_weight_sink=rollout_runtime,
            group_name=str(group_name),
            target_modules=target_modules,
            bucket_size_mb=int(bucket_size_mb),
            flush_cache=bool(flush_cache),
        )
        rank0 = results[0] if results else {}
        if not isinstance(rank0, dict):
            return {"buckets": 0, "broadcast_tensors": 0}
        return {
            "buckets": int(rank0.get("buckets", 0)),
            "broadcast_tensors": int(rank0.get("broadcast_tensors", 0)),
        }

    def get_train_backend_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._train_backend_info_cache is not None and not force_refresh:
            return dict(self._train_backend_info_cache)
        info = self._handle.call_rank0("get_train_backend_info")
        if isinstance(info, dict):
            self._train_backend_info_cache = dict(info)
            return dict(info)
        return {}

    def get_expected_global_batch_size(self, force_refresh: bool = False) -> int:
        if self._expected_global_batch_size_cache is not None and not force_refresh:
            return int(self._expected_global_batch_size_cache)
        expected_global_batch_size = self._handle.call_rank0("get_expected_global_batch_size")
        try:
            resolved = int(expected_global_batch_size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Invalid expected_global_batch_size payload: "
                f"{expected_global_batch_size!r}"
            ) from exc
        self._expected_global_batch_size_cache = resolved
        return resolved

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        *,
        export_format: str = "state_dict",
    ) -> str:
        refs = self._handle.call_all_async(
            "export_weights_to_path",
            checkpoint_path,
            export_format=export_format,
        )
        if len(refs) > 1:
            ray.get(refs[1:])
        if refs:
            ray.get(refs[0])
        return checkpoint_path

    def save_model(self, path: str) -> None:
        self._handle.call_all("save_model", path)

    def load_checkpoint(self, path: str) -> None:
        self._handle.call_all("load_checkpoint", path)

    def offload(self) -> None:
        self._handle.call_all("offload")

    def onload(self) -> None:
        self._handle.call_all("onload")

    def clear_memory(self) -> None:
        self._handle.call_all("clear_memory")

    def apply_ema_for_eval(self) -> None:
        self._handle.call_all("apply_ema_for_eval")

    def restore_from_eval(self) -> None:
        self._handle.call_all("restore_from_eval")


class RolloutGroupRuntime:
    """Rollout-side lifecycle and weight-sink runtime facade."""

    def __init__(
        self,
        *,
        handle: ActorGroupHandle,
        num_gpus_allocated: int = 1,
        sampler_engine_type: str = "unknown",
    ):
        self._handle = handle.snapshot()
        self._num_gpus_allocated = int(num_gpus_allocated or 1)
        self._sampler_engine_type = str(sampler_engine_type or "unknown")
        self._weight_update_target_by_actor: Dict[int, str] = {
            idx: f"actor_rank:{idx}" for idx in range(self._handle.num_actors)
        }
        self._weight_update_actor_indices: List[int] = list(range(self._handle.num_actors))
        self._weight_update_targets_ready = False

    @classmethod
    def from_group(cls, group: Any) -> "RolloutGroupRuntime":
        return cls(
            handle=group.snapshot(),
            num_gpus_allocated=int(getattr(group, "num_gpus_allocated", 1) or 1),
            sampler_engine_type=str(getattr(group, "sampler_engine_type", "unknown") or "unknown"),
        )

    def refresh_weight_update_targets(self) -> Dict[str, Any]:
        if self._handle.num_actors <= 0:
            self._weight_update_target_by_actor = {}
            self._weight_update_actor_indices = []
            self._weight_update_targets_ready = True
            return {
                "num_actors": 0,
                "num_unique_targets": 0,
                "selected_actor_indices": [],
            }

        try:
            payloads = self._handle.call_all("get_weight_update_target")
        except Exception as exc:
            raise RuntimeError(
                "Failed to collect rollout weight-update targets. "
                "Refusing to fall back to implicit per-actor updates."
            ) from exc

        target_by_actor: Dict[int, str] = {}
        first_actor_by_target: Dict[str, int] = {}
        for idx, payload in enumerate(payloads):
            target = None
            if isinstance(payload, dict):
                raw_target = payload.get("target")
                if isinstance(raw_target, str) and raw_target.strip():
                    target = raw_target.strip()
            if not target:
                target = f"actor_rank:{idx}"
            target_by_actor[idx] = target
            if target not in first_actor_by_target:
                first_actor_by_target[target] = idx

        selected = sorted(int(v) for v in first_actor_by_target.values())
        self._weight_update_target_by_actor = target_by_actor
        self._weight_update_actor_indices = selected
        self._weight_update_targets_ready = True

        if len(selected) < self._handle.num_actors:
            logger.info(
                "RolloutGroupRuntime(%s) deduplicates weight updates: %d actors -> %d logical targets",
                self._sampler_engine_type,
                self._handle.num_actors,
                len(selected),
            )

        return {
            "num_actors": int(self._handle.num_actors),
            "num_unique_targets": int(len(selected)),
            "selected_actor_indices": list(selected),
        }

    def _ensure_weight_update_targets(self) -> None:
        if not self._weight_update_targets_ready:
            self.refresh_weight_update_targets()

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        self._ensure_weight_update_targets()
        results = self._handle.call_subset(
            self._weight_update_actor_indices,
            "update_weights_from_path",
            checkpoint_path,
        )

        checksum_rows: List[tuple[int, tuple[tuple[str, str], ...]]] = []
        for idx, payload in enumerate(results):
            if not isinstance(payload, dict):
                continue
            raw_checksum = payload.get("checksum")
            if not isinstance(raw_checksum, dict) or not raw_checksum:
                continue
            normalized = tuple(sorted((str(k), str(v)) for k, v in raw_checksum.items()))
            rank = int(payload.get("rank", idx))
            checksum_rows.append((rank, normalized))

        if len(checksum_rows) <= 1:
            return

        checksum_groups: Dict[tuple[tuple[str, str], ...], List[int]] = {}
        for rank, normalized in checksum_rows:
            checksum_groups.setdefault(normalized, []).append(rank)

        if len(checksum_groups) > 1:
            details = {str(dict(items)): ranks for items, ranks in checksum_groups.items()}
            raise RuntimeError(
                "Checksum mismatch across rollout actors after update_weights_from_path: "
                f"{details}"
            )

        checksum_payload = dict(next(iter(checksum_groups.keys())))
        logger.info(
            "Verified consistent rollout checksum across %d actors: %s",
            len(checksum_rows),
            checksum_payload,
        )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        self._ensure_weight_update_targets()
        self._handle.call_subset(
            self._weight_update_actor_indices,
            "update_weights_from_tensor",
            serialized_named_tensors=list(serialized_named_tensors),
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        self._ensure_weight_update_targets()
        per_actor_args: List[Optional[tuple[Any, ...]]] = [None] * self._handle.num_actors
        per_actor_kwargs: List[Optional[Dict[str, Any]]] = [None] * self._handle.num_actors
        for idx, actor_idx in enumerate(self._weight_update_actor_indices):
            rank_offset = 1 + idx * self._num_gpus_allocated
            per_actor_args[actor_idx] = ()
            per_actor_kwargs[actor_idx] = {
                "master_address": master_address,
                "master_port": int(master_port),
                "rank_offset": int(rank_offset),
                "world_size": int(world_size),
                "group_name": str(group_name),
                "backend": str(backend),
            }
        self._handle.call_per_actor(
            "init_weights_update_group",
            per_actor_args=per_actor_args,
            per_actor_kwargs=per_actor_kwargs,
        )

    def destroy_weights_update_group(self, group_name: str) -> None:
        self._ensure_weight_update_targets()
        self._handle.call_subset(
            self._weight_update_actor_indices,
            "destroy_weights_update_group",
            group_name=str(group_name),
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
        self._ensure_weight_update_targets()
        self._handle.call_subset(
            self._weight_update_actor_indices,
            "update_weights_from_distributed",
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            target_modules=target_modules,
            flush_cache=flush_cache,
        )

    def get_weight_sync_topology(self) -> Dict[str, int]:
        self._ensure_weight_update_targets()
        num_unique_targets = int(len(self._weight_update_actor_indices))
        return {
            "num_actors": int(self._handle.num_actors),
            "num_weight_update_targets": num_unique_targets,
            "num_gpus_per_actor": int(self._num_gpus_allocated),
            "total_gpus": int(num_unique_targets * self._num_gpus_allocated),
        }

    def sleep(self) -> None:
        self._handle.call_all("sleep")

    def wake_up(self) -> None:
        self._handle.call_all("wake_up")


__all__ = ["TrainingGroupRuntime", "RolloutGroupRuntime"]
