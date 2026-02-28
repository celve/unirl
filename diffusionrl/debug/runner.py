"""Debug runtime entrypoints and artifact I/O."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ray
import torch

from diffusionrl.runtime.pipeline.partition_stage import maybe_partition_training_batch
from diffusionrl.types.training_batch import BackwardTrainingBatch, ForwardTrainingBatch, TrainingBatch

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _shape(value: Any) -> Optional[List[int]]:
    if torch.is_tensor(value):
        return [int(v) for v in value.shape]
    return None


def _tensor_stats(value: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value):
        return {}
    if value.numel() == 0:
        return {"shape": [int(v) for v in value.shape], "empty": True}
    flat = value.detach().to(dtype=torch.float32).reshape(-1)
    return {
        "shape": [int(v) for v in value.shape],
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _save_tensor_as_image(tensor: torch.Tensor, output_path: Path) -> None:
    from PIL import Image
    import numpy as np

    x = tensor.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x[:3].permute(1, 2, 0)
    elif x.ndim == 2:
        x = x.unsqueeze(-1)
    else:
        return

    arr = x.numpy()
    if arr.size == 0:
        return
    arr = arr.astype("float32")
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    denom = max(arr_max - arr_min, 1e-6)
    arr = (arr - arr_min) / denom
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(output_path)


def _save_media_from_sampler_outputs(
    *,
    sampler_outputs: Sequence[Any],
    output_dir: Path,
    max_media: int,
) -> Dict[str, Any]:
    media_dir = output_dir / "media"
    _ensure_dir(media_dir)

    saved_images = 0
    saved_videos = 0
    summaries: List[Dict[str, Any]] = []

    for out_idx, output in enumerate(sampler_outputs):
        item_summary: Dict[str, Any] = {
            "index": int(out_idx),
            "latents_shape": _shape(getattr(output, "latents", None)),
            "timesteps_shape": _shape(getattr(output, "timesteps", None)),
            "trajectories_shape": _shape(getattr(output, "trajectories", None)),
            "step_indices_shape": _shape(getattr(output, "step_indices", None)),
            "log_prob_steps": sorted(
                int(k) for k in (getattr(getattr(output, "log_probs", None), "data", {}) or {}).keys()
            ),
            "metadata_keys": sorted(list((getattr(output, "metadata", {}) or {}).keys())),
        }
        summaries.append(item_summary)

        decoded_images = getattr(output, "decoded_images", None) or []
        for image_idx, image in enumerate(decoded_images):
            if saved_images >= max_media:
                break
            output_path = media_dir / f"image_{out_idx:03d}_{image_idx:03d}.png"
            try:
                if hasattr(image, "save"):
                    image.save(output_path)
                    saved_images += 1
                    continue
                if torch.is_tensor(image):
                    _save_tensor_as_image(image, output_path)
                    saved_images += 1
            except Exception as exc:
                logger.warning(
                    "Failed to save decoded image (output=%s idx=%s): %s",
                    out_idx,
                    image_idx,
                    exc,
                )

        decoded_videos = None
        metadata = getattr(output, "metadata", {}) or {}
        if isinstance(metadata, dict):
            decoded_videos = metadata.get("decoded_videos")
        if torch.is_tensor(decoded_videos) and decoded_videos.ndim >= 4:
            if decoded_videos.ndim == 4:
                decoded_videos = decoded_videos.unsqueeze(0)
            for video_idx, video_tensor in enumerate(decoded_videos):
                if saved_videos >= max_media:
                    break
                video_path = media_dir / f"video_{out_idx:03d}_{video_idx:03d}.pt"
                try:
                    torch.save(video_tensor.cpu(), video_path)
                    # Also export the first frame for quick inspection.
                    if video_tensor.ndim == 4:
                        frame0 = video_tensor[0]
                        frame_path = media_dir / f"video_{out_idx:03d}_{video_idx:03d}_frame0.png"
                        _save_tensor_as_image(frame0, frame_path)
                    saved_videos += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to save decoded video (output=%s idx=%s): %s",
                        out_idx,
                        video_idx,
                        exc,
                    )

    return {
        "saved_images": int(saved_images),
        "saved_videos": int(saved_videos),
        "sampler_summaries": summaries,
    }


def _save_reward_csv(
    *,
    output_dir: Path,
    prompts: Sequence[str],
    rewards: Optional[torch.Tensor],
    advantages: Optional[torch.Tensor],
    reward_components: Optional[Dict[str, List[float]]],
) -> None:
    if rewards is None or not torch.is_tensor(rewards):
        return

    rewards_list = rewards.detach().cpu().reshape(-1).tolist()
    advantages_list: List[float] = []
    if torch.is_tensor(advantages):
        advantages_list = advantages.detach().cpu().reshape(-1).tolist()

    component_names = sorted((reward_components or {}).keys())
    path = output_dir / "rewards.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_idx", "prompt", "reward", "advantage"] + component_names
        writer.writerow(header)
        for idx, reward in enumerate(rewards_list):
            prompt = prompts[idx] if idx < len(prompts) else ""
            row = [
                int(idx),
                prompt,
                float(reward),
                float(advantages_list[idx]) if idx < len(advantages_list) else "",
            ]
            for name in component_names:
                comp_values = reward_components.get(name) or []
                row.append(float(comp_values[idx]) if idx < len(comp_values) else "")
            writer.writerow(row)


def _build_payload_for_disk(
    *,
    rollout_id: int,
    payload: Dict[str, Any],
    save_trajectories: bool,
    sampler_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    saved: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "created_at_utc": _now_utc_iso(),
        "rollout_id": int(rollout_id),
        "debug_mode": str(payload.get("debug_mode", "")),
        "prompts": list(payload.get("prompts", []) or []),
        "train_prompts": list(payload.get("train_prompts", []) or []),
        "base_prompts": list(payload.get("base_prompts", []) or []),
        "reward_prompts": list(payload.get("reward_prompts", []) or []),
        "sde_indices": sorted(int(v) for v in (payload.get("sde_indices") or [])),
        "rewards": payload.get("rewards"),
        "advantages": payload.get("advantages"),
        "reward_components": payload.get("reward_components", {}),
        "training_batch_type": type(payload.get("training_batch")).__name__,
        "training_batch_size": int(getattr(payload.get("training_batch"), "batch_size", 0) or 0),
        "training_batch_saved_separately": True,
        "training_batch_path": "training_batch.pt",
        "sampler_summaries": sampler_summaries,
    }
    if save_trajectories:
        saved["sampler_outputs"] = payload.get("sampler_outputs")
    return saved


def save_rollout_debug_payload(
    *,
    args: Any,
    payload: Dict[str, Any],
    rollout_id: int,
    source: str,
) -> Path:
    """Persist rollout debug payload, media, and summary artifacts."""
    root = Path(str(getattr(args, "debug_save_dir", "outputs/debug")))
    rollout_dir = root / f"rollout_{int(rollout_id):06d}"
    _ensure_dir(rollout_dir)

    sampler_outputs = payload.get("sampler_outputs", []) or []
    media_result = _save_media_from_sampler_outputs(
        sampler_outputs=sampler_outputs,
        output_dir=rollout_dir,
        max_media=max(1, int(getattr(args, "debug_max_media", 8))),
    )

    _save_reward_csv(
        output_dir=rollout_dir,
        prompts=list(payload.get("reward_prompts", []) or []),
        rewards=payload.get("rewards"),
        advantages=payload.get("advantages"),
        reward_components=payload.get("reward_components"),
    )

    training_batch = payload.get("training_batch")
    if training_batch is not None:
        torch.save(training_batch, rollout_dir / "training_batch.pt")

    disk_payload = _build_payload_for_disk(
        rollout_id=rollout_id,
        payload=payload,
        save_trajectories=bool(getattr(args, "debug_save_trajectories", False)),
        sampler_summaries=media_result.get("sampler_summaries", []),
    )
    payload_path = rollout_dir / "payload.pt"
    torch.save(disk_payload, payload_path)

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "source": str(source),
        "created_at_utc": _now_utc_iso(),
        "rollout_id": int(rollout_id),
        "training_batch_type": type(training_batch).__name__ if training_batch is not None else None,
        "training_batch_size": int(getattr(training_batch, "batch_size", 0) or 0),
        "rewards_stats": _tensor_stats(payload.get("rewards")),
        "advantages_stats": _tensor_stats(payload.get("advantages")),
        "saved_images": int(media_result.get("saved_images", 0)),
        "saved_videos": int(media_result.get("saved_videos", 0)),
        "num_sampler_outputs": int(len(sampler_outputs)),
    }
    with (rollout_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    logger.info(
        "Saved debug rollout payload: rollout_id=%s dir=%s batch_size=%s images=%s videos=%s",
        rollout_id,
        rollout_dir,
        summary["training_batch_size"],
        summary["saved_images"],
        summary["saved_videos"],
    )
    return rollout_dir


def _resolve_debug_load_path(load_path: str) -> Path:
    path = Path(load_path)
    if path.is_dir():
        batch_path = path / "training_batch.pt"
        payload_path = path / "payload.pt"
        if batch_path.exists():
            return batch_path
        if payload_path.exists():
            return payload_path
        raise FileNotFoundError(
            f"Directory {path} does not contain training_batch.pt or payload.pt."
        )
    return path


def load_debug_training_batch(
    *,
    load_path: str,
    subsample: int = 0,
) -> Tuple[TrainingBatch, Dict[str, Any]]:
    """Load a training batch from a debug payload path."""
    resolved = _resolve_debug_load_path(load_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Debug load path not found: {resolved}")

    obj = torch.load(resolved, map_location="cpu")
    meta: Dict[str, Any] = {"source_path": str(resolved)}
    if isinstance(obj, (BackwardTrainingBatch, ForwardTrainingBatch)):
        batch = obj
    elif isinstance(obj, dict):
        meta.update(
            {
                "rollout_id": obj.get("rollout_id"),
                "schema_version": obj.get("schema_version"),
                "created_at_utc": obj.get("created_at_utc"),
            }
        )
        inline_batch = obj.get("training_batch")
        if isinstance(inline_batch, (BackwardTrainingBatch, ForwardTrainingBatch)):
            batch = inline_batch
        else:
            candidate_paths: List[Path] = []
            declared = obj.get("training_batch_path")
            if isinstance(declared, str) and declared.strip():
                candidate_paths.append((resolved.parent / declared).resolve())
            candidate_paths.append((resolved.parent / "training_batch.pt").resolve())

            batch = None
            for candidate in candidate_paths:
                if not candidate.exists():
                    continue
                loaded = torch.load(candidate, map_location="cpu")
                if isinstance(loaded, (BackwardTrainingBatch, ForwardTrainingBatch)):
                    batch = loaded
                    meta["source_path"] = str(candidate)
                    meta["loaded_via"] = str(resolved)
                    break
            if batch is None:
                raise TypeError(
                    f"Debug payload at {resolved} does not contain inline training_batch and no valid "
                    "external training_batch.pt was found."
                )
    else:
        raise TypeError(
            f"Unsupported debug payload type at {resolved}: {type(obj).__name__}. "
            "Expected TrainingBatch or dict containing training_batch."
        )

    keep = int(subsample)
    if keep > 0 and int(batch.batch_size) > keep:
        batch = batch.slice(0, keep)
        meta["subsample"] = keep
    return batch, meta


def _resolve_consumer_spec(
    *,
    consumer_spec: Optional[Dict[str, Any]],
    default_dp_size: int,
) -> Tuple[Optional[int], bool, str]:
    dp_size: Optional[int] = int(default_dp_size)
    partition_train_data = True
    partition_mode = "data_parallel"

    if isinstance(consumer_spec, dict):
        if consumer_spec.get("dp_size") is not None:
            try:
                dp_size = int(consumer_spec["dp_size"])
            except (TypeError, ValueError):
                dp_size = int(default_dp_size)
        if consumer_spec.get("partition_train_data") is not None:
            partition_train_data = bool(consumer_spec["partition_train_data"])
        if consumer_spec.get("partition_mode") is not None:
            partition_mode = str(consumer_spec["partition_mode"]).strip().lower()

    return dp_size, partition_train_data, partition_mode


def put_training_data_for_consumer(
    *,
    train_data: TrainingBatch,
    consumer_spec: Optional[Dict[str, Any]],
    default_dp_size: int,
) -> Any:
    """Mirror rollout-buffer partition semantics when feeding debug data to training."""
    dp_size, partition_train_data, partition_mode = _resolve_consumer_spec(
        consumer_spec=consumer_spec,
        default_dp_size=default_dp_size,
    )
    if partition_mode in ("backend_managed", "replicated", "none"):
        return ray.put(train_data)
    if partition_mode != "data_parallel":
        raise ValueError(
            f"Unsupported partition_mode in consumer_spec: {partition_mode!r}. "
            "Expected one of data_parallel/backend_managed/replicated/none."
        )

    parts = maybe_partition_training_batch(
        train_data=train_data,
        dp_size=dp_size,
        partition_train_data=bool(partition_train_data),
    )
    if parts is None:
        return ray.put(train_data)
    return [ray.put(part) for part in parts]


def _maybe_init_ray(args: Any) -> None:
    if ray.is_initialized():
        return
    ray_address = getattr(args, "ray_address", None)
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init()


def _to_jsonable(value: Any) -> Any:
    """Recursively convert tensors and non-JSON objects to JSON-safe structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=lambda x: str(x))]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def run_debug_rollout_only(args: Any) -> None:
    """Run rollout-only debug mode and persist each rollout payload."""
    from diffusionrl.ray import create_placement_groups_from_args, create_rollout_manager
    from diffusionrl.utils import configure_logger, set_seed

    configure_logger()
    set_seed(args.seed)
    _maybe_init_ray(args)

    pgs = create_placement_groups_from_args(args)
    rollout_pg_result = pgs.get("rollout")
    if rollout_pg_result is None and not bool(getattr(args, "training_actor_direct_sampling", False)):
        raise ValueError(
            "debug_mode=rollout_only requires rollout placement resources."
        )

    rollout_manager, _ = create_rollout_manager(
        args,
        pg_result=rollout_pg_result,
        reward_pg_result=pgs.get("reward"),
    )
    logger.info(
        "Starting rollout_only debug run: num_rollouts=%s save_dir=%s",
        args.debug_num_rollouts,
        args.debug_save_dir,
    )

    start_rollout_id = int(getattr(args, "start_rollout_id", 0))
    try:
        for offset in range(int(args.debug_num_rollouts)):
            rollout_id = start_rollout_id + offset
            payload = ray.get(rollout_manager.build_training_debug_payload.remote(rollout_id))
            save_rollout_debug_payload(
                args=args,
                payload=payload,
                rollout_id=rollout_id,
                source="rollout_only",
            )
    finally:
        ray.get(rollout_manager.dispose.remote())

    logger.info("rollout_only debug run complete.")


def run_debug_train_only(args: Any) -> None:
    """Run train-only debug mode by replaying a saved debug training batch."""
    from diffusionrl.ray import create_placement_groups_from_args, create_training_actor_group
    from diffusionrl.utils import configure_logger, set_seed

    configure_logger()
    set_seed(args.seed)
    _maybe_init_ray(args)

    if not getattr(args, "debug_load_path", None):
        raise ValueError("debug_mode=train_only requires --debug-load-path.")

    pgs = create_placement_groups_from_args(args)
    training_pg_result = pgs.get("training")
    if training_pg_result is None:
        raise ValueError("debug_mode=train_only requires training placement resources.")

    training_group = create_training_actor_group(args, training_pg_result)
    try:
        batch, payload_meta = load_debug_training_batch(
            load_path=str(args.debug_load_path),
            subsample=int(getattr(args, "debug_subsample", 0)),
        )
        consumer_spec = training_group.get_buffer_consumer_spec()
        train_data_ref = put_training_data_for_consumer(
            train_data=batch,
            consumer_spec=consumer_spec,
            default_dp_size=int(getattr(training_group, "num_actors", 1)),
        )
        rollout_id = int(payload_meta.get("rollout_id", getattr(args, "start_rollout_id", 0)))
        metrics = training_group.train(rollout_id, train_data_ref)
        avg_loss = 0.0
        if metrics:
            avg_loss = sum(float(m.get("loss", 0.0)) for m in metrics) / float(len(metrics))
        logger.info(
            "train_only debug replay complete: source=%s rollout_id=%s batch_size=%s avg_loss=%.6f",
            payload_meta.get("source_path", args.debug_load_path),
            rollout_id,
            int(batch.batch_size),
            avg_loss,
        )

        out_dir = Path(str(getattr(args, "debug_save_dir", "outputs/debug"))) / "train_only"
        _ensure_dir(out_dir)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "created_at_utc": _now_utc_iso(),
                    "source_path": payload_meta.get("source_path", str(args.debug_load_path)),
                    "rollout_id": rollout_id,
                    "batch_size": int(batch.batch_size),
                    "avg_loss": float(avg_loss),
                    "metrics": _to_jsonable(metrics),
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    finally:
        training_group.dispose()
