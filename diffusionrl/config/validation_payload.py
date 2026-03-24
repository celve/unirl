"""Payload-shape validation for actor/service init configs."""

from __future__ import annotations

from typing import Any, Dict

from diffusionrl.training.update_schedule import coerce_training_execution_plan


def _require_dict_section(config: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got: {type(value).__name__}")
    return value


def validate_rollout_engine_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for dedicated rollout engine config."""
    if not isinstance(config, dict):
        raise ValueError(f"rollout_engine_config must be a dict, got: {type(config).__name__}")
    if not isinstance(config.get("engine_kwargs"), dict):
        raise ValueError(
            "rollout_engine_config.engine_kwargs must be a dict, "
            f"got: {type(config.get('engine_kwargs')).__name__}"
        )


def validate_rollout_actor_init_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for rollout actor init config."""
    if not isinstance(config, dict):
        raise ValueError(f"rollout_actor_init_config must be a dict, got: {type(config).__name__}")

    engine_runtime_config = _require_dict_section(config, name="engine_runtime_config")
    reward_config = _require_dict_section(config, name="reward_config")

    validate_rollout_engine_config(engine_runtime_config)

    sampler_path = str(engine_runtime_config.get("sampler_path", "") or "").strip()
    if not sampler_path:
        raise ValueError(
            "rollout_actor_init_config.engine_runtime_config.sampler_path is required."
        )
    sampler_engine_type = str(
        engine_runtime_config.get("sampler_engine_type", "") or ""
    ).strip()
    if not sampler_engine_type:
        raise ValueError(
            "rollout_actor_init_config.engine_runtime_config.sampler_engine_type is required."
        )
    for required_key in (
        "num_inference_steps",
        "eta",
        "sde_type",
        "shift",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
    ):
        if engine_runtime_config.get(required_key) is None:
            raise ValueError(
                f"rollout_actor_init_config.engine_runtime_config.{required_key} is required."
            )
    if not isinstance(reward_config, dict):
        raise ValueError(
            "rollout_actor_init_config.reward_config must be a dict, "
            f"got: {type(reward_config).__name__}"
        )


def validate_training_actor_init_config(config: Dict[str, Any]) -> None:
    """Minimal pre-dispatch validation for training actor config."""
    if not isinstance(config, dict):
        raise ValueError(f"training_actor_init_config must be a dict, got: {type(config).__name__}")

    for section in (
        "model_config",
        "reward_config",
        "optimizer_config",
        "scheduler_config",
        "algorithm_config",
        "training_config",
        "topology_config",
        "training_plan_config",
        "sampling_config",
        "train_backend_config",
    ):
        _require_dict_section(config, name=section)

    algorithm_config = config["algorithm_config"]
    if not isinstance(algorithm_config.get("algorithm_kwargs"), dict):
        raise ValueError(
            "algorithm_config.algorithm_kwargs must be a dict, "
            f"got: {type(algorithm_config.get('algorithm_kwargs')).__name__}"
        )

    backend_config = config["train_backend_config"]
    if not isinstance(backend_config.get("kwargs"), dict):
        raise ValueError(
            "train_backend_config.kwargs must be a dict, "
            f"got: {type(backend_config.get('kwargs')).__name__}"
        )
    backend_name = str(backend_config.get("name", "") or "").strip().lower()
    if not backend_name:
        raise ValueError("train_backend_config.name is required.")

    topology_config = config["topology_config"]
    for required_key in ("actor_count", "world_size", "dp_size"):
        value = topology_config.get(required_key)
        if value is None:
            raise ValueError(f"topology_config.{required_key} is required.")
        if int(value) < 1:
            raise ValueError(
                f"topology_config.{required_key} must be >= 1, got: {value!r}"
            )

    training_plan_config = config["training_plan_config"]
    try:
        coerce_training_execution_plan(training_plan_config)
    except ValueError as exc:
        raise ValueError(
            "Invalid training_plan_config in training actor init payload. "
            f"{exc}"
        ) from exc


__all__ = [
    "validate_rollout_engine_config",
    "validate_rollout_actor_init_config",
    "validate_training_actor_init_config",
]
