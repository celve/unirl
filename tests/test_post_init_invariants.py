"""Negative tests for every registered dataclass's ``__post_init__``.

One parametrized case per invariant. Each test builds a valid kwargs dict,
overrides one field with a bad value, and asserts the expected ``ValueError``
substring. Keeps the matrix shallow on purpose — exhaustiveness is covered
by the composition-level tests in ``test_cfg_validation.py``.

Trigger ``@register_config`` side effects via the module imports so the
validators themselves (which may depend on canonical aliases) are set up.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.algorithms.base import BaseAlgorithmConfig, EMAConfig
from diffusionrl.buffer.buffer_plugins import RewardRangeFilterPlugin
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.evaluation_config import EvaluationConfig
from diffusionrl.config.logging_config import LoggingConfig
from diffusionrl.config.resume_config import ResumeConfig
from diffusionrl.config.run_config import RunConfig
from diffusionrl.distributed.weight_sync import (
    CheckpointSyncConfig,
    NcclBroadcastSyncConfig,
    TensorPayloadSyncConfig,
)
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.ray.placement import PlacementConfig
from diffusionrl.ray.train_actor import TrainingExecutionConfig
from diffusionrl.reward.config import RewardConfig
from diffusionrl.reward.scorers.pickscore import PickScoreSpec
from diffusionrl.samplers.sglang.config import SGLangEngineConfig
from diffusionrl.training.backends.base import TrainTopology
from diffusionrl.training.plan import TrainingPlan
from diffusionrl.types.sampling import SamplingParams


def _build(cls: Any, base: Dict[str, Any], field: str, bad: Any) -> None:
    kwargs = {**base, field: bad}
    cls(**kwargs)


class TestTrainingPlan:
    _VALID = dict(
        global_batch_size=4,
        local_batch_size=4,
        local_mini_batch_size=4,
        micro_batch_size=4,
        num_updates_per_batch=1,
    )

    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("global_batch_size", 0, "global_batch_size must be >= 1"),
            ("local_batch_size", 0, "local_batch_size must be >= 1"),
            ("local_mini_batch_size", 0, "local_mini_batch_size must be >= 1"),
            ("micro_batch_size", 0, "micro_batch_size must be >= 1"),
            ("num_updates_per_batch", 0, "num_updates_per_batch must be >= 1"),
            ("local_batch_size", 2, "must equal"),
            ("micro_batch_size", 3, "must evenly divide"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(TrainingPlan, self._VALID, field, bad)


class TestBaseAlgorithmConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("samples_per_prompt", 0, "samples_per_prompt must be >= 1"),
            ("prompts_per_rollout", 0, "prompts_per_rollout must be >= 1"),
            ("epsilon", 0.0, "epsilon must be > 0"),
            ("clip_max", -1.0, "clip_max must be > 0"),
            ("trim_outliers_ratio", 0.5, r"trim_outliers_ratio must be in \[0.0, 0.5\)"),
            ("trim_outliers_ratio", -0.1, r"trim_outliers_ratio must be in \[0.0, 0.5\)"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(BaseAlgorithmConfig, {}, field, bad)


class TestEMAConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("eval_ema_decay", -1.0, "EMAConfig.eval_ema_decay must be >= 0"),
            ("eval_ema_update_interval", 0, "EMAConfig.eval_ema_update_interval must be >= 1"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(EMAConfig, {}, field, bad)


class TestSamplingParams:
    _VALID = dict(
        num_inference_steps=50,
        guidance_scale=7.5,
        height=256,
        width=256,
        num_frames=16,
        seed=42,
    )

    def test_rejects_reserved_sampler_kwarg(self):
        with pytest.raises(ValueError, match="cannot contain reserved keys"):
            SamplingParams(**{**self._VALID, "sampler_kwargs": {"autocast_precision": "bf16"}})


class TestSGLangEngineConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("num_gpus", 0, "num_gpus must be >= 1"),
            ("tp_size", 0, "tp_size must be >= 1 when set"),
            ("sp_degree", 0, "sp_degree must be >= 1 when set"),
            ("batch_size", 0, "batch_size must be >= 1 when set"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(SGLangEngineConfig, {}, field, bad)

    def test_rejects_remote_mode_without_host(self):
        with pytest.raises(ValueError, match="remote mode .* requires host and scheduler_port"):
            SGLangEngineConfig(local_mode=False)


class TestTrainTopology:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("dp_size", 0, "dp_size must be >= 1 when set"),
            ("dp_shard_size", 0, "dp_shard_size must be >= 1 when set"),
            ("dp_replicate_size", 0, "dp_replicate_size must be >= 1"),
            ("cp_size", 0, "cp_size must be >= 1"),
            ("actor_count", 0, "actor_count must be >= 1 when set"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(TrainTopology, {}, field, bad)


class TestTrainingExecutionConfig:
    _VALID = dict(max_grad_norm=1.0)

    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("max_grad_norm", 0.0, "max_grad_norm must be > 0"),
            ("training_autocast_precision", "bogus", "Unsupported"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(TrainingExecutionConfig, self._VALID, field, bad)


class TestPlacementConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("num_rollout_gpus_per_actor", 0, "num_rollout_gpus_per_actor must be >= 1"),
            ("num_train_nodes", 0, "num_train_nodes must be >= 1"),
            ("num_train_gpus_per_node", 0, "num_train_gpus_per_node must be >= 1"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(PlacementConfig, {}, field, bad)

    def test_accepts_zero_rollout_for_colocated_all_in_training(self):
        """NFT all-in colocated config: no separate rollout actor.

        ``num_rollout_nodes=0`` + ``num_rollout_gpus_per_node=0`` is the canonical
        shape; ``total_gpus`` falls back to ``total_train_gpus`` via the
        ``colocate=True`` branch.
        """
        cfg = PlacementConfig(
            num_rollout_nodes=0,
            num_rollout_gpus_per_node=0,
            num_train_nodes=2,
            num_train_gpus_per_node=8,
            colocate=True,
        )
        assert cfg.total_rollout_gpus == 0
        assert cfg.total_train_gpus == 16
        assert cfg.total_gpus == 16  # max(0, 16) under colocate


class TestModelBundleConfig:
    _VALID = dict(pretrained_model_ckpt_path="/some/path")

    def test_rejects_empty_lora_list(self):
        with pytest.raises(ValueError, match="must be None or a non-empty list"):
            ModelBundleConfig(**{**self._VALID, "lora_target_modules": []})

    def test_rejects_bad_precision(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ModelBundleConfig(**{**self._VALID, "model_precision": "bogus"})


class TestRewardConfig:
    """Flat RewardConfig invariants: aggregation_method, base_device, components."""

    @pytest.mark.parametrize("bad", ["bogus", "tpu", ""])
    def test_rejects_unknown_base_device(self, bad):
        with pytest.raises(ValueError, match="base_device must be cpu/cuda/auto"):
            RewardConfig(
                base_device=bad,
                components=(PickScoreSpec(weight=1.0),),
            )

    def test_rejects_unknown_aggregation_method(self):
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            RewardConfig(
                aggregation_method="bogus",
                components=(PickScoreSpec(weight=1.0),),
            )

    def test_rejects_empty_components(self):
        with pytest.raises(ValueError, match="components must be non-empty"):
            RewardConfig(aggregation_method="mean", base_device="cpu", components=())

    def test_accepts_non_empty_components(self):
        # Should construct without raising.
        RewardConfig(
            aggregation_method="mean",
            base_device="cpu",
            components=(PickScoreSpec(weight=1.0),),
        )


class TestTensorPayloadSyncConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("bucket_size", 0, "bucket_size must be >= 1"),
            ("target_modules", (), "must not be empty"),
            ("target_modules", ("",), "cannot contain empty entries"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(TensorPayloadSyncConfig, {}, field, bad)


class TestNcclBroadcastSyncConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("bucket_size", 0, "bucket_size must be >= 1"),
            ("target_modules", (), "must not be empty"),
            ("target_modules", ("",), "cannot contain empty entries"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(NcclBroadcastSyncConfig, {}, field, bad)


class TestCheckpointSyncConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("dir", "", "must be a non-empty path"),
            ("export_format", "unknown_format", "export_format must be one of"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(CheckpointSyncConfig, {}, field, bad)


class TestLoggingConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("logging_steps", -1, "logging_steps must be >= 0"),
            ("media_max_items", -1, "media_max_items must be >= 0"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(LoggingConfig, {}, field, bad)


class TestEvaluationConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("eval_steps", -1, "eval_steps must be >= 0"),
            ("eval_batch_size", 0, "eval_batch_size must be >= 1"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(EvaluationConfig, {}, field, bad)


class TestResumeConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("start_rollout_id", -1, "start_rollout_id must be >= 0"),
            ("save_steps", -1, "save_steps must be >= 0"),
            ("output_dir", "", "must be a non-empty string"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(ResumeConfig, {}, field, bad)


class TestRewardRangeFilterPlugin:
    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="min_reward must be <= max_reward"):
            RewardRangeFilterPlugin(min_reward=2.0, max_reward=1.0)


class TestRunConfig:
    @pytest.mark.parametrize(
        "field,bad,msg",
        [
            ("num_rollouts", 0, "num_rollouts must be >= 1"),
            ("weight_sync_interval", 0, "weight_sync_interval must be >= 1"),
            ("data_source_dotpath", "", "data_source_dotpath must be a non-empty dotpath"),
        ],
    )
    def test_rejects(self, field, bad, msg):
        with pytest.raises(ValueError, match=msg):
            _build(RunConfig, {}, field, bad)
