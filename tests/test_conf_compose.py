"""Hydra compose smoke tests for the Step-4 YAML surface.

Prove that ``conf/train.yaml`` composes against the registered-leaf defaults
and that each section with a registered ``_target_`` round-trips through
``build()`` into its runtime instance. The rollout/train bootstraps (Step 4
part 2) slice this exact cfg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Trigger @register_config side effects so ConfigStore has every leaf ready.
import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.algorithms.grpo import GRPOAlgorithm, GRPOAlgorithmConfig
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.instantiate import build, materialize
from diffusionrl.distributed import transfer_queue as _transfer_queue  # noqa: F401
from diffusionrl.distributed import weight_sync as _weight_sync  # noqa: F401
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.ray import placement as _placement  # noqa: F401
from diffusionrl.ray.train_actor import TrainingExecutionConfig  # noqa: F401
from diffusionrl.reward import config as _reward_config  # noqa: F401
from diffusionrl.samplers.fsdp import engine as _fsdp_engine  # noqa: F401
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.training.backends import fsdp as _fsdp  # noqa: F401
from diffusionrl.training.backends import veomni as _veomni  # noqa: F401
from diffusionrl.types import sampling as _sampling  # noqa: F401

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@pytest.fixture
def hydra_context():
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        yield


def test_train_yaml_composes_registered_leaves(hydra_context):
    cfg = compose(config_name="train")

    # Every defaults entry pointing at a leaf with a `target=` ends up with
    # `_target_` set on the composed section.
    assert cfg.algorithm._target_ == "diffusionrl.algorithms.grpo.GRPOAlgorithm"
    assert cfg.model._target_ == "diffusionrl.models.flux.FluxModelBundle"
    assert cfg.training.backend._target_ == "diffusionrl.training.backends.fsdp.FSDPBackend"

    # Schema-only leaves (registered without a target) resolve to their
    # dataclass's field defaults with no `_target_`.
    assert cfg.reward.get("_target_") is None
    assert cfg.sampling.get("_target_") is None
    # Reward dropped the provider sub-config; HTTP is now a component flavor.
    assert "provider" not in cfg.reward

    # Per-run scalars live under cfg.run (no top-level scalars).
    assert cfg.run.seed == 42


@pytest.mark.parametrize(
    "variant,target",
    [
        ("tensor_payload", "diffusionrl.distributed.weight_sync.tensor.UpdateWeightFromTensor"),
        ("nccl_broadcast", "diffusionrl.distributed.weight_sync.nccl.UpdateWeightFromDistributed"),
        ("checkpoint_path", "diffusionrl.distributed.weight_sync.checkpoint.UpdateWeightFromCheckpoint"),
    ],
)
def test_sync_variant_composes(hydra_context, variant, target):
    cfg = compose(config_name="train", overrides=[f"+sync={variant}"])
    assert cfg.sync._target_ == target


def test_transfer_queue_absent_by_default(hydra_context):
    """No defaults entry → cfg.transfer_queue is absent (TQ disabled)."""
    cfg = compose(config_name="train")
    assert cfg.get("transfer_queue") is None


@pytest.mark.parametrize(
    "variant,target",
    [
        ("simple", "diffusionrl.distributed.transfer_queue.simple.SimpleBackend"),
        ("mooncake", "diffusionrl.distributed.transfer_queue.mooncake.MooncakeBackend"),
    ],
)
def test_transfer_queue_variant_composes(hydra_context, variant, target):
    cfg = compose(config_name="train", overrides=[f"+transfer_queue={variant}"])
    assert cfg.transfer_queue._target_ == target


def test_mooncake_tuned_overlay_composes_on_colocate_experiment(hydra_context):
    """Yaml-side mooncake_tuned overlay inherits the structured config + applies tuned values."""
    from diffusionrl.distributed.transfer_queue.mooncake import MooncakeBackendConfig

    cfg = compose(
        config_name="train",
        overrides=[
            "+experiment=flowgrpo_sglang_sd3_colocate",
            "+transfer_queue=mooncake_tuned",
            "transfer_queue.metadata_server=http://h:1/metadata",
            "transfer_queue.master_server_address=h:2",
        ],
    )
    assert cfg.transfer_queue._target_ == "diffusionrl.distributed.transfer_queue.mooncake.MooncakeBackend"
    assert cfg.transfer_queue.global_segment_size_gb == 10
    assert cfg.transfer_queue.zero_copy.tensor_buffer_size_gb == 2.0
    assert cfg.transfer_queue.zero_copy.single_controller_tensor_buffer_size_gb == 10.0
    # device_name is auto-discovered at runtime (runtime.create_client), not
    # carried in the yaml — overlay leaves it None.
    assert cfg.transfer_queue.device_name is None
    # Schema preserved through the overlay (defaults: - mooncake) so validate(cfg)
    # walks it as a typed leaf and __post_init__ checks fire.
    assert OmegaConf.get_type(cfg.transfer_queue) is MooncakeBackendConfig


def test_mooncake_tuned_overlay_respects_device_name_cli_override(hydra_context):
    """Explicit transfer_queue.device_name=<name> at the CLI is preserved through compose."""
    cfg = compose(
        config_name="train",
        overrides=[
            "+transfer_queue=mooncake_tuned",
            "transfer_queue.metadata_server=http://h:1/metadata",
            "transfer_queue.master_server_address=h:2",
            "transfer_queue.device_name=mlx5_bond_3",
        ],
    )
    assert cfg.transfer_queue.device_name == "mlx5_bond_3"
    obj = materialize(cfg.transfer_queue)
    assert obj.device_name == "mlx5_bond_3"


def test_mooncake_tuned_overlay_fail_fasts_without_server_addresses(hydra_context):
    """__post_init__ require() fires when host-dependent server addresses aren't injected."""
    cfg = compose(
        config_name="train",
        overrides=["+transfer_queue=mooncake_tuned"],
    )
    with pytest.raises(ValueError, match="metadata_server must be set"):
        materialize(cfg.transfer_queue)


def test_build_on_composed_algorithm_returns_runtime_instance(hydra_context):
    """The core Step-4 primitive: cfg.section → build() → runtime object."""
    cfg = compose(config_name="train")

    algorithm = build(cfg.algorithm)
    assert isinstance(algorithm, GRPOAlgorithm)
    # Inside the algorithm, self.config is the registered dataclass.
    assert isinstance(algorithm.config, GRPOAlgorithmConfig)
    # SDE params live in a nested SDEConfig (same shape as SamplingParams.sde_config).
    # Strategy choice is its own Hydra group at cfg.sampling.sde_strategy.
    assert cfg.algorithm.sde_config.eta == pytest.approx(1.0)
    assert cfg.algorithm.sde_config.shift == pytest.approx(3.0)
    assert cfg.sampling.sde_strategy._target_ == "diffusionrl.sde.kernels.FlowSDEStrategy"
    # Property API still resolves through the nested struct.
    assert algorithm.eta == pytest.approx(1.0)
    assert algorithm.time_shift == pytest.approx(3.0)


def test_yaml_override_reaches_build(hydra_context):
    """Per-field YAML/CLI overrides flow through build() into the runtime object."""
    cfg = compose(
        config_name="train",
        overrides=["algorithm.kl_coef=0.25"],
    )
    algorithm = build(cfg.algorithm)
    assert algorithm.config.kl_coef == pytest.approx(0.25)


def test_switching_algorithm_preset_rewires_target(hydra_context):
    """Swap algorithm preset at compose time; _target_ and instance type follow."""
    from diffusionrl.algorithms.nft import NFTAlgorithm

    cfg = compose(
        config_name="train",
        overrides=["algorithm=nft"],
    )
    assert cfg.algorithm._target_ == "diffusionrl.algorithms.nft.NFTAlgorithm"
    algorithm = build(cfg.algorithm)
    assert isinstance(algorithm, NFTAlgorithm)


def test_rollout_section_ready_for_bootstrap(hydra_context):
    """The rollout cfg slice that ``RolloutActorGroup.bootstrap`` consumes.

    Verifies the shape: engine section with _target_ (for ``build(cfg.rollout.engine, rank=...)``).
    Sweeps to the SGLang engine to keep the SGLang-specific assertions meaningful;
    base default is fsdp (direct sampling), which is exercised by the bare-compose
    test elsewhere.
    """
    cfg = compose(config_name="train", overrides=["rollout/engine=sglang"])
    assert cfg.rollout.engine._target_ == "diffusionrl.samplers.sglang.engine.SGLangRolloutEngine"
    # forward_batch_size lives on RolloutPlan (engine-agnostic);
    # allow_noset_multi_gpu_inference lives on PlacementConfig.
    assert cfg.rollout.plan.forward_batch_size is None
    assert cfg.placement.allow_noset_multi_gpu_inference is False
    # Sampling is nested — same shape as FSDPEngineConfig.sampling.
    assert cfg.rollout.engine.sampling.num_inference_steps == 50
    assert cfg.rollout.engine.sampling.sde_config.shift == pytest.approx(3.0)


def test_switching_to_fsdp_engine_rewires_target(hydra_context):
    """Overriding rollout/engine=fsdp swaps the registered preset: _target_
    flips to FSDPSamplingEngine and the nested SamplingParams becomes
    the section payload.
    """
    cfg = compose(
        config_name="train",
        overrides=["rollout/engine=fsdp"],
    )
    assert cfg.rollout.engine._target_ == "diffusionrl.samplers.fsdp.engine.FSDPSamplingEngine"
    # FSDPEngineConfig carries sampling as a nested SamplingParams.
    assert cfg.rollout.engine.sampling.num_inference_steps == 50
    assert cfg.rollout.engine.sampling.guidance_scale == pytest.approx(7.5)


def test_train_sections_ready_for_bootstrap(hydra_context):
    """cfg.model / cfg.algorithm / cfg.training.backend carry _target_; the
    schema-only sections materialize via ``OmegaConf.to_object`` into their
    dataclass types.
    """
    from omegaconf import OmegaConf

    from diffusionrl.training.plan import TrainingPlan
    from diffusionrl.training.types import TrainTopology

    cfg = compose(
        config_name="train",
        overrides=["model.pretrained_model_ckpt_path=/tmp/model"],
    )
    assert cfg.model._target_ == "diffusionrl.models.flux.FluxModelBundle"
    assert cfg.algorithm._target_ == "diffusionrl.algorithms.grpo.GRPOAlgorithm"
    assert cfg.training.backend._target_ == "diffusionrl.training.backends.fsdp.FSDPBackend"

    plan = OmegaConf.to_object(cfg.training.plan)
    topo = OmegaConf.to_object(cfg.training.topology)
    assert isinstance(plan, TrainingPlan)
    assert isinstance(topo, TrainTopology)


def test_algorithm_sampling_inherits_from_top_level(hydra_context):
    """``cfg.algorithm.sampling`` interpolates from ``cfg.sampling`` by default.

    Pins the conf/train.yaml override that wires ``algorithm.sampling: ${sampling}``
    so an experiment that doesn't explicitly override ``algorithm.sampling.*``
    still gets the canonical sampling spec.
    """
    cfg = compose(config_name="train")
    OmegaConf.resolve(cfg)
    assert cfg.algorithm.sampling.num_inference_steps == cfg.sampling.num_inference_steps
    assert cfg.algorithm.sampling.guidance_scale == cfg.sampling.guidance_scale


def test_algorithm_sampling_per_field_override_writes_through_interpolation(hydra_context):
    """``algorithm.sampling`` is a live interpolation back to ``cfg.sampling``
    (declared as the dataclass default in ``BaseAlgorithmConfig``). Per-field
    CLI overrides on ``algorithm.sampling.*`` write *through* the
    interpolation to the top-level source — both nodes see the new value.

    This is OmegaConf's standard interpolation semantics. To genuinely diverge,
    overrides should target the canonical ``sampling.*`` (which both reflect)
    or replace the whole ``algorithm.sampling`` subtree.
    """
    cfg = compose(
        config_name="train",
        overrides=["algorithm.sampling.num_inference_steps=99"],
    )
    OmegaConf.resolve(cfg)
    assert cfg.algorithm.sampling.num_inference_steps == 99
    assert cfg.sampling.num_inference_steps == 99  # interpolation source updated too
