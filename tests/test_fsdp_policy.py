"""Unit tests for FSDPPolicy block-class discovery + duck-typed stage contract.

The tests here exercise the pure-Python discovery logic (HF
``_no_split_modules`` MRO walk → stage-class fallback) without invoking
``fully_shard`` (which requires a torch.distributed process group). The
real wrap path is exercised by the FSDP smoke
(``scripts/smoke_hunyuan_image3_t2i_replay_fsdp.py``).
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import pytest
import torch
from torch import nn

from diffusionrl.training_new.fsdp_policy import FSDPPolicy, FSDPPolicyConfig

# ---------------------------------------------------------------------------
# Test scaffolding: a FSDPPolicy subclass that skips the actual fully_shard
# call so we can construct it in a single-process test.
# ---------------------------------------------------------------------------


class _NoWrapPolicy(FSDPPolicy):
    """FSDPPolicy variant that exercises everything except the FSDP wrap.

    Lets us unit-test discovery + enumeration + property surface without
    needing a torch.distributed process group.
    """

    def _wrap_model(self) -> None:  # type: ignore[override]
        # Run the discovery side-effects so callers can assert on them, but
        # skip ``fully_shard`` which requires ``dist.init_process_group``.
        self._discovered_classes = self._discover_block_classes()
        self._discovered_instances = self._enumerate_block_instances(self._discovered_classes)


def _make_config() -> FSDPPolicyConfig:
    return FSDPPolicyConfig(
        cpu_offload=False,
        param_dtype="bf16",
        mixed_precision=True,
        fsdp_mode="full",
        reshard_after_forward=True,
    )


# ---------------------------------------------------------------------------
# Block-class discovery
# ---------------------------------------------------------------------------


def test_discovery_reads_hf_no_split_modules_from_root_mro():
    """When the trainable_root's class hierarchy has ``_no_split_modules``
    (HF ``PreTrainedModel`` convention), FSDPPolicy reads it from there
    and ignores the stage-class fallback.
    """

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

    class FakeBlockOther(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

    class FakeRoot(nn.Module):
        # HF convention. Note: class names, not classes.
        _no_split_modules = ["FakeBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.b0 = FakeBlock()
            self.b1 = FakeBlock()
            self.aux = FakeBlockOther()  # NOT a target.

    class FakeStage:
        # Stage-class fallback should be ignored when MRO discovery wins.
        _no_split_modules: ClassVar[Tuple[str, ...]] = ("StageOnlyBlock",)

        def __init__(self) -> None:
            self._root = FakeRoot()

        def trainable_module(self) -> nn.Module:
            return self._root

    stage = FakeStage()
    policy = _NoWrapPolicy(_make_config(), stage)

    assert policy._discovered_classes == ("FakeBlock",)
    assert len(policy._discovered_instances) == 2
    assert all(type(m).__name__ == "FakeBlock" for m in policy._discovered_instances)


def test_discovery_falls_back_to_stage_class_attribute():
    """When no ancestor in the trainable_root's MRO declares
    ``_no_split_modules``, FSDPPolicy reads from ``type(stage)._no_split_modules``.
    Used by SD3 (diffusers doesn't follow the HF convention).
    """

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class FakeRoot(nn.Module):
        # No _no_split_modules anywhere in the MRO.
        def __init__(self) -> None:
            super().__init__()
            self.b0 = FakeBlock()
            self.b1 = FakeBlock()
            self.b2 = FakeBlock()

    class FakeStage:
        _no_split_modules: ClassVar[Tuple[str, ...]] = ("FakeBlock",)

        def __init__(self) -> None:
            self._root = FakeRoot()

        def trainable_module(self) -> nn.Module:
            return self._root

    stage = FakeStage()
    policy = _NoWrapPolicy(_make_config(), stage)

    assert policy._discovered_classes == ("FakeBlock",)
    assert len(policy._discovered_instances) == 3


def test_discovery_empty_when_neither_source_declares(caplog):
    """Both sources empty → empty discovery + warning. FSDPPolicy will
    fall through to root-only wrap (not exercised here since we skip the
    actual fully_shard call)."""

    class FakeRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

    class FakeStage:
        # No _no_split_modules.
        def __init__(self) -> None:
            self._root = FakeRoot()

        def trainable_module(self) -> nn.Module:
            return self._root

    stage = FakeStage()
    with caplog.at_level("WARNING"):
        policy = _NoWrapPolicy(_make_config(), stage)
    assert policy._discovered_classes == ()
    assert policy._discovered_instances == ()


def test_init_raises_when_stage_lacks_trainable_module():
    """The duck-typed contract on the stage is the only thing FSDPPolicy
    needs. Missing it should fail loudly at construction time."""

    class BrokenStage:
        # No trainable_module() method.
        pass

    with pytest.raises(TypeError, match="trainable_module"):
        _NoWrapPolicy(_make_config(), BrokenStage())


# ---------------------------------------------------------------------------
# Module-proxy surface (no FSDP wrap involved — just delegation)
# ---------------------------------------------------------------------------


def test_policy_proxies_train_eval_and_parameters():
    """``policy.train()`` / ``.eval()`` / ``.parameters()`` proxy through
    to the FSDP-wrapped trainable_root. With _NoWrapPolicy, ``self.model``
    is the unwrapped root, so these reduce to plain nn.Module calls."""

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class FakeRoot(nn.Module):
        _no_split_modules = ["FakeBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.b = FakeBlock()

    class FakeStage:
        def __init__(self) -> None:
            self._root = FakeRoot()

        def trainable_module(self) -> nn.Module:
            return self._root

        def replay(self, *args, **kwargs) -> str:
            return "replayed"

    policy = _NoWrapPolicy(_make_config(), FakeStage())

    # Proxy methods.
    assert policy.training is True  # nn.Module default
    policy.eval()
    assert policy.training is False
    policy.train()
    assert policy.training is True

    # parameters() returns the same iterator as the wrapped module's.
    assert list(policy.parameters()) == list(policy.model.parameters())

    # replay forwards to stage.replay duck-typed.
    assert policy.replay() == "replayed"


def test_is_materialized_reflects_meta_state():
    """``is_materialized`` walks parameters and returns False if any is meta."""

    class FakeBlock(nn.Module):
        _no_split_modules = ["FakeBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class FakeStage:
        def __init__(self, root: nn.Module) -> None:
            self._root = root

        def trainable_module(self) -> nn.Module:
            return self._root

    real_stage = FakeStage(FakeBlock())
    real_policy = _NoWrapPolicy(_make_config(), real_stage)
    assert real_policy.is_materialized is True

    # Build a meta-device version.
    with torch.device("meta"):
        meta_root = FakeBlock()
    meta_stage = FakeStage(meta_root)
    meta_policy = _NoWrapPolicy(_make_config(), meta_stage)
    assert meta_policy.is_materialized is False
