"""Contract tests for ``validate_keep_local_contract``.

Keep-local (direct-sampling, no-driver-transfer) must be (a) direct-sampling
only, (b) mutually exclusive with TransferQueue, and (c) used only when the
rollout's prompt groups divide evenly across the train actors — otherwise its
prompt-group sharding produces a different per-actor partition (and gradient)
than the gathered path, so it would stop being a transparent optimization.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from diffusionrl.config.validation import validate_keep_local_contract

_TRAINSIDE = "diffusionrl.rollout.engine.trainside.engine.TrainsideRolloutEngine"
_SEPARATE = "diffusionrl.rollout.engine.sglang.engine.SGLangRolloutEngine"


def _cfg(*, keep_local=True, engine=_TRAINSIDE, prompts=32, actors=16, transfer_queue=False):
    d = {
        "training": {
            "execution": {"keep_local": keep_local},
            "topology": {"actor_count": actors},
        },
        "algorithm": {"prompts_per_rollout": prompts},
        "rollout": {"engine": {"_target_": engine}},
    }
    if transfer_queue:
        d["transfer_queue"] = {"backend": "simple"}
    return OmegaConf.create(d)


def test_disabled_is_noop_even_when_indivisible():
    # keep_local=False must short-circuit before any contract check.
    validate_keep_local_contract(_cfg(keep_local=False, prompts=3, actors=2))


def test_direct_sampling_divisible_passes():
    validate_keep_local_contract(_cfg(prompts=32, actors=16))


def test_single_actor_passes():
    # n==1 → every split is trivially even.
    validate_keep_local_contract(_cfg(prompts=7, actors=1))


def test_indivisible_prompts_rejected():
    with pytest.raises(ValueError, match="divisible"):
        validate_keep_local_contract(_cfg(prompts=3, actors=2))


def test_separate_sampling_rejected():
    with pytest.raises(ValueError, match="direct sampling"):
        validate_keep_local_contract(_cfg(engine=_SEPARATE))


def test_transfer_queue_mutually_exclusive():
    with pytest.raises(ValueError, match="transfer_queue"):
        validate_keep_local_contract(_cfg(transfer_queue=True))
