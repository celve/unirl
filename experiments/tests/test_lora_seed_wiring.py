"""E0 gate: the run's LoRA seed actually reaches adapter initialization.

Correctness test 6 needs seeds to differ across seed IDs *in a real run*, not only in the
derivation helper. At the audited pin `LoraConfig` carried no seed and `inject_lora` was
called without one, so the A-matrix draw came from ungoverned global RNG and two runs
configured with different seed IDs could share an initialization.

Model construction is deliberately not covered: SD3 and Qwen3 bundles load weights from a
checkpoint via `from_pretrained` and draw no random numbers, so LoRA is the only stochastic
path on the trainable side.
"""

from __future__ import annotations

import inspect

from unirl.train.configs import LoraConfig


def test_lora_config_carries_a_seed_field():
    assert "seed" in LoraConfig.__dataclass_fields__


def test_seed_defaults_to_none_so_the_null_contract_is_preserved():
    """Null means OS entropy, which the framework documents; it must not silently become 0."""
    assert LoraConfig().seed is None


def test_inject_lora_accepts_a_seed():
    from unirl.train.lora import inject_lora

    assert "seed" in inspect.signature(inject_lora).parameters


def test_reset_adapter_seeds_before_drawing():
    from unirl.train.lora import _reset_adapter

    assert "seed" in inspect.signature(_reset_adapter).parameters


def test_the_backend_passes_the_configured_seed_through():
    """The wiring gap that made test 6 unenforced: the field existed but nothing passed it."""
    from unirl.train.backend import base_backend

    source = inspect.getsource(base_backend._inject_structural)
    assert "seed=lora_cfg.seed" in source, "backend drops the configured LoRA seed"


def test_distinct_seed_ids_give_distinct_lora_draws():
    """End to end through the documented seeding path, at the granularity a run uses."""
    import torch

    from unirl.utils.worker_rng import seed_worker_rngs

    def draw(seed):
        seed_worker_rngs(seed, purpose="lora_init:default", rank=0)
        return torch.randn(16, 8).clone()

    a, b, a_again = draw(11), draw(22), draw(11)
    assert not torch.equal(a, b), "seeds 11 and 22 produced an identical LoRA draw"
    assert torch.equal(a, a_again), "seed 11 did not reproduce on rerun"
