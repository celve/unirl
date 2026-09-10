"""E0 gate: seeds actually differ across seed IDs and reproduce on rerun.

Correctness test 6 of the runbook. The failure this guards against is real at the
audited commit: LoRA initialization drew from ungoverned global RNG, so two runs
configured with different seed IDs could share an initialization.
"""

from __future__ import annotations

import random

import torch

from unirl.utils.worker_rng import (
    derive_worker_seed,
    reset_seeding_provenance,
    seed_worker_rngs,
    seeding_provenance,
)


def test_distinct_seed_ids_give_distinct_derived_seeds():
    seeds = {derive_worker_seed(s, rank=0, purpose="lora_init") for s in (11, 22, 33)}
    assert len(seeds) == 3


def test_distinct_ranks_give_distinct_derived_seeds():
    seeds = {derive_worker_seed(11, rank=r, purpose="lora_init") for r in range(8)}
    assert len(seeds) == 8


def test_distinct_purposes_give_distinct_derived_seeds():
    assert derive_worker_seed(11, rank=0, purpose="lora_init") != derive_worker_seed(11, rank=0, purpose="model_init")


def test_derivation_is_stable_across_calls():
    assert derive_worker_seed(11, rank=3, purpose="lora_init") == derive_worker_seed(11, rank=3, purpose="lora_init")


def test_derived_seed_is_a_valid_torch_seed():
    for base in (0, 11, 2**31, 2**52):
        seed = derive_worker_seed(base, rank=0, purpose="lora_init")
        assert 0 <= seed < 2**63
        torch.manual_seed(seed)


def test_seeding_reproduces_the_same_draws():
    seed_worker_rngs(11, purpose="lora_init", rank=0)
    first = (torch.randn(4).tolist(), random.random())
    seed_worker_rngs(11, purpose="lora_init", rank=0)
    second = (torch.randn(4).tolist(), random.random())
    assert first == second


def test_different_seed_ids_change_the_draws():
    seed_worker_rngs(11, purpose="lora_init", rank=0)
    first = torch.randn(8).tolist()
    seed_worker_rngs(22, purpose="lora_init", rank=0)
    second = torch.randn(8).tolist()
    assert first != second


def test_null_base_seed_is_recorded_rather_than_defaulted():
    reset_seeding_provenance()
    assert seed_worker_rngs(None, purpose="lora_init", rank=0) is None
    record = seeding_provenance()[-1]
    assert record["base_seed"] is None and record["derived_seed"] is None


def test_provenance_records_every_seeded_purpose():
    reset_seeding_provenance()
    seed_worker_rngs(11, purpose="model_init", rank=2)
    seed_worker_rngs(11, purpose="lora_init:default", rank=2)
    purposes = [record["purpose"] for record in seeding_provenance()]
    assert purposes == ["model_init", "lora_init:default"]
    assert all(record["rank"] == 2 for record in seeding_provenance())
