"""Real E2E test: SGLang rollout + FSDP training actor full pipeline.

Tests the full Ray pipeline:
  rollout actor → generate → buffer → compute_advantages → to_training_batch
  → training actor train (via train_from_buffer and train with ObjectRef)

Requires:
  - 2 GPUs (1 rollout, 1 training)
  - SD3.5 Medium model at MODEL_PATH
  - SGLang installed

Run: python tests/test_train_sglang_e2e.py
"""

from __future__ import annotations

import json
import math
import socket
import sys
import time
from pathlib import Path

import ray
import torch

from diffusionrl.config.training_sections import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import RolloutActorConfig
from diffusionrl.ray.rollout_actor import RolloutActor
from diffusionrl.ray.train_actor import TrainActor
from diffusionrl.reward.config import RewardSpec
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sampling import SamplingParams, SDEConfig
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.training.backends.fsdp import FSDPBackendConfig
from diffusionrl.algorithms.grpo import GRPOAlgorithmConfig

MODEL_PATH = "/mnt/bj/models/stable-diffusion-3.5-medium"
NUM_STEPS = 28
# Skip the very last SDE step — its log_prob is unstable (boundary at t->0)
# and produces NaN ratios during GRPO training. This matches the
# skip_last_timestep convention used by SD3 production configs.
SDE_INDICES = list(range(NUM_STEPS - 1))
NUM_SAMPLES_PER_PROMPT = 4
OCR_PROMPTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "samples" / "ocr_prompts_toy.json"
)


def load_ocr_prompts() -> list[str]:
    """Load OCR prompts (each contains a quoted target word the scorer extracts)."""
    with open(OCR_PROMPTS_PATH) as f:
        return [item["prompt"] for item in json.load(f)]


def _pick_free_port() -> int:
    """Bind to an ephemeral port on localhost and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_reward_spec(
    *,
    reward_components: list[str] | None,
) -> RewardSpec:
    return RewardSpec(
        reward_dotpath=None,
        reward_model_ckpt_path=None,
        reward_batch_size=1,
        local_reward_device="cpu",
        reward_backend="local",
        reward_service_urls=None,
        reward_components=reward_components,
        reward_weights=None,
        reward_aggregation_method="mean",
    )


# ---------------------------------------------------------------
# Config builders — rollout side
# ---------------------------------------------------------------

def build_engine_config() -> EngineConfig:
    return EngineConfig(
        pretrained_model_ckpt_path=MODEL_PATH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        sde_type="flow",
        shift=3.0,
        height=512,
        width=512,
        num_frames=1,
        local_mode=True,
        num_gpus=1,
        logprob_source="native",
    )


def build_rollout_actor_config() -> RolloutActorConfig:
    return RolloutActorConfig(
        engine_init_payload=ComponentInitPayload(
            component_dotpath="sglang",
            component_config=build_engine_config(),
        ),
        reward_config=_build_reward_spec(
            reward_components=["ocr"],
        ),
        algorithm_init_payload=ComponentInitPayload(
            component_dotpath="grpo",
            component_config=GRPOAlgorithmConfig(
                samples_per_prompt=NUM_SAMPLES_PER_PROMPT,
                num_inference_steps=NUM_STEPS,
            ),
        ),
    )


def build_sampling_params() -> SamplingParams:
    return SamplingParams(
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        height=512,
        width=512,
        num_frames=1,
        seed=42,
        num_samples_per_prompt=NUM_SAMPLES_PER_PROMPT,
        sde_config=SDEConfig(eta=1.0, sde_type="flow", shift=3.0),
        sde_indices=SDE_INDICES,
    )


def build_request(prompts: list[str]) -> RolloutRequest:
    p = Prompts.from_unique_prompts(prompts)
    if NUM_SAMPLES_PER_PROMPT > 1:
        p = p.expand(NUM_SAMPLES_PER_PROMPT)
    return RolloutRequest(prompts=p, sampling_params=build_sampling_params())


# ---------------------------------------------------------------
# Config builders — training side
# ---------------------------------------------------------------

def build_train_actor_kwargs(
    *,
    rank: int,
    master_addr: str,
    master_port: int,
    batch_size: int = NUM_SAMPLES_PER_PROMPT,
) -> dict:
    return dict(
        world_size=1,
        rank=rank,
        master_addr=master_addr,
        master_port=master_port,
        mini_batch_size=batch_size,
        micro_batch_size=batch_size,
        max_grad_norm=1.0,
        backend_config=FSDPBackendConfig(),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
        ),
        scheduler_config=LrSchedulerConfig(
            type="constant",
            warmup_steps=0,
            total_steps=100,
        ),
        reward_spec=_build_reward_spec(
            reward_components=None,
        ),
        algorithm_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.algorithms.grpo.GRPOAlgorithm",
            component_config=GRPOAlgorithmConfig(
                num_inference_steps=NUM_STEPS,
                sde_config=SDEConfig(eta=1.0, sde_type="flow", shift=3.0),
                model_type="sd3",
                skip_last_timestep=True,
            ),
        ),
        model_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.models.sd3.SD3ModelBundle",
            component_config=ModelBundleConfig(
                pretrained_model_ckpt_path=MODEL_PATH,
                model_precision="bf16",
                training_only=True,
            ),
        ),
        training_autocast_precision="bf16",
    )


# ---------------------------------------------------------------
# Helper: produce a TrainingBatch from rollout
# ---------------------------------------------------------------

def produce_training_batch(rollout_actor) -> TrainingBatch:
    """Run rollout, attach real OCR rewards, compute advantages, convert to TrainingBatch."""
    prompts = load_ocr_prompts()
    request = build_request(prompts)
    handles = ray.get(rollout_actor.generate_buffered.remote(request))
    assert len(handles) == len(prompts), f"expected {len(prompts)} group handles, got {len(handles)}"

    # Attach real OCR rewards on the rollout actor (one group at a time),
    # then z-score normalize per group via compute_advantages.
    for h in handles:
        ray.get(rollout_actor.attach_reward.remote(h))
        ray.get(rollout_actor.compute_advantages.remote(h))

    # Use the first group as the training batch (4 samples for one prompt).
    response = ray.get(rollout_actor.get_buffer.remote(handles[0]))
    assert response.samples.rewards is not None, "OCR rewards must be attached"
    assert not torch.isnan(response.samples.rewards).any(), "OCR rewards should not be NaN"
    print(f"  OCR rewards (group 0): {response.samples.rewards.tolist()}")
    if getattr(response.samples, "component_rewards", None):
        print(f"  component_rewards: {response.samples.component_rewards}")

    return response.to_training_batch(sde_indices=set(SDE_INDICES))


# ---------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------

def test_rollout_to_training_batch(rollout_actor):
    """Validate TrainingBatch produced from rollout."""
    print("\n[test] rollout_to_training_batch")
    batch = produce_training_batch(rollout_actor)

    assert isinstance(batch, TrainingBatch)
    assert batch.batch_size == NUM_SAMPLES_PER_PROMPT
    assert batch.trajectory_store is not None
    assert batch.advantages is not None
    assert batch.forward_context is not None
    assert batch.timesteps is not None
    assert batch.rewards is not None
    assert batch.prompts is not None

    print(f"  batch_size={batch.batch_size}, "
          f"traj_positions={batch.trajectory_store.total_positions}, "
          f"log_probs={batch.log_probs is not None}")
    print("  PASS")


def test_train_from_buffer(training_actor, rollout_actor):
    """Full pipeline: rollout → buffer → train_from_buffer."""
    print("\n[test] train_from_buffer — full pipeline")
    batch = produce_training_batch(rollout_actor)

    # Store batch in rollout actor's buffer, get handle
    handle = ray.get(rollout_actor.put_buffer.remote(batch, batch))
    assert handle is not None
    # BufferHandle.actor_handle is not set automatically by Ray —
    # patch it with the known rollout actor reference.
    handle.actor_handle = rollout_actor

    # Train via buffer handle
    t0 = time.perf_counter()
    result = ray.get(training_actor.train_from_buffer.remote(0, handle))
    elapsed = time.perf_counter() - t0

    assert result.rollout_step == 0
    assert not math.isnan(result.loss),      f"loss is NaN: {result}"
    assert not math.isinf(result.loss),      f"loss is Inf: {result}"
    assert not math.isnan(result.grad_norm), f"grad_norm is NaN: {result}"
    assert result.has_backward
    assert result.optimizer_steps >= 1

    print(f"  loss={result.loss:.4f}, grad_norm={result.grad_norm:.4f}, "
          f"lr={result.lr:.2e}, elapsed={elapsed:.1f}s")
    print("  PASS")


def test_train_via_objectref(training_actor, rollout_actor):
    """Train using existing train() with ray.put(batch)."""
    print("\n[test] train via ObjectRef — backward compat")
    batch = produce_training_batch(rollout_actor)

    batch_ref = ray.put(batch)
    t0 = time.perf_counter()
    result = ray.get(training_actor.train.remote(1, batch_ref))
    elapsed = time.perf_counter() - t0

    assert result.rollout_step == 1
    assert not math.isnan(result.loss)
    assert not math.isnan(result.grad_norm)

    print(f"  loss={result.loss:.4f}, grad_norm={result.grad_norm:.4f}, "
          f"elapsed={elapsed:.1f}s")
    print("  PASS")


def test_train_direct_batch(training_actor, rollout_actor):
    """Train by passing TrainingBatch directly (no ObjectRef)."""
    print("\n[test] train with direct TrainingBatch")
    batch = produce_training_batch(rollout_actor)

    result = ray.get(training_actor.train.remote(2, batch))

    assert result.rollout_step == 2
    assert not math.isnan(result.loss)

    print(f"  loss={result.loss:.4f}")
    print("  PASS")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("SGLang Rollout + FSDP Training E2E Test")
    print("=" * 60)

    ray.init(ignore_reinit_error=True)
    resources = ray.cluster_resources()
    num_gpus = int(resources.get("GPU", 0))
    print(f"Ray resources: {resources}")

    if num_gpus < 2:
        print(f"SKIP: need 2 GPUs, have {num_gpus}")
        ray.shutdown()
        sys.exit(0)

    # --- Create rollout actor ---
    print("\n--- Rollout actor setup ---")
    rollout_actor = RolloutActor.options(num_gpus=1).remote(
        rank=0, world_size=1, num_gpus_allocated=1,
    )
    t0 = time.time()
    ray.get(rollout_actor.init.remote(build_rollout_actor_config()))
    print(f"Rollout actor initialized in {time.time() - t0:.1f}s")

    # --- Create training actor ---
    print("\n--- Training actor setup ---")
    master_addr = "127.0.0.1"
    master_port = _pick_free_port()
    t0 = time.time()
    training_actor = TrainActor.remote(
        **build_train_actor_kwargs(
            rank=0,
            master_addr=master_addr,
            master_port=master_port,
        )
    )
    # Force eager init to complete by round-tripping a no-op call.
    assert ray.get(training_actor.health_check.remote())
    print(f"Training actor initialized in {time.time() - t0:.1f}s")

    # --- Run tests ---
    print("\n--- Tests ---")
    test_rollout_to_training_batch(rollout_actor)
    test_train_from_buffer(training_actor, rollout_actor)
    test_train_via_objectref(training_actor, rollout_actor)
    test_train_direct_batch(training_actor, rollout_actor)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    ray.shutdown()


if __name__ == "__main__":
    main()
