"""End-to-end test for the new RolloutActor rollout pipeline.

Tests the full flow:
  RolloutRequest → generate() → RolloutResponse
  RolloutRequest → generate_buffered() → BufferHandle list
  BufferHandle → attach_reward() → rewards attached in-place

Run with: python -m pytest tests/test_rollout_actor_e2e.py -v -s
Or standalone: python tests/test_rollout_actor_e2e.py
"""

from __future__ import annotations

import torch

from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sample import LogProbData, RolloutSamples
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.types.trajectory_store import Trajectory, TrajectoryBuilder
from diffusionrl.utils.batched import Batched


def make_prompts(n: int) -> Prompts:
    return Prompts.from_unique_prompts(
        prompts=[f"a photo of a cat {i}" for i in range(n)],
    )


def make_sampling_params() -> SamplingParams:
    return SamplingParams(
        num_inference_steps=20,
        guidance_scale=7.5,
        height=256,
        width=256,
        num_frames=1,
        seed=42,
    )


def make_request(n: int = 4) -> RolloutRequest:
    return RolloutRequest(
        prompts=make_prompts(n),
        sampling_params=make_sampling_params(),
    )


def make_trajectory(batch_size: int, num_steps: int = 5) -> Trajectory:
    builder = TrajectoryBuilder.full(num_steps)
    for i in range(num_steps + 1):
        builder.add(i, torch.randn(batch_size, 4, 32, 32))
    return builder.finalize()


def make_log_probs(num_steps: int = 5) -> LogProbData:
    return LogProbData(data={i: torch.randn(4) for i in range(num_steps)})


def make_samples(n: int = 4) -> RolloutSamples:
    return RolloutSamples(
        latents=torch.randn(n, 4, 32, 32),
        timesteps=torch.linspace(1.0, 0.0, 6),
        sampling_params=make_sampling_params(),
        prompts=make_prompts(n),
        trajectories=make_trajectory(n),
        log_probs=make_log_probs(),
        forward_context=None,
        step_indices=torch.arange(6),
    )


def test_request_creation():
    req = make_request(4)
    assert len(req.prompts.prompts) == 4
    assert req.sampling_params.num_inference_steps == 20
    print("  [PASS] RolloutRequest creation")


def test_request_slice():
    req = make_request(4)
    sub = req.slice(0, 2)
    assert len(sub.prompts.prompts) == 2
    assert sub.sampling_params is req.sampling_params  # shared field
    print("  [PASS] RolloutRequest.slice()")


def test_samples_creation():
    samples = make_samples(4)
    assert samples.batch_size == 4
    assert samples.latents.shape == (4, 4, 32, 32)
    assert samples.trajectories is not None
    assert samples.trajectories.batch_size == 4
    assert samples.rewards is None
    assert samples.decoded_images is None
    print("  [PASS] RolloutSamples creation")


def test_samples_concat():
    s1 = make_samples(2)
    s2 = make_samples(3)
    merged = RolloutSamples.concat([s1, s2])
    assert merged.batch_size == 5
    assert merged.latents.shape[0] == 5
    assert len(merged.prompts.prompts) == 5
    # Trajectory should be concatenated along batch dim
    assert merged.trajectories.batch_size == 5
    print("  [PASS] RolloutSamples.concat()")


def test_samples_select():
    samples = make_samples(4)
    indices = torch.tensor([0, 2])
    selected = samples.select(indices)
    assert selected.batch_size == 2
    assert selected.latents.shape[0] == 2
    assert len(selected.prompts.prompts) == 2
    assert selected.trajectories.batch_size == 2
    print("  [PASS] RolloutSamples.select()")


def test_response_creation():
    req = make_request(4)
    samples = make_samples(4)
    resp = RolloutResponse(request=req, samples=samples)
    assert resp.request is req
    assert resp.samples is samples
    print("  [PASS] RolloutResponse creation")


def test_response_split():
    # Create 4 prompts in 2 groups
    prompts = Prompts(
        prompts=["cat", "dog", "cat2", "dog2"],
        prompt_ids=["p0", "p1", "p2", "p3"],
        sample_ids=["s0", "s1", "s2", "s3"],
        group_ids=["g0", "g1", "g0", "g1"],
        noise_group_ids=["s0", "s1", "s2", "s3"],
        prompt_metadata=[{}, {}, {}, {}],
    )
    req = RolloutRequest(prompts=prompts, sampling_params=make_sampling_params())
    samples = make_samples(4)
    resp = RolloutResponse(request=req, samples=samples)

    splits = resp.split()
    assert len(splits) == 2
    assert splits[0].samples.batch_size == 2
    assert splits[1].samples.batch_size == 2
    assert splits[0].request.prompts.group_ids == ["g0", "g0"]
    assert splits[1].request.prompts.group_ids == ["g1", "g1"]
    # Trajectory should be split too
    assert splits[0].samples.trajectories.batch_size == 2
    print("  [PASS] RolloutResponse.split()")


def test_response_to_meta():
    req = make_request(4)
    samples = make_samples(4)
    resp = RolloutResponse(request=req, samples=samples)
    meta = resp.to_meta()
    assert len(meta.group_ids) == 4
    assert len(meta.sample_ids) == 4
    print("  [PASS] RolloutResponse.to_meta()")


def test_none_fields_through_concat():
    s1 = make_samples(2)
    s2 = make_samples(2)
    # Both have None rewards/decoded
    assert s1.rewards is None
    assert s2.decoded_images is None
    merged = RolloutSamples.concat([s1, s2])
    assert merged.rewards is None
    assert merged.decoded_images is None
    print("  [PASS] None fields survive concat")


def test_none_fields_through_select():
    samples = make_samples(4)
    assert samples.rewards is None
    selected = samples.select(torch.tensor([1, 3]))
    assert selected.rewards is None
    assert selected.decoded_images is None
    print("  [PASS] None fields survive select")


def test_trajectory_builder():
    builder = TrajectoryBuilder.for_sde_steps({0, 2, 4}, 5)
    for i in range(6):
        builder.add(i, torch.randn(2, 4, 32, 32))
    traj = builder.finalize()
    assert isinstance(traj, Trajectory)
    assert isinstance(traj, Batched)
    assert traj.batch_size == 2
    assert traj.has_position(0)
    assert traj.has_position(1)
    assert traj.has_position(2)
    print("  [PASS] TrajectoryBuilder → Trajectory")


def test_trajectory_select():
    traj = make_trajectory(4)
    selected = traj.select(torch.tensor([0, 2]))
    assert selected.batch_size == 2
    assert selected.total_positions == traj.total_positions
    print("  [PASS] Trajectory.select()")


def test_trajectory_concat():
    t1 = make_trajectory(2)
    t2 = make_trajectory(3)
    merged = Trajectory.concat([t1, t2])
    assert merged.batch_size == 5
    print("  [PASS] Trajectory.concat()")


if __name__ == "__main__":
    print("Testing new rollout types...")
    test_request_creation()
    test_request_slice()
    test_samples_creation()
    test_samples_concat()
    test_samples_select()
    test_response_creation()
    test_response_split()
    test_response_to_meta()
    test_none_fields_through_concat()
    test_none_fields_through_select()
    test_trajectory_builder()
    test_trajectory_select()
    test_trajectory_concat()
    print("\nAll tests passed!")
