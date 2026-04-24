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
from diffusionrl.types.sample import LogProbData, MediaPreview, RolloutSamples
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


# ---------------------------------------------------------------------------
# Media-preview plumbing (wandb eval + actor-side image drop)
# ---------------------------------------------------------------------------


class _FakePIL:
    """Minimal PIL-image stand-in that only exposes ``save`` (duck-typed)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def save(self, *args, **kwargs) -> None:  # pragma: no cover - never called
        return None


def test_request_carries_media_preview_knobs():
    req = RolloutRequest(
        prompts=make_prompts(2),
        sampling_params=make_sampling_params(),
        collect_media_preview=True,
        media_max_items=3,
    )
    assert req.collect_media_preview is True
    assert req.media_max_items == 3
    # Defaults should be off so unrelated callers keep legacy behaviour.
    default_req = make_request(2)
    assert default_req.collect_media_preview is False
    assert default_req.media_max_items == 8
    # Shared-field semantics should survive a slice.
    sub = req.slice(0, 1)
    assert sub.collect_media_preview is True
    assert sub.media_max_items == 3
    print("  [PASS] RolloutRequest carries media-preview knobs")


def test_samples_media_preview_concat_merges_lists():
    s1 = make_samples(2)
    s2 = make_samples(3)
    s1.media_preview = MediaPreview(
        images=[_FakePIL("a"), _FakePIL("b")],
        prompts=["p0", "p1"],
        rewards=[0.1, 0.2],
    )
    s2.media_preview = MediaPreview(
        images=[_FakePIL("c")],
        prompts=["p2"],
        rewards=[0.3],
    )
    merged = RolloutSamples.concat([s1, s2])
    assert merged.media_preview is not None
    assert [im.tag for im in merged.media_preview.images] == ["a", "b", "c"]
    assert merged.media_preview.prompts == ["p0", "p1", "p2"]
    assert merged.media_preview.rewards == [0.1, 0.2, 0.3]
    print("  [PASS] RolloutSamples.concat() merges media_preview lists")


def test_samples_media_preview_concat_all_none():
    s1 = make_samples(2)
    s2 = make_samples(2)
    # Explicit Nones.
    s1.media_preview = None
    s2.media_preview = None
    merged = RolloutSamples.concat([s1, s2])
    assert merged.media_preview is None
    print("  [PASS] RolloutSamples.concat() yields None when all previews are None")


def test_samples_media_preview_single_item_is_noop():
    s1 = make_samples(2)
    original = MediaPreview(
        images=[_FakePIL("x")],
        prompts=["only"],
        rewards=[0.5],
    )
    s1.media_preview = original
    merged = RolloutSamples.concat([s1])
    # Single-item concat should not allocate or rewrite the preview dataclass.
    assert merged.media_preview is original
    print("  [PASS] Single-item RolloutSamples.concat() preserves preview identity")


def test_samples_cap_media_preview():
    s = make_samples(2)
    s.media_preview = MediaPreview(
        images=[_FakePIL(f"img{i}") for i in range(5)],
        prompts=[f"p{i}" for i in range(5)],
        rewards=[float(i) for i in range(5)],
    )
    s.cap_media_preview(2)
    assert len(s.media_preview.images) == 2
    assert s.media_preview.prompts == ["p0", "p1"]
    assert s.media_preview.rewards == [0.0, 1.0]
    # Capping below the current length is idempotent at the new cap.
    s.cap_media_preview(2)
    assert len(s.media_preview.images) == 2
    # Capping above the current length is a no-op.
    s.cap_media_preview(10)
    assert len(s.media_preview.images) == 2
    print("  [PASS] RolloutSamples.cap_media_preview() truncates lists")


def test_attach_reward_builds_preview_and_drops_images():
    """Exercise the mixin's attach_reward logic end-to-end without Ray.

    Uses stub host/engine/reward objects so the test only covers the new
    preview-build + image-drop branch that this change added.
    """
    from diffusionrl.ray.mixins.rollout_pipeline import RolloutPipelineMixin

    class _StubEngine:
        def decode_latents(self, latents):  # pragma: no cover - not exercised
            raise AssertionError("decode_latents should not be called; images are pre-set")

    class _StubRewardPipeline:
        def score_and_attach(self, response):
            n = response.samples.latents.shape[0]
            response.samples.rewards = torch.arange(n, dtype=torch.float32)
            return response

    class _Host(RolloutPipelineMixin):
        def __init__(self, response):
            self.engine = _StubEngine()
            self._reward = _StubRewardPipeline()
            self._stored = response
            # Short-circuit the one-shot decode-diag block in attach_reward —
            # it numpy-casts decoded images, which our _FakePIL can't support.
            self._logged_decode_diag = True

        def _ensure_reward_pipeline(self):
            return self._reward

        def get_buffer(self, handle):
            return self._stored

    # --- Case A: collect_media_preview=True ---
    req_on = RolloutRequest(
        prompts=make_prompts(3),
        sampling_params=make_sampling_params(),
        collect_media_preview=True,
        media_max_items=2,
    )
    samples_on = make_samples(3)
    samples_on.decoded_images = [_FakePIL(f"img{i}") for i in range(3)]
    response_on = RolloutResponse(request=req_on, samples=samples_on)

    host_on = _Host(response_on)
    host_on.attach_reward(handle=object())

    assert response_on.samples.rewards is not None
    assert response_on.samples.rewards.tolist() == [0.0, 1.0, 2.0]
    assert isinstance(response_on.samples.media_preview, MediaPreview)
    assert len(response_on.samples.media_preview.images) == 2
    assert [im.tag for im in response_on.samples.media_preview.images] == ["img0", "img1"]
    assert response_on.samples.media_preview.prompts[:2] == [
        req_on.prompts.prompts[0],
        req_on.prompts.prompts[1],
    ]
    assert response_on.samples.media_preview.rewards == [0.0, 1.0]
    # Actor-side drop: decoded media must be gone even though scoring kept
    # a reference inside the preview.
    assert response_on.samples.decoded_images is None
    assert response_on.samples.decoded_videos is None

    # --- Case B: collect_media_preview=False ---
    req_off = RolloutRequest(
        prompts=make_prompts(3),
        sampling_params=make_sampling_params(),
        collect_media_preview=False,
        media_max_items=2,
    )
    samples_off = make_samples(3)
    samples_off.decoded_images = [_FakePIL(f"off{i}") for i in range(3)]
    response_off = RolloutResponse(request=req_off, samples=samples_off)

    host_off = _Host(response_off)
    host_off.attach_reward(handle=object())

    assert response_off.samples.media_preview is None
    # Still drop decoded images — the actor should never return them.
    assert response_off.samples.decoded_images is None
    assert response_off.samples.decoded_videos is None
    print("  [PASS] attach_reward: builds preview + drops decoded images")


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
    test_request_carries_media_preview_knobs()
    test_samples_media_preview_concat_merges_lists()
    test_samples_media_preview_concat_all_none()
    test_samples_media_preview_single_item_is_noop()
    test_samples_cap_media_preview()
    test_attach_reward_builds_preview_and_drops_images()
    print("\nAll tests passed!")
