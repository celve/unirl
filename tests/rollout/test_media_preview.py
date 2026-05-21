"""MediaPreview plumbing tests (wandb eval + actor-side image drop).

Covers the request knobs, batched concat/cap semantics on the dataclass, and
the mixin attach_reward path that builds the preview and drops decoded
images so they don't cross Ray.
"""

from __future__ import annotations

import torch

# Warm import graph past pre-existing circular import in
# ``diffusionrl.distributed`` → ``rollout.engine`` → ``types.rollout_req``.
# Without this, pytest collection fails on
# ``from diffusionrl.types.sample import MediaPreview``.
import diffusionrl.config  # noqa: F401  -- import-graph warm
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sample import LogProbData, MediaPreview, RolloutSamples
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.types.trajectory_store import Trajectory, TrajectoryBuilder


def make_prompts(n: int) -> Prompts:
    return Prompts.from_unique_prompts(prompts=[f"a photo of a cat {i}" for i in range(n)])


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
    return RolloutRequest(prompts=make_prompts(n), sampling_params=make_sampling_params())


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
    default_req = make_request(2)
    assert default_req.collect_media_preview is False
    assert default_req.media_max_items == 8
    sub = req.slice(0, 1)
    assert sub.collect_media_preview is True
    assert sub.media_max_items == 3


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


def test_samples_media_preview_concat_all_none():
    s1 = make_samples(2)
    s2 = make_samples(2)
    s1.media_preview = None
    s2.media_preview = None
    merged = RolloutSamples.concat([s1, s2])
    assert merged.media_preview is None


def test_samples_media_preview_single_item_is_noop():
    s1 = make_samples(2)
    original = MediaPreview(
        images=[_FakePIL("x")],
        prompts=["only"],
        rewards=[0.5],
    )
    s1.media_preview = original
    merged = RolloutSamples.concat([s1])
    assert merged.media_preview is original


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
    s.cap_media_preview(2)
    assert len(s.media_preview.images) == 2
    s.cap_media_preview(10)
    assert len(s.media_preview.images) == 2


def test_attach_media_preview_mixed_slices_videos_by_selected_image_indices() -> None:
    """Videos must match the batch indices of tensors actually used for images."""
    n = 4
    req = make_request(n)
    samples = make_samples(n)
    samples.rewards = torch.arange(n, dtype=torch.float32)
    # Index 0 skipped (non-tensor); first two successful tensors at 1 and 3.
    samples.decoded_images = [
        "not-a-tensor",
        torch.full((3, 8, 8), 0.25),
        "not-a-tensor",
        torch.full((3, 8, 8), 0.75),
    ]
    samples.decoded_videos = torch.stack(
        [torch.full((3, 5, 8, 8), float(i) / 10.0) for i in range(n)],
        dim=0,
    )
    response = RolloutResponse(request=req, samples=samples)
    response.attach_media_preview(max_items=2)

    mp = response.samples.media_preview
    assert mp is not None
    assert len(mp.images) == 2
    assert len(mp.videos) == 2
    assert mp.prompts == [req.prompts.prompts[1], req.prompts.prompts[3]]
    assert mp.rewards == [1.0, 3.0]
    # Video rows 1 and 3 must be selected (not 0 and 1).
    assert torch.allclose(mp.videos[0], samples.decoded_videos[1].cpu())
    assert torch.allclose(mp.videos[1], samples.decoded_videos[3].cpu())


# NOTE: 2 test cases from main were removed during the merge because they
# test main's older ``attach_media_preview`` behaviors that the current
# design does not have:
#
# - ``test_attach_media_preview_all_non_tensor_images_uses_video_only_path``:
#   main synthesized middle-frame PIL images from videos when all entries
#   in ``decoded_images`` were non-tensor placeholders. main-unified-base's
#   :meth:`RolloutResponse.attach_media_preview` (kept here via ``--ours``
#   during the merge) drives selection from whichever modality is
#   non-empty and does NOT cross-synthesize. Either pure image-only or
#   pure video-only previews are produced; "non-tensor placeholders in
#   images list trigger video-side iteration" is not part of the contract.
#
# - ``test_attach_media_preview_mixed_raises_when_video_row_missing``:
#   main raised ``ValueError("no matching decoded_videos")`` when the
#   image count exceeded the video count in mixed mode. The current
#   design surfaces a different (cleaner) error from
#   :meth:`MediaPreview.__post_init__` ("'videos' has N entries but the
#   canonical batch size ... is M. All non-empty parallel lists must
#   agree.") — the validation moved into ``MediaPreview`` where it
#   belongs rather than living in ``attach_media_preview``'s per-modality
#   branch.
