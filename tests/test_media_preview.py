"""MediaPreview plumbing tests (wandb eval + actor-side image drop).

Covers the request knobs, batched concat/cap semantics on the dataclass, and
the mixin attach_reward path that builds the preview and drops decoded
images so they don't cross Ray.
"""

from __future__ import annotations

import torch

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


def test_attach_reward_builds_preview_and_drops_images():
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
            self._logged_decode_diag = True

        def _ensure_reward_pipeline(self):
            return self._reward

        def get_buffer(self, handle):
            return self._stored

    req_on = RolloutRequest(
        prompts=make_prompts(3),
        sampling_params=make_sampling_params(),
        collect_media_preview=True,
        media_max_items=2,
    )
    samples_on = make_samples(3)
    # ``decoded_images`` now carries canonical 3D tensors; ``attach_media_preview``
    # converts them to PIL on the wandb boundary.
    samples_on.decoded_images = [torch.full((3, 8, 8), float(i) / 10.0) for i in range(3)]
    response_on = RolloutResponse(request=req_on, samples=samples_on)

    host_on = _Host(response_on)
    host_on.attach_reward(handle=object())

    from PIL.Image import Image as _PILImage

    assert response_on.samples.rewards is not None
    assert response_on.samples.rewards.tolist() == [0.0, 1.0, 2.0]
    assert isinstance(response_on.samples.media_preview, MediaPreview)
    assert len(response_on.samples.media_preview.images) == 2
    assert all(isinstance(im, _PILImage) for im in response_on.samples.media_preview.images)
    assert response_on.samples.media_preview.prompts[:2] == [
        req_on.prompts.prompts[0],
        req_on.prompts.prompts[1],
    ]
    assert response_on.samples.media_preview.rewards == [0.0, 1.0]
    assert response_on.samples.decoded_images is None
    assert response_on.samples.decoded_videos is None

    req_off = RolloutRequest(
        prompts=make_prompts(3),
        sampling_params=make_sampling_params(),
        collect_media_preview=False,
        media_max_items=2,
    )
    samples_off = make_samples(3)
    samples_off.decoded_images = [torch.full((3, 8, 8), float(i) / 10.0) for i in range(3)]
    response_off = RolloutResponse(request=req_off, samples=samples_off)

    host_off = _Host(response_off)
    host_off.attach_reward(handle=object())

    assert response_off.samples.media_preview is None
    assert response_off.samples.decoded_images is None
    assert response_off.samples.decoded_videos is None
