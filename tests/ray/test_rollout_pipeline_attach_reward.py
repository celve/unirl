"""Multi-track attach_reward path through RolloutPipelineMixin.

Builds a minimal host class that satisfies the mixin's host contract
(``get_buffer`` / ``_ensure_reward_pipeline`` / ``_handle_state``) and
exercises :meth:`RolloutPipelineMixin.attach_reward` against a fabricated
multi-track ``RolloutResp`` (refined root + image leaf). The mocked
reward pipeline writes deterministic per-sample rewards onto the track
so we can assert:

- Only the scorable track (LatentSegment-bearing) is scored; the parent
  track is filled by :meth:`RolloutResp.propagate_rewards`.
- ``component_rewards`` and ``media_preview`` live only on scored leaves.
- Propagation aggregates leaf rewards over each parent's Z-sized group
  (mean), so parent rewards line up with the configured op.

The buffered-pipeline machinery (generate / split / TransferQueue) is
out of scope here — only ``attach_reward`` is the focus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# Warm import graph past pre-existing circular-import seam in diffusionrl.distributed.
import diffusionrl.config  # noqa: F401
from diffusionrl.ray.mixins.rollout_pipeline import (
    SCORER_BY_SEGMENT_TYPE,
    RolloutPipelineMixin,
)
from diffusionrl.types.primitives import Image, Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.segments.latent import LatentSegment


@dataclass
class _FakeHandle:
    id: str


class _FakeRewardPipeline:
    """Mock RewardPipeline.

    Each call to ``score_and_attach`` stamps each sample with an
    incrementing reward derived from the call index so per-track and
    per-shard provenance is observable from the test side.
    """

    def __init__(self) -> None:
        self.calls: List[int] = []

    def score_and_attach(self, *, req: RolloutReq, track: RolloutTrack) -> None:
        decoded = track.decoded
        n = int(decoded.pixels.shape[0]) if isinstance(decoded, Images) else 0
        call_idx = len(self.calls)
        self.calls.append(n)
        track.rewards = torch.tensor(
            [call_idx * 100.0 + float(i) for i in range(n)],
            dtype=torch.float32,
        )
        track.component_rewards = {
            "fake_metric": track.rewards.clone(),
        }


class _MixinHost(RolloutPipelineMixin):
    """Minimal host satisfying the mixin's contract for the attach_reward path.

    Implements just enough Buffer + reward-pipeline plumbing for one
    handle's worth of state; doesn't go near TransferQueue, Ray, or the
    generate path.
    """

    def __init__(self, resp: RolloutResp, req: RolloutReq, reward_pipeline) -> None:
        self._buffer: Dict[str, RolloutResp] = {"h0": resp}
        self._handle_state: Dict[str, RolloutReq] = {"h0": req}
        self._reward_pipeline = reward_pipeline

    def get_buffer(self, handle):
        return self._buffer[handle.id]

    def _ensure_reward_pipeline(self):
        return self._reward_pipeline


def _make_image_track_with_decoded(parent_sids: List[str], z: int) -> RolloutTrack:
    """Build an image leaf forked Z-from-each-parent, with decoded pixels.

    ``parent_sids`` are the refined track's sample IDs (the parent
    track's natural sample identity). Returns an image track whose
    samples are group-by-parent-ordered and whose segment carries
    minimal valid LatentSegment metadata.
    """
    image_sids = [f"{p}/i{j}" for p in parent_sids for j in range(z)]
    image_pids = [p for p in parent_sids for _ in range(z)]
    n = len(image_sids)
    return RolloutTrack(
        sample_ids=image_sids,
        parent_ids=image_pids,
        parent_track="refined",
        segment=LatentSegment(
            sample_indices=torch.arange(n, dtype=torch.long),
            positions=torch.zeros(n, dtype=torch.long),
            latents=torch.zeros(n, 2, 4, 4, 4),
            sigmas=torch.linspace(1.0, 0.0, 3),
            indices=torch.arange(2, dtype=torch.long),
        ),
        decoded=Images.from_list([Image(pixels=torch.zeros(3, 4, 4)) for _ in range(n)]),
    )


def _refined_image_resp(prompt_ids: List[str], y: int, z: int) -> Tuple[RolloutResp, RolloutReq]:
    """Build a refined+image RolloutResp + a matching RolloutReq shard.

    refined: root track, no segment (text refiner output would carry a
    TextSegment in a real pipeline — irrelevant here since refined is
    never scored).
    image: child of refined, LatentSegment-bearing scorable leaf.
    req: aligned with the image track so the new ``(req, track)``
    reward path passes its text-vs-track-sids alignment check.
    """
    refined_sids = [f"{p}/r{j}" for p in prompt_ids for j in range(y)]
    refined_pids = [p for p in prompt_ids for _ in range(y)]
    refined = RolloutTrack(
        sample_ids=refined_sids,
        parent_ids=refined_pids,
        segment=None,
    )
    image = _make_image_track_with_decoded(refined_sids, z=z)
    resp = RolloutResp(tracks={"refined": refined, "image": image})

    # Texts aligned 1:1 with the image track — that's the alignment the
    # new (req, track) reward path checks. In a real prompt-enhancement
    # pipeline the driver would source these texts from refined.decoded;
    # here we just stamp synthetic strings.
    req = RolloutReq(
        sample_ids=list(image.sample_ids),
        group_ids=list(image.group_ids),
        primitives={"text": Texts(texts=[f"text-for-{sid}" for sid in image.sample_ids])},
        collect_media_preview=False,
        media_max_items=8,
    )
    return resp, req


def test_attach_reward_scores_only_latent_segment_track():
    """The reward pipeline should be called once — for the image leaf."""
    resp, req = _refined_image_resp(prompt_ids=["p0"], y=2, z=3)
    pipeline = _FakeRewardPipeline()
    host = _MixinHost(resp, req, pipeline)
    host.attach_reward(_FakeHandle(id="h0"))
    assert len(pipeline.calls) == 1
    # Image is the leaf: 2 refined * 3 images = 6.
    assert pipeline.calls[0] == 6


def test_attach_reward_writes_rewards_and_component_rewards_on_image():
    resp, req = _refined_image_resp(prompt_ids=["p0"], y=2, z=3)
    host = _MixinHost(resp, req, _FakeRewardPipeline())
    host.attach_reward(_FakeHandle(id="h0"))
    image = resp.tracks["image"]
    assert image.rewards is not None
    assert image.rewards.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert "fake_metric" in image.component_rewards


def test_attach_reward_propagates_mean_to_refined_parent():
    """Refined gets rewards = mean over each Z-sized group of image rewards."""
    resp, req = _refined_image_resp(prompt_ids=["p0"], y=2, z=3)
    host = _MixinHost(resp, req, _FakeRewardPipeline())
    host.attach_reward(_FakeHandle(id="h0"))
    refined = resp.tracks["refined"]
    # mean of [0,1,2] = 1; mean of [3,4,5] = 4.
    assert refined.rewards is not None
    assert torch.allclose(refined.rewards, torch.tensor([1.0, 4.0]))
    # Propagated parents have no component_rewards (only direct scoring does).
    assert refined.component_rewards is None


def test_attach_reward_media_preview_only_on_scored_leaf():
    """media_preview is set only on scored leaves, never on propagated parents."""
    resp, req = _refined_image_resp(prompt_ids=["p0"], y=2, z=2)
    req.collect_media_preview = True
    req.media_max_items = 8
    host = _MixinHost(resp, req, _FakeRewardPipeline())
    host.attach_reward(_FakeHandle(id="h0"))
    assert resp.tracks["image"].media_preview is not None
    assert resp.tracks["refined"].media_preview is None


def test_attach_reward_single_track_regression_unchanged():
    """Single-track resp (today's diffusion shape) still works end-to-end."""
    image = _make_image_track_with_decoded(["p0"], z=4)
    # Convert image to a root track (parent_track=None) for the single-track shape.
    image.parent_track = None
    resp = RolloutResp(tracks={"image": image})
    req = RolloutReq(
        sample_ids=list(image.sample_ids),
        group_ids=list(image.group_ids),
        primitives={"text": Texts(texts=[f"t-{sid}" for sid in image.sample_ids])},
        collect_media_preview=False,
        media_max_items=8,
    )
    host = _MixinHost(resp, req, _FakeRewardPipeline())
    host.attach_reward(_FakeHandle(id="h0"))
    assert resp.tracks["image"].rewards is not None
    assert resp.tracks["image"].rewards.shape == (4,)


def test_attach_reward_registry_keys_are_segment_types():
    """Sanity: SCORER_BY_SEGMENT_TYPE is keyed by Type[Segment], not strings."""
    assert LatentSegment in SCORER_BY_SEGMENT_TYPE


def test_attach_reward_uses_tracks_with_segment_types():
    """``RolloutResp.tracks_with_segment_types`` returns only the image track."""
    resp, _ = _refined_image_resp(prompt_ids=["p0", "p1"], y=2, z=2)
    scorable = resp.tracks_with_segment_types(SCORER_BY_SEGMENT_TYPE.keys())
    assert [name for name, _ in scorable] == ["image"]
