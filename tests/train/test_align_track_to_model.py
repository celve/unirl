"""Dropping the reward-only ``decoded`` payload before training.

``decoded`` (generated Images/Videos/Texts) is consumed upstream by
``reward.score_and_attach`` and never read by training (which uses only
segment / conditions / advantages). Two layers free it:

- ``BaseTrainer._drop_decoded(resp)`` runs on the driver before dispatch, so
  the driver-held TensorStore refs don't keep the (often GPU-resident,
  trainside) storage alive through the optimizer-step peak.
- ``_align_track_to_model`` nulls it again at the single-stage device-alignment
  chokepoint as a backstop.

``media_preview`` (the logging payload) and the training inputs must survive.
"""

from __future__ import annotations

import torch

from unirl.train.stack import _align_track_to_model
from unirl.trainer.base import BaseTrainer
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import LatentSegment


def test_align_track_to_model_drops_decoded_keeps_training_inputs():
    n = 2
    track = RolloutTrack(
        sample_ids=[f"s{i}" for i in range(n)],
        conditions={"image_latent": ImageLatentCondition(latents=torch.zeros(n, 4, 4, 4))},
        segment=LatentSegment(latents=torch.zeros(n, 2, 4, 4, 4)),
        decoded=Images(pixels=torch.zeros(n, 3, 4, 4)),
        advantages=torch.zeros(n),
    )
    track.media_preview = MediaPreview()
    media = track.media_preview
    assert track.decoded is not None  # precondition: rollout produced pixels

    _align_track_to_model(track, device=torch.device("cpu"))

    # Reward-only payload dropped …
    assert track.decoded is None
    # … training inputs preserved and on the requested device.
    assert track.segment is not None
    assert int(track.segment.latents.shape[0]) == n
    assert track.segment.latents.device.type == "cpu"
    assert "image_latent" in track.conditions
    assert track.conditions["image_latent"].latents.device.type == "cpu"
    assert track.advantages is not None
    assert track.advantages.device.type == "cpu"
    # … media_preview (logging payload) untouched.
    assert track.media_preview is media


def test_drop_decoded_nulls_all_tracks_keeps_rest():
    """``BaseTrainer._drop_decoded`` nulls decoded on every track (PE/HI3 shape:
    a Texts-bearing root + an Images-bearing leaf) while leaving training inputs
    and ``media_preview`` intact."""
    ar = RolloutTrack(
        sample_ids=["p0/r0", "p0/r1"],
        parent_ids=["p0", "p0"],
        parent_track=None,
        decoded=Texts(texts=["a", "b"]),
    )
    image = RolloutTrack(
        sample_ids=["p0/r0/i0", "p0/r1/i0"],
        parent_ids=["p0/r0", "p0/r1"],
        parent_track="ar",
        segment=LatentSegment(latents=torch.zeros(2, 2, 4, 4, 4)),
        decoded=Images(pixels=torch.zeros(2, 3, 4, 4)),
        advantages=torch.zeros(2),
    )
    image.media_preview = MediaPreview()
    media = image.media_preview
    resp = RolloutResp(tracks={"ar": ar, "image": image})

    BaseTrainer._drop_decoded(resp)

    # decoded dropped on BOTH the Texts root and the Images leaf …
    assert resp.tracks["ar"].decoded is None
    assert resp.tracks["image"].decoded is None
    # … training inputs + logging payload survive.
    assert resp.tracks["image"].segment is not None
    assert resp.tracks["image"].advantages is not None
    assert resp.tracks["image"].media_preview is media
