"""``BaseTrainer._drop_decoded`` frees the reward-only ``decoded`` payload.

``decoded`` (generated Images/Videos/Texts) is consumed upstream by
``reward.score_and_attach`` and never read by training (which uses only
segment / conditions / advantages). Each trainer's ``train_step`` calls
``_drop_decoded`` on the driver right before dispatching to ``train_track``,
releasing the driver-held TensorStore refs so the (trainside, GPU-resident)
storage frees before the optimizer-step peak. ``media_preview`` and the
training inputs must survive.
"""

from __future__ import annotations

import torch

from unirl.trainer.base import BaseTrainer
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import LatentSegment


def test_drop_decoded_nulls_all_tracks_keeps_rest():
    """PE/HI3 shape: a Texts-bearing root + an Images-bearing leaf. Both lose
    ``decoded`` while training inputs and ``media_preview`` survive."""
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
