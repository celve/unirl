#!/usr/bin/env python3
"""Smoke test: Geo3K VLM rollout → text reward pipeline.

Validates the full path: load VLM → generate text answers → MC reward scorer
scores them → rewards are 0.0 or 1.0. Bypasses the full training loop to
isolate the rollout+reward changes from Hydra config resolution issues.

Usage:
    cd ~/diffusionrl && source .venv/bin/activate
    QWEN_VL_PATH=models/local/Qwen2.5-VL-3B-Instruct \
    DATA_PATH=data/datasets/geo3k_test/train.jsonl \
      python scripts/smoke_geo3k_reward.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("smoke_geo3k")


def main() -> int:
    model_path = os.environ.get("QWEN_VL_PATH", "models/local/Qwen2.5-VL-3B-Instruct")
    data_path = os.environ.get("DATA_PATH", "data/datasets/geo3k_test/train.jsonl")

    # Phase 1: Load dataset
    logger.info("== Phase 1: Load dataset ==")
    from diffusionrl.data.datasets import TextPromptDataset

    dataset = TextPromptDataset(file_path=data_path)
    logger.info("Loaded %d samples from %s", len(dataset), data_path)

    # Take first 4 samples
    samples = [dataset[i] for i in range(min(4, len(dataset)))]
    prompts = [s["prompt"] for s in samples]
    metadata_list = [s.get("metadata") for s in samples]
    media_refs_list = [s.get("media_refs", []) for s in samples]
    logger.info("Prompts: %s", [p[:60] + "..." for p in prompts])
    logger.info("Metadata answers: %s", [m.get("answer") if m else None for m in metadata_list])

    # Phase 2: Load condition images
    logger.info("== Phase 2: Load condition images ==")
    from diffusionrl.data.data_source import _load_condition_images

    images_per_prompt = _load_condition_images(media_refs_list)
    if images_per_prompt is None:
        logger.error("No condition images loaded!")
        return 1
    n_images = sum(1 for img in images_per_prompt if img is not None)
    logger.info("Loaded %d condition images", n_images)

    # Phase 3: Build primitives (Texts + Images)
    logger.info("== Phase 3: Build primitives ==")
    from diffusionrl.types.primitives import Images, Texts

    texts_prim = Texts(texts=prompts)
    images_prim = Images.from_list([img for img in images_per_prompt if img is not None])
    logger.info("Texts: %d, Images: %s", len(texts_prim.texts), images_prim.pixels.shape)

    # Phase 4: Load VLM
    logger.info("== Phase 4: Load VLM (%s) ==", model_path)
    from diffusionrl.models.qwen_vl.config import QwenVLPipelineConfig
    from diffusionrl.models.qwen_vl.pipeline import QwenVLPipeline

    config = QwenVLPipelineConfig(
        pretrained_model_ckpt_path=model_path,
        model_precision="bf16",
        freeze_vision_tower=True,
        max_prompt_length=2048,
    )
    pipeline = QwenVLPipeline.from_config(config)
    logger.info("VLM loaded on %s", pipeline.bundle.device)

    # Phase 5: Generate (rollout)
    logger.info("== Phase 5: VLM generate ==")
    from diffusionrl.types.rollout_req import RolloutReq

    req = RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": texts_prim, "image": images_prim},
        metadata=metadata_list,
    )
    resp = pipeline.generate(req)

    track = resp.tracks.get("text")
    if track is None:
        logger.error("No 'text' track in response!")
        return 2

    decoded = track.decoded
    logger.info("Generated %d answers:", len(decoded.texts))
    for i, text in enumerate(decoded.texts):
        logger.info("  [%d] %r", i, text[:100])

    # Phase 6: Reward scoring
    logger.info("== Phase 6: MC Exact-Match Reward ==")
    from diffusionrl.reward.local.mc_exact_match import MCExactMatchRewardScorer, MCExactMatchSpec

    scorer = MCExactMatchRewardScorer(config=MCExactMatchSpec(), base_device="cpu")
    from diffusionrl.types.reward import RewardRequest

    reward_req = RewardRequest(
        primitives={"text": Texts(texts=prompts)},
        generated={"text": decoded},
        metadata=metadata_list,
    )
    reward_resp = scorer.compute_rewards(reward_req)

    logger.info("Rewards: %s", reward_resp.rewards)
    logger.info("Successes: %s", reward_resp.successes)

    n_correct = sum(1 for r in reward_resp.rewards if r == 1.0)
    logger.info(
        "Accuracy: %d/%d (%.1f%%)",
        n_correct,
        len(reward_resp.rewards),
        100 * n_correct / len(reward_resp.rewards) if reward_resp.rewards else 0,
    )

    # Phase 7: Validate reward service integration
    logger.info("== Phase 7: RewardService.score_and_attach ==")
    from diffusionrl.reward.service import RewardService

    service = RewardService(backend=scorer)

    assert service.preferred_input_kind == "text", f"Expected 'text', got {service.preferred_input_kind}"
    service.score_and_attach(req=req, track=track)

    logger.info("track.rewards = %s", track.rewards.tolist())
    assert track.rewards is not None
    assert all(r in (0.0, 1.0) for r in track.rewards.tolist()), f"Unexpected rewards: {track.rewards.tolist()}"

    logger.info("== ALL PHASES PASSED ==")
    logger.info("Rollout → Reward pipeline validated end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
