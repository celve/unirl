"""End-to-end smoke for the new ``NewRolloutActorGroup`` + ``NewRolloutPipeline``.

Workload: 48 prompts × 16 samples per prompt = 768 samples on a single Ray
``NewRolloutActor`` wrapping ``VLLMOmniRolloutEngine`` for HunyuanImage-3 t2i,
across 8 H20 GPUs (NOSET multi-GPU).

Usage on pod (after vllm + vllm-omni 0.20.0 are installed in ``.venv``):

    cd ~/diffusionrl && source .venv/bin/activate
    FBS=4 python scripts/smoke_new_rollout_group_e2e.py \\
        2>&1 | tee /mnt/gz/logs/smoke-new-rollout-group.log

Calibration knob: ``FBS`` env var sets ``cfg.rollout.plan.forward_batch_size``
(default 4). Drop to 2 / 1 if the smoke OOMs in the vllm-omni worker
subprocess.

This script is rollout-side only — it never bootstraps a ``TrainActorGroup``
and never calls ``train.py``. Verifies that:

1. ``NewRolloutActorGroup(cfg=cfg, placement=placement)`` brings up the actor
   in one shot (the bootstrap-into-__init__ change from commit ``29ab56a``).
2. ``NewRolloutPipeline.run_once`` runs generate → reward → advantages on
   ``RolloutReq``/``RolloutResp`` end-to-end.
3. Per-group advantage normalization (z-score) holds across all 48 groups.
4. ``RolloutResponse.to_training_batch`` validates with the empty
   ``ForwardContext()`` stub injected by ``convert_training_data``.

The Hydra cfg is composed from ``conf/experiment/smoke_vllm_omni_hi3_t2i.yaml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _build_cfg():
    """Compose the Hydra cfg from the smoke experiment YAML.

    Uses ``hydra.initialize_config_dir`` against ``conf/`` so the smoke can
    run from anywhere, plus an ``experiment=smoke_vllm_omni_hi3_t2i`` override
    to swap in the rollout-only config.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Force register_all_configs so all @register_config decorators run before
    # compose. Otherwise Hydra raises "could not find ... in config search path"
    # for groups like rollout/engine: vllm_omni.
    from diffusionrl.config import register_all_configs
    from diffusionrl.config.polymorphic import expand_polymorphic_lists

    register_all_configs()

    conf_dir = (Path(__file__).resolve().parent.parent / "conf").as_posix()
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["+experiment=smoke_vllm_omni_hi3_t2i"],
        )
    OmegaConf.resolve(cfg)
    # Resolve polymorphic lists like ``reward.components: [{name: pickscore}]``
    # into typed structured elements with ``_target_``. ``train.py`` does this
    # right after compose; the smoke needs to mirror that or RewardService
    # raises ``cfg at 'reward.components0' has no _target_``.
    expand_polymorphic_lists(cfg)
    return cfg


_FORTY_EIGHT_PROMPTS: List[str] = [
    "a red apple on a wooden table",
    "a city skyline at sunset",
    "a black cat sitting on a windowsill",
    "a steaming bowl of ramen with chopsticks",
    "a sailboat on a calm blue lake",
    "a cozy library with floor-to-ceiling bookshelves",
    "a snowy mountain peak under clear skies",
    "a vibrant coral reef teeming with fish",
    "an old wooden door with brass handles",
    "a field of sunflowers under a bright sun",
    "a rainy night street scene with neon signs",
    "a hot air balloon over a green valley",
    "a vintage typewriter on a desk",
    "a crystal-clear stream in a forest",
    "a futuristic robot in a sleek silver design",
    "a steampunk pocket watch close-up",
    "a tropical beach with palm trees",
    "a small village covered in fresh snow",
    "a dragon perched atop a stone tower",
    "a bowl of fresh strawberries with cream",
    "a lonely lighthouse on a rocky coast",
    "an astronaut floating in outer space",
    "a child flying a colorful kite at the park",
    "a misty bamboo forest at dawn",
    "an antique map spread on a wooden surface",
    "a samurai standing in a cherry blossom garden",
    "a glowing campfire under a starry sky",
    "a market stall full of fresh vegetables",
    "an iron bridge over a river at twilight",
    "a violin resting on a velvet cushion",
    "a hot springs onsen surrounded by snow",
    "a giraffe on the African savanna",
    "a futuristic cityscape with flying cars",
    "a pair of running shoes on a forest trail",
    "a cup of latte art on a cafe table",
    "a magical waterfall in an enchanted forest",
    "a polar bear walking across the arctic",
    "an old grand piano in a sunlit room",
    "a calligrapher writing with an ink brush",
    "a flock of birds flying across orange clouds",
    "a cathedral interior with stained glass windows",
    "a koi pond with lotus flowers",
    "a fishing village at the edge of a fjord",
    "a chef preparing sushi at a counter",
    "a bookstore alley with hanging lights",
    "a desert oasis under a starry night",
    "a glowing lantern festival on a river",
    "a winding mountain road in autumn",
]
assert len(_FORTY_EIGHT_PROMPTS) == 48, f"expected 48 prompts, got {len(_FORTY_EIGHT_PROMPTS)}"


class _HardcodedPromptSource:
    """In-process data source for the smoke. Returns the 48 hardcoded prompts.

    Mirrors the contract that
    :func:`diffusionrl.rollout.request_builders.load_prompt_batch_from_source`
    expects: ``get_samples(batch_size) -> Dict[str, Any]`` with at least a
    ``prompts`` key.
    """

    def __init__(self, prompts: List[str]) -> None:
        self._prompts = list(prompts)
        self._cursor = 0

    def get_samples(self, batch_size: int) -> Dict[str, Any]:
        n = int(batch_size)
        # Cycle prompts so the request always has exactly batch_size entries.
        out_prompts: List[str] = []
        out_ids: List[str] = []
        for i in range(n):
            idx = (self._cursor + i) % len(self._prompts)
            out_prompts.append(self._prompts[idx])
            out_ids.append(f"prompt:{idx}")
        self._cursor = (self._cursor + n) % len(self._prompts)
        return {"prompts": out_prompts, "prompt_ids": out_ids}


def main() -> int:
    # Quick-test knob: PROMPT_BATCH_SIZE env var overrides the default of 48
    # so we can run the smoke against, e.g., 2 prompts × 16 samples = 32 in
    # ~minutes instead of hours. Also overrides cfg.algorithm.prompts_per_rollout
    # below so GRPO's group-normalization scope stays consistent.
    prompt_batch_size = int(os.environ.get("PROMPT_BATCH_SIZE", "48"))
    samples_per_prompt = 16
    total_samples = prompt_batch_size * samples_per_prompt

    print("[smoke] Phase 0: imports + Hydra compose ...")
    cfg = _build_cfg()
    # Keep the algorithm's group-scope aware of the smaller batch.
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "algorithm.prompts_per_rollout", prompt_batch_size, force_add=True)
    print("[smoke] cfg loaded — rollout/engine target =", cfg.rollout.engine.get("_target_"))
    print(
        "[smoke] prompt_batch_size=%d samples_per_prompt=%d total=%d"
        % (
            prompt_batch_size,
            samples_per_prompt,
            total_samples,
        )
    )
    print("[smoke] forward_batch_size =", cfg.rollout.plan.forward_batch_size)
    print(
        "[smoke] sampling: steps=%d, h=%d, w=%d, eta=%.2f"
        % (
            int(cfg.sampling.num_inference_steps),
            int(cfg.sampling.height),
            int(cfg.sampling.width),
            float(cfg.sampling.sde_config.eta),
        )
    )

    # Materialize the locals the driver pipeline needs (sampling_spec + algo).
    from diffusionrl.config.instantiate import build, materialize

    sampling_spec = materialize(cfg.sampling)
    algorithm = build(cfg.algorithm)
    print("[smoke] algorithm =", type(algorithm).__name__)

    # Init Ray locally and bring up the placement.
    print("[smoke] Phase 1: Ray init + placement ...")
    import ray

    from diffusionrl.ray.placement import Placement

    if not ray.is_initialized():
        ray.init(num_gpus=8, ignore_reinit_error=True)

    placement_cfg = materialize(cfg.placement)
    placement = Placement.from_config(placement_cfg)
    print(
        "[smoke] placement: %d rollout actors, %d train actors (placeholder)"
        % (
            placement_cfg.num_rollout_actors,
            placement_cfg.num_train_actors,
        )
    )
    print("[smoke] placement.gpu_ids=", placement.gpu_ids)
    for i, a in enumerate(placement.rollout_actors):
        print(
            f"[smoke] rollout_actor[{i}]: rank={a.rank}, bundle_idx={a.bundle_idx}, gpu_ids={a.gpu_ids}, node_ip={a.node_ip}"
        )

    # Bootstrap the new group (single-step __init__).
    print("[smoke] Phase 2: NewRolloutActorGroup(cfg, placement) ...")
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup

    group = NewRolloutActorGroup(cfg=cfg, placement=placement)
    print("[smoke] group ready — %d actor handles" % group.num_actors)

    # Drive the rollout.
    print(
        "[smoke] Phase 3: NewRolloutPipeline.run_once (%d prompts × %d samples = %d) ..."
        % (
            prompt_batch_size,
            samples_per_prompt,
            total_samples,
        )
    )
    from diffusionrl.rollout.new_pipeline import NewRolloutPipeline

    pipeline = NewRolloutPipeline()
    data_source = _HardcodedPromptSource(_FORTY_EIGHT_PROMPTS)

    import time

    t0 = time.time()
    training_batch, sample_count, combined = pipeline.run_once(
        rollout_group=group,
        data_source=data_source,
        prompt_batch_size=prompt_batch_size,
        samples_per_prompt=samples_per_prompt,
        sampling_spec=sampling_spec,
        control_algorithm=algorithm,
        rollout_id=0,
        collect_media_preview=False,
        media_max_items=8,
    )
    dt = time.time() - t0
    print("[smoke] run_once done in %.1fs (sample_count=%d)" % (dt, sample_count))

    # Assertions — what "the group works well" looks like.
    print("[smoke] Phase 4: assertions ...")
    assert sample_count == total_samples, f"sample_count={sample_count} expected {total_samples}"
    assert combined.batch_size == total_samples, f"combined.batch_size={combined.batch_size}"
    assert combined.rewards is not None, "rewards not populated"
    assert combined.rewards.shape == (total_samples,), f"rewards.shape={tuple(combined.rewards.shape)}"
    assert combined.advantages is not None, "advantages not populated"
    assert combined.advantages.shape == (total_samples,), f"advantages.shape={tuple(combined.advantages.shape)}"
    print(
        "[smoke]   rewards: mean=%.4f std=%.4f"
        % (
            float(combined.rewards.mean().item()),
            float(combined.rewards.std().item()),
        )
    )
    print(
        "[smoke]   advantages: mean=%.4f std=%.4f"
        % (
            float(combined.advantages.mean().item()),
            float(combined.advantages.std().item()),
        )
    )

    # Per-group advantage z-score: mean ≈ 0 within each group.
    import torch

    unique_groups = list(dict.fromkeys(combined.group_ids))
    assert len(unique_groups) == prompt_batch_size, f"unique groups={len(unique_groups)}, expected {prompt_batch_size}"
    for gid in unique_groups:
        mask = torch.tensor(
            [i for i, g in enumerate(combined.group_ids) if g == gid],
            dtype=torch.long,
        )
        group_adv_mean = float(combined.advantages.index_select(0, mask).mean().item())
        if abs(group_adv_mean) > 1e-3:
            raise AssertionError(f"group {gid!r} advantage mean = {group_adv_mean:.4e} (expected ≈ 0 for z-score)")

    # TrainingBatch validates (validate runs inside to_training_batch).
    assert training_batch.batch_size == total_samples, f"training_batch.batch_size={training_batch.batch_size}"

    print("[smoke] ALL OK — group + pipeline + RolloutReq/RolloutResp end-to-end working")

    # Cleanup.
    print("[smoke] Phase 5: cleanup ...")
    placement.destroy()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
