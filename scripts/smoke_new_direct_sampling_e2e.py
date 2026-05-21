"""End-to-end smoke for the new Policy-stack direct-sampling pipeline.

Mirror of :mod:`scripts.smoke_new_rollout_group_e2e` but bootstraps a
:class:`NewTrainActorGroup` rather than a :class:`NewRolloutActorGroup`,
then constructs the rollout group via
:meth:`NewRolloutActorGroup.from_train_group` so the same train actor
handles serve rollouts (in-process via :class:`TrainsideRolloutEngine`).

Phases:

1. Hydra compose with ``+experiment=smoke_direct_sampling_sd3``.
2. ``ray.init`` + :class:`Placement` from cfg.
3. ``train_group = NewTrainActorGroup(cfg, placement)`` — actors build
   the Pipeline + Policy stack and install the trainside engine.
4. ``rollout_group = NewRolloutActorGroup.from_train_group(train_group, cfg=cfg)``.
5. ``NewRolloutPipeline().run_once(...)`` for the workload defined in cfg.
6. ``train_group.train(rollout_id=0, combined)`` — one optimizer step.
7. Assertions:
   * ``sample_count`` matches ``prompts_per_rollout * samples_per_prompt``.
   * Per-group advantage z-score mean ≈ 0.
   * Each rank's ``TrainOptimizerStepResult`` reports ``has_backward=True``
     and a finite loss.

Run on a pod with 1 H20 GPU::

    cd ~/diffusionrl && source .venv/bin/activate
    .venv/bin/python scripts/smoke_new_direct_sampling_e2e.py \\
      2>&1 | tee /mnt/gz/logs/smoke-direct-sampling.log
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _build_cfg():
    """Compose cfg via Hydra; mirrors smoke_new_rollout_group_e2e._build_cfg."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from diffusionrl.config import register_all_configs
    from diffusionrl.config.polymorphic import expand_polymorphic_lists

    register_all_configs()

    conf_dir = (Path(__file__).resolve().parent.parent / "conf").as_posix()
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["+experiment=smoke_direct_sampling_sd3"],
        )
    OmegaConf.resolve(cfg)
    expand_polymorphic_lists(cfg)
    return cfg


_PROMPTS: List[str] = [
    "a red apple on a wooden table",
    "a black cat sitting on a windowsill",
]


class _HardcodedPromptSource:
    """In-process prompt source. Matches the contract expected by
    :func:`load_prompt_batch_from_source`: ``get_samples(batch_size)``
    returns a dict with a ``prompts`` key.
    """

    def __init__(self, prompts: List[str]) -> None:
        self._prompts = list(prompts)
        self._cursor = 0

    def get_samples(self, batch_size: int) -> Dict[str, Any]:
        n = int(batch_size)
        out_prompts: List[str] = []
        out_ids: List[str] = []
        for i in range(n):
            idx = (self._cursor + i) % len(self._prompts)
            out_prompts.append(self._prompts[idx])
            out_ids.append(f"prompt:{idx}")
        self._cursor = (self._cursor + n) % len(self._prompts)
        return {"prompts": out_prompts, "prompt_ids": out_ids}


def main() -> int:
    print("[smoke] Phase 0: imports + Hydra compose ...")
    cfg = _build_cfg()
    print("[smoke] cfg loaded — rollout.engine._target_ =", cfg.rollout.engine.get("_target_"))

    prompt_batch_size = int(cfg.algorithm.prompts_per_rollout)
    samples_per_prompt = int(cfg.algorithm.samples_per_prompt)
    total_samples = prompt_batch_size * samples_per_prompt
    print(
        "[smoke] prompt_batch_size=%d samples_per_prompt=%d total=%d"
        % (prompt_batch_size, samples_per_prompt, total_samples)
    )

    from diffusionrl.config.instantiate import build, materialize
    from diffusionrl.config.validation import is_direct_sampling

    if not is_direct_sampling(cfg):
        print(
            "[smoke] ERROR: cfg.rollout.engine._target_=%s is not a "
            "direct-sampling engine; expected trainside." % cfg.rollout.engine.get("_target_")
        )
        return 1

    print("[smoke] Phase 1: Ray init + placement ...")
    import ray

    from diffusionrl.ray.placement import Placement

    if not ray.is_initialized():
        ray.init(num_gpus=1, ignore_reinit_error=True)

    placement_cfg = materialize(cfg.placement)
    placement = Placement.from_config(placement_cfg)
    print("[smoke] placement: %d train actors" % placement_cfg.num_train_actors)

    print("[smoke] Phase 2: NewTrainActorGroup(cfg, placement) ...")
    from diffusionrl.ray.group.new_train import NewTrainActorGroup

    train_group = NewTrainActorGroup(cfg=cfg, placement=placement)
    print("[smoke] train_group ready — %d actor handle(s)" % train_group.num_actors)

    print("[smoke] Phase 3: NewRolloutActorGroup.from_train_group ...")
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup

    rollout_group = NewRolloutActorGroup.from_train_group(train_group, cfg=cfg)
    print("[smoke] rollout_group ready (adopted train handles, no spawn)")

    print("[smoke] Phase 4: NewRolloutPipeline.run_once ...")
    from diffusionrl.rollout.new_pipeline import NewRolloutPipeline

    # ``cfg.sampling`` is a schema-only registered dataclass (no ``_target_``);
    # use ``materialize`` (= OmegaConf.to_object) to roundtrip into the typed
    # SamplingParams instance. ``cfg.algorithm`` carries ``_target_`` so build
    # is the right call.
    sampling_spec = materialize(cfg.sampling)
    control_algorithm = build(cfg.algorithm)
    data_source = _HardcodedPromptSource(_PROMPTS[:prompt_batch_size])
    pipeline = NewRolloutPipeline()

    t0 = time.time()
    with train_group.use_eval_ema():
        training_batch, sample_count, combined = pipeline.run_once(
            rollout_group=rollout_group,
            data_source=data_source,
            prompt_batch_size=prompt_batch_size,
            samples_per_prompt=samples_per_prompt,
            sampling_spec=sampling_spec,
            control_algorithm=control_algorithm,
            rollout_id=0,
        )
    dt = time.time() - t0
    print("[smoke] run_once done in %.1fs (sample_count=%d)" % (dt, sample_count))

    print("[smoke] Phase 5: assertions on rollout output ...")
    assert sample_count == total_samples, f"sample_count={sample_count} expected {total_samples}"
    assert combined.batch_size == total_samples, combined.batch_size
    assert combined.rewards is not None
    assert combined.rewards.shape == (total_samples,)
    assert combined.advantages is not None
    assert combined.advantages.shape == (total_samples,)
    print("[smoke]   rewards mean=%.4f std=%.4f" % (float(combined.rewards.mean()), float(combined.rewards.std())))
    print(
        "[smoke]   advantages mean=%.4f std=%.4f"
        % (float(combined.advantages.mean()), float(combined.advantages.std()))
    )

    # Per-group advantage z-score: mean ≈ 0 within each group.
    import torch

    unique_groups = list(dict.fromkeys(combined.group_ids))
    for gid in unique_groups:
        mask = torch.tensor(
            [i for i, g in enumerate(combined.group_ids) if g == gid],
            dtype=torch.long,
        )
        group_mean = float(combined.advantages.index_select(0, mask).mean().item())
        assert abs(group_mean) < 1e-3, f"group {gid!r} advantage mean={group_mean:.4e}, expected ≈ 0"
    print(f"[smoke]   per-group z-score OK across {len(unique_groups)} groups")

    print("[smoke] Phase 6: train_group.train(rollout_id=0, combined) ...")
    t0 = time.time()
    per_update_results = train_group.train(rollout_id=0, training_resp=combined)
    dt = time.time() - t0
    num_updates = len(per_update_results)
    print("[smoke] train done in %.1fs (num_updates=%d)" % (dt, num_updates))

    for u, per_actor_results in enumerate(per_update_results):
        for i, r in enumerate(per_actor_results):
            assert getattr(r, "has_backward", False), f"update {u} rank {i}: has_backward False"
            loss = float(getattr(r, "loss", math.nan))
            grad_norm = float(getattr(r, "grad_norm", math.nan))
            assert math.isfinite(loss), f"update {u} rank {i}: loss not finite: {loss}"
            assert math.isfinite(grad_norm), f"update {u} rank {i}: grad_norm not finite: {grad_norm}"
            print("[smoke]   update=%d rank=%d loss=%.6f grad_norm=%.4f" % (u, i, loss, grad_norm))

    print("[smoke] ALL OK — direct-sampling on the new stack end-to-end")
    print("[smoke] Phase 7: cleanup ...")
    train_group.dispose()
    placement.destroy()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
