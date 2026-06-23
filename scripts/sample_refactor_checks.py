"""CPU type-level checks for the Sample/Part endomorphism refactor (LIN-446).

Path-robust: imports the unirl package from the repo that contains this script
(works in a git worktree on the pod and in a local checkout). Run with any env
that has torch installed:

    /root/unirl/.venv/bin/python scripts/sample_refactor_checks.py

Covers the locally-verifiable core type layer (rollout_resp / batch / segments).
Engine/pipeline/trainer paths and the training golden-diff are NOT covered here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from unirl.types.rollout_resp import Part, RolloutResp, RolloutTrack, Sample, _track_with_field  # noqa: E402
from unirl.types.segments.latent import make_image_segment  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402


def _text_part(sample_ids, parent_ids=None, parent_track=None, stage="generation", **extra):
    seg = TextSegment.pack(tokens=[torch.arange(3 + i) for i in range(len(sample_ids))])
    return RolloutTrack(
        sample_ids=list(sample_ids), parent_ids=parent_ids, parent_track=parent_track, segment=seg, stage=stage, **extra
    )


def _img_input_part(sample_ids):
    n = len(sample_ids)
    latents = torch.stack([torch.full((1, 4), float(i)) for i in range(n)])  # [n, K=1, 4]
    return RolloutTrack(sample_ids=list(sample_ids), stage="input", segment=make_image_segment(latents=latents))


def check_aliases():
    assert Sample is RolloutResp and Part is RolloutTrack
    print("[ok] aliases: Sample is RolloutResp, Part is RolloutTrack")


def check_backward_compat_defaults():
    p = _text_part(["pr0/a0", "pr1/a0"], parent_ids=["pr0", "pr1"], parent_track="input")
    assert p.primitives == {} and p.metadata == [] and p.sampling_params == {}
    assert p.sigmas is None and p.stage_config == {} and p.init_noise_group_ids == []
    assert p.init_noise_latent_shape is None and p.stage == "generation" and p.batch_size == 2
    print("[ok] field-fold backward-compatible (new fields default; old construction unaffected)")


def check_new_fields_roundtrip():
    inp = _text_part(["pr0", "pr1"], stage="input", metadata=[{"k": 0}, {"k": 1}], stage_config={"task": "t2i"})
    assert inp.clone().stage == "input" and inp.clone().metadata == [{"k": 0}, {"k": 1}]
    s = inp.slice(0, 1)
    assert s.batch_size == 1 and s.stage == "input" and s.metadata == [{"k": 0}] and s.stage_config == {"task": "t2i"}
    cc = RolloutTrack.concat([inp.slice(0, 1), inp.slice(1, 2)])
    assert cc.batch_size == 2 and cc.metadata == [{"k": 0}, {"k": 1}] and cc.stage == "input"
    print("[ok] new fields round-trip through clone / slice / concat with correct kinds")


def check_tree_with_input_root():
    inp = _text_part(["pr0", "pr1"], stage="input")
    ar = _text_part(["pr0/a0", "pr1/a0"], parent_ids=["pr0", "pr1"], parent_track="input")
    s = RolloutResp(tracks={"input": inp, "ar": ar})
    assert s.root_track() is s.tracks["input"]
    shards = s.split()
    assert len(shards) == 2 and all(sh.tracks["input"].batch_size == 1 for sh in shards)
    print("[ok] input-as-root: root_track()=input Part; split() groups by prompt (tree-complete)")


def check_fork():
    inp = _img_input_part(["pr0", "pr1"])
    child = inp.fork("input", "ar", 3, stage_config={"task": "t2i"})
    assert child.sample_ids == ["pr0/a0", "pr0/a1", "pr0/a2", "pr1/a0", "pr1/a1", "pr1/a2"]
    assert child.parent_ids == ["pr0", "pr0", "pr0", "pr1", "pr1", "pr1"] and child.parent_track == "input"
    assert child.segment is None and child.conditions == {} and child.stage == "generation"
    ft = inp.fork_track("input", "ar", 3)
    assert child.sample_ids == ft.sample_ids and child.parent_ids == ft.parent_ids
    print("[ok] fork: ids/lineage match fork_track; no conditions; control on shell; stage=generation")


def check_conditions_for_single_level():
    inp = _img_input_part(["pr0", "pr1"])
    s = RolloutResp(tracks={"input": inp, "ar": inp.fork("input", "ar", 3)})
    got = s.conditions_for("ar")["input"].latents[:, 0]
    assert torch.equal(got, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float)), got
    print("[ok] conditions_for (1-level): row-aligned promotion from the input Part")


def check_conditions_for_two_level():
    inp = _img_input_part(["p0", "p1"])
    mid_seg = make_image_segment(latents=torch.stack([torch.full((1, 4), float(10 + i)) for i in range(4)]))
    mid = RolloutTrack(
        sample_ids=["p0/m0", "p0/m1", "p1/m0", "p1/m1"], parent_ids=["p0", "p0", "p1", "p1"], parent_track="input", segment=mid_seg
    )
    s = RolloutResp(tracks={"input": inp, "mid": mid, "leaf": mid.fork("mid", "leaf", 1)})
    conds = s.conditions_for("leaf")
    assert torch.equal(conds["mid"].latents[:, 0], torch.tensor([10, 11, 12, 13], dtype=torch.float))
    assert torch.equal(conds["input"].latents[:, 0], torch.tensor([0, 0, 1, 1], dtype=torch.float))
    print("[ok] conditions_for (2-level): chain walk + alignment through input->mid->leaf")


def check_reward_advantage_input_as_root():
    inp = _img_input_part(["p0", "p1"])
    ar = _track_with_field(inp.fork("input", "ar", 2), "rewards", torch.tensor([1.0, 3.0, 5.0, 9.0]))
    s = RolloutResp(tracks={"input": inp, "ar": ar})
    adv = s.compute_track_advantages("ar", group_key="root", normalize=True).advantages
    assert adv[0] < 0 < adv[1] and adv[2] < 0 < adv[3], adv
    s2 = s.propagate_rewards(op="mean")
    assert torch.equal(s2.tracks["input"].rewards, torch.tensor([2.0, 7.0])), s2.tracks["input"].rewards
    print("[ok] reward/advantage (input-as-root): GRPO groups by prompt; propagate_rewards aggregates ar->input")


if __name__ == "__main__":
    check_aliases()
    check_backward_compat_defaults()
    check_new_fields_roundtrip()
    check_tree_with_input_root()
    check_fork()
    check_conditions_for_single_level()
    check_conditions_for_two_level()
    check_reward_advantage_input_as_root()
    print(f"\nALL CORE-TYPE CHECKS PASSED (torch {torch.__version__})")
