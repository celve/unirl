"""Sample + Part — the rollout endomorphism types (LIN-446).

See ``docs/rollout-sample-refactor.md``. One recursive type for the rollout
boundary (``step: Sample -> Sample``), replacing the ``RolloutReq → RolloutResp``
pair. A ``Sample`` holds an ordered ``parts: List[Part]`` whose position *is* the
lineage chain — a part's parent is the entry before it (index ``i-1``), so parts
carry no name/key. A *request* is a ``Sample`` with only input Part(s); ``fork``
appends a generation shell and the fill step populates it. Conditioning is collected
from the ancestor prefix as primitives (:meth:`Sample.conditioning`), not stored.
Reward/advantage/split machinery is ported from ``rollout_resp.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, TypeVar, Union

import torch

from unirl.distributed.tensor.batch import (
    Batch,
    FieldKind,
    concat_field,
    field,
    max_field,
    shared_field,
)
from unirl.distributed.tensor.ref import hydrate
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.sampling import BaseSamplingParams
from unirl.types.segments import Segment
from unirl.utils.shard_balance import lpt_shard_permutation, shard_token_spread

logger = logging.getLogger(__name__)

TP = TypeVar("TP", bound="Part")
TS = TypeVar("TS", bound="Sample")

# A Part's content in raw/primitive form (text / image / …) — the counterpart of
# the encoded ``segment``: given content on an input Part, decoded output on a
# generation Part. One value (a Part is single-modality).
Primitive = Union[Texts, Images, Videos, Audios]

# Per-Part lifecycle: "input" (given prompt, untrained) vs "generation" (produced).
STAGE_INPUT = "input"
STAGE_GENERATION = "generation"
_VALID_STAGES = frozenset({STAGE_INPUT, STAGE_GENERATION})


@dataclass
class Part(Batch):
    """One rollout slice — single modality, single lifecycle stage.

    A node in a :class:`Sample`'s positional chain: its parent is the preceding
    entry in ``Sample.parts`` (index ``i-1``); ``parent_index[i]`` is sample ``i``'s
    parent *row* in that part, and ``parent_index is None`` marks a root. ``segment`` holds
    the encoded payload, ``primitive`` the same content in raw form. Conditioning is
    collected from the prefix (:meth:`Sample.conditioning`), not stored.
    """

    sample_ids: List[str] = concat_field(default_factory=list)
    parent_index: Optional[List[int]] = concat_field(default=None)

    segment: Optional[Segment] = field(kind=FieldKind.CONCAT, default=None)
    primitive: Optional[Primitive] = field(kind=FieldKind.CONCAT, default=None)
    media_preview: Optional[MediaPreview] = concat_field(default=None)

    rewards: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    status: Optional[torch.Tensor] = concat_field(default=None)

    # Generation control — rides a generation Part's shell, set at fork. ``shared``
    # params/σ encode the aligned-batch assumption (every row same schedule).
    metadata: List[Optional[Dict[str, Any]]] = concat_field(default_factory=list)
    sampling_params: Dict[str, BaseSamplingParams] = shared_field(default_factory=dict)
    sigmas: Optional[torch.Tensor] = shared_field(default=None)
    stage_config: Dict[str, Any] = shared_field(default_factory=dict)
    init_noise_group_ids: List[str] = concat_field(default_factory=list)
    init_noise_latent_shape: Optional[List[int]] = shared_field(default=None)

    stage: str = shared_field(default=STAGE_GENERATION)

    @classmethod
    def input(
        cls: Type[TP],
        sample_ids: List[str],
        segment: Optional[Segment] = None,
        *,
        primitive: Optional[Primitive] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        stage_config: Optional[Dict[str, Any]] = None,
        init_noise_group_ids: Optional[List[str]] = None,
        init_noise_latent_shape: Optional[List[int]] = None,
    ) -> TP:
        """Build a turn-0 input Part (the prompt, a root; untrained).

        ``segment`` is the encoded prompt; ``primitive`` optionally carries the same
        content in raw form (what :meth:`Sample.conditioning` surfaces). Always a
        root (``parent_index=None``).
        """
        return cls(
            sample_ids=list(sample_ids),
            segment=segment,
            stage=STAGE_INPUT,
            primitive=primitive,
            metadata=list(metadata) if metadata else [],
            stage_config=dict(stage_config) if stage_config else {},
            init_noise_group_ids=list(init_noise_group_ids) if init_noise_group_ids else [],
            init_noise_latent_shape=list(init_noise_latent_shape) if init_noise_latent_shape else None,
        )

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    @property
    def group_ids(self) -> List[Union[int, str]]:
        """Grouping labels: ``parent_index`` (int; siblings = one group), or
        ``sample_ids`` (str) for a root (each sample its own group)."""
        if self.parent_index is not None:
            return list(self.parent_index)
        return list(self.sample_ids)

    def split(self: TP) -> List[TP]:
        """Split into one Part per :attr:`group_ids` equivalence class."""
        gids = self.group_ids
        if not gids:
            return [self]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(gids):
            groups.setdefault(gid, []).append(i)
        results: List[TP] = []
        for gid in dict.fromkeys(gids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(self.select(indices))
        return results

    def balance_shards(self: TP, num_shards: int, *, min_spread: float = 0.05) -> TP:
        """Reorder samples so ``num_shards`` equal contiguous shards carry ~equal
        tokens (verl ``balance_batch`` parity, greedy LPT). No-op when balancing
        can't apply or shards are already within ``min_spread``."""
        if self.segment is None or self.segment.lengths is None or num_shards <= 1:
            return self
        total = self.batch_size
        if total % num_shards != 0:
            return self
        lengths = [int(x) for x in self.segment.lengths.tolist()]
        if len(lengths) != total:
            logger.warning("balance_shards: lengths (%d) != batch_size (%d); skipping.", len(lengths), total)
            return self

        before = shard_token_spread(lengths, num_shards)
        if before < min_spread:
            return self

        perm = lpt_shard_permutation(lengths, num_shards)
        after = shard_token_spread([lengths[i] for i in perm], num_shards)
        logger.info("balance_shards: token spread %.1f%% -> %.1f%%", 100 * before, 100 * after)
        return self.select(perm)

    def fork(
        self,
        branch: int,
        *,
        sampling_params: Optional[Dict[str, BaseSamplingParams]] = None,
        sigmas: Optional[torch.Tensor] = None,
        stage_config: Optional[Dict[str, Any]] = None,
        init_noise_group_ids: Optional[List[str]] = None,
        init_noise_latent_shape: Optional[List[int]] = None,
        new_segment: Optional[Segment] = None,
    ) -> "Part":
        """Create a child generation shell — ``N`` self-samples → ``N*branch``.

        The mechanic behind :meth:`Sample.fork` (which appends the child so its
        parent is positionally ``self``). Child ``parent_index`` is each parent
        row repeated ``branch``× group-by-parent; ids ``f"{sid}/{j}"``. Control rides
        the shell; ``segment`` is left for the fill step. No conditioning assembled
        (collected via :meth:`Sample.conditioning`).
        """
        if not self.sample_ids:
            raise ValueError("Part.fork: parent has no sample_ids")
        if branch < 1:
            raise ValueError(f"Part.fork: branch must be >= 1, got {branch}")

        child_sample_ids = [f"{pid}/{j}" for pid in self.sample_ids for j in range(branch)]
        child_parent_index = [r for r in range(len(self.sample_ids)) for _ in range(branch)]

        return Part(
            sample_ids=child_sample_ids,
            parent_index=child_parent_index,
            segment=new_segment,
            stage=STAGE_GENERATION,
            sampling_params=dict(sampling_params) if sampling_params else {},
            sigmas=sigmas,
            stage_config=dict(stage_config) if stage_config else {},
            init_noise_group_ids=list(init_noise_group_ids) if init_noise_group_ids else [],
            init_noise_latent_shape=list(init_noise_latent_shape) if init_noise_latent_shape else None,
        )

    def compute_advantages(
        self: TP,
        normalize: bool = True,
        eps: float = 1e-8,
        scope: str = "group",
        use_global_std: bool = False,
        group_ids: Optional[List[str]] = None,
    ) -> TP:
        """GRPO per-group advantage ``(reward - group_mean) / (group_std + eps)``.

        Groups are :attr:`group_ids` (``parent_index``, or per-sample for a root) and
        must be group-by-parent contiguous, so the reduce is one ``view`` reshape.
        ``scope="global"`` normalizes over the whole batch; ``use_global_std`` keeps
        per-group means but one batch-wide std; ``group_ids`` overrides the grouping
        (e.g. to group at a coarser lineage level). Population std
        (``unbiased=False``) makes ``branch=1`` degenerate to advantage 0.
        """
        if self.rewards is None:
            raise ValueError("Part.compute_advantages: part has no rewards")
        n = len(self.sample_ids)
        if n == 0:
            return self

        # rewards may arrive as a TensorRef proxy from the reward workers; hydrate.
        rewards_local = hydrate(self.rewards)

        if scope == "global":
            rewards_g = rewards_local.to(torch.float32)
            if normalize:
                adv_g = (rewards_g - rewards_g.mean()) / (rewards_g.std() + eps)
            else:
                adv_g = rewards_g - rewards_g.mean()
            return _part_with_field(self, "advantages", adv_g)

        if group_ids is not None and len(group_ids) != n:
            raise ValueError(
                f"compute_advantages: group_ids length {len(group_ids)} != sample count {n}; "
                f"the override must be one label per sample."
            )
        group_labels = self.parent_index if group_ids is None else list(group_ids)

        # Root (no grouping labels): each sample is its own group → advantage 0.
        if group_labels is None:
            return _part_with_field(self, "advantages", torch.zeros_like(rewards_local, dtype=torch.float32))

        unique_pids = list(dict.fromkeys(group_labels))
        n_groups = len(unique_pids)
        if n_groups == 0 or n % n_groups != 0:
            raise ValueError(
                f"compute_advantages: non-uniform group sizes (n={n}, n_groups={n_groups}). "
                f"Expected uniform branching with group-by-parent ordering — use fork to build the Part."
            )
        branch = n // n_groups
        expected = [pid for pid in unique_pids for _ in range(branch)]
        if list(group_labels) != expected:
            raise ValueError(
                "compute_advantages: grouping labels not in group-by-parent contiguous order. "
                "Siblings must be consecutive (use fork), got interleaved ordering."
            )

        rewards = rewards_local.to(torch.float32)
        reshaped = rewards.view(n_groups, branch)
        mean = reshaped.mean(dim=1, keepdim=True)
        if normalize:
            if use_global_std:
                std = rewards.std() + eps
            else:
                std = (reshaped.var(dim=1, unbiased=False, keepdim=True) + eps).sqrt()
            adv = (reshaped - mean) / std
        else:
            adv = reshaped - mean
        return _part_with_field(self, "advantages", adv.flatten())


def _part_with_field(part: TP, field_name: str, value: Any) -> TP:
    """Copy of ``part`` with one field replaced."""
    kwargs: Dict[str, Any] = {f.name: getattr(part, f.name) for f in dc_fields(part)}
    kwargs[field_name] = value
    return type(part)(**kwargs)


@dataclass
class Sample(Batch):
    """Rollout container — an ordered ``parts: List[Part]`` (the merged
    ``RolloutReq`` + ``RolloutResp``). Position is lineage (parent = the preceding
    part); per-Part invariants (chain foreign-keys, ``stage``) are checked in
    :meth:`__post_init__`."""

    parts: List[Part] = field(kind=FieldKind.CONCAT, default_factory=list)
    reward_compute_s: float = max_field(default=0.0)

    def __post_init__(self) -> None:
        for i, p in enumerate(self.parts):
            if p.stage not in _VALID_STAGES:
                raise ValueError(f"Sample.parts[{i}].stage={p.stage!r} not in {sorted(_VALID_STAGES)}")
            n = len(p.sample_ids)
            if p.parent_index is None:
                # Only the head (index 0) may be a root — position is lineage.
                if i != 0:
                    raise ValueError(
                        f"Sample.parts[{i}] has parent_index=None but is not the head; only the first part may be a root."
                    )
                continue
            if len(p.parent_index) != n:
                raise ValueError(
                    f"Sample.parts[{i}]: parent_index length {len(p.parent_index)} != sample_ids length {n}"
                )
            if i == 0:
                raise ValueError("Sample.parts[0] has parent_index set; the head has no parent.")
            prev_n = len(self.parts[i - 1].sample_ids)
            bad = [r for r in p.parent_index if not (0 <= r < prev_n)]
            if bad:
                raise ValueError(
                    f"Sample.parts[{i}].parent_index: {len(bad)} rows out of range [0, {prev_n}) for the "
                    f"preceding part (index {i - 1}); first bad: {bad[:3]!r}"
                )

    @classmethod
    def concat(cls: Type[TS], items: Sequence[TS]) -> TS:
        """Concat Samples (e.g. DP gather): concat each part position-wise across
        shards. All shards carry the same parts in the same order (shards of one
        Sample). ``reward_compute_s`` reduces by max."""
        if not items:
            raise ValueError("Sample.concat: cannot concat an empty sequence")
        if len(items) == 1:
            return items[0]
        n_parts = len(items[0].parts)
        merged = [Part.concat([it.parts[i] for it in items]) for i in range(n_parts)]
        return cls(parts=merged, reward_compute_s=max(it.reward_compute_s for it in items))

    @classmethod
    def request(cls: Type[TS], *input_parts: Part) -> TS:
        """A *request* — a ``Sample`` of only input Part(s), e.g.
        ``Sample.request(Part.input(ids, seg))``. (Multi-input multimodal needs the
        inputs chained so only the head is a root; future work, §3.)"""
        for i, p in enumerate(input_parts):
            if p.stage != STAGE_INPUT:
                raise ValueError(f"Sample.request: part {i} has stage {p.stage!r}; a request holds only input Parts.")
        return cls(parts=list(input_parts))

    @property
    def batch_size(self) -> int:
        """Size of the root Part (one prompt + its fan-out = "one sample"); max
        across parts when the root isn't unique."""
        if not self.parts:
            return 0
        roots = [p for p in self.parts if p.parent_index is None]
        if len(roots) == 1:
            return roots[0].batch_size
        return max(p.batch_size for p in self.parts)

    def split(self: TS) -> List[TS]:
        """Split into one ``Sample`` per root-group, tree-complete: each shard holds
        one prompt's whole subtree across all parts. Requires a unique root."""
        if not self.parts:
            return [self]

        root_gids = self.parts[0].group_ids
        if not root_gids:
            return [self]

        # Root-group label per sample, propagated forward along the chain (parent =
        # preceding part): head is its own group; each child indexes the parent's labels.
        per_part_root_groups: List[List[str]] = []
        for part in self.parts:
            if part.parent_index is None:
                per_part_root_groups.append(list(part.group_ids))
            else:
                prev = per_part_root_groups[-1]
                per_part_root_groups.append([prev[r] for r in part.parent_index])

        results: List[TS] = []
        for rgid in dict.fromkeys(root_gids):
            shard_parts: List[Part] = []
            for i, part in enumerate(self.parts):
                indices = [k for k, rg in enumerate(per_part_root_groups[i]) if rg == rgid]
                if not indices:
                    raise RuntimeError(
                        f"Sample.split: part {i} has no samples in root group {rgid!r}; lineage tree is malformed."
                    )
                shard_parts.append(part.select(torch.tensor(indices, dtype=torch.long)))
            results.append(type(self)(parts=shard_parts))
        return results

    def fork(
        self: TS,
        branch: int,
        *,
        sampling_params: Optional[Dict[str, BaseSamplingParams]] = None,
        sigmas: Optional[torch.Tensor] = None,
        stage_config: Optional[Dict[str, Any]] = None,
        init_noise_group_ids: Optional[List[str]] = None,
        init_noise_latent_shape: Optional[List[int]] = None,
        new_segment: Optional[Segment] = None,
    ) -> TS:
        """Append a generation shell forked from the frontier (the last part) — the
        sole fan-out edge (§5), the "fork" half of ``step: fork → fill``."""
        if not self.parts:
            raise ValueError("Sample.fork: no parts to fork from (empty Sample)")
        child = self.parts[-1].fork(
            branch,
            sampling_params=sampling_params,
            sigmas=sigmas,
            stage_config=stage_config,
            init_noise_group_ids=init_noise_group_ids,
            init_noise_latent_shape=init_noise_latent_shape,
            new_segment=new_segment,
        )
        return type(self)(parts=[*self.parts, child], reward_compute_s=self.reward_compute_s)

    def propagate_rewards(self: TS, op: Literal["mean", "max", "sum"] = "mean") -> TS:
        """Aggregate child rewards up the chain (leaf → root) into unscored parents.
        Walks ``parts`` in reverse; per parent, reduces the successor's rewards
        ``view(n_parent, branch).reduce(dim=1)``. Direct rewards win. Single-child
        only (the chain guarantees it; §7)."""
        new_parts = list(self.parts)
        for i in range(len(new_parts) - 1, -1, -1):
            part = self.parts[i]
            if part.rewards is not None:
                continue
            if i + 1 >= len(new_parts):
                continue
            child = new_parts[i + 1]
            if child.parent_index is None:  # successor isn't a child of this part
                continue
            if child.rewards is None:
                raise ValueError(
                    f"propagate_rewards: cannot aggregate from part {i + 1} to {i} — child.rewards is None. "
                    f"Score the leaf parts first."
                )
            n_parent = len(part.sample_ids)
            n_child = len(child.sample_ids)
            if n_parent == 0 or n_child % n_parent != 0:
                raise ValueError(
                    f"propagate_rewards: non-uniform branching from part {i + 1} ({n_child} samples) "
                    f"to {i} ({n_parent} samples). Group-by-parent ordering requires n_child % n_parent == 0."
                )
            branch = n_child // n_parent
            reshaped = child.rewards.view(n_parent, branch)
            if op == "mean":
                aggregated = reshaped.mean(dim=1)
            elif op == "max":
                aggregated = reshaped.amax(dim=1)
            elif op == "sum":
                aggregated = reshaped.sum(dim=1)
            else:
                raise ValueError(f"propagate_rewards: unknown op {op!r}; expected 'mean', 'max', or 'sum'.")
            new_parts[i] = _part_with_field(part, "rewards", aggregated)

        return type(self)(parts=new_parts, reward_compute_s=self.reward_compute_s)

    def conditioning(self) -> List[Primitive]:
        """Conditioning inputs for generating the frontier (last) part: each
        ancestor's ``primitive`` (raw text/image), row-aligned to the frontier's
        samples, in chronological order (root → frontier-parent). The model encodes
        these into its own conditions, mapping position → turn via the fixed
        schedule. Undecoded ancestors (``primitive is None``) are skipped; to get a
        non-frontier part's conditioning (replay), call this on ``parts[:i+1]``."""
        if not self.parts:
            return []
        out: List[Primitive] = []
        ridx = self.parts[-1].parent_index
        anc = len(self.parts) - 2 if ridx is not None else None
        while anc is not None:
            prim = self.parts[anc].primitive
            if prim is not None:
                out.append(prim.select(torch.tensor(ridx, dtype=torch.long)))
            nxt = self.parts[anc].parent_index
            ridx = [nxt[r] for r in ridx] if nxt is not None else None
            anc = anc - 1 if nxt is not None else None
        out.reverse()  # chronological: root → frontier-parent
        return out


__all__ = [
    "Sample",
    "Part",
    "Primitive",
    "STAGE_INPUT",
    "STAGE_GENERATION",
]
