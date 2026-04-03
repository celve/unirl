"""
TrajectoryStore — compact trajectory storage with integrated builder lifecycle.

Two creation paths:

1. **Builder** (used inside sampler denoising loops)::

       store = TrajectoryStore.for_sde_steps(sde_indices, num_steps)
       store.add(0, initial_latents)
       for i in range(num_steps):
           ...
           store.add(i + 1, latents)
       store.finalize()

2. **Direct** (pre-built data, already finalized)::

       store = TrajectoryStore.from_full(trajectories)
       store = TrajectoryStore.from_selective(trajectories, positions, T+1)
       store = TrajectoryStore.from_clean_latents(clean_latents)
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple, Union

import torch


def compute_trajectory_positions(sde_indices: Set[int], num_steps: int) -> List[int]:
    """Return sorted positions needed for ``(x_t, x_{t+1})`` pairs at SDE boundaries.

    For each SDE step index ``i`` in *sde_indices*, both position ``i`` and
    ``i + 1`` are required.  Results are clamped to ``[0, num_steps]``.

    >>> compute_trajectory_positions({0, 2, 4}, 5)
    [0, 1, 2, 3, 4, 5]
    >>> compute_trajectory_positions({3}, 5)
    [3, 4]
    """
    positions: Set[int] = set()
    for i in sde_indices:
        positions.add(max(0, min(i, num_steps)))
        positions.add(max(0, min(i + 1, num_steps)))
    return sorted(positions)


class TrajectoryStore:
    """Compact trajectory storage with index map and integrated builder lifecycle.

    Attributes (available after finalization):
        data: Dense trajectory tensor [B, K, ...] where K <= T+1
            (only collected positions).
        index_map: 1-D LongTensor of size ``total_positions`` where
            ``index_map[i]`` gives the compact index into ``data``
            for original position ``i``, or -1 if not stored.
        total_positions: Total number of positions in the full
            trajectory (T+1).
    """

    # ---- constructors -------------------------------------------------------

    def __init__(
        self,
        data: torch.Tensor,
        index_map: torch.Tensor,
        total_positions: int,
    ) -> None:
        self.data = data
        self.index_map = index_map
        self.total_positions = total_positions
        self._finalized = True
        self._total_steps: Optional[int] = None
        self._needed: Optional[Set[int]] = None
        self._collected: Optional[List[Tuple[int, torch.Tensor]]] = None

    @classmethod
    def _collecting(cls, total_steps: int, needed_positions: Set[int]) -> TrajectoryStore:
        """Internal: create a store in collecting (not-yet-finalized) state."""
        obj = object.__new__(cls)
        obj.data = None  # type: ignore[assignment]
        obj.index_map = None  # type: ignore[assignment]
        obj.total_positions = total_steps + 1
        obj._finalized = False
        obj._total_steps = total_steps
        obj._needed = needed_positions
        obj._collected = []
        return obj

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError(
                "TrajectoryStore has not been finalized. "
                "Call finalize() after the denoising loop."
            )

    # ---- builder classmethods -----------------------------------------------

    @classmethod
    def for_sde_steps(
        cls, sde_indices: Set[int], total_steps: int
    ) -> TrajectoryStore:
        """Create a collecting store that keeps only positions needed for SDE pairs.

        When *sde_indices* is empty (e.g. NFT with deterministic solver),
        the store falls back to keeping only the final position so that
        ``finalize()`` still produces a valid clean-latents store.
        """
        needed = set(compute_trajectory_positions(sde_indices, total_steps))
        if not needed:
            needed = {total_steps}
        return cls._collecting(total_steps, needed)

    @classmethod
    def full(cls, total_steps: int) -> TrajectoryStore:
        """Create a collecting store that keeps all positions."""
        return cls._collecting(total_steps, set(range(total_steps + 1)))

    # ---- builder methods ----------------------------------------------------

    def add(self, position: int, latents: torch.Tensor) -> None:
        """Record latents at *position*.  Silently drops unneeded positions."""
        if self._finalized:
            raise RuntimeError("Cannot add() to a finalized TrajectoryStore.")
        if position in self._needed:  # type: ignore[operator]
            self._collected.append((position, latents))  # type: ignore[union-attr]

    def finalize(self) -> TrajectoryStore:
        """Freeze collected latents into a usable trajectory store.

        Returns *self* for convenience chaining.
        """
        if self._finalized:
            raise RuntimeError("TrajectoryStore is already finalized.")
        if not self._collected:
            raise ValueError(
                "finalize() called with no collected positions. "
                f"needed={sorted(self._needed)}"  # type: ignore[arg-type]
            )
        positions = [p for p, _ in self._collected]
        data = torch.stack([t for _, t in self._collected], dim=1)
        total_positions = self._total_steps + 1  # type: ignore[operator]

        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, orig_pos in enumerate(positions):
            if 0 <= orig_pos < total_positions:
                index_map[orig_pos] = compact_idx

        self.data = data
        self.index_map = index_map
        self.total_positions = total_positions
        self._finalized = True
        self._total_steps = None
        self._needed = None
        self._collected = None
        return self

    # ---- direct factories (produce finalized stores) ------------------------

    @classmethod
    def from_full(cls, trajectories: torch.Tensor) -> TrajectoryStore:
        """Wrap full trajectories [B, T+1, ...] with an identity index map."""
        t_plus_1 = int(trajectories.shape[1])
        index_map = torch.arange(t_plus_1, dtype=torch.long)
        return cls(data=trajectories, index_map=index_map, total_positions=t_plus_1)

    @classmethod
    def from_clean_latents(
        cls,
        clean_latents: torch.Tensor,
        total_positions: int = 1,
    ) -> TrajectoryStore:
        """NFT path: store only clean latents as a single-position trajectory.

        The latents are placed at position ``total_positions - 1``
        (the final denoised state).
        """
        data = clean_latents.unsqueeze(1)  # [B, 1, ...]
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        index_map[total_positions - 1] = 0
        return cls(data=data, index_map=index_map, total_positions=total_positions)

    @classmethod
    def from_selective(
        cls,
        trajectories: torch.Tensor,
        collected_positions: List[int],
        total_positions: int,
    ) -> TrajectoryStore:
        """Selective storage: K positions out of T+1.

        Args:
            trajectories: Dense tensor [B, K, ...]
            collected_positions: Sorted list of original position indices
                corresponding to each column in *trajectories*.
            total_positions: Total number of positions (T+1).
        """
        if int(trajectories.shape[1]) != len(collected_positions):
            raise ValueError(
                f"trajectories dim-1 ({trajectories.shape[1]}) != "
                f"len(collected_positions) ({len(collected_positions)})"
            )
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, orig_pos in enumerate(collected_positions):
            if 0 <= orig_pos < total_positions:
                index_map[orig_pos] = compact_idx
        return cls(data=trajectories, index_map=index_map, total_positions=total_positions)

    # ---- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        self._require_finalized()
        return int(self.data.shape[0])

    @property
    def num_stored(self) -> int:
        """Number of positions actually stored (K)."""
        self._require_finalized()
        return int(self.data.shape[1])

    @property
    def device(self) -> torch.device:
        self._require_finalized()
        return self.data.device

    @property
    def is_full(self) -> bool:
        """True when all positions are stored (K == T+1)."""
        return self.num_stored == self.total_positions

    @property
    def is_clean_latents_only(self) -> bool:
        """True when only a single position is stored (NFT path)."""
        return self.num_stored == 1

    @property
    def is_selective(self) -> bool:
        """True when storing a subset of positions (not full, not clean-only)."""
        return not self.is_full and not self.is_clean_latents_only

    @property
    def clean_latents(self) -> torch.Tensor:
        """Final denoised latents — last stored position."""
        self._require_finalized()
        last_stored = int((self.index_map >= 0).nonzero(as_tuple=False)[-1].item())
        compact_idx = int(self.index_map[last_stored].item())
        return self.data[:, compact_idx]

    # ---- position access ----------------------------------------------------

    def has_position(self, pos: int) -> bool:
        """Check whether original position *pos* is stored."""
        self._require_finalized()
        if pos < 0 or pos >= self.total_positions:
            return False
        return int(self.index_map[pos].item()) >= 0

    def get_position(self, pos: int) -> torch.Tensor:
        """Get latents at original position *pos*.  O(1) via index_map."""
        self._require_finalized()
        if pos < 0 or pos >= self.total_positions:
            raise IndexError(
                f"Position {pos} out of range [0, {self.total_positions})"
            )
        compact_idx = int(self.index_map[pos].item())
        if compact_idx < 0:
            raise IndexError(
                f"Position {pos} was not collected (index_map=-1). "
                f"Stored positions: {self.stored_positions}"
            )
        return self.data[:, compact_idx]

    def get_pair(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (x_t, x_{t+1}) latent pair for an SDE training step.

        Both *pos* and *pos+1* must be stored positions.
        """
        return self.get_position(pos), self.get_position(pos + 1)

    @property
    def stored_positions(self) -> List[int]:
        """Sorted list of original positions that are stored."""
        self._require_finalized()
        return sorted(
            int(i) for i in range(self.total_positions)
            if int(self.index_map[i].item()) >= 0
        )

    # ---- batch ops ----------------------------------------------------------

    def slice_batch(self, start: int, end: int) -> TrajectoryStore:
        """Slice along the batch (dim-0) dimension."""
        self._require_finalized()
        return TrajectoryStore(
            data=self.data[start:end].clone(),
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def slice(self, start: int, end: int) -> TrajectoryStore:
        """Alias for slice_batch — enables slice_columnar_value protocol."""
        return self.slice_batch(start, end)

    def reindex_batch(self, indices: torch.Tensor) -> TrajectoryStore:
        """Reindex along the batch (dim-0) dimension."""
        self._require_finalized()
        return TrajectoryStore(
            data=self.data[indices],
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def index_select_batch(self, idx: torch.Tensor) -> TrajectoryStore:
        """Select samples by index along the batch dimension."""
        self._require_finalized()
        return TrajectoryStore(
            data=self.data.index_select(0, idx.to(self.data.device)),
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def to_device(self, device: Union[str, torch.device]) -> TrajectoryStore:
        """Move data tensor to *device*. index_map stays on CPU."""
        self._require_finalized()
        return TrajectoryStore(
            data=self.data.to(device=device),
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def cast_dtype(self, dtype: torch.dtype) -> TrajectoryStore:
        """Cast data tensor to *dtype* if it is floating-point and differs."""
        self._require_finalized()
        if self.data.is_floating_point() and self.data.dtype != dtype:
            return TrajectoryStore(
                data=self.data.to(dtype=dtype),
                index_map=self.index_map,
                total_positions=self.total_positions,
            )
        return self

    @classmethod
    def concat(cls, stores: List[TrajectoryStore]) -> TrajectoryStore:
        """Concatenate stores along the batch dimension.

        All stores must have identical index_map and total_positions.
        """
        if not stores:
            raise ValueError("Cannot concat empty store list.")
        first = stores[0]
        first._require_finalized()
        for s in stores[1:]:
            s._require_finalized()
            if s.total_positions != first.total_positions:
                raise ValueError("Inconsistent total_positions across stores.")
            if not torch.equal(s.index_map, first.index_map):
                raise ValueError("Inconsistent index_map across stores.")
        data = torch.cat([s.data for s in stores], dim=0)
        return cls(
            data=data,
            index_map=first.index_map,
            total_positions=first.total_positions,
        )

    # ---- modality detection -------------------------------------------------

    def detect_modality(self) -> str:
        """Detect media modality from the stored data shape."""
        self._require_finalized()
        if self.is_full:
            return "video" if int(self.data.ndim) >= 6 else "image"
        return "video" if int(self.data.ndim) >= 5 else "image"


__all__ = ["TrajectoryStore", "compute_trajectory_positions"]
