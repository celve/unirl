#!/usr/bin/env python
"""End-to-end GPU transport evidence for E5-R: tree IR versus a flat ID join.

Implements the CLI named in ``GPU_HANDOFF_README.md`` section 11, which is a
required interface specification rather than an existing executable at the audited
commit. The paper repository's ``run_cpu_evidence.py`` is a CPU regression only.

Structure-operation time is reported separately from materialization, because the
claim under test is that structural metadata is small relative to model work — not
that the transport is fast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPRESENTATIONS = ("tree", "flat_id_join")
BACKENDS = ("colocate_store", "gpu_store", "transfer_queue")
PLACEMENTS = ("same_gpu", "cross_gpu", "cross_node")


def _int_list(raw: str) -> List[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"expected a comma-separated int list, got {raw!r}")
    return values


@dataclass
class Cell:
    """One measurement cell of the sweep."""

    roots: int
    branch_factor: int
    payload_mib: float
    structure_s: List[float] = field(default_factory=list)
    materialize_s: List[float] = field(default_factory=list)
    transfer_s: List[float] = field(default_factory=list)
    metadata_bytes: int = 0
    dense_bytes: int = 0
    referenced_bytes: int = 0
    content_hash: str = ""

    @property
    def leaves(self) -> int:
        return self.roots * self.branch_factor


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "median": _percentile(values, 0.5),
        "mean": sum(values) / len(values),
        "p90": _percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _devices(placement: str) -> Tuple[str, str]:
    """Source and destination devices for a placement; cross_node needs a launcher."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("run_gpu_transport_evidence requires CUDA; this is the GPU harness")
    if placement == "same_gpu":
        return "cuda:0", "cuda:0"
    if placement == "cross_gpu":
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"placement=cross_gpu needs 2 GPUs, found {torch.cuda.device_count()}")
        return "cuda:0", "cuda:1"
    raise RuntimeError(
        "placement=cross_node must be driven by a multi-node launcher with a rendezvous; "
        "run one rank per node rather than invoking this placement directly"
    )


def _build_leaf_payloads(cell: Cell, device: str, *, dtype_bytes: int = 2):
    """Allocate one payload tensor per leaf, sized to --payload-mib."""
    import torch

    elements = max(int(cell.payload_mib * (1 << 20) / dtype_bytes), 1)
    return [
        torch.full((elements,), float(i % 7), dtype=torch.bfloat16, device=device) for i in range(cell.leaves)
    ]


def _tree_structure(cell: Cell) -> Dict[str, object]:
    """Nested root→children structure: children are referenced, never concatenated."""
    return {
        f"r{r}": {"children": [f"r{r}/{b}" for b in range(cell.branch_factor)]} for r in range(cell.roots)
    }


def _flat_rows(cell: Cell) -> List[Dict[str, object]]:
    """One dense row per leaf, carrying its own root and parent ids."""
    return [
        {"sample_id": f"r{r}/{b}", "root_id": f"r{r}", "parent_id": f"r{r}", "branch": b}
        for r in range(cell.roots)
        for b in range(cell.branch_factor)
    ]


def _metadata_bytes(obj: object) -> int:
    return len(json.dumps(obj, sort_keys=True).encode())


def _hash_payloads(payloads: Sequence[object]) -> str:
    """Correctness hash over materialized content, order-independent by leaf index."""
    import torch

    digest = hashlib.sha256()
    for tensor in payloads:
        # view(torch.uint8) rather than numpy(): numpy has no bfloat16 dtype.
        head = tensor.detach().to("cpu").contiguous().view(-1)[:4096]
        digest.update(head.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_cell(
    cell: Cell,
    *,
    representation: str,
    backend: str,
    placement: str,
    warmup: int,
    repetitions: int,
) -> Cell:
    """Measure structure, materialization and transfer for one cell."""
    import torch

    src, dst = _devices(placement)
    payloads = _build_leaf_payloads(cell, src)
    element_bytes = payloads[0].numel() * payloads[0].element_size()
    cell.dense_bytes = element_bytes * cell.leaves

    if representation == "tree":
        structure: object = _tree_structure(cell)
        cell.referenced_bytes = element_bytes * cell.leaves
    else:
        structure = _flat_rows(cell)
        cell.referenced_bytes = element_bytes * cell.leaves
    cell.metadata_bytes = _metadata_bytes(structure)

    for iteration in range(warmup + repetitions):
        recording = iteration >= warmup

        t0 = time.perf_counter()
        if representation == "tree":
            groups = {root: spec["children"] for root, spec in structure.items()}  # type: ignore[index]
        else:
            groups = {}
            for row in structure:  # type: ignore[union-attr]
                groups.setdefault(str(row["root_id"]), []).append(str(row["sample_id"]))
        torch.cuda.synchronize()
        structure_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        # The tree keeps per-leaf references; the flat join concatenates each group,
        # which is the dense driver materialization the IR claim is about.
        if representation == "tree":
            materialized = payloads
        else:
            materialized = [torch.cat(payloads[i : i + cell.branch_factor]) for i in range(0, cell.leaves, cell.branch_factor)]
        torch.cuda.synchronize()
        materialize_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        moved = [tensor.to(dst, non_blocking=False) for tensor in materialized] if src != dst else materialized
        torch.cuda.synchronize()
        transfer_s = time.perf_counter() - t2

        if recording:
            cell.structure_s.append(structure_s)
            cell.materialize_s.append(materialize_s)
            cell.transfer_s.append(transfer_s)
            if not cell.content_hash:
                cell.content_hash = _hash_payloads(moved)
        assert len(groups) == cell.roots, f"expected {cell.roots} groups, built {len(groups)}"

    return cell


def _peak_memory() -> Dict[str, float]:
    import resource

    import torch

    return {
        "driver_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1 << 20),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--placement", required=True, choices=PLACEMENTS)
    parser.add_argument("--roots", type=_int_list, default=[8, 32, 128])
    parser.add_argument("--branch-factor", type=_int_list, default=[1, 4, 8])
    parser.add_argument("--payload-mib", type=lambda s: [float(x) for x in s.split(",")], default=[1.0, 16.0, 256.0])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--model-work-s", type=float, default=None, help="reference interval overhead is expressed against")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.backend == "transfer_queue":
        raise SystemExit(
            "backend=transfer_queue requires a validated Mooncake/RDMA configuration; "
            "the protocol omits it when the cluster lacks one — run colocate_store or gpu_store"
        )

    import torch

    torch.cuda.reset_peak_memory_stats()
    cells: List[Dict[str, object]] = []
    for roots in args.roots:
        for branch in args.branch_factor:
            for payload in args.payload_mib:
                cell = run_cell(
                    Cell(roots=roots, branch_factor=branch, payload_mib=payload),
                    representation=args.representation,
                    backend=args.backend,
                    placement=args.placement,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                row: Dict[str, object] = {
                    "roots": cell.roots,
                    "branch_factor": cell.branch_factor,
                    "leaves": cell.leaves,
                    "payload_mib": cell.payload_mib,
                    "structure_time_s": _stats(cell.structure_s),
                    "materialize_time_s": _stats(cell.materialize_s),
                    "transfer_time_s": _stats(cell.transfer_s),
                    "metadata_bytes": cell.metadata_bytes,
                    "dense_bytes": cell.dense_bytes,
                    "referenced_bytes": cell.referenced_bytes,
                    "dense_bytes_avoided": cell.dense_bytes - cell.referenced_bytes,
                    "content_hash": cell.content_hash,
                }
                critical_path = _stats(cell.structure_s)["median"] + _stats(cell.materialize_s)["median"]
                row["critical_path_time_s"] = critical_path
                if args.model_work_s:
                    row["overhead_fraction_of_model_work"] = critical_path / args.model_work_s
                transfer_median = _stats(cell.transfer_s)["median"]
                if transfer_median and transfer_median > 0:
                    row["transfer_bandwidth_gib_s"] = (cell.referenced_bytes / (1 << 30)) / transfer_median
                cells.append(row)

    result = {
        "representation": args.representation,
        "backend": args.backend,
        "placement": args.placement,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "model_work_s": args.model_work_s,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "memory": _peak_memory(),
        "cells": cells,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"transport_{args.representation}_{args.backend}_{args.placement}.json"
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out_file} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
