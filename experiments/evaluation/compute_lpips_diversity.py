#!/usr/bin/env python
"""Within-prompt mean pairwise LPIPS — the preregistered E2 diversity guard.

Implements the CLI named in ``GPU_HANDOFF_README.md`` section 8.4, which is an
interface requirement rather than a checked-in executable at the audited commit.
The guard is against within-prompt collapse; it is not an image-quality metric and
must not be reported as one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 120 = C(16, 2), the unordered pair count the protocol fixes for 16 images/prompt.
EXPECTED_PAIRS_PER_PROMPT = 120


@dataclass(frozen=True)
class PromptGroup:
    """One prompt's image set, as named by the group manifest."""

    prompt_id: str
    prompt: str
    images: Tuple[str, ...]


def load_group_manifest(path: Path) -> List[PromptGroup]:
    """Read the JSONL manifest built from the registry's deduplicated prompt order."""
    groups: List[PromptGroup] = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"prompt_id", "prompt", "images"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{lineno}: group manifest row missing {sorted(missing)}")
            groups.append(
                PromptGroup(
                    prompt_id=str(row["prompt_id"]),
                    prompt=str(row["prompt"]),
                    images=tuple(str(name) for name in row["images"]),
                )
            )
    if not groups:
        raise ValueError(f"{path}: group manifest is empty")
    return groups


def _load_image_tensor(path: Path):
    """Load one RGB image normalized to [-1, 1] as 1x3xHxW, per the protocol."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def mean_pairwise_lpips(
    image_dir: Path,
    group: PromptGroup,
    metric,
    device: str,
    *,
    strict_pair_count: bool,
) -> float:
    """Average the unordered pair distances within one prompt's image set."""
    import torch

    paths = [image_dir / name for name in group.images]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"prompt {group.prompt_id}: missing images {missing[:4]}")
    if len(paths) < 2:
        raise ValueError(f"prompt {group.prompt_id}: {len(paths)} image(s), need at least 2")

    pairs = list(combinations(range(len(paths)), 2))
    if strict_pair_count and len(pairs) != EXPECTED_PAIRS_PER_PROMPT:
        raise ValueError(
            f"prompt {group.prompt_id}: {len(pairs)} pairs from {len(paths)} images, "
            f"protocol fixes {EXPECTED_PAIRS_PER_PROMPT} (16 images). Pass --allow-partial-groups to override."
        )

    tensors = [_load_image_tensor(p).to(device) for p in paths]
    total = 0.0
    with torch.no_grad():
        for i, j in pairs:
            total += float(metric(tensors[i], tensors[j]).item())
    return total / len(pairs)


def paired_prompt_bootstrap(
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, float]:
    """Paired prompt bootstrap over the relative change, resampling prompts not images."""
    if len(base) != len(candidate):
        raise ValueError(f"bootstrap: {len(base)} base prompts against {len(candidate)} candidate prompts")
    n = len(base)
    rng = random.Random(seed)
    ratios: List[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        base_mean = sum(base[i] for i in idx) / n
        cand_mean = sum(candidate[i] for i in idx) / n
        if base_mean == 0.0:
            continue
        ratios.append((cand_mean - base_mean) / base_mean)
    if not ratios:
        return {"relative_change_ci_low": float("nan"), "relative_change_ci_high": float("nan"), "resamples_used": 0}
    ratios.sort()
    return {
        "relative_change_ci_low": ratios[int(0.025 * len(ratios))],
        "relative_change_ci_high": ratios[min(int(0.975 * len(ratios)), len(ratios) - 1)],
        "resamples_used": len(ratios),
    }


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metric(network: str, device: str):
    """Frozen LPIPS with pretrained features; the protocol fixes AlexNet for this guard."""
    import lpips

    return lpips.LPIPS(net=network).to(device).eval()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-images", type=Path, required=True)
    parser.add_argument("--candidate-images", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--network", default="alex", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None, help="defaults to cuda when available")
    parser.add_argument(
        "--allow-partial-groups",
        action="store_true",
        help="permit prompt groups that are not the protocol's 16 images",
    )
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    groups = load_group_manifest(args.group_manifest)
    metric = build_metric(args.network, device)

    per_prompt: List[Dict[str, object]] = []
    base_values: List[float] = []
    candidate_values: List[float] = []
    for group in groups:
        base_mean = mean_pairwise_lpips(
            args.base_images, group, metric, device, strict_pair_count=not args.allow_partial_groups
        )
        cand_mean = mean_pairwise_lpips(
            args.candidate_images, group, metric, device, strict_pair_count=not args.allow_partial_groups
        )
        base_values.append(base_mean)
        candidate_values.append(cand_mean)
        per_prompt.append({**asdict(group), "base_lpips": base_mean, "candidate_lpips": cand_mean})

    base_overall = sum(base_values) / len(base_values)
    cand_overall = sum(candidate_values) / len(candidate_values)
    absolute_change = cand_overall - base_overall
    relative_defined = base_overall > 1e-8

    result: Dict[str, object] = {
        "network": args.network,
        "device": device,
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "num_prompts": len(groups),
        "images_per_prompt": len(groups[0].images),
        "group_manifest_sha256": _sha256_of(args.group_manifest),
        "base_mean_lpips": base_overall,
        "candidate_mean_lpips": cand_overall,
        "absolute_change": absolute_change,
        "relative_change": (absolute_change / base_overall) if relative_defined else None,
        "relative_criterion_defined": relative_defined,
        "investigation_trigger_10pct_decrease": bool(relative_defined and (absolute_change / base_overall) < -0.10),
        "per_prompt": per_prompt,
    }
    if relative_defined:
        result.update(paired_prompt_bootstrap(base_values, candidate_values, resamples=args.bootstrap, seed=args.seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
