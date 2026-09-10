#!/usr/bin/env python
"""Root-conditioned and rewrite-conditioned scoring of PE lineage checkpoints.

Implements the CLI named in ``GPU_HANDOFF_README.md`` section 10, which is an
interface requirement rather than an existing executable at the audited commit.

The stock PE recipe has ``eval_interval=0`` and so produces no held-out images at
all; training-trace samples cannot substitute for checkpoint evaluation. Two rules
from the protocol are enforced here rather than left to the analyst:
aggregation is images then rewrites then roots, and the bootstrap resamples roots —
32 image descendants of one root are not 32 independent prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EVALUATORS = ("hpsv3", "imagereward")


@dataclass(frozen=True)
class Rewrite:
    """One cached rewrite of one held-out root prompt."""

    root_id: str
    root_text: str
    rewrite_id: str
    rewrite_text: str


def load_rewrite_manifest(path: Path) -> List[Rewrite]:
    """Read the frozen rewrite manifest cached once with the frozen Qwen model."""
    rewrites: List[Rewrite] = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"root_id", "root_text", "rewrite_id", "rewrite_text"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{lineno}: rewrite manifest row missing {sorted(missing)}")
            rewrites.append(
                Rewrite(
                    root_id=str(row["root_id"]),
                    root_text=str(row["root_text"]),
                    rewrite_id=str(row["rewrite_id"]),
                    rewrite_text=str(row["rewrite_text"]),
                )
            )
    if not rewrites:
        raise ValueError(f"{path}: rewrite manifest is empty")
    return rewrites


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def image_seed(base_seed: int, rewrite_id: str, index: int) -> int:
    """Identical image seeds across scopes, seeds and checkpoints — a fixed eval RNG."""
    digest = hashlib.sha256(f"{base_seed}:{rewrite_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def build_pipeline(base: str, adapter: Optional[str], device: str):
    """Load the common frozen diffusion evaluation pipeline, optionally with a PE adapter.

    PE checkpoints place the trainable side under ``checkpoint-<rollout>/diffusion``,
    so the caller passes that subdirectory, not the checkpoint root.
    """
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipeline = StableDiffusion3Pipeline.from_pretrained(base, torch_dtype=torch.bfloat16).to(device)
    if adapter:
        pipeline.load_lora_weights(adapter)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_images(
    pipeline,
    rewrite: Rewrite,
    *,
    samples: int,
    seed: int,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    out_dir: Path,
) -> List[Path]:
    """Generate ``samples`` images for one rewrite from fixed per-image seeds."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index in range(samples):
        this_seed = image_seed(seed, rewrite.rewrite_id, index)
        generator = torch.Generator(device=pipeline.device).manual_seed(this_seed)
        image = pipeline(
            prompt=rewrite.rewrite_text,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        path = out_dir / f"{rewrite.rewrite_id}_{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


def assert_service_serves(reward_url: str, evaluators: Sequence[str]) -> List[str]:
    """Fail before generating anything if the service does not serve every evaluator.

    A scorer whose optional dependency failed to install is absent from ``/rewards``
    rather than erroring per request, so an unchecked run would quietly report on
    fewer metrics than it claims.
    """
    import requests

    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{reward_url.rstrip('/')}/rewards", timeout=60)
    response.raise_for_status()
    served = list(response.json()["rewards"])
    missing = [name for name in evaluators if name not in served]
    if missing:
        raise RuntimeError(
            f"reward service at {reward_url} serves {served} but the run requires {list(evaluators)}; "
            f"missing {missing} — install the scorer rather than dropping the metric"
        )
    return served


def score_images(
    image_paths: Sequence[Path],
    prompt: str,
    *,
    evaluator: str,
    reward_url: Optional[str],
    batch_size: int = 8,
) -> List[float]:
    """Score each image against one prompt through the pinned reward service.

    Speaks ``reward_service.schemas``: a batch of ``{history: [{text, image_b64}],
    required_rewards: [...]}`` under ``requests``, answered as
    ``results[i][reward][sub_metric]``.
    """
    if not reward_url:
        raise RuntimeError(
            f"evaluator {evaluator} needs --reward-url pointing at the pinned reward service; "
            "local scorer weights are not part of this evaluator's contract"
        )
    import base64

    import requests

    session = requests.Session()
    session.trust_env = False
    endpoint = f"{reward_url.rstrip('/')}/score"

    scores: List[float] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = list(image_paths[start : start + batch_size])
        payload = {
            "protocol_version": "1",
            "requests": [
                {
                    "history": [
                        {"text": prompt, "image_b64": base64.b64encode(path.read_bytes()).decode()}
                    ],
                    "required_rewards": [evaluator],
                    "sample_id": path.name,
                }
                for path in chunk
            ],
        }
        response = session.post(endpoint, json=payload, timeout=1800)
        response.raise_for_status()
        body = response.json()
        results = body.get("results") or []
        errors = body.get("errors") or []
        if len(results) != len(chunk):
            raise RuntimeError(f"reward service returned {len(results)} results for {len(chunk)} images")
        for index, (path, result) in enumerate(zip(chunk, results)):
            failure = errors[index].get(evaluator) if index < len(errors) else None
            if failure:
                raise RuntimeError(f"{evaluator} failed on {path.name}: {failure}")
            metrics = result.get(evaluator)
            if not metrics:
                raise RuntimeError(f"{evaluator} returned no metrics for {path.name}: {result}")
            # One finite score per metric per image; a dropped sub-metric must not average away.
            values = [float(v) for v in metrics.values()]
            if len(values) != 1 or not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{evaluator} on {path.name} is not one finite score: {metrics}")
            scores.append(values[0])
    return scores


def aggregate(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, float]:
    """Images then rewrites then roots — never a flat mean over images."""
    by_rewrite: Dict[str, List[float]] = {}
    rewrite_root: Dict[str, str] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        by_rewrite.setdefault(str(row["rewrite_id"]), []).append(float(value))
        rewrite_root[str(row["rewrite_id"])] = str(row["root_id"])

    rewrite_means = {rid: sum(vals) / len(vals) for rid, vals in by_rewrite.items()}
    by_root: Dict[str, List[float]] = {}
    for rid, mean in rewrite_means.items():
        by_root.setdefault(rewrite_root[rid], []).append(mean)
    root_means = {root: sum(vals) / len(vals) for root, vals in by_root.items()}
    overall = sum(root_means.values()) / len(root_means) if root_means else float("nan")
    return {"mean": overall, "num_roots": len(root_means), "num_rewrites": len(rewrite_means)}


def bootstrap_roots(rows: Sequence[Dict[str, object]], key: str, *, resamples: int, seed: int) -> Dict[str, float]:
    """Bootstrap over roots, the independent unit; rewrites and images are nested."""
    by_root: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            by_root.setdefault(str(row["root_id"]), []).append(float(value))
    roots = sorted(by_root)
    if len(roots) < 2:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "resamples": 0}

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(resamples):
        picked = [roots[rng.randrange(len(roots))] for _ in roots]
        per_root = [sum(by_root[r]) / len(by_root[r]) for r in picked]
        means.append(sum(per_root) / len(per_root))
    means.sort()
    return {
        "ci_low": means[int(0.025 * len(means))],
        "ci_high": means[min(int(0.975 * len(means)), len(means) - 1)],
        "resamples": len(means),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="frozen SD3.5 base checkpoint")
    parser.add_argument("--adapter", default=None, help="checkpoint-<n>/diffusion; omit for the base row")
    parser.add_argument("--rewrite-manifest", type=Path, required=True)
    parser.add_argument("--samples-per-rewrite", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--evaluators", default="hpsv3,imagereward")
    parser.add_argument("--reward-url", default=None)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--scope", default=None, help="recorded on every row: prompt or rewrite")
    parser.add_argument("--checkpoint", default=None, help="recorded on every row, e.g. checkpoint-300")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    evaluators = [name.strip() for name in args.evaluators.split(",") if name.strip()]
    unknown = [name for name in evaluators if name not in EVALUATORS]
    if unknown:
        raise SystemExit(f"unknown evaluators {unknown}; supported: {list(EVALUATORS)}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    served = assert_service_serves(args.reward_url, evaluators) if args.reward_url else []
    rewrites = load_rewrite_manifest(args.rewrite_manifest)
    pipeline = build_pipeline(args.base, args.adapter, device)

    args.output.mkdir(parents=True, exist_ok=True)
    image_root = args.output / "images"
    rows: List[Dict[str, object]] = []

    for rewrite in rewrites:
        paths = generate_images(
            pipeline,
            rewrite,
            samples=args.samples_per_rewrite,
            seed=args.seed,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance=args.guidance,
            out_dir=image_root / rewrite.root_id,
        )
        # Root-conditioned and rewrite-conditioned scores must be shown separately:
        # the training request uses the rewrite, so it cannot establish root intent.
        for evaluator in evaluators:
            root_scores = score_images(
                paths, rewrite.root_text, evaluator=evaluator,
                reward_url=args.reward_url, batch_size=args.score_batch_size,
            )
            rewrite_scores = score_images(
                paths, rewrite.rewrite_text, evaluator=evaluator,
                reward_url=args.reward_url, batch_size=args.score_batch_size,
            )
            for index, path in enumerate(paths):
                rows.append(
                    {
                        "root_id": rewrite.root_id,
                        "parent_id": rewrite.root_id,
                        "rewrite_id": rewrite.rewrite_id,
                        "root_text": rewrite.root_text,
                        "rewrite_text": rewrite.rewrite_text,
                        "root_text_sha256": _sha256_text(rewrite.root_text),
                        "rewrite_text_sha256": _sha256_text(rewrite.rewrite_text),
                        "image_path": str(path),
                        "image_sha256": _sha256_file(path),
                        "image_index": index,
                        "image_seed": image_seed(args.seed, rewrite.rewrite_id, index),
                        "scope": args.scope,
                        "checkpoint": args.checkpoint,
                        "evaluator": evaluator,
                        f"{evaluator}_root_conditioned": root_scores[index],
                        f"{evaluator}_rewrite_conditioned": rewrite_scores[index],
                    }
                )

    rows_path = args.output / "rows.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary: Dict[str, object] = {
        "base": args.base,
        "adapter": args.adapter,
        "scope": args.scope,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "samples_per_rewrite": args.samples_per_rewrite,
        "rewrite_manifest_sha256": _sha256_file(args.rewrite_manifest),
        "num_rows": len(rows),
        "reward_service_url": args.reward_url,
        "reward_service_rewards": served,
        "generation": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "guidance": args.guidance,
        },
    }
    for evaluator in evaluators:
        for conditioning in ("root_conditioned", "rewrite_conditioned"):
            key = f"{evaluator}_{conditioning}"
            summary[key] = {
                **aggregate(rows, key),
                "bootstrap_roots": bootstrap_roots(rows, key, resamples=args.bootstrap, seed=args.seed),
            }

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
