#!/usr/bin/env python
"""
Analyze debug tensors dumped by GRPO train-inference consistency debugging.

This script compares per-step tensors saved during sampling and training
to find where the first inconsistency appears.

Usage:
    python scripts/analyze_debug_tensors.py /path/to/debug_output
    python scripts/analyze_debug_tensors.py /path/to/debug_output --verbose
    python scripts/analyze_debug_tensors.py /path/to/debug_output --top-k 3

The script will:
1. Load sampling and training tensors for each matching step
2. Compare noise_pred, latents_input, latents_output, prev_sample_mean, log_prob
3. Report per-step max/mean absolute differences
4. Identify the first step where divergence exceeds a threshold
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

# Tensor names that should match between sampling and training
COMPARISON_TENSORS = [
    "noise_pred",
    "latents_input",
    "latents_output",
    "prev_sample_mean",
    "sigma",
    "sigma_next",
]

# Tensors with per-sample scalar values (1D, [B])
SCALAR_TENSORS = [
    "log_prob",  # sampling: old log prob; training: new log prob from replayed model
]

# Training-only tensors for diagnostics
TRAINING_ONLY_TENSORS = [
    "old_log_prob",
    "new_log_prob",
    "ratio",
    "sigma_max",
]


def load_step_tensors(step_dir: str) -> Dict[str, torch.Tensor]:
    """Load all .pt tensors from a step directory."""
    tensors = {}
    if not os.path.isdir(step_dir):
        return tensors
    for fname in sorted(os.listdir(step_dir)):
        if fname.endswith(".pt"):
            name = fname[:-3]
            tensors[name] = torch.load(os.path.join(step_dir, fname), map_location="cpu")
    return tensors


def discover_steps(base_dir: str) -> List[int]:
    """Discover all step_XXX directories and return sorted step indices."""
    steps = []
    if not os.path.isdir(base_dir):
        return steps
    for entry in os.listdir(base_dir):
        if entry.startswith("step_") and os.path.isdir(os.path.join(base_dir, entry)):
            try:
                step_idx = int(entry.split("_")[1])
                steps.append(step_idx)
            except (IndexError, ValueError):
                continue
    return sorted(steps)


def tensor_stats(t: torch.Tensor) -> str:
    """Return a compact stats string for a tensor."""
    if t.numel() == 0:
        return "empty"
    t_f = t.float()
    return (
        f"shape={list(t.shape)} dtype={t.dtype} "
        f"min={t_f.min().item():.6e} max={t_f.max().item():.6e} "
        f"mean={t_f.mean().item():.6e} std={t_f.std().item():.6e}"
    )


def compare_tensors(
    name: str,
    sampling_t: Optional[torch.Tensor],
    training_t: Optional[torch.Tensor],
    allow_prefix_match: bool = False,
) -> Dict[str, float]:
    """Compare two tensors and return difference metrics.

    Args:
        name: Name of the tensor for metric keys.
        sampling_t: Tensor from sampling side.
        training_t: Tensor from training side.
        allow_prefix_match: If True and tensors differ only in batch dim (dim 0),
            compare the overlapping prefix (min batch size). This is useful when
            sampling and training have different micro-batch sizes.
    """
    result = {}
    if sampling_t is None and training_t is None:
        return result
    if sampling_t is None:
        result[f"{name}_missing_in_sampling"] = 1.0
        return result
    if training_t is None:
        result[f"{name}_missing_in_training"] = 1.0
        return result

    # Ensure same shape for comparison
    s = sampling_t.float()
    t = training_t.float()

    if s.shape != t.shape:
        # 当 allow_prefix_match=True 且仅 batch 维度不同时，取前缀子集比较
        if allow_prefix_match and s.ndim == t.ndim and s.ndim >= 1 and s.shape[1:] == t.shape[1:]:
            min_b = min(s.shape[0], t.shape[0])
            result[f"{name}_shape_mismatch"] = 1.0
            result[f"{name}_sampling_shape"] = str(list(s.shape))
            result[f"{name}_training_shape"] = str(list(t.shape))
            result[f"{name}_prefix_compared"] = float(min_b)
            s = s[:min_b]
            t = t[:min_b]
        else:
            result[f"{name}_shape_mismatch"] = 1.0
            result[f"{name}_sampling_shape"] = str(list(s.shape))
            result[f"{name}_training_shape"] = str(list(t.shape))
            return result

    diff = (s - t).abs()
    result[f"{name}_max_abs_diff"] = diff.max().item()
    result[f"{name}_mean_abs_diff"] = diff.mean().item()
    result[f"{name}_max_rel_diff"] = (diff / (t.abs() + 1e-12)).max().item()

    # Check for exact equality
    result[f"{name}_exact_match"] = float(torch.equal(s, t))

    return result


def analyze_step(
    step_idx: int, sampling_dir: str, training_dir: str, verbose: bool = False, threshold: float = 1e-6
) -> Tuple[Dict[str, float], bool]:
    """Analyze a single step for train-inference consistency.

    Returns:
        Tuple of (metrics_dict, has_divergence)
    """
    sampling_step_dir = os.path.join(sampling_dir, f"step_{step_idx:03d}")
    training_step_dir = os.path.join(training_dir, f"step_{step_idx:03d}")

    sampling_tensors = load_step_tensors(sampling_step_dir)
    training_tensors = load_step_tensors(training_step_dir)

    if not sampling_tensors and not training_tensors:
        return {}, False

    metrics = {}
    has_divergence = False

    # Compare matching tensors
    for name in COMPARISON_TENSORS:
        s_t = sampling_tensors.get(name)
        t_t = training_tensors.get(name)
        step_metrics = compare_tensors(name, s_t, t_t, allow_prefix_match=True)
        metrics.update(step_metrics)

        max_diff_key = f"{name}_max_abs_diff"
        shape_mismatch_key = f"{name}_shape_mismatch"
        if shape_mismatch_key in step_metrics and f"{name}_prefix_compared" not in step_metrics:
            # 完全 shape 不匹配（非 batch 维度差异），视为 divergence
            has_divergence = True
        elif max_diff_key in step_metrics and step_metrics[max_diff_key] > threshold:
            has_divergence = True

    # Compare log_prob between sampling old_log_prob and training old_log_prob
    # Sampling saves "log_prob" (the old log prob computed during sampling)
    # Training saves "old_log_prob" (the same old log prob passed to training)
    # These MUST be identical (same data, just transported)
    # 注意：sampling 和 training 的 micro-batch 大小通常不同，使用前缀匹配比较
    s_log_prob = sampling_tensors.get("log_prob")
    t_old_log_prob = training_tensors.get("old_log_prob")
    if s_log_prob is not None and t_old_log_prob is not None:
        lp_metrics = compare_tensors("old_log_prob_transport", s_log_prob, t_old_log_prob, allow_prefix_match=True)
        metrics.update(lp_metrics)
        max_diff_key = "old_log_prob_transport_max_abs_diff"
        if max_diff_key in lp_metrics and lp_metrics[max_diff_key] > threshold:
            has_divergence = True
        elif (
            "old_log_prob_transport_shape_mismatch" in lp_metrics
            and "old_log_prob_transport_prefix_compared" not in lp_metrics
        ):
            has_divergence = True

    # Compare sampling log_prob vs training new_log_prob
    # For on-policy step (first step before any gradient update), these should match
    t_new_log_prob = training_tensors.get("new_log_prob")
    if s_log_prob is not None and t_new_log_prob is not None:
        new_lp_metrics = compare_tensors(
            "log_prob_sampling_vs_training_new", s_log_prob, t_new_log_prob, allow_prefix_match=True
        )
        metrics.update(new_lp_metrics)

    # Report ratio stats from training
    t_ratio = training_tensors.get("ratio")
    if t_ratio is not None:
        metrics["ratio_mean"] = t_ratio.mean().item()
        metrics["ratio_std"] = t_ratio.std().item()
        metrics["ratio_max"] = t_ratio.max().item()
        metrics["ratio_min"] = t_ratio.min().item()
        metrics["ratio_max_deviation"] = (t_ratio - 1.0).abs().max().item()
    if "ratio_max_deviation" in metrics and metrics["ratio_max_deviation"] > threshold:
        has_divergence = True

    if verbose:
        print(f"\n  === Step {step_idx} detailed tensor stats ===")
        print(f"  Sampling tensors available: {sorted(sampling_tensors.keys())}")
        print(f"  Training tensors available: {sorted(training_tensors.keys())}")
        for name in sorted(set(list(sampling_tensors.keys()) + list(training_tensors.keys()))):
            if name in sampling_tensors:
                print(f"    [sampling] {name}: {tensor_stats(sampling_tensors[name])}")
            if name in training_tensors:
                print(f"    [training] {name}: {tensor_stats(training_tensors[name])}")

    return metrics, has_divergence


def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO train-inference consistency debug tensors")
    parser.add_argument(
        "debug_dir",
        type=str,
        help="Root debug output directory (containing sampling/ and training/ subdirs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-tensor statistics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-13,
        help="Divergence threshold for flagging inconsistencies (default: 1e-6)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top-K largest per-sample log_prob divergences (default: 5)",
    )
    args = parser.parse_args()

    debug_dir = args.debug_dir
    sampling_dir = os.path.join(debug_dir, "sampling")
    training_dir = os.path.join(debug_dir, "training")

    # Load config if available
    config_path = os.path.join(sampling_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print("=== Sampling Config ===")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print()

    # Discover steps
    sampling_steps = set(discover_steps(sampling_dir))
    training_steps = set(discover_steps(training_dir))
    all_steps = sorted(sampling_steps | training_steps)

    if not all_steps:
        print(f"ERROR: No step directories found in {sampling_dir} or {training_dir}")
        print("  Make sure you ran the debug script first.")
        sys.exit(1)

    print("=== Step Discovery ===")
    print(f"  Sampling steps: {sorted(sampling_steps)}")
    print(f"  Training steps: {sorted(training_steps)}")
    common_steps = sorted(sampling_steps & training_steps)
    print(f"  Common steps:   {common_steps}")
    sampling_only = sorted(sampling_steps - training_steps)
    training_only = sorted(training_steps - sampling_steps)
    if sampling_only:
        print(f"  Sampling-only steps: {sampling_only}")
    if training_only:
        print(f"  Training-only steps: {training_only}")
    print()

    # Analyze each common step
    first_divergence_step = None
    all_step_metrics = {}

    print(f"=== Per-Step Consistency Analysis (threshold={args.threshold:.0e}) ===")
    print(
        f"{'Step':>6} | {'noise_pred':>14} | {'latents_in':>14} | {'latents_out':>14} | {'prev_mean':>14} | {'lp_transport':>14} | {'lp_s_vs_t_new':>14} | {'ratio_dev':>14} | {'status'}"
    )
    print("-" * 140)

    for step_idx in common_steps:
        metrics, has_divergence = analyze_step(
            step_idx, sampling_dir, training_dir, verbose=args.verbose, threshold=args.threshold
        )
        all_step_metrics[step_idx] = metrics

        if has_divergence and first_divergence_step is None:
            first_divergence_step = step_idx

        def _fmt(key, precision=6):
            val = metrics.get(key)
            if val is None:
                return "N/A".rjust(14)
            return f"{val:.{precision}e}".rjust(14)

        status = "DIVERGED" if has_divergence else "OK"
        print(
            f"{step_idx:>6} | "
            f"{_fmt('noise_pred_max_abs_diff')} | "
            f"{_fmt('latents_input_max_abs_diff')} | "
            f"{_fmt('latents_output_max_abs_diff')} | "
            f"{_fmt('prev_sample_mean_max_abs_diff')} | "
            f"{_fmt('old_log_prob_transport_max_abs_diff')} | "
            f"{_fmt('log_prob_sampling_vs_training_new_max_abs_diff')} | "
            f"{_fmt('ratio_max_deviation')} | "
            f"{status}"
        )

    print()

    # ── Training On-Policy Consistency (primary metric) ──
    # This uses only training-side tensors (old_log_prob vs new_log_prob) which
    # are always correctly aligned regardless of sampling sub-batch ordering.
    #
    # Iterate over ``training_steps`` (not ``common_steps``) so that SGLang
    # rollout runs — where ``sampling/`` is always empty because the SGLang
    # scheduler / worker processes don't expose per-step tensors to the
    # training process — still get a meaningful on-policy readout.  For
    # FSDP direct-sampling this is equivalent to iterating common_steps,
    # since training_steps ⊆ sampling_steps in that topology.
    onpolicy_steps = sorted(training_steps)
    print("=== Training On-Policy Consistency (primary metric) ===")
    any_onpolicy_divergence = False
    onpolicy_step_count = 0
    for step_idx in onpolicy_steps:
        training_step_dir = os.path.join(training_dir, f"step_{step_idx:03d}")
        t_tensors = load_step_tensors(training_step_dir)
        t_old = t_tensors.get("old_log_prob")
        t_new = t_tensors.get("new_log_prob")
        t_ratio = t_tensors.get("ratio")
        if t_old is not None and t_new is not None:
            lp_diff = (t_old - t_new).abs()
            max_diff = lp_diff.max().item()
            mean_diff = lp_diff.mean().item()
            ratio_dev = (t_ratio - 1.0).abs().max().item() if t_ratio is not None else float("nan")
            status = "OK" if max_diff <= args.threshold else "DRIFT"
            if max_diff > args.threshold:
                any_onpolicy_divergence = True
            onpolicy_step_count += 1
            print(
                f"  step {step_idx}: |old_lp - new_lp| max={max_diff:.8e}  mean={mean_diff:.8e}  "
                f"ratio_max_dev={ratio_dev:.8e}  [{status}]"
            )
    if onpolicy_step_count == 0:
        print("  (no training-side old_log_prob/new_log_prob tensors found — "
              "did the run reach GRPOAlgorithm.compute_loss with debug_output_dir set?)")
    elif not any_onpolicy_divergence:
        print("  >> All steps PASS: old_log_prob == new_log_prob (ratio == 1.0)")
        print("     Training replay is perfectly consistent with transported sampling log-probs.")
    else:
        print("  >> WARNING: ratio drift detected — old_log_prob != new_log_prob on training side.")
    print()

    # ── Shape alignment check ──
    has_shape_mismatch = False
    for step_idx in common_steps:
        m = all_step_metrics.get(step_idx, {})
        for name in COMPARISON_TENSORS:
            if m.get(f"{name}_shape_mismatch"):
                has_shape_mismatch = True
                break
        if has_shape_mismatch:
            break
    if has_shape_mismatch:
        print("=== WARNING: Sampling / Training Batch Shape Mismatch ===")
        m0 = all_step_metrics.get(common_steps[0], {})
        s_shape = m0.get("latents_input_sampling_shape", "?")
        t_shape = m0.get("latents_input_training_shape", "?")
        print(f"  Sampling debug tensor shape: {s_shape}")
        print(f"  Training debug tensor shape: {t_shape}")
        print("  The sampling and training debug tensors have different batch sizes.")
        print("  Cross-side comparisons (noise_pred, latents_in/out, log_prob transport)")
        print("  use prefix matching but may compare NON-CORRESPONDING samples if the")
        print("  sample ordering differs between sampling sub-batches and training DP shards.")
        print("  Rely on the 'Training On-Policy Consistency' section above as the")
        print("  authoritative consistency metric.")
        print()

    # Summary
    print("=== Cross-Side Summary (sampling vs training debug tensors) ===")
    if has_shape_mismatch:
        print("  NOTE: batch shape mismatch detected; cross-side diffs may reflect")
        print("        non-corresponding samples rather than real divergence.")
    if first_divergence_step is not None:
        print(f"  FIRST DIVERGENCE at step {first_divergence_step}")
        m = all_step_metrics[first_divergence_step]
        print("  Metrics at divergence step:")
        for key in sorted(m.keys()):
            val = m[key]
            if isinstance(val, float):
                print(f"    {key}: {val:.8e}")
            else:
                print(f"    {key}: {val}")
    else:
        if common_steps:
            print("  All common steps are consistent (within threshold).")
        else:
            print("  No common steps to compare!")

    # Top-K log_prob divergences
    if common_steps:
        print(f"\n=== Top-{args.top_k} Steps by Log-Prob Divergence (sampling vs training_new) ===")
        lp_divergences = []
        for step_idx in common_steps:
            m = all_step_metrics.get(step_idx, {})
            lp_diff = m.get("log_prob_sampling_vs_training_new_max_abs_diff")
            if lp_diff is not None:
                lp_divergences.append((step_idx, lp_diff))

        lp_divergences.sort(key=lambda x: x[1], reverse=True)
        for rank, (step_idx, diff) in enumerate(lp_divergences[: args.top_k], 1):
            m = all_step_metrics[step_idx]
            ratio_dev = m.get("ratio_max_deviation", 0.0)
            print(f"  #{rank} step={step_idx}: log_prob_diff={diff:.8e}, ratio_max_deviation={ratio_dev:.8e}")

    # Per-sample analysis for the worst step
    if first_divergence_step is not None:
        print(f"\n=== Per-Sample Analysis at Step {first_divergence_step} ===")
        sampling_step_dir = os.path.join(sampling_dir, f"step_{first_divergence_step:03d}")
        training_step_dir = os.path.join(training_dir, f"step_{first_divergence_step:03d}")

        s_tensors = load_step_tensors(sampling_step_dir)
        t_tensors = load_step_tensors(training_step_dir)

        s_lp = s_tensors.get("log_prob")
        t_new_lp = t_tensors.get("new_log_prob")
        t_old_lp = t_tensors.get("old_log_prob")
        t_ratio = t_tensors.get("ratio")

        if s_lp is not None and t_new_lp is not None:
            # 处理 sampling 和 training 的 batch_size 不同的情况
            min_b = min(s_lp.shape[0], t_new_lp.shape[0])
            if s_lp.shape[0] != t_new_lp.shape[0]:
                print(
                    f"  NOTE: sampling log_prob shape={list(s_lp.shape)} vs training new_log_prob shape={list(t_new_lp.shape)}"
                )
                print(f"        Comparing first {min_b} samples (prefix match)")
            s_lp_cmp = s_lp[:min_b]
            t_new_lp_cmp = t_new_lp[:min_b]
            per_sample_diff = (s_lp_cmp - t_new_lp_cmp).abs()
            print("  Per-sample |sampling_log_prob - training_new_log_prob|:")
            for i in range(min(args.top_k, per_sample_diff.shape[0])):
                worst_idx = per_sample_diff.argmax().item()
                print(
                    f"    sample[{worst_idx}]: diff={per_sample_diff[worst_idx].item():.8e} "
                    f"sampling_lp={s_lp_cmp[worst_idx].item():.6f} "
                    f"training_new_lp={t_new_lp_cmp[worst_idx].item():.6f}"
                )
                per_sample_diff[worst_idx] = 0  # Mask for next iteration

        if t_old_lp is not None and t_new_lp is not None:
            print("\n  Per-sample |old_log_prob - new_log_prob| (should be ~0 for on-policy):")
            on_policy_diff = (t_old_lp - t_new_lp).abs()
            for i in range(min(args.top_k, on_policy_diff.shape[0])):
                worst_idx = on_policy_diff.argmax().item()
                ratio_val = t_ratio[worst_idx].item() if t_ratio is not None else float("nan")
                print(
                    f"    sample[{worst_idx}]: diff={on_policy_diff[worst_idx].item():.8e} "
                    f"old_lp={t_old_lp[worst_idx].item():.6f} "
                    f"new_lp={t_new_lp[worst_idx].item():.6f} "
                    f"ratio={ratio_val:.6f}"
                )
                on_policy_diff[worst_idx] = 0

        # Check noise_pred consistency per-sample
        s_noise = s_tensors.get("noise_pred")
        t_noise = t_tensors.get("noise_pred")
        if s_noise is not None and t_noise is not None:
            # 支持 batch 维度不同时的前缀匹配
            if s_noise.ndim == t_noise.ndim and s_noise.shape[1:] == t_noise.shape[1:]:
                min_b = min(s_noise.shape[0], t_noise.shape[0])
                if s_noise.shape[0] != t_noise.shape[0]:
                    print(
                        f"\n  NOTE: noise_pred sampling shape={list(s_noise.shape)} vs training shape={list(t_noise.shape)}"
                    )
                    print(f"        Comparing first {min_b} samples")
                s_noise_cmp = s_noise[:min_b].float()
                t_noise_cmp = t_noise[:min_b].float()
                per_sample_noise_diff = (s_noise_cmp - t_noise_cmp).abs()
                # Reduce spatial dims to get per-sample max
                spatial_dims = tuple(range(1, per_sample_noise_diff.ndim))
                per_sample_max = per_sample_noise_diff.amax(dim=spatial_dims)
                per_sample_mean = per_sample_noise_diff.mean(dim=spatial_dims)
                print("\n  Per-sample noise_pred max_abs_diff:")
                for i in range(min(args.top_k, per_sample_max.shape[0])):
                    worst_idx = per_sample_max.argmax().item()
                    print(
                        f"    sample[{worst_idx}]: max={per_sample_max[worst_idx].item():.8e} "
                        f"mean={per_sample_mean[worst_idx].item():.8e}"
                    )
                    per_sample_max[worst_idx] = 0
            elif s_noise.shape != t_noise.shape:
                print(
                    f"\n  WARNING: noise_pred shape mismatch: sampling={list(s_noise.shape)} training={list(t_noise.shape)}"
                )
                print("           Cannot compare per-sample noise_pred")

    print("\nDone.")


if __name__ == "__main__":
    main()
