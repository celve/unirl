"""Atomic checkpoint-path weight sync helpers."""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict

import torch


READY_MARKER_SUFFIX = ".ready"


def checkpoint_ready_marker_path(checkpoint_path: str) -> str:
    """Return ready-marker path for a published checkpoint."""
    return f"{checkpoint_path}{READY_MARKER_SUFFIX}"


def publish_checkpoint_atomic(state_dict: Dict[str, torch.Tensor], checkpoint_path: str) -> str:
    """
    Atomically publish a checkpoint and ready marker.

    Writer path:
        tmp checkpoint -> fsync -> rename(final)
        tmp marker -> fsync -> rename(final marker)
    """
    directory = os.path.dirname(checkpoint_path) or "."
    os.makedirs(directory, exist_ok=True)

    pid = os.getpid()
    nonce = int(time.time_ns())
    tmp_checkpoint = f"{checkpoint_path}.tmp.{pid}.{nonce}"
    ready_marker = checkpoint_ready_marker_path(checkpoint_path)
    tmp_marker = f"{ready_marker}.tmp.{pid}.{nonce}"

    # Remove stale ready marker from previous publish attempt.
    try:
        os.remove(ready_marker)
    except OSError:
        pass

    with open(tmp_checkpoint, "wb") as f:
        torch.save(state_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_checkpoint, checkpoint_path)

    with open(tmp_marker, "w", encoding="utf-8") as f:
        f.write(str(nonce))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_marker, ready_marker)

    return checkpoint_path


def publish_sglang_transformer_checkpoint_atomic(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_path: str,
    *,
    module_name: str = "transformer",
    filename: str = "model.safetensors",
) -> str:
    """
    Atomically publish an sglang-compatible module checkpoint directory.

    Layout:
        <checkpoint_path>/<module_name>/<filename>
        <checkpoint_path>.ready
    """
    directory = os.path.dirname(checkpoint_path) or "."
    os.makedirs(directory, exist_ok=True)

    pid = os.getpid()
    nonce = int(time.time_ns())
    tmp_checkpoint_dir = f"{checkpoint_path}.tmp.{pid}.{nonce}"
    ready_marker = checkpoint_ready_marker_path(checkpoint_path)
    tmp_marker = f"{ready_marker}.tmp.{pid}.{nonce}"

    # Best-effort cleanup from a previous failed publish.
    try:
        os.remove(ready_marker)
    except OSError:
        pass
    if os.path.isdir(checkpoint_path):
        shutil.rmtree(checkpoint_path, ignore_errors=True)
    elif os.path.isfile(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    module_dir = os.path.join(tmp_checkpoint_dir, module_name)
    os.makedirs(module_dir, exist_ok=True)

    from safetensors.torch import save_file

    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    target_file = os.path.join(module_dir, filename)
    try:
        save_file(cpu_state, target_file)
        os.replace(tmp_checkpoint_dir, checkpoint_path)
    except Exception:
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
        raise

    with open(tmp_marker, "w", encoding="utf-8") as f:
        f.write(str(nonce))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_marker, ready_marker)

    return checkpoint_path


def wait_for_published_checkpoint(
    checkpoint_path: str,
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.05,
) -> None:
    """Wait until checkpoint and ready marker are visible to reader."""
    ready_marker = checkpoint_ready_marker_path(checkpoint_path)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # New protocol: checkpoint + ready marker.
        if os.path.exists(checkpoint_path) and os.path.exists(ready_marker):
            return
        # Backward compatibility: checkpoint-only publication.
        if os.path.exists(checkpoint_path) and not os.path.exists(ready_marker):
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out waiting for published checkpoint: path={checkpoint_path}, marker={ready_marker}"
    )


def cleanup_published_checkpoint(checkpoint_path: str) -> None:
    """Best-effort cleanup for checkpoint and marker files."""
    marker_path = checkpoint_ready_marker_path(checkpoint_path)

    if os.path.isdir(checkpoint_path):
        shutil.rmtree(checkpoint_path, ignore_errors=True)
    else:
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    try:
        os.remove(marker_path)
    except OSError:
        pass
