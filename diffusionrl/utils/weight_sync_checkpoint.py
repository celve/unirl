"""Atomic checkpoint-path weight sync helpers."""

from __future__ import annotations

import os
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
    paths = (checkpoint_path, checkpoint_ready_marker_path(checkpoint_path))
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
