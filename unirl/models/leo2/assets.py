"""Node-local mirrors of the heavy hymm aux assets; see README.md '## Environment'."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

_ASSET_ENVS = (("TEXT_ENCODER_BASE", "text_encoder_base"), ("VAE_BASE", "vae_base"))


def export_local_asset_bases(
    *, text_encoder_base: Optional[str], vae_base: Optional[str], log: Callable[[str], None] = print
) -> Dict[str, str]:
    """Point hymm at node-local asset mirrors that exist; returns the env vars that were set."""
    wanted = {"text_encoder_base": text_encoder_base, "vae_base": vae_base}
    applied: Dict[str, str] = {}
    for env, field in _ASSET_ENVS:
        path = wanted[field]
        if not path:
            continue
        if env in os.environ:
            applied[env] = os.environ[env]
            continue
        if os.path.isdir(path):
            os.environ[env] = str(path)
            applied[env] = str(path)
        else:
            log(f"[leo2 assets] {field}={path} not on this node; hymm keeps the ceph copy under ASSETS_BASE")
    if applied:
        log(f"[leo2 assets] local mirrors: {applied}")
    return applied


__all__ = ["export_local_asset_bases"]
