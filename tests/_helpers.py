"""Shared test helpers for cfg fixture setup.

Tests that compose a fresh cfg via Hydra and need to mutate it for fixture
setup (injecting fake sync sections, scrubbing fields to test validators)
hit the post-compose readonly/struct seal applied by ``freeze``. This
module centralizes the unseal logic so tests don't scatter raw OmegaConf
flag toggles.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def unseal_for_testing(node) -> None:
    """Recursively clear ``readonly`` and ``struct`` on every nested container.

    Walks via ``_get_node`` so MISSING values aren't resolved during the
    traversal. Use this in test setup when fixture code needs to mutate a
    composed cfg — e.g. assign new fields, inject schema sections that are
    absent from the default composition, or scrub a value to provoke a
    validator failure. Production code never calls this; the entry point
    runs ``freeze`` instead.
    """
    if OmegaConf.is_dict(node):
        OmegaConf.set_readonly(node, False)
        OmegaConf.set_struct(node, False)
        for key in list(node.keys()):
            child = node._get_node(key)
            if child is not None:
                unseal_for_testing(child)
    elif OmegaConf.is_list(node):
        OmegaConf.set_readonly(node, False)
        OmegaConf.set_struct(node, False)
        for i in range(len(node)):
            unseal_for_testing(node._get_node(i))


__all__ = ["unseal_for_testing"]
