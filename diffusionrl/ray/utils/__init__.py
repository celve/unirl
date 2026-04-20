"""Stateless utility helpers used by diffusionrl Ray actors.

The public boundary for these helpers is ``diffusionrl.ray``'s lazy facade
(``diffusionrl/ray/__init__.py``). This ``__init__`` is intentionally empty;
import from the specific submodule (``net``, ``gpu``, ``node``) at call sites.
"""
