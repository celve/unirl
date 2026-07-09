"""Shared forward-numerics kernels.

One implementation of the numerics-critical ops (attention, qk-norm), imported
by BOTH the FSDP trainer and the rollout-engine worker so their forwards are
bitwise-identical on the same GPU. See ``unirl/kernels/sd3.py``.
"""
