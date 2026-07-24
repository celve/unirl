"""Per-prompt ``initial_noise`` / ``denoise_seeds`` slice for the de-expanded
rollout path (LIN-513).

With de-expand removed (``ImageAdapter.build_prompts`` now emits all B prompts and
never ``num_outputs_per_prompt``), SGLang fans the request into B single-output
``Req``s via ``expand_request_outputs`` and runs each as its own ``B=1``
``pipeline.forward`` -- no grouped forward is ever issued
(``GPUWorker.execute_forward`` takes the ``len(batch) == 1`` branch). But the
driver still ships ``initial_noise`` as one ``[B, ...]`` tensor and
``denoise_seeds`` as one length-B list (``adapters/image.py:build_inputs``), and
``patch_sampling_io``'s ``prepare_request`` copies the FULL batch onto every
per-prompt ``Req`` (``dataclasses.replace`` in ``diffusion_generator.generate``
overrides only the prompt/file/image fields). So the i-th prompt's ``Req`` would
carry the whole ``[B, ...]`` noise at ``batch_size=1`` and crash in
``LatentPreparationStage`` (``patch_latent_prep._expand_initial_noise``: ``n=B``
matches neither ``batch_size=1`` nor ``num_prompts=1`` nor ``1`` with ``nopp=1``
-> ``ValueError``).

This relocates the per-output slice the (now-deleted) grouped worker slice
``_patch_grouped_initial_noise_slice`` used to do -- but at the per-prompt request
seam instead of inside the grouped forward. ``expand_request_outputs`` already
receives the GLOBAL ``prompt_index`` and already slices the sibling scalar
``seed`` field there (``normalize_output_seeds``), so we mirror that for
``latents`` and ``denoise_seeds``. Keying on the global index makes it correct for
multi-group (G*K) batches, which the old within-group worker slice was not.

Idempotent; AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations

_SENTINEL = "_unirl_req_noise_slice"


def _slice_request_drivers(latents, denoise_seeds, *, prompt_index, num_prompts):
    """Pure: this prompt's row of a batched driver noise/seed, else pass-through.

    Slice ONLY when a full ``[num_prompts, ...]`` noise tensor / length-``num_prompts``
    seed list is present (the de-expanded path's batched drivers); a tensor/list
    that is already this prompt's own row, ``None``, or any other shape passes
    through unchanged. Returns ``(latents, denoise_seeds)``.

    Duck-typed on ``.shape`` (mirrors the deleted grouped worker slice) so it
    carries no torch import and is unit-testable without torch or a live server.
    """
    if num_prompts <= 1:
        return latents, denoise_seeds

    new_latents = latents
    shape = getattr(latents, "shape", None)
    if shape is not None and len(shape) >= 1 and int(shape[0]) == num_prompts:
        new_latents = latents[prompt_index : prompt_index + 1]

    new_seeds = denoise_seeds
    if isinstance(denoise_seeds, (list, tuple)) and len(denoise_seeds) == num_prompts:
        new_seeds = [denoise_seeds[prompt_index]]

    return new_latents, new_seeds


def patch_request_noise_slice() -> None:
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod

    orig = utils_mod.expand_request_outputs
    if getattr(orig, _SENTINEL, False):
        return

    def expand_request_outputs(req, *, num_prompts=1, prompt_index=0):
        out = orig(req, num_prompts=num_prompts, prompt_index=prompt_index)
        # Only the de-expand-removed path: many prompts, one output each, with a
        # batched driver noise/seed that must be split to this prompt's row. A
        # genuine multi-output request has len(out) > 1 and is left alone (its
        # per-output split is the worker's job); a true single-sample request has
        # num_prompts == 1 and needs no slice; absent/None drivers pass through.
        if num_prompts > 1 and len(out) == 1:
            r = out[0]
            r.latents, r.denoise_seeds = _slice_request_drivers(
                getattr(r, "latents", None),
                getattr(r, "denoise_seeds", None),
                prompt_index=prompt_index,
                num_prompts=num_prompts,
            )
        return out

    setattr(expand_request_outputs, _SENTINEL, True)
    utils_mod.expand_request_outputs = expand_request_outputs

    # CRITICAL: ``from ...entrypoints.utils import expand_request_outputs`` binds
    # the name BY VALUE at import time -- ``diffusion_generator.generate`` calls
    # its own module-level ``expand_request_outputs``. Patching ``utils_mod``
    # alone never reaches that caller, so the Req would ship the full ``[B, ...]``
    # noise and crash. Rebind every already-imported module that holds the
    # original -- the exact sweep ``patch_sampling_io`` uses for ``prepare_request``.
    import sys

    for _mod in list(sys.modules.values()):
        try:
            if getattr(_mod, "expand_request_outputs", None) is orig:
                _mod.expand_request_outputs = expand_request_outputs
        except Exception:  # pragma: no cover - defensive
            pass


__all__ = ["patch_request_noise_slice"]
