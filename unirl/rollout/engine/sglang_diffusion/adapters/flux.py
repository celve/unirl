"""FLUX-family image adapters: plain FLUX + FLUX.2-Klein.

``FluxAdapter`` is the default image path (5-D passthrough). ``Flux2KleinAdapter``
overrides ``build_segment`` (Klein's transformer is a pure sequence model emitting
packed ``[B, T, H*W, C_packed]`` tokens, so the trajectory is unpacked to image form
before segment assembly) and the schedule policy (Klein needs a model-specific
``compute_mu`` the generic FlowMatch path can't synthesize). The Dance-GRPO SDE label
rides on the base ``resolve_sde_label``.

``Flux2KleinAdapter`` serves BOTH modalities off one checkpoint — text→image and
text+image→image (SGLang's ``ModelTaskType.TI2I``) — branching on whether the request
``Sample`` carries an image input (``Sample.has_image_input``), exactly like the
trainside ``Flux2KleinPipeline``. That is why there is no separate edit adapter key:
upstream's pipeline config is TI2I unconditionally, its ``ImageVAEEncodingStage``
no-ops when no condition image is set, and ``Flux2KleinConditions`` already declares
the image slots optional. See ``build_prompts`` / ``build_condition`` for the two ti2i
deltas — both no-ops when the sample carries no image.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


@register_adapter("flux")
class FluxAdapter(ImageAdapter):
    """FLUX — image-form 5-D trajectory throughout; default path."""

    pass


# FLUX.2 patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
_KLEIN_DOWNSAMPLE = 16


@register_adapter("flux2_klein")
class Flux2KleinAdapter(ImageAdapter):
    """FLUX.2-Klein — packed sequence-style trajectory + model-specific schedule."""

    def validate(self) -> None:
        super().validate()
        require(
            callable(getattr(self.model_config, "build_schedule_policy", None)),
            "flux2_klein adapter requires model_config.build_schedule_policy() "
            "(Klein needs a model-specific compute_mu the generic FlowMatch path "
            "cannot synthesize from scheduler_config.json).",
        )

    def schedule_policy(self):
        return self.model_config.build_schedule_policy()

    # ------------------------------------------------------------------ #
    # Request side — the ti2i delta
    # ------------------------------------------------------------------ #

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """T2I prompt payload, plus the source image when the request carries one.

        Klein is one checkpoint serving both modalities, so the image is
        OPTIONAL: absent (``not sample.has_image_input()``), this is the inherited
        T2I payload verbatim. Present, the PIL rides the ``condition_image``
        sampling kwarg — a SamplingParams field injected by
        :mod:`._patches.patch_sampling_io`, copied onto ``Req.condition_image``,
        and indexed per prompt by that same patch (so a multi-prompt batch does
        not condition every prompt on the last image). Upstream's
        ``InputValidationStage`` then resizes it and ``ImageVAEEncodingStage``
        VAE-encodes it into ``batch.image_latent`` +
        ``batch.condition_image_latent_ids``. The driver never replicates that
        preprocessing.

        Mirrors the Edit-Plus adapter's ingestion, but branch-guarded: Edit-Plus
        is edit-only and fails without an image, whereas Klein degrades cleanly to
        text-to-image. A batch where only SOME prompts carry an image already
        fails earlier, in the data source's homogeneous-image check.
        """
        if not sample.has_image_input():
            return super().build_prompts(sample)

        turns, image_batches = sample.vision_conditioning()
        text_turns = [turn.content for turn in turns if isinstance(turn.content, Texts)]
        if len(text_turns) != 1 or len(image_batches) != 1:
            raise ValueError(
                f"modality={self.model_family!r} ti2i requires exactly one text turn and one "
                f"image turn; got {len(text_turns)} text and {len(image_batches)} image turns."
            )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        prompts = list(text_turns[0].texts)
        unique_prompts, k = self._deexpand_prompts(prompts, gen_part.group_ids)
        pil_images = image_batches[0].to_pils()
        if len(pil_images) != len(prompts):
            raise ValueError(f"build_prompts: image batch {len(pil_images)} != prompt count {len(prompts)}")
        # Collapse the PILs in lockstep with the prompt collapse: one source image
        # per group, in first-seen group order. Indexing ``pil_images[::k]`` would
        # misalign images vs prompts when ``group_ids`` interleave ([A,B,A,B,...]).
        # ``k == 1`` means no collapse happened, so the PILs stay 1:1.
        if k > 1:
            unique_pils = self._first_per_group(pil_images, list(gen_part.group_ids))
            if len(unique_pils) != len(unique_prompts):
                raise ValueError(
                    f"build_prompts: collapsed image count {len(unique_pils)} != unique prompt "
                    f"count {len(unique_prompts)} (group_ids/image misalignment)."
                )
        else:
            unique_pils = pil_images
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": unique_pils if len(unique_pils) > 1 else unique_pils[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Inherited text conditions, plus the ti2i source-image slots.

        ``Flux2KleinConditions.from_dict`` reads ``image_latent`` (the packed
        condition tokens ``[B, N, 128]``) and ``image_latent_ids`` (their 4-axis
        RoPE ids ``[B, N, 4]``) as two separate ``ImageLatentCondition`` entries,
        and ``predict_noise`` requires both — hence the paired emission.

        Unlike Edit-Plus, no unpack is needed: FLUX.2 hands back the packed token
        form the trainside slot already wants. The ids are shipped rather than
        recomputed because ``N`` alone does not determine the ``h_pat × w_pat``
        factorization and FLUX.2 never populates ``vae_image_sizes`` (its pipeline
        config does not override the base ``preprocess_vae_image`` no-op), so the
        grid cannot be recovered driver-side without duplicating upstream's
        condition-image resize math. Both stacks are ``None`` for pure T2I.
        """
        cond_dict = super().build_condition(results)
        tokens = self._stack_condition_field(results, "image_latent")
        if tokens is None:
            return cond_dict  # pure T2I
        ids = self._stack_condition_field(results, "condition_image_latent_ids")
        require(
            ids is not None,
            "build_condition: FLUX.2-Klein ti2i rollout returned image_latent but no "
            "condition_image_latent_ids. Check that patch_conditions captured "
            "batch.condition_image_latent_ids (set by upstream's "
            "prepare_condition_image_latent_ids) — replay's predict_noise requires "
            "both the condition tokens and their RoPE ids.",
        )
        cond_dict["image_latent"] = ImageLatentCondition(latents=tokens)
        cond_dict["image_latent_ids"] = ImageLatentCondition(latents=ids)
        return cond_dict

    def _stack_condition_field(self, results: List[RawResult], name: str) -> Optional[torch.Tensor]:
        """Concatenate a per-result single-tensor conditions field over dim 0.

        ``patch_conditions`` ships each field as a one-element list holding that
        result's own ``[1, N, ...]`` slice, so the dim-0 concat over results
        reconstructs the batch. Returns ``None`` when no result carries the field
        (the T2I path). Accessed via ``getattr`` because these are patch-injected
        fields, not declared on the ``RawResult`` protocol.
        """
        tensors: List[torch.Tensor] = []
        for r in results:
            value = getattr(r, name, None)
            if not value:
                continue
            tensors.append(value[0])
        if not tensors:
            return None
        require(
            len(tensors) == len(results),
            f"build_condition: {name} present on {len(tensors)}/{len(results)} results — "
            f"expected all or none. A partial capture means the merge/slice path dropped "
            f"a sample; fix the source rather than padding.",
        )
        shapes = {tuple(t.shape) for t in tensors}
        require(
            len(shapes) == 1,
            f"build_condition: FLUX.2-Klein {name} tensors have heterogeneous shapes "
            f"{sorted(shapes)} — expected a uniform grid. Upstream sizes each source image "
            f"to ~1024² preserving ITS OWN aspect ratio, so a batch mixing aspect ratios "
            f"produces different token counts. Bucket the dataloader by aspect ratio, or "
            f"normalize the dataset to one.",
        )
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _first_per_group(items: List[Any], group_ids: List[str]) -> List[Any]:
        """First item of each group, in first-seen group order.

        Mirrors how :func:`utils.deexpand_prompts_from_groups` collapses prompts,
        so the source-image collapse aligns with the prompt collapse regardless of
        the sample layout (contiguous or interleaved ``group_ids``).
        """
        seen: set[str] = set()
        out: List[Any] = []
        for item, gid in zip(items, group_ids):
            if gid not in seen:
                seen.add(gid)
                out.append(item)
        return out

    def _deexpand_prompts(self, prompts: List[str], group_ids: List[str]):
        """Collapse K-expanded prompts back to unique + repeat count."""
        return utils.deexpand_prompts_from_groups(prompts, list(group_ids))

    # ------------------------------------------------------------------ #
    # Segment (inherited shape, Klein packed-token unpack)
    # ------------------------------------------------------------------ #

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Collect, unpack Klein's packed ``[B, T, H*W, C]`` to image form, assemble.

        Klein keeps the packed channels at patch resolution (a token→spatial
        reshape). 5-D arrivals (image-form) skip the unpack. The ti2i condition
        tokens never enter the recorded trajectory — SGLang concats them into a
        separate ``latent_model_input`` per step, never back into ``latents`` — so
        this unpack is identical for t2i and ti2i.
        """
        diffusion = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 5:
            B, T, S, C, h_pat, w_pat = utils.validate_packed_trajectory(
                traj, diffusion, family="flux2_klein", downsample=_KLEIN_DOWNSAMPLE, require_divisible=True
            )
            from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

            flat = traj.reshape(B * T, S, C)
            traj = unpack_latents(flat, h_pat, w_pat).reshape(B, T, C, h_pat, w_pat).contiguous()
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=diffusion.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )


__all__ = ["FluxAdapter", "Flux2KleinAdapter"]
