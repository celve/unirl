from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from diffusionrl.types.media import MediaRef
from diffusionrl.types.primitives import Image
from diffusionrl.utils.batched import Batched, concat_field


@dataclass
class Prompts(Batched):
    prompts: List[str] = concat_field()
    prompt_ids: List[str] = concat_field()
    sample_ids: List[str] = concat_field()
    group_ids: List[str] = concat_field()
    noise_group_ids: List[str] = concat_field()
    prompt_metadata: List[Dict[str, Any]] = concat_field()
    # ``default_factory=list`` keeps the field optional for NEW callers
    # that never set per-sample media refs (text-only rollouts on the
    # ``RolloutPipeline.convert_training_data`` /
    # ``RolloutPipelineMixin.resp_to_response_shard`` construct
    # ``Prompts(...)`` with the 6 original fields). Without a default,
    # the merge of PR #100 broke every NEW ``Prompts(...)`` site with
    # ``TypeError: missing 1 required positional argument: 'media_refs'``.
    # OLD callers always go through ``from_unique_prompts(media_refs=...)``
    # which fills it explicitly, so the default-factory only kicks in on
    # the direct-construction path.
    #
    # ``media_refs`` is OLD-only conceptually; NEW image-as-condition
    # goes through ``RolloutReq.primitives["image"]: Images`` + typed
    # ``ImageLatentCondition`` / ``ImageEmbedCondition``. When OLD is
    # deleted, this whole field goes away.
    media_refs: List[List[MediaRef]] = concat_field(default_factory=list)
    # NEW-path image conditioning channel. Per-prompt ``Image`` (None for
    # prompts that carry no condition image). The data layer
    # (:meth:`MultimodalRLDataSource._collate_text`) loads these from
    # ``(image, condition)`` MediaRefs;
    # :meth:`RolloutPipeline.plan_requests` then injects them as
    # ``RolloutReq.primitives["image"]: Images``. Default-factory is
    # ``list`` (same rationale as ``media_refs``).
    images: List[Optional[Image]] = concat_field(default_factory=list)

    @classmethod
    def from_unique_prompts(
        cls,
        prompts: List[str],
        prompt_ids: Optional[List[str]] = None,
        prompt_metadata: Optional[List[Dict[str, Any]]] = None,
        media_refs: Optional[List[List[MediaRef]]] = None,
        images: Optional[List[Optional[Image]]] = None,
    ) -> Prompts:
        """Create a pre-expansion Prompts from a list of unique prompt strings."""
        if prompt_ids is not None and len(prompt_ids) != len(prompts):
            raise ValueError(f"prompt_ids length {len(prompt_ids)} != prompts length {len(prompts)}")
        if prompt_metadata is not None and len(prompt_metadata) != len(prompts):
            raise ValueError(f"prompt_metadata length {len(prompt_metadata)} != prompts length {len(prompts)}")
        if media_refs is not None and len(media_refs) != len(prompts):
            raise ValueError(f"media_refs length {len(media_refs)} != prompts length {len(prompts)}")
        if images is not None and len(images) != len(prompts):
            raise ValueError(f"images length {len(images)} != prompts length {len(prompts)}")
        ids = (
            [str(pid) for pid in prompt_ids] if prompt_ids is not None else [f"prompt:{i}" for i in range(len(prompts))]
        )
        sample_ids = [f"prompt:{pid}:sample:0" for pid in ids]
        metadata = list(prompt_metadata) if prompt_metadata is not None else [{} for _ in prompts]
        media = [list(refs) for refs in media_refs] if media_refs is not None else [[] for _ in prompts]
        imgs = list(images) if images is not None else [None for _ in prompts]
        return cls(
            prompts=list(prompts),
            prompt_ids=ids,
            sample_ids=sample_ids,
            group_ids=list(ids),
            noise_group_ids=list(ids),
            prompt_metadata=metadata,
            media_refs=media,
            images=imgs,
        )

    def expand(
        self,
        samples_per_prompt: int,
        init_same_noise: bool = False,
    ) -> Prompts:
        """Expand each prompt into samples_per_prompt entries."""
        k = int(samples_per_prompt)
        if k < 1:
            raise ValueError(f"samples_per_prompt must be >= 1, got {k}")

        expanded_prompts = [p for p in self.prompts for _ in range(k)]
        expanded_prompt_ids = [pid for pid in self.prompt_ids for _ in range(k)]
        expanded_group_ids = [gid for gid in self.group_ids for _ in range(k)]
        expanded_metadata = [m for m in self.prompt_metadata for _ in range(k)]
        expanded_media_refs = [list(refs) for refs in self.media_refs for _ in range(k)]
        # Per-prompt ``Image`` refs propagate by reference (no tensor
        # copy) — every sample in a group shares the same conditioning
        # image, mirroring the prompt-string duplication above.
        expanded_images = [img for img in self.images for _ in range(k)] if self.images else []

        sample_ids = [f"prompt:{pid}:sample:{j}" for pid in self.prompt_ids for j in range(k)]

        if init_same_noise:
            noise_group_ids = expanded_prompt_ids
        else:
            noise_group_ids = list(sample_ids)

        return Prompts(
            prompts=expanded_prompts,
            prompt_ids=expanded_prompt_ids,
            sample_ids=sample_ids,
            group_ids=expanded_group_ids,
            noise_group_ids=noise_group_ids,
            prompt_metadata=expanded_metadata,
            media_refs=expanded_media_refs,
            images=expanded_images,
        )
