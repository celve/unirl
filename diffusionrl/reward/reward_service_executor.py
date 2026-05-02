"""Reward executor that calls a RewardService HTTP API.

Bridges the DiffusionRL reward interface (flat images + prompts) with the
RewardService wire format (history turns + required_rewards).  One executor
instance handles *all* requested reward components in a single HTTP round
trip because the server multiplexes them via ``required_rewards``.

Typical config::

    reward:
      reward_backend: reward_service
      reward_service_url: "http://reward-server:8080"
      reward_components: [hpsv2, clip]
      reward_weights: [0.6, 0.4]
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests as http_requests
import torch
from PIL import Image

from diffusionrl.reward.base import BaseRewardExecutor
from diffusionrl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW or HW torch.Tensor to a PIL RGB image.

    Handles both [0, 1] float and [0, 255] float/byte tensors.
    Always moves to CPU before conversion.
    """
    from torchvision.transforms.functional import to_pil_image

    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        if tensor.max() <= 1.0:
            tensor = tensor.clamp(0.0, 1.0)
        else:
            # Assume 0–255 range; normalize to 0–1 for to_pil_image.
            tensor = (tensor / 255.0).clamp(0.0, 1.0)
    return to_pil_image(tensor)


def _encode_image_b64(
    image: Union[Image.Image, torch.Tensor],
    image_format: str = "JPEG",
    quality: int = 95,
) -> str:
    """Encode an image to a base64 string for the RewardService wire format."""
    if isinstance(image, torch.Tensor):
        image = _pil_from_tensor(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# RewardServiceExecutor
# ---------------------------------------------------------------------------


class RewardServiceExecutor(BaseRewardExecutor):
    """Executor that calls a remote RewardService ``POST /score`` endpoint.

    Converts DiffusionRL's ``RewardRequest`` (flat images + prompts) into
    the RewardService wire format (list of per-sample history-turn requests),
    calls the service, and converts the nested response back into a flat
    ``RewardResponse``.

    One instance handles all ``required_rewards`` in a single HTTP call,
    because the RewardService server multiplexes multiple reward models
    via the ``required_rewards`` field per request.

    Args:
        base_url: Root URL of the RewardService (e.g. ``http://host:8080``).
        required_rewards: Reward model names to request
            (e.g. ``["hpsv2", "clip"]``).
        reward_weights: Per-reward aggregation weights, keyed by reward name.
            Rewards not listed default to weight 1.0.
        sub_metric_reduce: Strategy for collapsing multiple sub-metrics
            within a single reward into one float.  ``"first"`` takes the
            first value, ``"mean"`` averages all, ``"max"`` takes the max.
        image_format: Image encoding for the wire payload
            (``"JPEG"`` or ``"PNG"``).
        image_quality: JPEG quality (1-95); ignored for PNG.
        max_retries: Number of retry attempts on transient HTTP errors.
        retry_delay: Seconds between retries.
        aggregation_method: How to aggregate across rewards for each sample.
            ``"weighted_sum"`` (default), ``"mean"``, ``"min"``, ``"max"``.
    """

    _REDUCE_STRATEGIES = {"first", "mean", "max"}
    _AGGREGATION_METHODS = {"weighted_sum", "mean", "min", "max"}

    def __init__(
        self,
        base_url: str,
        required_rewards: List[str],
        reward_weights: Optional[Dict[str, float]] = None,
        model_name: str = "reward_service",
        weight: float = 1.0,
        batch_size: int = 8,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        sub_metric_reduce: str = "first",
        image_format: str = "JPEG",
        image_quality: int = 95,
        raise_on_failure: bool = True,
        aggregation_method: str = "weighted_sum",
    ) -> None:
        super().__init__(
            model_name=model_name,
            weight=weight,
            batch_size=batch_size,
            timeout=timeout,
        )
        if not required_rewards:
            raise ValueError("required_rewards must be a non-empty list of reward names.")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}.")
        if retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {retry_delay}.")
        if sub_metric_reduce not in self._REDUCE_STRATEGIES:
            raise ValueError(
                f"sub_metric_reduce must be one of {sorted(self._REDUCE_STRATEGIES)}, got {sub_metric_reduce!r}."
            )
        if aggregation_method not in self._AGGREGATION_METHODS:
            raise ValueError(
                f"aggregation_method must be one of {sorted(self._AGGREGATION_METHODS)}, got {aggregation_method!r}."
            )

        self.base_url = base_url.rstrip("/")
        self.required_rewards = list(required_rewards)
        self.reward_weights = dict(reward_weights or {})
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sub_metric_reduce = sub_metric_reduce
        self.image_format = image_format
        self.image_quality = image_quality
        self.raise_on_failure = raise_on_failure
        self.aggregation_method = aggregation_method

        self._remote_rewards_validated = False

        # Disable proxy env vars — reward services are typically on an internal
        # network where corporate HTTP proxies (squid etc.) would return 503.
        self._session = http_requests.Session()
        self._session.trust_env = False

    # ------------------------------------------------------------------
    # Public interface (BaseRewardExecutor)
    # ------------------------------------------------------------------

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Convert a DiffusionRL request, call the remote service, and
        convert the response back.

        Video requests are rejected explicitly: this executor only knows the
        image-history wire format, and silently falling through would yield
        all-zero rewards that corrupt downstream advantage computation.

        By default (``raise_on_failure=True``) HTTP errors and malformed
        responses are propagated as exceptions so that corrupted zero-rewards
        never silently enter the training loop.  Set ``raise_on_failure=False``
        for a degraded mode that returns zeroed rewards on failure.

        Returns:
            A ``RewardResponse`` with ``rewards`` (one float per sample —
            the weighted aggregation of all required_rewards), and
            ``component_rewards`` keyed by reward name.
        """
        if request.is_video:
            n = len(request.videos or [])
            raise NotImplementedError(
                f"RewardServiceExecutor does not support video requests "
                f"(received {n} video(s)). Use a local video reward scorer instead."
            )
        start = time.time()
        bs = request.batch_size
        try:
            payload = self._build_score_payload(request)
            raw = self._post_score(payload)
            return self._parse_score_response(raw, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("RewardServiceExecutor.compute_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=["RewardServiceExecutor failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def is_available(self) -> bool:
        """Ping ``/health``; ``True`` iff the server is reachable.

        On the first successful ping also runs roster validation
        (see ``_validate_required_rewards_once``); transport errors and
        non-200 still return ``False``.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
        except http_requests.exceptions.RequestException:
            return False
        if resp.status_code != 200:
            return False
        self._validate_required_rewards_once(resp)
        return True

    def _validate_required_rewards_once(self, health_resp: http_requests.Response) -> None:
        """One-shot: ``raise ValueError`` if any required reward is not in the
        ``/health`` roster (catches component-name typos at startup, not at
        first ``/score`` call); log the full roster at INFO on success.

        Expected ``/health`` body shape:
        ``{"status": "ok", "rewards": {<name>: [<readiness>, ...], ...}}``.
        """
        if self._remote_rewards_validated:
            return

        try:
            body = health_resp.json()
        except ValueError as e:
            raise ValueError(f"RewardServiceExecutor: /health at {self.base_url} returned non-JSON body.") from e

        if not isinstance(body, dict) or not isinstance(body.get("rewards"), dict):
            raise ValueError(
                f"RewardServiceExecutor: /health at {self.base_url} returned unexpected shape: "
                f"{body!r}. Expected {{'status': 'ok', 'rewards': {{<name>: [...]}}}}."
            )

        available = sorted(body["rewards"].keys())
        available_set = set(available)
        missing = [name for name in self.required_rewards if name not in available_set]
        if missing:
            raise ValueError(
                f"RewardServiceExecutor: required_rewards={missing} not served by "
                f"{self.base_url}; server reports available={available}. "
                f"Check REWARD_COMPONENTS for typos "
                f"(e.g. 'unifiedreward' vs 'unified_reward')."
            )

        logger.info(
            "RewardServiceExecutor: %s serves rewards=%s",
            self.base_url,
            available,
        )
        self._remote_rewards_validated = True

    def dispose(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Request conversion: DiffusionRL → RewardService wire format
    # ------------------------------------------------------------------

    def _build_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a DiffusionRL ``RewardRequest`` into the RewardService
        ``ScoreRequest`` JSON payload.

        Each sample ``(images[i], prompts[i])`` becomes one entry in the
        wire-format ``requests`` list, with
        ``history = [{"text": prompt, "image_b64": ...}]`` and
        ``required_rewards`` set to ``self.required_rewards``.

        Per-sample metadata from ``request.metadata`` is forwarded when present.
        """
        images = request.images or []
        prompts = request.prompts
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        for idx in range(len(images)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            image_b64 = _encode_image_b64(
                images[idx],
                image_format=self.image_format,
                quality=self.image_quality,
            )
            sample_metadata = None
            if metadata_list is not None and idx < len(metadata_list):
                sample_metadata = metadata_list[idx]
            wire_requests.append(
                {
                    "history": [{"text": prompt, "image_b64": image_b64}],
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                }
            )

        return {"requests": wire_requests}

    # ------------------------------------------------------------------
    # HTTP call with retries
    # ------------------------------------------------------------------

    def _post_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to ``/score`` with retry logic.

        Raises ``RuntimeError`` chained from the last underlying exception
        if all retries are exhausted.
        """
        url = f"{self.base_url}/score"
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except http_requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "RewardServiceExecutor: request timed out (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )
            except http_requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning(
                    "RewardServiceExecutor: %s (attempt %d/%d)",
                    e,
                    attempt + 1,
                    self.max_retries,
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"RewardServiceExecutor: failed after {self.max_retries} retries calling {url}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Response conversion: RewardService wire format → DiffusionRL
    # ------------------------------------------------------------------

    def _parse_score_response(
        self,
        raw: Dict[str, Any],
        batch_size: int,
        compute_time: float,
    ) -> RewardResponse:
        """Convert the RewardService ``ScoreResponse`` JSON into a DiffusionRL
        ``RewardResponse``.

        For each sample *i*:

        1. Extract ``results[i][reward_name] → {sub_metric: float}``.
        2. Reduce sub-metrics to one float per reward via
           ``_reduce_sub_metrics``.
        3. Store per-reward scores in ``component_rewards``.
        4. Aggregate across rewards → ``rewards[i]`` using
           ``self.aggregation_method``.
        """
        results: List[Dict[str, Dict[str, float]]] = raw.get("results", [])
        errors_list: List[Dict[str, str]] = raw.get("errors", [])

        # Pad if server returned fewer entries than expected.
        while len(results) < batch_size:
            results.append({})
        while len(errors_list) < batch_size:
            errors_list.append({})

        component_rewards: Dict[str, List[float]] = {name: [] for name in self.required_rewards}
        aggregated_rewards: List[float] = []
        successes: List[bool] = []
        sample_errors: List[Optional[str]] = []

        for i in range(batch_size):
            sample_result = results[i]
            sample_errors_dict = errors_list[i]

            scores: List[float] = []
            weights: List[float] = []
            error_parts: List[str] = []

            # Validation contract: every reward this sample asked for must come
            # back. Today every sample asks for the same self.required_rewards
            # (set at executor construction). When per-sample required_rewards
            # arrives for multi-turn rollouts (the wire format already supports
            # it via wire_requests[i]["required_rewards"]), replace the loop
            # source with `request.required_rewards[i]` — the failure semantics
            # ("asked-for not returned") stay identical.
            for reward_name in self.required_rewards:
                if reward_name in sample_result:
                    sub_metrics = sample_result[reward_name]
                    score = self._reduce_sub_metrics(sub_metrics)
                    component_rewards[reward_name].append(score)
                    scores.append(score)
                    weights.append(self.reward_weights.get(reward_name, 1.0))
                else:
                    component_rewards[reward_name].append(0.0)
                    if reward_name in sample_errors_dict:
                        error_parts.append(f"{reward_name}: {sample_errors_dict[reward_name]}")
                    else:
                        # Asked-for reward absent without explanation: server bug, not legitimate omission.
                        error_parts.append(f"{reward_name}: missing from server response without error")

            if scores:
                aggregated_rewards.append(self._aggregate_scores(scores, weights))
                successes.append(len(error_parts) == 0)
            else:
                aggregated_rewards.append(0.0)
                successes.append(False)

            sample_errors.append("; ".join(error_parts) if error_parts else None)

        return RewardResponse(
            rewards=aggregated_rewards,
            component_rewards=component_rewards,
            successes=successes,
            errors=sample_errors,
            compute_time=compute_time,
        )

    def _aggregate_scores(self, scores: List[float], weights: List[float]) -> float:
        """Aggregate per-reward scores for one sample.

        Strategies:
            ``"weighted_sum"``: ``Σ(score_k * w_k) / Σ(w_k)``.
            ``"mean"``: arithmetic mean (ignores weights).
            ``"min"``: minimum score across rewards.
            ``"max"``: maximum score across rewards.
        """
        if not scores:
            return 0.0
        if self.aggregation_method == "weighted_sum":
            total_w = sum(weights)
            return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
        if self.aggregation_method == "mean":
            return sum(scores) / len(scores)
        if self.aggregation_method == "min":
            return min(scores)
        # "max"
        return max(scores)

    def _reduce_sub_metrics(self, sub_metrics: Dict[str, float]) -> float:
        """Collapse a reward's sub-metric dict into a single float.

        Strategies:
            ``"first"``: value of the first sub-metric
                (stable iteration order in Python 3.7+).
            ``"mean"``: arithmetic mean of all sub-metric values.
            ``"max"``: maximum sub-metric value.
        """
        if not sub_metrics:
            return 0.0
        values = list(sub_metrics.values())
        if self.sub_metric_reduce == "first":
            return float(values[0])
        if self.sub_metric_reduce == "mean":
            return float(sum(values) / len(values))
        # "max"
        return float(max(values))


__all__ = [
    "RewardServiceExecutor",
]
