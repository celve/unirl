"""SGLang Inference Engine (external-service placeholder)."""

import logging
from typing import Any, Dict, List, Optional, Set
import torch

from ..engine import BaseInferenceEngine, EngineConfig, EngineCapabilities, register_engine
from diffusionrl.types import SamplerOutput
from .client_adapter import SGLangClientAdapter

logger = logging.getLogger(__name__)


@register_engine("sglang")
class SGLangInferenceEngine(BaseInferenceEngine):
    """
    SGLang Inference Engine (Placeholder).

    This engine is designed for distributed inference using SGLang.
    Currently not implemented - serves as a placeholder for future development.

    Planned Features:
    - Connect to SGLang server for distributed inference
    - Support for tensor and pipeline parallelism
    - Dynamic batching via SGLang runtime

    Ray Scheduling:
    - Acts as a client to SGLang service
    - Minimal GPU usage per actor (client mode)
    - External service manages actual GPU resources
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize SGLang engine.

        Args:
            config: Engine configuration
        """
        super().__init__(config)
        self.client = None
        self._server_url = config.engine_kwargs.get("server_url")
        self._device = None
        self._server_capabilities: Dict[str, Any] = {}

    def initialize(self, device: torch.device) -> None:
        """
        Initialize connection to SGLang server and negotiate capabilities.

        Args:
            device: Target device (used for local operations)
        """
        self._device = device
        logger.info("Initializing SGLang engine")

        if self._server_url is None:
            raise ValueError(
                "SGLang engine requires 'server_url' in engine_kwargs. "
                "Example: engine_kwargs={'server_url': 'http://localhost:30000'}"
            )

        engine_kwargs = dict(self.config.engine_kwargs or {})
        self.client = SGLangClientAdapter(
            server_url=self._server_url,
            handshake_timeout_s=float(engine_kwargs.get("handshake_timeout_s", 5.0)),
            request_timeout_s=float(engine_kwargs.get("request_timeout_s", 60.0)),
            max_retries=int(engine_kwargs.get("max_retries", 1)),
            retry_backoff_s=float(engine_kwargs.get("retry_backoff_s", 0.5)),
            max_outstanding_requests=int(engine_kwargs.get("max_outstanding_requests", 1)),
            queue_timeout_s=engine_kwargs.get("queue_timeout_s"),
            handshake_paths=engine_kwargs.get("handshake_paths"),
        )
        self._server_capabilities = self.client.handshake()

        self._is_initialized = True
        logger.info(
            "SGLang engine initialized (external placeholder), capabilities=%s",
            self._server_capabilities,
        )

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> SamplerOutput:
        """
        Generate samples using SGLang server.

        NOT YET IMPLEMENTED - Placeholder for future development.
        """
        raise NotImplementedError(
            "SGLang inference request routing is not implemented yet in diffusionrl. "
            "Capability handshake is active, but sample response-to-SamplerOutput "
            "conversion is pending. Use FSDP or FastVideo engine for training."
        )

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode prompts using SGLang server.

        NOT YET IMPLEMENTED - Placeholder for future development.
        """
        raise NotImplementedError("SGLang prompt encoding RPC is not yet implemented.")

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights on SGLang server.

        For SGLang, this would trigger a checkpoint reload on the server.

        NOT YET IMPLEMENTED - Placeholder for future development.
        """
        raise NotImplementedError(
            "SGLang weight update RPC is not yet implemented. "
            "This requires server-side hot reload support."
        )

    def offload(self) -> None:
        """
        Offload is not applicable for SGLang client.

        SGLang engine acts as a client - GPU resources are managed by the server.
        """
        self._is_offloaded = True
        logger.info("SGLang engine offload (no-op for client mode)")

    def onload(self) -> None:
        """
        Onload is not applicable for SGLang client.

        SGLang engine acts as a client - GPU resources are managed by the server.
        """
        self._is_offloaded = False
        logger.info("SGLang engine onload (no-op for client mode)")

    def onload_weights(self) -> None:
        self.onload()

    @property
    def supports_distributed(self) -> bool:
        """SGLang supports distributed inference via server."""
        return True

    @property
    def requires_external_service(self) -> bool:
        """SGLang requires external server."""
        return True

    def get_capabilities(self) -> EngineCapabilities:
        caps = dict(self._server_capabilities or {})
        return EngineCapabilities(
            supports_logprob=bool(caps.get("supports_logprob", False)),
            supports_trajectory=bool(caps.get("supports_trajectory", False)),
            supports_prompt_embeddings=bool(
                caps.get(
                    "supports_prompt_embeddings",
                    caps.get("supports_encode_prompt", False),
                )
            ),
            supports_guidance_scale=bool(caps.get("supports_guidance_scale", False)),
            supports_staged_onload=False,
            weight_sync_mode="external",
        )
