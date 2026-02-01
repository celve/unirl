"""
HTTP reward worker for GRPO training.

Connects to remote reward services via HTTP API.
"""

import base64
import io
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import torch
from PIL import Image

from .base import BaseRewardWorker, RewardRequest, RewardResponse, RewardType


@dataclass
class HTTPRewardConfig:
    """Configuration for HTTP reward service."""
    base_url: str
    endpoint: str = "/compute_reward"
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 8


class HTTPRewardWorker(BaseRewardWorker):
    """
    HTTP-based reward worker that connects to remote reward services.

    This worker sends images/videos to a remote server and receives
    reward scores. Useful for:
    - Scaling reward computation across multiple GPUs/machines
    - Using proprietary reward models hosted as services
    - Integrating with existing reward APIs

    Example usage:
        worker = HTTPRewardWorker(
            base_url="http://reward-server:8000",
            endpoint="/compute_reward",
            api_key="your-api-key",
        )

        response = worker.compute_rewards(
            RewardRequest(
                images=[img1, img2],
                prompts=["a cat", "a dog"],
            )
        )

    Expected API Format:
        POST /compute_reward
        Content-Type: application/json

        Request:
        {
            "images": ["base64_encoded_image", ...],  # or "videos"
            "prompts": ["prompt1", "prompt2", ...],
            "reward_types": ["image_text_alignment"],
            "return_components": false
        }

        Response:
        {
            "rewards": [0.8, 0.7, ...],
            "reward_components": {...},  # optional
            "successes": [true, true, ...],
            "errors": [null, null, ...]
        }
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/compute_reward",
        model_name: str = "http",
        weight: float = 1.0,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 8,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP reward worker.

        Args:
            base_url: Base URL of the reward server
            endpoint: API endpoint for reward computation
            model_name: Name identifier for this worker
            weight: Weight for this worker in multi-reward aggregation
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
            batch_size: Maximum batch size per request
            verify_ssl: Whether to verify SSL certificates
            headers: Additional HTTP headers
        """
        super().__init__(
            model_name=model_name,
            weight=weight,
            batch_size=batch_size,
            timeout=timeout,
        )

        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl

        # Build headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.headers.update(headers)

        # Check for requests library
        try:
            import requests
            self._requests = requests
        except ImportError:
            self._requests = None

        # Check for aiohttp for async support
        try:
            import aiohttp
            self._aiohttp = aiohttp
        except ImportError:
            self._aiohttp = None

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards by sending request to remote server.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        if self._requests is None:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["requests library not installed"] * request.batch_size,
                compute_time=0.0,
            )

        start = time.time()

        try:
            # Prepare request payload
            payload = self._prepare_payload(request)

            # Send request with retries
            response = self._send_request(payload)

            # Parse response
            return self._parse_response(response, time.time() - start)

        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(e)] * request.batch_size,
                compute_time=time.time() - start,
            )

    def _prepare_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Prepare JSON payload for the API request."""
        payload = {
            "prompts": request.prompts,
            "reward_types": [rt.value for rt in request.reward_types],
            "return_components": request.return_components,
        }

        if request.is_video:
            payload["videos"] = [
                self._encode_video(v) for v in request.videos
            ]
        else:
            payload["images"] = [
                self._encode_image(img) for img in request.images
            ]

        if request.metadata:
            payload["metadata"] = request.metadata

        return payload

    def _encode_image(self, image: Union[Image.Image, torch.Tensor]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            from torchvision.transforms.functional import to_pil_image
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = to_pil_image(image)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _encode_video(self, video: torch.Tensor) -> str:
        """
        Encode video to base64 string.

        For simplicity, we encode as a sequence of frames.
        A more efficient implementation could use video codecs.
        """
        # Handle different video formats
        if video.dim() == 4:  # [C, T, H, W]
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        elif video.dim() == 5:  # [1, C, T, H, W]
            video = video.squeeze(0).permute(1, 0, 2, 3)

        # Encode as sequence of frame base64 strings
        frames = []
        for frame in video:
            frames.append(self._encode_image(frame))

        # Return as JSON-encoded list
        return json.dumps(frames)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request with retries."""
        url = f"{self.base_url}{self.endpoint}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                return response.json()

            except self._requests.exceptions.Timeout:
                last_error = "Request timed out"
            except self._requests.exceptions.RequestException as e:
                last_error = str(e)

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_error}")

    def _parse_response(
        self,
        response: Dict[str, Any],
        compute_time: float,
    ) -> RewardResponse:
        """Parse API response into RewardResponse."""
        return RewardResponse(
            rewards=response.get("rewards", []),
            reward_components=response.get("reward_components", {}),
            successes=response.get("successes", [True] * len(response.get("rewards", []))),
            errors=response.get("errors", [None] * len(response.get("rewards", []))),
            compute_time=compute_time,
        )

    def is_available(self) -> bool:
        """Check if the reward server is reachable."""
        if self._requests is None:
            return False

        try:
            # Try to reach health endpoint
            health_url = f"{self.base_url}/health"
            response = self._requests.get(
                health_url,
                headers=self.headers,
                timeout=5.0,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except Exception:
            return False


class AsyncHTTPRewardWorker(HTTPRewardWorker):
    """
    Async version of HTTPRewardWorker using aiohttp.

    Useful for high-throughput scenarios where multiple
    reward requests can be made concurrently.
    """

    async def compute_rewards_async(
        self,
        request: RewardRequest,
    ) -> RewardResponse:
        """
        Async version of compute_rewards.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        if self._aiohttp is None:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["aiohttp library not installed"] * request.batch_size,
                compute_time=0.0,
            )

        start = time.time()

        try:
            payload = self._prepare_payload(request)
            response = await self._send_request_async(payload)
            return self._parse_response(response, time.time() - start)

        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(e)] * request.batch_size,
                compute_time=time.time() - start,
            )

    async def _send_request_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send async HTTP request with retries."""
        import asyncio

        url = f"{self.base_url}{self.endpoint}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with self._aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=self.headers,
                        timeout=self._aiohttp.ClientTimeout(total=self.timeout),
                        ssl=self.verify_ssl,
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

            except asyncio.TimeoutError:
                last_error = "Request timed out"
            except self._aiohttp.ClientError as e:
                last_error = str(e)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_error}")

    async def compute_rewards_batch_async(
        self,
        requests: List[RewardRequest],
    ) -> List[RewardResponse]:
        """
        Compute rewards for multiple requests concurrently.

        Args:
            requests: List of reward requests

        Returns:
            List of reward responses
        """
        import asyncio

        tasks = [self.compute_rewards_async(req) for req in requests]
        return await asyncio.gather(*tasks)
