"""
Local reward worker for GRPO training.

Computes rewards using locally loaded models (PickScore, CLIP, etc.)
"""

import time
from typing import Dict, List, Optional, Any, Union, Callable
import torch
from PIL import Image

from .base import BaseRewardWorker, RewardRequest, RewardResponse, RewardType


class LocalRewardWorker(BaseRewardWorker):
    """
    Local reward computation using pre-loaded models.

    Supports various reward models:
    - PickScore: Image-text alignment
    - CLIP: Image-text similarity
    - Aesthetic: Aesthetic quality scoring
    - Custom: User-defined reward functions

    Example usage:
        # Using PickScore
        worker = LocalRewardWorker(
            model_name="pickscore",
            device="cuda",
        )

        # Using custom function
        def my_reward(images, prompts):
            return [1.0] * len(images)

        worker = LocalRewardWorker(
            reward_fn=my_reward,
        )
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weight: float = 1.0,
        reward_fn: Optional[Callable] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        timeout: float = 60.0,
        **model_kwargs,
    ):
        """
        Initialize local reward worker.

        Args:
            model_name: Name of the reward model to load
                       ("pickscore", "clip", "aesthetic", etc.)
            weight: Weight for this worker in multi-reward aggregation
            reward_fn: Custom reward function (overrides model_name)
            device: Device to run on
            dtype: Model dtype
            batch_size: Maximum batch size
            timeout: Timeout for reward computation
            **model_kwargs: Additional arguments for model loading
        """
        super().__init__(
            model_name=model_name or "",
            weight=weight,
            batch_size=batch_size,
            timeout=timeout,
        )

        self.reward_fn = reward_fn
        self.device = device
        self.dtype = dtype
        self.model_kwargs = model_kwargs

        self.model = None
        self.processor = None
        self._is_loaded = False

        if reward_fn is not None:
            self._is_loaded = True
            self.reward_types = [RewardType.CUSTOM]
        elif model_name is not None:
            self._load_model()

    def _load_model(self):
        """Load the reward model based on model_name."""
        if self.model_name == "pickscore":
            self._load_pickscore()
        elif self.model_name == "clip":
            self._load_clip()
        elif self.model_name == "aesthetic":
            self._load_aesthetic()
        elif self.model_name == "hpsv2":
            self._load_hpsv2()
        elif self.model_name == "ocr":
            self._load_ocr()
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        self._is_loaded = True

    def _load_pickscore(self):
        """Load PickScore model."""
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise ImportError("transformers is required for PickScore")

        model_id = self.model_kwargs.get(
            "model_id", "yuvalkirstain/PickScore_v1"
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()

        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _load_clip(self):
        """Load CLIP model."""
        try:
            import clip
        except ImportError:
            raise ImportError("openai-clip is required for CLIP")

        model_name = self.model_kwargs.get("model_name", "ViT-B/32")
        self.model, self.processor = clip.load(model_name, device=self.device)
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _load_aesthetic(self):
        """Load aesthetic predictor model."""
        # Implementation depends on specific aesthetic model
        # This is a placeholder for models like LAION aesthetic predictor
        raise NotImplementedError("Aesthetic model loading not yet implemented")

    def _load_hpsv2(self):
        """Load HPSv2 model."""
        try:
            import hpsv2
            self._hpsv2_module = hpsv2
        except ImportError:
            raise ImportError("hpsv2 is required for HPSv2 reward")

        # HPSv2 uses module-level score function, not a class
        self.model = "hpsv2"  # Placeholder to indicate model is loaded
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards for the given request.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        start = time.time()

        if not self._is_loaded:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["Model not loaded"] * request.batch_size,
                compute_time=0.0,
            )

        try:
            if self.reward_fn is not None:
                rewards = self._compute_custom(request)
            elif self.model_name == "pickscore":
                rewards = self._compute_pickscore(request)
            elif self.model_name == "clip":
                rewards = self._compute_clip(request)
            elif self.model_name == "hpsv2":
                rewards = self._compute_hpsv2(request)
            elif self.model_name == "ocr":
                rewards = self._compute_ocr(request)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            return RewardResponse(
                rewards=rewards,
                successes=[True] * len(rewards),
                errors=[None] * len(rewards),
                compute_time=time.time() - start,
            )

        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(e)] * request.batch_size,
                compute_time=time.time() - start,
            )

    def _compute_custom(self, request: RewardRequest) -> List[float]:
        """Compute rewards using custom function."""
        if request.is_video:
            return self.reward_fn(request.videos, request.prompts)
        else:
            return self.reward_fn(request.images, request.prompts)

    def _compute_pickscore(self, request: RewardRequest) -> List[float]:
        """Compute PickScore rewards."""
        images = request.images
        prompts = request.prompts

        # Process in batches
        all_rewards = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            # Process images
            image_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            # Process text
            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # Get embeddings
                image_embeds = self.model.get_image_features(**image_inputs)
                text_embeds = self.model.get_text_features(**text_inputs)

                # Normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                # Compute similarity
                scores = (image_embeds * text_embeds).sum(dim=-1)

                all_rewards.extend(scores.cpu().tolist())

        return all_rewards

    def _compute_clip(self, request: RewardRequest) -> List[float]:
        """Compute CLIP similarity rewards."""
        import clip

        images = request.images
        prompts = request.prompts

        all_rewards = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            # Preprocess images
            image_inputs = torch.stack([
                self.processor(img) for img in batch_images
            ]).to(self.device)

            # Tokenize text
            text_inputs = clip.tokenize(batch_prompts).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_inputs)
                text_features = self.model.encode_text(text_inputs)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (image_features * text_features).sum(dim=-1)

                all_rewards.extend(similarity.cpu().tolist())

        return all_rewards

    def _compute_hpsv2(self, request: RewardRequest) -> List[float]:
        """Compute HPSv2 rewards."""
        images = request.images
        prompts = request.prompts

        all_rewards = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            # HPSv2 uses module-level score function
            # It accepts (images, prompts) and returns list of scores
            for img, prompt in zip(batch_images, batch_prompts):
                try:
                    # hpsv2.score expects single image and prompt
                    score = self._hpsv2_module.score(img, prompt)
                    if isinstance(score, (list, tuple)):
                        score = score[0]
                    all_rewards.append(float(score))
                except Exception as e:
                    # Fall back to 0.0 on error
                    all_rewards.append(0.0)

        return all_rewards

    def _load_ocr(self):
        """Load OCR model for text detection reward (used for text-in-image tasks)."""
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr is required for OCR reward. Install with: pip install easyocr")

        # Initialize EasyOCR reader
        self._ocr_reader = easyocr.Reader(
            ['en'],
            gpu=torch.cuda.is_available() and self.device != "cpu",
        )
        self.model = "ocr"  # Placeholder to indicate model is loaded
        self.reward_types = [RewardType.CUSTOM]

    def _compute_ocr(self, request: RewardRequest) -> List[float]:
        """
        Compute OCR reward for text-in-image tasks.

        The reward measures how well the generated image contains the text
        mentioned in the prompt. This is useful for flow_grpo text rendering tasks.

        Scoring:
        - Extract expected text from prompt (e.g., "an image with text 'Hello World'")
        - Run OCR on the image
        - Compute word-level overlap between expected and detected text
        """
        import numpy as np

        images = request.images
        prompts = request.prompts

        all_rewards = []

        for img, prompt in zip(images, prompts):
            try:
                # Convert image to numpy array if needed
                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                elif isinstance(img, torch.Tensor):
                    # Convert tensor to numpy
                    if img.dim() == 4:
                        img = img.squeeze(0)
                    if img.max() <= 1.0:
                        img = (img * 255).byte()
                    img_array = img.permute(1, 2, 0).cpu().numpy()
                else:
                    img_array = np.array(img)

                # Run OCR detection
                ocr_results = self._ocr_reader.readtext(img_array)
                detected_text = " ".join([r[1] for r in ocr_results]).lower()

                # Extract expected text from prompt
                expected_words = self._extract_text_keywords(prompt)

                # Compute word-level match score
                if expected_words:
                    detected_words = set(detected_text.split())
                    matched = len(expected_words & detected_words)
                    score = matched / len(expected_words)
                else:
                    # If no specific text expected, reward having any readable text
                    score = min(len(detected_text) / 50, 1.0)  # Normalize by expected length

                all_rewards.append(score)

            except Exception as e:
                # Fall back to 0.0 on error
                all_rewards.append(0.0)

        return all_rewards

    def _extract_text_keywords(self, prompt: str) -> set:
        """
        Extract expected text keywords from prompt.

        Looks for patterns like:
        - "text 'Hello World'"
        - "with the word 'Example'"
        - "saying 'Text Here'"
        """
        import re

        keywords = set()

        # Pattern 1: text in quotes
        quote_patterns = [
            r"text\s*['\"]([^'\"]+)['\"]",
            r"word\s*['\"]([^'\"]+)['\"]",
            r"saying\s*['\"]([^'\"]+)['\"]",
            r"reads?\s*['\"]([^'\"]+)['\"]",
            r"written\s*['\"]([^'\"]+)['\"]",
        ]

        for pattern in quote_patterns:
            matches = re.findall(pattern, prompt.lower())
            for match in matches:
                keywords.update(match.split())

        # If no quoted text found, try to extract capitalized words
        # (often used for text rendering prompts)
        if not keywords:
            # Find words that might be intended text (all caps or title case)
            words = prompt.split()
            for word in words:
                if word.isupper() and len(word) > 1:
                    keywords.add(word.lower())

        return keywords

    def is_available(self) -> bool:
        """Check if the worker is ready."""
        return self._is_loaded

    def offload(self):
        """Offload model from GPU to CPU."""
        if self.model is not None and hasattr(self.model, 'cpu'):
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

    def onload(self):
        """Load model back to GPU."""
        if self.model is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)


class VideoRewardWorker(LocalRewardWorker):
    """
    Specialized reward worker for video generation.

    Computes rewards based on:
    - Frame quality (aesthetic)
    - Temporal consistency
    - Text-video alignment
    """

    def __init__(
        self,
        frame_reward_model: Optional[str] = "pickscore",
        weight: float = 1.0,
        temporal_weight: float = 0.3,
        alignment_weight: float = 0.7,
        sample_frames: int = 8,
        **kwargs,
    ):
        """
        Initialize video reward worker.

        Args:
            frame_reward_model: Model for per-frame rewards
            weight: Weight for this worker in multi-reward aggregation
            temporal_weight: Weight for temporal consistency
            alignment_weight: Weight for text alignment
            sample_frames: Number of frames to sample for evaluation
        """
        super().__init__(model_name=frame_reward_model, weight=weight, **kwargs)

        self.temporal_weight = temporal_weight
        self.alignment_weight = alignment_weight
        self.sample_frames = sample_frames

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Compute rewards for videos."""
        if not request.is_video:
            # Fall back to image reward
            return super().compute_rewards(request)

        start = time.time()
        videos = request.videos  # [B, C, T, H, W] or [B, T, C, H, W]
        prompts = request.prompts

        try:
            rewards = []
            reward_components = {
                "alignment": [],
                "temporal": [],
            }

            for video, prompt in zip(videos, prompts):
                # Sample frames
                frames = self._sample_frames(video)

                # Compute frame-level alignment rewards
                frame_request = RewardRequest(
                    images=frames,
                    prompts=[prompt] * len(frames),
                )
                frame_response = super().compute_rewards(frame_request)
                alignment_reward = sum(frame_response.rewards) / len(frame_response.rewards)

                # Compute temporal consistency
                temporal_reward = self._compute_temporal_consistency(video)

                # Combine
                total_reward = (
                    self.alignment_weight * alignment_reward +
                    self.temporal_weight * temporal_reward
                )

                rewards.append(total_reward)
                reward_components["alignment"].append(alignment_reward)
                reward_components["temporal"].append(temporal_reward)

            return RewardResponse(
                rewards=rewards,
                reward_components=reward_components,
                successes=[True] * len(rewards),
                errors=[None] * len(rewards),
                compute_time=time.time() - start,
            )

        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * len(videos),
                successes=[False] * len(videos),
                errors=[str(e)] * len(videos),
                compute_time=time.time() - start,
            )

    def _sample_frames(self, video: torch.Tensor) -> List[Image.Image]:
        """Sample frames from video for evaluation."""
        from torchvision.transforms.functional import to_pil_image

        # Handle different video formats
        if video.dim() == 4:  # [C, T, H, W]
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        elif video.dim() == 5:  # [1, C, T, H, W]
            video = video.squeeze(0).permute(1, 0, 2, 3)

        num_frames = video.shape[0]
        indices = torch.linspace(0, num_frames - 1, self.sample_frames).long()

        frames = []
        for idx in indices:
            frame = video[idx]
            if frame.max() <= 1.0:
                frame = (frame * 255).byte()
            frames.append(to_pil_image(frame))

        return frames

    def _compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """
        Compute temporal consistency score.

        Uses frame-to-frame similarity as a proxy for smoothness.
        """
        # Handle different video formats
        if video.dim() == 4:  # [C, T, H, W]
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        elif video.dim() == 5:  # [1, C, T, H, W]
            video = video.squeeze(0).permute(1, 0, 2, 3)

        # Compute frame differences
        frame_diffs = []
        for i in range(len(video) - 1):
            diff = (video[i] - video[i + 1]).abs().mean()
            frame_diffs.append(diff.item())

        # Lower difference = higher consistency
        # Normalize to [0, 1] range (assuming max diff is ~1)
        avg_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0
        consistency = max(0, 1 - avg_diff)

        return consistency
