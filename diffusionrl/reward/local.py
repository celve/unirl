"""
Local reward worker for GRPO training.

Computes rewards using locally loaded models (PickScore, CLIP, etc.)
"""

import os
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union, Callable
import torch
from PIL import Image
import logging

from .base import BaseRewardWorker, RewardRequest, RewardResponse, RewardType

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

_REWARD_LOADERS = {
    "pickscore": "_load_pickscore",
    "clip": "_load_clip",
    "aesthetic": "_load_aesthetic",
    "hpsv2": "_load_hpsv2",
    "ocr": "_load_ocr",
}

_REWARD_COMPUTERS = {
    "pickscore": "_compute_pickscore",
    "clip": "_compute_clip",
    "hpsv2": "_compute_hpsv2",
    "ocr": "_compute_ocr",
}


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
        loader_name = _REWARD_LOADERS.get(self.model_name)
        if loader_name is None:
            raise ValueError(
                f"Unknown model_name: {self.model_name}. "
                f"Available: {sorted(_REWARD_LOADERS.keys())}"
            )
        getattr(self, loader_name)()
        self._is_loaded = True

    def _load_pickscore(self):
        """Load PickScore model (aligned with flow_grpo PickScoreScorer)."""
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("transformers is required for PickScore")

        processor_path = self.model_kwargs.get(
            "processor_id", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        model_path = self.model_kwargs.get(
            "model_id", "yuvalkirstain/PickScore_v1"
        )

        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        # flow_grpo uses float32 by default
        self.model = self.model.to(dtype=torch.float32)

        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _load_clip(self):
        """Load CLIP model (aligned with flow_grpo ClipScorer)."""
        try:
            from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor
            import torchvision.transforms as T
            import torch.nn as nn
        except ImportError:
            raise ImportError("transformers and torchvision are required for CLIP")

        model_name = self.model_kwargs.get(
            "model_name", "openai/clip-vit-large-patch14"
        )
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Build image transform consistent with flow_grpo (get_image_transform)
        def _get_size(size):
            if isinstance(size, int):
                return (size, size)
            elif "height" in size and "width" in size:
                return (size["height"], size["width"])
            elif "shortest_edge" in size:
                return size["shortest_edge"]
            else:
                raise ValueError(f"Invalid size: {size}")

        config = self.processor.image_processor.to_dict()
        resize = T.Resize(_get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
        crop = T.CenterCrop(_get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
        normalise = T.Normalize(
            mean=self.processor.image_processor.image_mean,
            std=self.processor.image_processor.image_std
        ) if config.get("do_normalize") else nn.Identity()
        self._clip_tform = T.Compose([resize, crop, normalise])

        self.model.eval()
        self.reward_types = [RewardType.IMAGE_TEXT_ALIGNMENT]

    def _load_aesthetic(self):
        """Load aesthetic predictor model."""
        # Implementation depends on specific aesthetic model
        # This is a placeholder for models like LAION aesthetic predictor
        raise NotImplementedError("Aesthetic model loading not yet implemented")

    def _load_hpsv2(self):
        """Load HPSv2 model (aligned with DanceGRPO train_grpo_sd.py)."""
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError:
            raise ImportError("hpsv2 is required for HPSv2 reward")

        # Manually load open_clip model and weights, consistent with DanceGRPO
        open_clip_path = self.model_kwargs.get(
            "open_clip_path", "./hps_ckpt/open_clip_pytorch_model.bin"
        )
        checkpoint_path = self.model_kwargs.get(
            "checkpoint_path", "./hps_ckpt/HPS_v2.1_compressed.pt"
        )

        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            open_clip_path,
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        self._hpsv2_tokenizer = get_tokenizer('ViT-H-14')
        self._hpsv2_preprocess_val = preprocess_val
        self.model = model.to(self.device)
        self.model.eval()
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
            else:
                computer_name = _REWARD_COMPUTERS.get(self.model_name)
                if computer_name is None:
                    raise ValueError(
                        f"Unknown model: {self.model_name}. "
                        f"Available: {sorted(_REWARD_COMPUTERS.keys())}"
                    )
                rewards = getattr(self, computer_name)(request)

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
        """Compute PickScore rewards (aligned with flow_grpo PickScoreScorer)."""
        images = request.images
        prompts = request.prompts

        # Process all images and texts
        all_rewards = []

        def _extract_tensor(output):
            """Compatible with both transformers < 5.0 and >= 5.0. This is an ugly temp solution."""
            if isinstance(output, torch.Tensor):
                return output  # transformers < 5.0: the model returns tensor
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                return output.pooler_output  # transformers >= 5.0: the model returns a wrapped class, which may has pooler_output attr
            if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                return output.last_hidden_state[:, 0]  # fallback: get the first token
            if isinstance(output, (tuple, list)):
                return output[0]
            raise TypeError(f"Unexpected output format: {type(output)}")

        rank = int(os.environ.get("RANK", 0))
        for i in tqdm(
            range(0, len(images), self.batch_size),
            desc="Computing PickScore rewards",
            disable=True
        ):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            # Preprocess images
            image_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

            # Preprocess texts
            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                # Get feature embeddings
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = _extract_tensor(image_embs)
                image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

                text_embs = self.model.get_text_features(**text_inputs)
                text_embs = _extract_tensor(text_embs)
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

                # Scale with logit_scale + matrix multiply then take diagonal, consistent with flow_grpo
                logit_scale = self.model.logit_scale.exp()
                scores = logit_scale * (text_embs @ image_embs.T)
                scores = scores.diag()
                # Normalize to 0-1 range, consistent with flow_grpo
                scores = scores / 26
                all_rewards.extend(scores.cpu().tolist())

        return all_rewards

    def _compute_clip(self, request: RewardRequest) -> List[float]:
        """Compute CLIP similarity rewards (aligned with flow_grpo ClipScorer)."""
        import numpy as np

        images = request.images
        prompts = request.prompts

        all_rewards = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            # Convert PIL Image to tensor [B, C, H, W], value range [0, 1]
            # Consistent with clip_score input conversion in flow_grpo rewards.py
            img_arrays = [np.array(img) for img in batch_images]
            img_arrays = np.array(img_arrays)
            pixels = img_arrays.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            pixels = torch.tensor(pixels, dtype=torch.uint8).float() / 255.0

            # Apply image transform consistent with flow_grpo
            pixels = self._clip_tform(pixels).to(self.device, dtype=pixels.dtype)

            # Text preprocessing
            texts = self.processor(
                text=batch_prompts,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                # Use CLIPModel's forward method, which includes logit_scale internally
                outputs = self.model(pixel_values=pixels, **texts)
                # Take diagonal and normalize by dividing by 30, consistent with flow_grpo
                scores = outputs.logits_per_image.diagonal() / 30

                all_rewards.extend(scores.cpu().tolist())

        return all_rewards

    def _compute_hpsv2(self, request: RewardRequest) -> List[float]:
        """Compute HPSv2 rewards (aligned with DanceGRPO train_grpo_sd.py)."""
        images = request.images
        prompts = request.prompts

        all_rewards = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]

            for img, prompt in zip(batch_images, batch_prompts):
                try:
                    # Image preprocessing: consistent with DanceGRPO
                    if isinstance(img, Image.Image):
                        img_pil = img.convert("RGB")
                    else:
                        img_pil = Image.fromarray(img).convert("RGB")

                    image_input = self._hpsv2_preprocess_val(img_pil).unsqueeze(0).to(
                        device=self.device, non_blocking=True
                    )
                    # Text preprocessing
                    text_input = self._hpsv2_tokenizer([prompt]).to(
                        device=self.device, non_blocking=True
                    )

                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(image_input, text_input)
                            image_features = outputs["image_features"]
                            text_features = outputs["text_features"]
                            # Compute cosine similarity directly without logit_scale (consistent with DanceGRPO)
                            logits_per_image = image_features @ text_features.T
                            hps_score = torch.diagonal(logits_per_image)

                    all_rewards.append(float(hps_score.item()))
                except Exception as e:
                    # Fall back to 0.0 on error
                    all_rewards.append(0.0)

        return all_rewards

    def _load_ocr(self):
        """Load OCR model for text detection reward (PP-OCRv5)."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

        try:
            from Levenshtein import distance
        except ImportError:
            raise ImportError("python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein")

        # Initialize PP-OCRv5 reader with new API
        self._ocr_reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
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
        - Run OCR on the image using PP-OCRv5 predict API
        - Compute word-level overlap between expected and detected text
        """
        import numpy as np
        from Levenshtein import distance

        images = request.images
        prompts = request.prompts

        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        rank = int(os.environ.get("RANK", 0))
        for img, prompt in tqdm(
            zip(images, prompts),
            desc="Computing OCR rewards",
            disable=(rank != 0)
        ):
            # Convert image format to np.ndarray
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition using PP-OCRv5 predict API
                result = self._ocr_reader.predict(img)
                # Extract recognized text from PP-OCRv5 result
                recognized_text = ''
                for res in result:
                    recognized_text += ''.join(res['rec_texts'])

                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1 - dist / (len(prompt))
            rewards.append(reward)

        return rewards

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


if __name__ == '__main__':

    def test_pickscore():
        """Test PickScore reward model."""
        print("=" * 60)
        print("Testing PickScore Reward Model")
        print("=" * 60)

        # Create a simple test image
        test_image = Image.new("RGB", (512, 512), color=(135, 206, 235))

        # Initialize PickScore worker
        worker = LocalRewardWorker(
            model_name="pickscore",
            device="cuda",
        )

        # Build request
        test_prompts = [
            "a beautiful blue sky",
            "a red sports car on the highway",
        ]
        test_images = [test_image, test_image]

        request = RewardRequest(
            images=test_images,
            prompts=test_prompts,
        )

        # Compute rewards
        response = worker.compute_rewards(request)

        print(f"Prompts: {test_prompts}")
        print(f"Rewards: {response.rewards}")
        print(f"Successes: {response.successes}")
        print(f"Errors: {response.errors}")
        print(f"Compute time: {response.compute_time:.4f}s")
        print()

    def test_ocr():
        """Test OCR (PP-OCRv5) reward model."""
        print("=" * 60)
        print("Testing OCR (PP-OCRv5) Reward Model")
        print("=" * 60)

        from PIL import ImageDraw, ImageFont

        # Create a test image with text rendered on it
        test_image = Image.new("RGB", (512, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(test_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except IOError:
            font = ImageFont.load_default()
        draw.text((50, 80), "Hello World", fill=(0, 0, 0), font=font)
        test_image.save("test_ocr_input.png")
        print("Saved test image to test_ocr_input.png")

        # Initialize OCR worker
        worker = LocalRewardWorker(
            model_name="ocr",
            device="cuda",
        )

        # Test 1: prompt matches image text
        request_match = RewardRequest(
            images=[test_image],
            prompts=['an image with text "Hello World"'],
        )
        response_match = worker.compute_rewards(request_match)
        print(f"[Match test] Prompt: 'an image with text \"Hello World\"'")
        print(f"  Reward: {response_match.rewards}")
        print(f"  Success: {response_match.successes}")
        print()

        # Test 2: prompt does NOT match image text
        request_mismatch = RewardRequest(
            images=[test_image],
            prompts=['an image with text "Goodbye"'],
        )
        response_mismatch = worker.compute_rewards(request_mismatch)
        print(f"[Mismatch test] Prompt: 'an image with text \"Goodbye\"'")
        print(f"  Reward: {response_mismatch.rewards}")
        print(f"  Success: {response_mismatch.successes}")
        print()

        # Also demonstrate raw PP-OCRv5 predict API
        print("-" * 40)
        print("Raw PP-OCRv5 predict output:")
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        result = ocr.predict("test_ocr_input.png")
        for res in result:
            res.print()
        print()

    # Run tests
    print("Starting reward model tests...\n")
    # test_pickscore()
    test_ocr()
    print("All tests completed!")
