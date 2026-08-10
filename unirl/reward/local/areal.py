"""AReaL-compatible answer judge for Tongyi DeepResearch."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List

import aiohttp

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = (
    "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.\n"
    "You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).\n\n"
    "\n"
    "question: {question}\n"
    "ground truth answers: {gt_answer}\n"
    "pred_answer: {pred_answer}\n\n"
    "Did the model give an answer **equivalent** to the labeled answer? \n\nThe output should in the following json format:\n"
    "```json\n"
    "{{\n"
    '    "rationale": "your rationale for the judgement, as a text",\n'
    "    \"judgement\": \"your judgement result, can only be 'correct' or 'incorrect'\n"
    "}}\n"
    "```\n"
    "Your output:"
)


def _parse_judge_result(raw_response: str) -> float:
    parsed: Any = None
    candidate = raw_response.split("```json")[-1].split("```")[0].strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
            break
        except Exception:
            pass
    if parsed is None and '"judgement": "incorrect"' in raw_response:
        parsed = {"judgement": "incorrect"}
    if parsed is None and '"judgement": "correct"' in raw_response:
        parsed = {"judgement": "correct"}
    return float(isinstance(parsed, dict) and parsed.get("judgement") == "correct")


class ARealJudgeRewardScorer(LocalRewardBackend):
    """Binary AReaL judge with ordered concurrent submission to an external service."""

    canonical_model_name = "areal_judge"
    input_kind = "text"

    def __init__(self, *, config: "ARealJudgeSpec", base_device: str) -> None:
        del base_device
        self._spec = config
        super().__init__(timeout=config.timeout_seconds)

    def _load_model(self) -> None:
        self.model = "areal_judge"
        self._endpoint = os.environ.get("JUDGE_URL", self._spec.endpoint)
        self._judge_model = os.environ.get("JUDGE_MODEL", self._spec.model)
        self._api_key = os.environ.get("JUDGE_API_KEY", self._spec.api_key)
        if not self._endpoint or not self._judge_model:
            raise ValueError("ARealJudgeRewardScorer requires endpoint and model")

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        predictions = request.texts
        if predictions is None:
            raise ValueError("ARealJudgeRewardScorer requires request.texts")
        questions = request.prompts or [""] * len(predictions)
        metadata = request.metadata or [None] * len(predictions)
        answers = [(item or {}).get("answer") for item in metadata]
        return asyncio.run(self._judge_batch(questions, predictions, answers))

    async def _judge_batch(
        self,
        questions: list[str],
        predictions: list[str],
        answers: list[Any],
    ) -> list[float]:
        timeout = aiohttp.ClientTimeout(
            total=self._spec.timeout_seconds,
            connect=self._spec.connect_timeout_seconds,
            sock_connect=self._spec.connect_timeout_seconds,
            sock_read=self._spec.timeout_seconds,
        )
        semaphore = asyncio.Semaphore(max(1, self._spec.submission_concurrency))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._judge_one(question, prediction, answer, session, semaphore)
                for question, prediction, answer in zip(questions, predictions, answers)
            ]
            return list(await asyncio.gather(*tasks))

    async def _judge_one(
        self,
        question: str,
        prediction: str,
        answer: Any,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> float:
        if isinstance(answer, list) and len(answer) == 1:
            answer = str(answer[0])
        prompt = _JUDGE_PROMPT.format(
            question=question,
            gt_answer=str(answer),
            pred_answer=prediction[: self._spec.max_prediction_chars],
        )
        payload = {
            "model": self._judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._spec.temperature,
            "top_p": self._spec.top_p,
            "max_completion_tokens": self._spec.max_completion_tokens,
            "store": self._spec.store,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with semaphore:
            for attempt in range(max(1, self._spec.transport_attempts)):
                try:
                    async with session.post(self._endpoint, json=payload, headers=headers) as response:
                        if response.status < 200 or response.status >= 300:
                            await response.read()
                            raise RuntimeError("judge service returned a non-success status")
                        body = await response.json(content_type=None)
                        content = body["choices"][0]["message"]["content"]
                        if not isinstance(content, str):
                            return 0.0
                        return _parse_judge_result(content)
                except Exception:
                    if attempt + 1 < max(1, self._spec.transport_attempts):
                        await asyncio.sleep(self._spec.retry_backoff_seconds)
            logger.warning(
                "AReaL judge request failed after %d transport attempt(s); scoring zero",
                max(1, self._spec.transport_attempts),
            )
            return 0.0


@dataclass
class ARealJudgeSpec(BaseRewardComponentSpec):
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    timeout_seconds: float = 200.0
    connect_timeout_seconds: float = 10.0
    transport_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    submission_concurrency: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    max_prediction_chars: int = 200
    max_completion_tokens: int = 8192
    store: bool = False


__all__ = ["ARealJudgeRewardScorer", "ARealJudgeSpec"]
