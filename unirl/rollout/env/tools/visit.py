"""VisitTool — read webpage(s) through the Polaris gateway and summarise toward a goal (LIN-892).

Page retrieval goes through the gateway's ``jina_ai`` provider; the extraction step calls
an out-of-band OpenAI-compatible service (the same one the answer judge uses), so the
observation is a structured evidence/summary block rather than raw HTML.

Reach for :class:`~unirl.rollout.env.tools.fetch.FetchTool` instead when the content does
not need summarising — an image URL, or a page small enough to read whole.

The extractor prompt, retry budgets, truncation policy and result formatting are inherited
verbatim from AReaL's Tongyi DeepResearch ``tool_visit.py``, including its ``feilds``
typo. **Do not reword the output while LIN-714's AReaL comparison is running.**
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict

import aiohttp

from unirl.rollout.env.tools.base import Tool
from unirl.rollout.env.tools.polaris import POLARIS_URL, READER_PATH, polaris_headers, run_sync

_JINA_ATTEMPTS = 3
_JINA_TIMEOUT_SECONDS = 50.0
_CONTENT_VALIDITY_ATTEMPTS = 8
_SUMMARY_TRANSPORT_ATTEMPTS = 3
_SHORT_OUTPUT_REGENERATIONS = 4
_PARSE_REGENERATIONS = 3
_RETRY_BACKOFF_SECONDS = 0.5
_SUMMARY_TEMPERATURE = 0.7
_SUMMARY_TOP_P = 1.0
_SUMMARY_MAX_COMPLETION_TOKENS = 512
_SUMMARY_CONTEXT_LIMIT = 32768
_SUMMARY_CONTEXT_SAFETY_TOKENS = 256
_SUMMARY_CONNECT_TIMEOUT_SECONDS = 10.0
_SUMMARY_TIMEOUT_SECONDS = 200.0
_MULTI_URL_TIMEOUT_SECONDS = 900.0

_EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

_READ_FAILURE = "[visit] Failed to read page."
_EVIDENCE_FAILURE = "The provided webpage content could not be accessed. Please check the URL or file format."
_SUMMARY_FAILURE = "The webpage content could not be processed, and therefore, no information is available."


class VisitTool(Tool):
    """Read webpage(s) via the gateway and return an evidence/summary extraction."""

    name = "visit"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer_path: str,
    ) -> None:
        if not endpoint or not model or not tokenizer_path:
            raise ValueError("VisitTool requires endpoint, model, and tokenizer_path")
        self._endpoint = str(endpoint)
        self._model = str(model)
        self._tokenizer_path = str(tokenizer_path)
        self._summary_timeout = aiohttp.ClientTimeout(
            total=_SUMMARY_TIMEOUT_SECONDS,
            connect=_SUMMARY_CONNECT_TIMEOUT_SECONDS,
            sock_connect=_SUMMARY_CONNECT_TIMEOUT_SECONDS,
            sock_read=_SUMMARY_TIMEOUT_SECONDS,
        )
        self._tokenizer = None

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Visit webpage(s) and return the summary of the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": ["string", "array"],
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": (
                                "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                            ),
                        },
                        "goal": {
                            "type": "string",
                            "description": "The goal of the visit for webpage(s).",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        return run_sync(lambda: self._execute(arguments))

    async def _execute(self, arguments: Dict[str, Any]) -> str:
        try:
            url = arguments["url"]
            goal = arguments["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        async with aiohttp.ClientSession(timeout=self._summary_timeout) as summary_session:
            if isinstance(url, str):
                try:
                    response = await self._visit_one(url, goal, summary_session)
                except Exception:
                    response = self._failed_response(url, goal)
            else:
                assert isinstance(url, list)
                started = time.monotonic()
                responses = []
                for item in url:
                    item = str(item)
                    if time.monotonic() - started > _MULTI_URL_TIMEOUT_SECONDS:
                        response = self._failed_response(item, goal)
                    else:
                        try:
                            response = await self._visit_one(item, goal, summary_session)
                        except Exception:
                            response = self._failed_response(item, goal)
                    responses.append(response)
                response = "\n=======\n".join(responses)
        return response.strip()

    async def _visit_one(
        self,
        url: str,
        goal: Any,
        summary_session: aiohttp.ClientSession,
    ) -> str:
        original_content = await self._html_readpage_jina(url)
        if not self._valid_content(original_content):
            return self._failed_response(url, goal)

        content = self._truncate_for_summary(original_content, goal)
        messages = [{"role": "user", "content": self._extractor_prompt(content, goal)}]
        raw = await self._call_summary(messages, summary_session)

        for regeneration in range(_SHORT_OUTPUT_REGENERATIONS):
            if not isinstance(raw, str) or len(raw) >= 10:
                break
            if regeneration + 1 < _SHORT_OUTPUT_REGENERATIONS:
                truncate_length = int(0.7 * len(content))
            else:
                truncate_length = 25000
            content = content[:truncate_length]
            messages = [{"role": "user", "content": self._extractor_prompt(content, goal)}]
            raw = await self._call_summary(messages, summary_session)

        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()

        raw_object: Any = None
        parse_attempt = 0
        while True:
            try:
                raw_object = json.loads(raw) if isinstance(raw, str) else raw
                if raw_object is not None and not isinstance(raw_object, dict):
                    raise TypeError("summary response must be an object")
                break
            except Exception:
                if parse_attempt >= _PARSE_REGENERATIONS:
                    raw_object = None
                    break
                raw = await self._call_summary(messages, summary_session)
                parse_attempt += 1
                if parse_attempt >= _PARSE_REGENERATIONS:
                    # AReaL exits without parsing its final regeneration.
                    raw_object = None
                    break

        if raw_object is None:
            return f"The provided webpage content: {original_content[:1000]}"

        response = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        response += "Evidence in page: \n" + str(raw_object.get("evidence", "")) + "\n\n"
        response += "Summary: \n" + str(raw_object.get("summary", "")) + "\n\n"
        return response

    async def _call_summary(
        self,
        messages: list[dict[str, str]],
        session: aiohttp.ClientSession,
    ) -> str | None:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": _SUMMARY_TEMPERATURE,
            "top_p": _SUMMARY_TOP_P,
            "max_completion_tokens": _SUMMARY_MAX_COMPLETION_TOKENS,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("JUDGE_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(_SUMMARY_TRANSPORT_ATTEMPTS):
            try:
                async with session.post(self._endpoint, json=payload, headers=headers) as response:
                    if response.status < 200 or response.status >= 300:
                        await response.read()
                        raise RuntimeError("summary service returned a non-success status")
                    body = await response.json(content_type=None)
                    content = body["choices"][0]["message"]["content"]
                    if content:
                        content = str(content)
                        try:
                            json.loads(content)
                        except Exception:
                            left = content.find("{")
                            right = content.rfind("}")
                            if left != -1 and right != -1 and left <= right:
                                content = content[left : right + 1]
                        return content
                    return None
            except Exception:
                if attempt + 1 < _SUMMARY_TRANSPORT_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        return ""

    async def _jina_readpage(self, url: str) -> str:
        timeout = aiohttp.ClientTimeout(total=_JINA_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(_JINA_ATTEMPTS):
                try:
                    async with session.post(
                        f"{POLARIS_URL}{READER_PATH}",
                        json={"url": url},
                        headers=polaris_headers("jina_ai"),
                    ) as response:
                        if response.status == 200:
                            body = await response.json(content_type=None)
                            content = body["data"]["content"]
                            if not isinstance(content, str):
                                raise TypeError("page reader content must be text")
                            return content
                        await response.read()
                        raise RuntimeError("page reader returned a non-success status")
                except Exception:
                    if attempt + 1 < _JINA_ATTEMPTS:
                        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        return _READ_FAILURE

    async def _html_readpage_jina(self, url: str) -> str:
        for _ in range(_CONTENT_VALIDITY_ATTEMPTS):
            content = await self._jina_readpage(url)
            if self._valid_content(content):
                return content
        return _READ_FAILURE

    @staticmethod
    def _valid_content(content: Any) -> bool:
        return bool(
            content
            and not content.startswith(_READ_FAILURE)
            and content != "[visit] Empty content."
            and not content.startswith("[document_parser]")
        )

    @staticmethod
    def _extractor_prompt(content: str, goal: Any) -> str:
        return _EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)

    def _truncate_for_summary(self, content: str, goal: Any) -> str:
        tokenizer = self._get_tokenizer()
        max_prompt_tokens = _SUMMARY_CONTEXT_LIMIT - _SUMMARY_MAX_COMPLETION_TOKENS - _SUMMARY_CONTEXT_SAFETY_TOKENS
        if max_prompt_tokens <= 0:
            raise ValueError("summary completion allowance leaves no prompt capacity")

        def prompt_length(page: str) -> int:
            messages = [{"role": "user", "content": self._extractor_prompt(page, goal)}]
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(token_ids)

        if prompt_length(content) <= max_prompt_tokens:
            return content
        if prompt_length("") > max_prompt_tokens:
            raise ValueError("summary prompt exceeds configured context headroom")

        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        low, high = 0, len(content_tokens)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = tokenizer.decode(
                content_tokens[:middle],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if prompt_length(candidate) <= max_prompt_tokens:
                low = middle
            else:
                high = middle - 1
        return tokenizer.decode(
            content_tokens[:low],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self._tokenizer

    @staticmethod
    def _failed_response(url: str, goal: Any) -> str:
        response = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        response += "Evidence in page: \n" + _EVIDENCE_FAILURE + "\n\n"
        response += "Summary: \n" + _SUMMARY_FAILURE + "\n\n"
        return response


__all__ = ["VisitTool"]
