"""AReaL-compatible search and visit tools for Tongyi DeepResearch."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, TypeVar

import aiohttp

from unirl.rollout.env.tools.base import Tool

_T = TypeVar("_T")

_SERPER_URL = "https://google.serper.dev/search"
_JINA_URL = "https://r.jina.ai/"

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


def _run_sync(factory: Callable[[], Awaitable[_T]]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    raise RuntimeError("AReaL tools require synchronous Tool.execute dispatch")


class ARealSearchTool(Tool):
    """Pinned AReaL Serper behavior with bounded, credential-safe failures."""

    name = "search"

    def __init__(
        self,
        *,
        endpoint: str = _SERPER_URL,
        max_attempts: int = 5,
        retry_backoff_seconds: float = 0.5,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        total_timeout_seconds: float = 45.0,
    ) -> None:
        self._endpoint = str(endpoint)
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._timeout = aiohttp.ClientTimeout(
            total=float(total_timeout_seconds),
            connect=float(connect_timeout_seconds),
            sock_connect=float(connect_timeout_seconds),
            sock_read=float(read_timeout_seconds),
        )

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Performs batched web searches: supply an array 'query'; the tool "
                    "retrieves the top 10 results for each query in one call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Array of query strings. Include multiple complementary "
                                "search queries in a single call."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        try:
            query = arguments["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        if isinstance(query, str):
            return _run_sync(lambda: self._search_one(query))
        assert isinstance(query, list)
        return _run_sync(lambda: self._search_many(query))

    async def _search_many(self, queries: list[Any]) -> str:
        responses = await asyncio.gather(*(self._search_one(query) for query in queries))
        return "\n=======\n".join(responses)

    async def _search_one(self, query: str) -> str:
        payload = (
            {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
            if any("\u4e00" <= char <= "\u9fff" for char in query)
            else {"q": query, "location": "United States", "gl": "us", "hl": "en"}
        )
        headers = {
            "X-API-KEY": os.environ.get("SERPER_KEY_ID", ""),
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for attempt in range(self._max_attempts):
                try:
                    async with session.post(self._endpoint, json=payload, headers=headers) as response:
                        text = await response.text()
                        if response.status < 200 or response.status >= 300:
                            raise RuntimeError("search service returned a non-success status")
                        try:
                            results = json.loads(text)
                        except Exception:
                            return f"[Search] Failed to parse response for '{query}'."
                        if "organic" not in results:
                            return f"No results found for query: '{query}'. Use a less specific query."
                        snippets = []
                        for index, page in enumerate(results.get("organic", []), start=1):
                            date = f"\nDate published: {page['date']}" if page.get("date") else ""
                            source = f"\nSource: {page['source']}" if page.get("source") else ""
                            snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
                            item = (
                                f"{index}. [{page.get('title', '')}]({page.get('link', '')}){date}{source}\n{snippet}"
                            ).replace("Your browser can't play this video.", "")
                            snippets.append(item)
                        return (
                            f"A Google search for '{query}' found {len(snippets)} results:"
                            "\n\n## Web Results\n" + "\n\n".join(snippets)
                        )
                except Exception:
                    if attempt + 1 < self._max_attempts:
                        await asyncio.sleep(self._retry_backoff_seconds)
        return "Google search Timeout or error; return None, Please try again later."


class ARealVisitTool(Tool):
    """Pinned AReaL Jina/extractor behavior with context-safe truncation."""

    name = "visit"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer_path: str,
        jina_endpoint: str = _JINA_URL,
        jina_attempts: int = 3,
        jina_timeout_seconds: float = 50.0,
        content_validity_attempts: int = 8,
        summary_transport_attempts: int = 3,
        short_output_regenerations: int = 4,
        parse_regenerations: int = 3,
        retry_backoff_seconds: float = 0.5,
        summary_temperature: float = 0.7,
        summary_top_p: float = 1.0,
        summary_max_completion_tokens: int = 512,
        summary_context_limit: int = 32768,
        summary_context_safety_tokens: int = 256,
        summary_connect_timeout_seconds: float = 10.0,
        summary_timeout_seconds: float = 200.0,
        multi_url_timeout_seconds: float = 900.0,
    ) -> None:
        if not endpoint or not model or not tokenizer_path:
            raise ValueError("ARealVisitTool requires endpoint, model, and tokenizer_path")
        self._endpoint = str(endpoint)
        self._model = str(model)
        self._tokenizer_path = str(tokenizer_path)
        self._jina_endpoint = str(jina_endpoint)
        self._jina_attempts = max(1, int(jina_attempts))
        self._jina_timeout_seconds = float(jina_timeout_seconds)
        self._content_validity_attempts = max(1, int(content_validity_attempts))
        self._summary_transport_attempts = max(1, int(summary_transport_attempts))
        self._short_output_regenerations = max(0, int(short_output_regenerations))
        self._parse_regenerations = max(0, int(parse_regenerations))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._summary_temperature = float(summary_temperature)
        self._summary_top_p = float(summary_top_p)
        self._summary_max_completion_tokens = int(summary_max_completion_tokens)
        self._summary_context_limit = int(summary_context_limit)
        self._summary_context_safety_tokens = int(summary_context_safety_tokens)
        self._summary_timeout = aiohttp.ClientTimeout(
            total=float(summary_timeout_seconds),
            connect=float(summary_connect_timeout_seconds),
            sock_connect=float(summary_connect_timeout_seconds),
            sock_read=float(summary_timeout_seconds),
        )
        self._multi_url_timeout_seconds = float(multi_url_timeout_seconds)
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
        return _run_sync(lambda: self._execute(arguments))

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
                    if time.monotonic() - started > self._multi_url_timeout_seconds:
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

        for regeneration in range(self._short_output_regenerations):
            if not isinstance(raw, str) or len(raw) >= 10:
                break
            if regeneration + 1 < self._short_output_regenerations:
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
                if parse_attempt >= self._parse_regenerations:
                    raw_object = None
                    break
                raw = await self._call_summary(messages, summary_session)
                parse_attempt += 1
                if parse_attempt >= self._parse_regenerations:
                    # The pinned AReaL loop issues its final regeneration but
                    # exits without parsing that response.
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
            "temperature": self._summary_temperature,
            "top_p": self._summary_top_p,
            "max_completion_tokens": self._summary_max_completion_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("JUDGE_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(self._summary_transport_attempts):
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
                if attempt + 1 < self._summary_transport_attempts:
                    await asyncio.sleep(self._retry_backoff_seconds)
        return ""

    async def _jina_readpage(self, url: str) -> str:
        timeout = aiohttp.ClientTimeout(total=self._jina_timeout_seconds)
        headers = {"Authorization": f"Bearer {os.environ.get('JINA_API_KEYS', '')}"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(self._jina_attempts):
                try:
                    async with session.get(f"{self._jina_endpoint}{url}", headers=headers) as response:
                        if response.status == 200:
                            return await response.text()
                        await response.read()
                        raise RuntimeError("page reader returned a non-success status")
                except Exception:
                    if attempt + 1 < self._jina_attempts:
                        await asyncio.sleep(self._retry_backoff_seconds)
        return _READ_FAILURE

    async def _html_readpage_jina(self, url: str) -> str:
        for _ in range(self._content_validity_attempts):
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
        max_prompt_tokens = (
            self._summary_context_limit - self._summary_max_completion_tokens - self._summary_context_safety_tokens
        )
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


__all__ = ["ARealSearchTool", "ARealVisitTool"]
