"""AReaL-compatible turn semantics for the Tongyi deep-research task."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import json5

from unirl.rollout.env.tool_environment import ToolEnvironment
from unirl.rollout.env.tools.base import StatefulTool, Tool
from unirl.types.primitives import Texts
from unirl.types.sample import Primitive, Sample

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INVALID_TOOL_CALL_SUFFIX = 'Tool call must be a valid json contain a valid "name" and "arguments" field.'


def extract_first_answer(text: str) -> Optional[str]:
    """Return the first closed answer interior without trimming it."""
    match = _ANSWER_RE.search(text or "")
    return match.group(1) if match is not None else None


class ARealDeepResearchEnvironment(ToolEnvironment):
    """Single-trajectory AReaL parser, tool dispatcher, and termination rule."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        super().__init__(tools=tools, max_turns=0)

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        frontier = sample.parts[-1].primitives.get("text")
        if not isinstance(frontier, Texts) or len(frontier.texts) != 1:
            raise TypeError("ARealDeepResearchEnvironment requires one Texts trajectory")
        text = frontier.texts[0] or ""
        sessions = (sample.parts[0].control or {}).get("tool_sessions", {})

        observation: Optional[Texts] = None
        call: Optional[Dict[str, Any]] = None
        result: Optional[str] = None
        parse_failed = False
        match = _TOOL_CALL_RE.search(text)
        if match is not None:
            try:
                payload = json5.loads(match.group(1))
                if not isinstance(payload, dict):
                    raise TypeError("tool call must be an object")
                name = payload["name"]
                arguments = payload.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise ValueError('tool call "name" must be a non-empty string')
                if not isinstance(arguments, dict):
                    raise TypeError('tool call "arguments" must be an object')
                call = {"name": name, "arguments": arguments}
                result = self._run_areal(call, sessions)
            except (KeyError, TypeError, ValueError) as exc:
                parse_failed = True
                result = f"Error: {exc} {_INVALID_TOOL_CALL_SUFFIX}"
            observation = Texts(texts=[f"<tool_response>\n{result}\n</tool_response>"])

        prediction = extract_first_answer(text)
        info = {
            "tool_call": call,
            "tool_result": result,
            "tool_parse_failed": parse_failed,
            "prediction": prediction,
            "answer_tag_found": prediction is not None,
        }
        return observation, prediction is not None, info

    def _run_areal(self, call: Dict[str, Any], sessions: Dict[str, str]) -> str:
        name = call["name"]
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Tool {name} not found"
        try:
            if isinstance(tool, StatefulTool):
                session_id = sessions.get(name)
                if session_id is None:
                    raise RuntimeError(f"no active session for stateful tool {name}")
                return str(tool.execute_session(session_id, call["arguments"]))
            return str(tool.execute(call["arguments"]))
        except Exception:  # noqa: BLE001 - corrected failure text must not expose service credentials
            logger.warning("AReaL deep-research tool execution failed", exc_info=False)
            return f"Error: tool {name} execution failed"


__all__ = ["ARealDeepResearchEnvironment", "extract_first_answer"]
