"""Tool — the abstract interface a ``ToolEnvironment`` dispatches to (LIN-492).

A ``Tool`` is the two things the environment needs: a JSON schema (so the rollout prompt can
advertise the tool to the model via ``tokenizer.apply_chat_template(tools=...)``) and an executor
(run the parsed call, return a text result). One concrete tool per module; see
:class:`~unirl.rollout.loop.tools.calculator.CalculatorTool`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ToolExecutionResult:
    """Text returned to the model plus optional, model-invisible diagnostics.

    ``Tool.execute`` remains the compatibility surface for existing tools and
    callers.  Web tools override :meth:`Tool.execute_with_info` to attach safe,
    aggregate transport counters; :class:`ToolEnvironment` consumes the richer
    result when available while continuing to expose the exact same observation
    text to the policy.

    Diagnostics must never contain request inputs, response bodies, endpoints,
    or credentials.  ``ToolEnvironment`` applies a second allow-list before
    publishing them in ``info``.
    """

    text: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """A single callable tool: a name, a JSON function-schema, and an executor."""

    #: The tool name the model emits in ``<tool_call>{"name": ...}</tool_call>``.
    name: str

    @abstractmethod
    def json_schema(self) -> Dict[str, Any]:
        """The OpenAI function-tool schema — the shape ``apply_chat_template(tools=[...])`` consumes:
        ``{"type": "function", "function": {"name", "description", "parameters": {JSON Schema}}}``."""
        ...

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Run the tool on parsed ``arguments`` and return the result as text.

        May raise on bad input — :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment`
        catches and surfaces the error to the model as the observation, so the policy can recover.
        """
        ...

    def execute_with_info(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Run the tool and optionally return safe execution diagnostics.

        The default adapter preserves every existing :class:`Tool`
        implementation: subclasses only implementing ``execute`` automatically
        get an empty diagnostic mapping.
        """

        return ToolExecutionResult(text=self.execute(arguments))


class StatefulTool(Tool):
    """A :class:`Tool` that holds **per-trajectory state** behind a session handle (LIN-533).

    Where :class:`Tool` is a pure function (args in, text out, holds nothing), a ``StatefulTool``
    carries state across turns — a code-interpreter namespace, an editing canvas, a connection.
    :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment` dispatches on
    ``isinstance(tool, StatefulTool)`` (one protocol, no code fork), so the stateless ``Tool`` path
    is byte-for-byte unchanged.

    Lifecycle per trajectory, keyed by a ``session_id`` minted at ``reset`` and carried in the root
    Sample's *control* bag:

    - ``session_start(session_id, context)`` — once, on the loop thread. Cheap and infallible:
      record the session and stash ``context``. Do **no** blocking I/O here (spawning a subprocess
      or opening a connection would stall the shared worker loop) — open the handle lazily in
      ``execute_session``, which runs off-loop in an executor.
    - ``execute_session(session_id, arguments)`` — per turn. Operates on the (lazily opened)
      per-session handle; runs in the executor via
      :meth:`~unirl.rollout.loop.tool_environment.ToolEnvironment.step`.
    - ``session_end(session_id)`` — once, guaranteed: the engine's ``finally`` hook calls it even on
      a crashed/aborted trajectory (via ``ToolEnvironment.close``). Must be **idempotent**, a no-op
      on an unknown/never-opened id, and **must not raise**.

    Implementations own a handle store keyed by ``session_id`` and guarded by a lock (``reset`` runs
    on the loop thread; ``execute_session``/``session_end`` on executor threads) — never keep
    per-trajectory state unkeyed on ``self``, since one instance serves many concurrent trajectories.
    """

    def session_start(self, session_id: str, context: Dict[str, Any]) -> None:
        """Open a session. Default no-op — a light tool may allocate lazily in ``execute_session``."""

    @abstractmethod
    def execute_session(self, session_id: str, arguments: Dict[str, Any]) -> str:
        """Run the tool for ``session_id`` on parsed ``arguments``; return the result as text.

        May raise on bad input — :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment`
        catches and surfaces the error to the model as the observation.
        """
        ...

    def execute_session_with_info(self, session_id: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Session-scoped equivalent of :meth:`Tool.execute_with_info`."""

        return ToolExecutionResult(text=self.execute_session(session_id, arguments))

    def session_end(self, session_id: str) -> None:
        """Tear down a session. Default no-op. Idempotent, no-op on unknown ids, never raises."""

    def execute(self, arguments: Dict[str, Any]) -> str:  # pragma: no cover - guard
        """Stateless entrypoint — a programming error for a session-scoped tool; use
        :meth:`execute_session`. ``ToolEnvironment`` never routes here for a ``StatefulTool``."""
        raise NotImplementedError("StatefulTool is session-scoped; call execute_session(session_id, ...)")


__all__ = ["Tool", "StatefulTool", "ToolExecutionResult"]
