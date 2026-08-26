"""Tools a :class:`~unirl.rollout.env.tool_environment.ToolEnvironment` dispatches to (LIN-492).

One class per module: the interfaces :class:`~unirl.rollout.env.tools.base.Tool` (stateless) and
:class:`~unirl.rollout.env.tools.base.StatefulTool` (session-scoped), the reference
:class:`~unirl.rollout.env.tools.calculator.CalculatorTool`, the persistent-REPL
:class:`~unirl.rollout.env.tools.sandbox.SandboxTool`, and the web tools
:class:`~unirl.rollout.env.tools.search.SearchTool`,
:class:`~unirl.rollout.env.tools.image_search.ImageSearchTool`,
:class:`~unirl.rollout.env.tools.fetch.FetchTool` and
:class:`~unirl.rollout.env.tools.visit.VisitTool`.

The web tools all reach their providers through the gateway in
:mod:`~unirl.rollout.env.tools.polaris`, which owns credentials and the endpoint.
"""

from unirl.rollout.env.tools.base import StatefulTool, Tool
from unirl.rollout.env.tools.calculator import CalculatorTool
from unirl.rollout.env.tools.fetch import FetchTool
from unirl.rollout.env.tools.image_search import ImageSearchTool
from unirl.rollout.env.tools.sandbox import SandboxTool
from unirl.rollout.env.tools.search import SearchTool
from unirl.rollout.env.tools.visit import VisitTool

__all__ = [
    "CalculatorTool",
    "FetchTool",
    "ImageSearchTool",
    "SandboxTool",
    "SearchTool",
    "StatefulTool",
    "Tool",
    "VisitTool",
]
