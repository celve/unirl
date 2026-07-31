"""DrawTool — the agent's image-generation action (LIN-577).

Unlike every other :class:`~unirl.rollout.loop.tools.tool.Tool`, this one does not
produce its own observation. The **engine** owns the diffusion child, so
:class:`~unirl.rollout.engine.agentic.image_engine.AgenticImageRolloutEngine`
intercepts the parsed call — surfaced by
:meth:`~unirl.rollout.loop.tool_environment.ToolEnvironment.step` in
``info["tool_calls"]`` — and runs the generation as a **trainable diffusion gen
Part** on the trajectory's own lineage, rather than as a mask-0 observation.

That split is deliberate: the environment keeps owning the *decision* (parse the
call, decide the loop is not done), and the engine keeps owning the *mechanism*
(which engine renders the turn), exactly as ``ToolEnvironment``'s contract states.
The tool itself exists for two things: its schema, so the rollout prompt advertises
the action through ``apply_chat_template(tools=...)``, and its argument validation.

``execute`` is a fallback, reached only if a recipe registers this tool on an
environment driven by an engine that does *not* handle draw turns (e.g. the plain
``AgenticRolloutEngine``). It returns an explanatory string rather than raising, so
such a misconfiguration degrades to a visible message in the transcript instead of
sinking the trajectory.
"""

from __future__ import annotations

from typing import Any, Dict

from unirl.rollout.loop.tools.tool import Tool

#: Default tool name; mirrored by ``AgenticImageRolloutEngineConfig.draw_tool_name``.
DRAW_TOOL_NAME = "draw"


class DrawTool(Tool):
    """Render an image from a text prompt, optionally editing the previous one."""

    name = DRAW_TOOL_NAME

    def __init__(self, *, name: str = DRAW_TOOL_NAME, description: str = "") -> None:
        self.name = name
        self._description = description or (
            "Render an image from a text prompt. If you have already drawn an image earlier in "
            "this conversation, the new image edits that most recent image; pass edit=false to "
            "start from scratch instead. The rendered image is returned to you so you can inspect "
            "it and draw again to refine it."
        )

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "What to render. When editing a previous image, describe the "
                                "desired result, not just the change."
                            ),
                        },
                        "edit": {
                            "type": "boolean",
                            "description": (
                                "Edit the most recent image in the conversation (default) rather "
                                "than rendering from scratch. Ignored on the first draw."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Fallback only — a draw-aware engine intercepts the call before dispatch."""
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("draw: 'prompt' must be a non-empty string")
        return (
            f"Error: the {self.name!r} tool needs a rollout engine that renders image turns "
            "(AgenticImageRolloutEngine with in_loop_images=true); this run has none, so no "
            "image was produced."
        )


__all__ = ["DrawTool", "DRAW_TOOL_NAME"]
