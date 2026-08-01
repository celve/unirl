#!/usr/bin/env python
"""Does the agent's prompt actually advertise the draw tool? (LIN-577)

Rollout-only diagnostic for a near-zero render rate. Answers, in order, the three
things that could make an in-loop image agent never call ``draw``:

1. **Do the tool schemas reach the prompt at all?** ``ToolEnvironment`` exposes them
   and the engine injects them into ``chat_template_kwargs``, but each adapter
   decides whether to forward those to ``apply_chat_template``. Prints whether the
   tool name survives into the rendered prompt, for the text and VLM paths
   separately — they use different call sites.
2. **Does the model's chat template even support tools?** Many VL templates silently
   ignore a ``tools=`` kwarg. Compared against a with/without render.
3. **Given a prompt that does advertise the tool, does the model call it?** Boots
   the real engine and prints raw generations plus parsed tool calls.

Run (needs a free GPU):
  python scripts/agentic_image_prompt_probe.py
"""

from __future__ import annotations

import json
import os

VLM = os.environ.get("VLM_MODEL", "/root/unirl/models/local/Qwen2.5-VL-3B-Instruct")

SYSTEM = (
    "You are an image-generation agent. Call the `draw` tool with a detailed visual prompt. "
    'For each function call, return a json object within <tool_call></tool_call> XML tags:\n'
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
TASK = "a photograph of a patterdale dog driving a land rover in a cave"


def main() -> None:
    from transformers import AutoProcessor, AutoTokenizer

    from unirl.rollout.loop.tool_environment import ToolEnvironment, parse_tool_call
    from unirl.rollout.loop.tools.draw import DrawTool

    env = ToolEnvironment(tools=[DrawTool()], max_turns=6)
    schemas = env.tool_schemas()
    print(f"== tool_schemas from env: {[s['function']['name'] for s in schemas]}")

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]

    # --- 1/2. Does the tool name survive into the rendered prompt? ---
    tok = AutoTokenizer.from_pretrained(VLM)
    proc = AutoProcessor.from_pretrained(VLM)

    def rendered(obj, label: str, **kw) -> str:
        try:
            out = obj.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
        except Exception as exc:  # noqa: BLE001 — a template that rejects tools is a finding
            print(f"  {label}: RAISED {type(exc).__name__}: {str(exc)[:120]}")
            return ""
        has = "draw" in out
        print(f"  {label}: len={len(out):5d}  advertises 'draw'={has}")
        return out

    print("\n== 1/2. tool advertisement in the rendered prompt ==")
    print(" tokenizer (the TEXT adapter's call site — it forwards chat_template_kwargs):")
    with_tools = rendered(tok, "with tools=  ", tools=schemas)
    rendered(tok, "without tools")
    print(" processor (the VLM adapter's call site — encode_mm passes NO template kwargs):")
    rendered(proc, "with tools=  ", tools=schemas)
    rendered(proc, "without tools")

    if with_tools:
        head = with_tools[: with_tools.find(TASK)] if TASK in with_tools else with_tools[:1200]
        print("\n--- prompt prefix actually shown to the model (tokenizer + tools) ---")
        print(head[:1200])

    # --- 3. Given a tool-advertising prompt, does the model call it? ---
    print("\n== 3. does the model emit a tool call? ==")
    from sglang.srt.entrypoints.engine import Engine

    eng = Engine(
        model_path=VLM, tp_size=1, mem_fraction_static=0.3, skip_server_warmup=True, attention_backend="triton"
    )
    try:
        for label, prompt in (("WITH tool schemas", with_tools), ("prose-only (no tools=)", None)):
            text = prompt or tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            outs = eng.generate(
                [text] * 4, {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 256, "stop": ["</tool_call>"]}
            )
            calls = 0
            for i, o in enumerate(outs):
                gen = o["text"] if isinstance(o, dict) else str(o)
                parsed = parse_tool_call(gen + "</tool_call>")
                calls += int(bool(parsed and parsed.get("name") == "draw"))
                if i == 0:
                    print(f"\n--- {label} sample generation ---\n{gen[:500]}")
                    print(f"  parsed: {json.dumps(parsed)[:200] if parsed else None}")
            print(f"  => {label}: {calls}/4 emitted a draw call")
    finally:
        eng.shutdown()


if __name__ == "__main__":
    main()
