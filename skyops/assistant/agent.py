"""Optional Claude provider for the SkyOps assistant (set SKYOPS_ASSISTANT_PROVIDER=claude).

Uses the Anthropic SDK's beta tool runner, so the agentic loop (call tool -> feed result -> continue) is
handled by the SDK. Requires `pip install anthropic` and ANTHROPIC_API_KEY. The default provider is Gemini
(see gemini_agent.py); both share the same tools and system prompt.
"""
from __future__ import annotations

import json
import os
import time

from skyops.assistant.prompts import SYSTEM_PROMPT
from skyops.assistant.tools import FUNCTIONS
from skyops.config import settings


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClaudeAssistant:
    provider = "claude"

    def __init__(self, model: str | None = None, max_tokens: int = 4096, effort: str = "medium"):
        import anthropic
        from anthropic import beta_tool

        self.anthropic = anthropic
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic()
        self.tools = [beta_tool(f) for f in FUNCTIONS]
        self.messages: list[dict] = []
        self.transcript: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()
        self.transcript.clear()

    def ask(self, text: str, t_idx: int | None = None, context: dict | None = None) -> dict:
        anthropic = self.anthropic
        ctx = dict(context or {})
        if t_idx is not None:
            ctx["current_snapshot_index"] = int(t_idx)
        content = f"[context: {json.dumps(ctx)}]\n{text}" if ctx else text
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        t0 = time.time()
        tool_calls: list[dict] = []
        final_text, stop_reason = "", None
        params = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT, tools=self.tools,
                      messages=self.messages, output_config={"effort": self.effort})
        if self.model.startswith(("claude-opus-5", "claude-fable")):
            # server-side refusal fallbacks (routes a policy decline to another model inside the same call)
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            runner = self.client.beta.messages.tool_runner(**params)
            for message in runner:
                stop_reason = message.stop_reason
                self.messages.append({"role": "assistant", "content": message.content})
                for block in message.content:
                    if block.type == "tool_use":
                        tool_calls.append(dict(name=block.name, input=block.input))
                    elif block.type == "text" and block.text.strip():
                        final_text = block.text
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    self.messages.append(tool_response)
        except anthropic.AuthenticationError:
            del self.messages[checkpoint:]
            return dict(error="Anthropic API key missing or invalid. Set ANTHROPIC_API_KEY in .env.", answer=None, tool_calls=[])
        except anthropic.RateLimitError:
            del self.messages[checkpoint:]
            return dict(error="Rate limited by the Claude API, try again in a few seconds.", answer=None, tool_calls=[])
        except anthropic.APIStatusError as e:
            del self.messages[checkpoint:]
            return dict(error=f"Claude API error {e.status_code}: {e.message}", answer=None, tool_calls=[])
        except anthropic.APIConnectionError:
            del self.messages[checkpoint:]
            return dict(error="Could not reach the Claude API (network).", answer=None, tool_calls=[])
        if stop_reason == "refusal":
            final_text = final_text or "I can't help with that request."
        rec = dict(question=text, answer=final_text, tool_calls=tool_calls, provider=self.provider, model=self.model,
                   seconds=round(time.time() - t0, 1))
        self.transcript.append(rec)
        return rec
