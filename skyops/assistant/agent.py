"""SkyOps operations assistant: Claude with tools over the live airspace, fleet, camera and land-use layers.

Uses the Anthropic SDK's beta tool runner, so the agentic loop (call tool -> feed result -> continue) is
handled by the SDK. The conversation is mirrored locally so multi-turn chat works and the UI can show
which tools were called. Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import json
import os
import time

import anthropic

from skyops.config import settings
from skyops.assistant.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are SkyOps, the operations assistant of an AI control tower for the drone era. You sit
between manned air traffic (live ADS-B over the Indian subcontinent), a drone fleet with on-board cameras and
telemetry, land-use maps, and the maintenance state of an aircraft fleet.

Ground rules:
- Always use the tools for facts. Never invent callsigns, distances, altitudes, scores or engine numbers.
- Answer like a calm, precise duty controller: short paragraphs or tight bullet lists, units stated (ft, kt, NM, km, m).
- When asked whether a drone can launch or fly somewhere, call mission_risk_brief and lead with the verdict
  (GO / CAUTION / NO-GO), then the top reasons. Offer the safest alternative (later time, lower altitude, other site)
  when the verdict is not GO.
- When the user refers to "now", use the current replay snapshot index given in the context line, unless they name
  another time.
- The airspace data is a 350-second replay of a real OpenSky capture (20 snapshots, ~18 s apart). The drone flight
  (AU-AIR) was recorded in Aarhus, Denmark; the tower can relocate it onto an Indian site for the demo.
- Fleet health numbers are Remaining Useful Life in engine cycles: ground <= 15, maintenance <= 40, watch <= 80.
- If a tool errors, say what failed and answer with what you have. Keep answers under 180 words unless asked for detail.
"""


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.exists(os.path.expanduser("~/.config/anthropic")))


class Assistant:
    def __init__(self, model: str | None = None, max_tokens: int = 4096, effort: str = "medium"):
        self.model = model or settings.assistant_model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic()
        self.messages: list[dict] = []
        self.transcript: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()
        self.transcript.clear()

    def ask(self, text: str, t_idx: int | None = None, context: dict | None = None) -> dict:
        ctx = dict(context or {})
        if t_idx is not None:
            ctx["current_snapshot_index"] = int(t_idx)
        content = f"[context: {json.dumps(ctx)}]\n{text}" if ctx else text
        self.messages.append({"role": "user", "content": content})
        t0 = time.time()
        tool_calls: list[dict] = []
        usage = dict(input_tokens=0, output_tokens=0)
        final_text = ""
        stop_reason = None
        params = dict(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT, tools=ALL_TOOLS,
            messages=self.messages, output_config={"effort": self.effort},
        )
        if self.model.startswith(("claude-opus-5", "claude-fable")):
            # server-side refusal fallbacks (routes a policy decline to another model inside the same call)
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            runner = self.client.beta.messages.tool_runner(**params)
            for message in runner:
                stop_reason = message.stop_reason
                usage["input_tokens"] += int(getattr(message.usage, "input_tokens", 0) or 0)
                usage["output_tokens"] += int(getattr(message.usage, "output_tokens", 0) or 0)
                self.messages.append({"role": "assistant", "content": message.content})
                for block in message.content:
                    if block.type == "tool_use":
                        tool_calls.append(dict(name=block.name, input=block.input))
                    elif block.type == "text" and block.text.strip():
                        final_text = block.text
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    self.messages.append(tool_response)
                    for i, blk in enumerate(tool_response.get("content", [])):
                        if isinstance(blk, dict) and blk.get("type") == "tool_result" and i < len(tool_calls):
                            preview = blk.get("content")
                            tool_calls[-len(tool_response["content"]) + i]["result_preview"] = (str(preview)[:300] if preview else "")
        except anthropic.AuthenticationError:
            self.messages.pop()
            return dict(error="Anthropic API key missing or invalid. Set ANTHROPIC_API_KEY in .env.", answer=None, tool_calls=[])
        except anthropic.RateLimitError:
            self.messages.pop()
            return dict(error="Rate limited by the Claude API, try again in a few seconds.", answer=None, tool_calls=[])
        except anthropic.APIStatusError as e:
            self.messages.pop()
            return dict(error=f"Claude API error {e.status_code}: {e.message}", answer=None, tool_calls=[])
        except anthropic.APIConnectionError:
            self.messages.pop()
            return dict(error="Could not reach the Claude API (network).", answer=None, tool_calls=[])
        if stop_reason == "refusal":
            final_text = final_text or "I can't help with that request."
        rec = dict(question=text, answer=final_text, tool_calls=tool_calls, stop_reason=stop_reason, usage=usage,
                   model=self.model, seconds=round(time.time() - t0, 1))
        self.transcript.append(rec)
        return rec


_singleton: Assistant | None = None


def get_assistant() -> Assistant:
    global _singleton
    if _singleton is None:
        _singleton = Assistant()
    return _singleton
