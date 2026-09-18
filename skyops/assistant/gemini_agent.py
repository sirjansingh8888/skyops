"""SkyOps operations assistant on the Gemini API (free tier friendly).

Uses the google-genai SDK's Interactions API in *stateless* mode, exactly as documented at
https://ai.google.dev/gemini-api/docs/function-calling : the conversation history is kept on our side
(`store=False`), every model step is appended back unchanged, and tool results are returned as
`function_result` steps until the model answers in text.

The key is read by the SDK from GEMINI_API_KEY (or GOOGLE_API_KEY); put it in skyops/.env.
"""
from __future__ import annotations

import json
import os
import time

from skyops.assistant.prompts import SYSTEM_PROMPT
from skyops.assistant.tools import declarations, run_tool
from skyops.config import settings


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


class GeminiAssistant:
    provider = "gemini"

    def __init__(self, model: str | None = None, max_tool_rounds: int = 8):
        from google import genai

        self.client = genai.Client()
        self.model = model or settings.gemini_model
        self.fallback_model = settings.gemini_fallback_model
        self.max_tool_rounds = max_tool_rounds
        self.tools = declarations()
        self.history: list[dict] = []
        self.transcript: list[dict] = []

    def reset(self) -> None:
        self.history.clear()
        self.transcript.clear()

    def _create(self, model: str):
        return self.client.interactions.create(model=model, store=False, input=self.history, tools=self.tools,
                                               system_instruction=SYSTEM_PROMPT, generation_config={"thinking_level": "low"})

    def ask(self, text: str, t_idx: int | None = None, context: dict | None = None) -> dict:
        from google.genai import errors

        ctx = dict(context or {})
        if t_idx is not None:
            ctx["current_snapshot_index"] = int(t_idx)
        content = f"[context: {json.dumps(ctx)}]\n{text}" if ctx else text
        checkpoint = len(self.history)
        self.history.append({"type": "user_input", "content": [{"type": "text", "text": content}]})
        t0, tool_calls, answer, model_used = time.time(), [], "", self.model
        try:
            for _ in range(self.max_tool_rounds):
                try:
                    interaction = self._create(model_used)
                except errors.APIError as e:
                    # free-tier quota or a temporarily overloaded model: try the lighter fallback once
                    if getattr(e, "code", None) in (429, 503) and model_used != self.fallback_model and self.fallback_model:
                        model_used = self.fallback_model
                        interaction = self._create(model_used)
                    else:
                        raise
                calls = []
                for step in interaction.steps:
                    self.history.append(step.model_dump())
                    if step.type == "function_call":
                        calls.append(step)
                if not calls:
                    answer = interaction.output_text or ""
                    break
                for fc in calls:
                    result = run_tool(fc.name, dict(fc.arguments or {}))
                    tool_calls.append(dict(name=fc.name, input=dict(fc.arguments or {}), result_preview=result[:300]))
                    self.history.append({"type": "function_result", "name": fc.name, "call_id": fc.id,
                                         "result": [{"type": "text", "text": result}]})
            else:
                answer = "I hit the tool-call limit before finishing. Try a narrower question."
        except errors.APIError as e:
            del self.history[checkpoint:]
            code = getattr(e, "code", None)
            if code == 429:
                msg = "Gemini free-tier rate limit reached. Wait a minute and try again (limits reset per minute and per day)."
            elif code in (400, 401, 403):
                msg = f"Gemini rejected the request ({code}). Check GEMINI_API_KEY in skyops/.env. Details: {getattr(e, 'message', e)}"
            else:
                msg = f"Gemini API error {code}: {getattr(e, 'message', e)}"
            return dict(error=msg, answer=None, tool_calls=tool_calls)
        except Exception as e:  # noqa: BLE001 - network problems and the like
            del self.history[checkpoint:]
            return dict(error=f"Could not reach the Gemini API: {type(e).__name__}: {e}", answer=None, tool_calls=tool_calls)
        rec = dict(question=text, answer=answer, tool_calls=tool_calls, provider=self.provider, model=model_used,
                   seconds=round(time.time() - t0, 1))
        self.transcript.append(rec)
        return rec
