"""SkyOps operations assistant on the Gemini API (free tier friendly).

Uses the google-genai SDK's Interactions API in *stateless* mode, exactly as documented at
https://ai.google.dev/gemini-api/docs/function-calling : the conversation history is kept on our side
(`store=False`), every model step is appended back unchanged, and tool results are returned as
`function_result` steps until the model answers in text.

Latency is the enemy of a live demo, so every call is bounded:
  * the SDK's own retries (up to 4, with backoff) are switched off;
  * each call has a hard timeout (SKYOPS_GEMINI_TIMEOUT_S, default 25 s);
  * on a timeout, 429 or 5xx the rest of the question runs on the lighter fallback model;
  * if that fails too, the rule-based offline assistant answers from the same tools.
Measured on the free tier (Sept 2026): gemini-3.6-flash answers a two-round tool question in about 10 s and
gemini-3.5-flash-lite in about 4 s, while gemini-3.7/3.8-flash were overloaded (90-160 s or timeouts).

The key is read by the SDK from GEMINI_API_KEY (or GOOGLE_API_KEY); put it in skyops/.env.
"""
from __future__ import annotations

import json
import os
import time

from skyops.assistant.prompts import SYSTEM_PROMPT
from skyops.assistant.tools import declarations, run_tool
from skyops.config import settings

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _error_code(e: Exception) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


def _is_retryable(e: Exception) -> bool:
    return _error_code(e) in RETRYABLE_CODES or "timeout" in type(e).__name__.lower() or isinstance(e, (TimeoutError, ConnectionError))


class GeminiAssistant:
    provider = "gemini"

    def __init__(self, model: str | None = None, max_tool_rounds: int = 8):
        from google import genai

        self.client = genai.Client()
        self._disable_sdk_retries()
        self.model = model or settings.gemini_model
        self.fallback_model = settings.gemini_fallback_model
        self.timeout_s = float(settings.gemini_timeout_s)
        self.cooldown_s = 300.0          # how long to avoid the main model after it proved slow
        self._main_slow_until = 0.0
        self.max_tool_rounds = max_tool_rounds
        self.tools = declarations()
        self.history: list[dict] = []
        self.transcript: list[dict] = []

    def _disable_sdk_retries(self) -> None:
        """The generated client retries 408/409/429/5xx up to four times with backoff, which turned one slow
        call into minutes. We do our own single fallback instead. Private API, so failure here is non-fatal."""
        try:
            from google.genai._gaos import utils

            self.client.interactions.sdk_configuration.retry_config = utils.RetryConfig("none", None, False)
        except Exception:  # noqa: BLE001
            pass

    def reset(self) -> None:
        self.history.clear()
        self.transcript.clear()

    @staticmethod
    def _offline(text: str, t_idx, context, why: str) -> dict:
        """Keep the demo alive: answer from the same tools with templates when the LLM cannot be reached."""
        from skyops.assistant.offline import OfflineAssistant

        return OfflineAssistant().ask(text, t_idx=t_idx, context=context, note=f"{why}. Offline answer from the live tools:")

    def _create(self, model: str):
        return self.client.interactions.create(model=model, store=False, input=self.history, tools=self.tools,
                                               system_instruction=SYSTEM_PROMPT, generation_config={"thinking_level": "low"},
                                               timeout=self.timeout_s)

    def ask(self, text: str, t_idx: int | None = None, context: dict | None = None) -> dict:
        ctx = dict(context or {})
        if t_idx is not None:
            ctx["current_snapshot_index"] = int(t_idx)
        content = f"[context: {json.dumps(ctx)}]\n{text}" if ctx else text
        checkpoint = len(self.history)
        self.history.append({"type": "user_input", "content": [{"type": "text", "text": content}]})
        # circuit breaker: after the main model times out or rate-limits, skip it for a few minutes
        main_ok = time.time() >= self._main_slow_until or not self.fallback_model
        t0, tool_calls, answer, model_used = time.time(), [], "", (self.model if main_ok else self.fallback_model)
        try:
            for _ in range(self.max_tool_rounds):
                try:
                    interaction = self._create(model_used)
                except Exception as e:  # noqa: BLE001
                    # slow / overloaded / rate-limited main model: finish this question on the lighter fallback
                    if _is_retryable(e) and self.fallback_model and model_used != self.fallback_model:
                        self._main_slow_until = time.time() + self.cooldown_s
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
        except Exception as e:  # noqa: BLE001
            del self.history[checkpoint:]
            code = _error_code(e)
            if code == 429:
                why = "Gemini free-tier rate limit reached (limits reset per minute and per day)"
            elif code in (400, 401, 403):
                why = f"Gemini rejected the request ({code}): check GEMINI_API_KEY in skyops/.env"
            elif "timeout" in type(e).__name__.lower():
                why = f"Gemini did not answer within {self.timeout_s:.0f} s"
            elif code:
                why = f"Gemini API error {code}"
            else:
                why = f"Could not reach the Gemini API ({type(e).__name__})"
            return self._offline(text, t_idx, context, why)
        rec = dict(question=text, answer=answer, tool_calls=tool_calls, provider=self.provider, model=model_used,
                   seconds=round(time.time() - t0, 1))
        self.transcript.append(rec)
        return rec
