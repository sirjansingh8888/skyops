"""Assistant provider switch. Default: Gemini (free tier). Optional: Claude.

    SKYOPS_ASSISTANT_PROVIDER=gemini   needs GEMINI_API_KEY   (pip install google-genai)
    SKYOPS_ASSISTANT_PROVIDER=claude   needs ANTHROPIC_API_KEY (pip install anthropic)
"""
from __future__ import annotations

from skyops.config import settings

_singleton = None


def provider() -> str:
    return (settings.assistant_provider or "gemini").lower()


def available() -> bool:
    if provider() == "claude":
        from skyops.assistant import agent

        return agent.available()
    from skyops.assistant import gemini_agent

    return gemini_agent.available()


def status() -> dict:
    p = provider()
    model = settings.claude_model if p == "claude" else settings.gemini_model
    key = "ANTHROPIC_API_KEY" if p == "claude" else "GEMINI_API_KEY"
    return dict(available=available(), provider=p, model=model, key_env=key)


def get_assistant():
    global _singleton
    if _singleton is None or _singleton.provider != provider():
        if provider() == "claude":
            from skyops.assistant.agent import ClaudeAssistant

            _singleton = ClaudeAssistant()
        else:
            from skyops.assistant.gemini_agent import GeminiAssistant

            _singleton = GeminiAssistant()
    return _singleton
