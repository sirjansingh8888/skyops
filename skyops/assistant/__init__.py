"""Assistant provider switch. Default: Gemini (free tier). Optional: Claude. Always available: offline rules.

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
    ok = available()
    return dict(available=ok, provider=p if ok else "offline", model=model if ok else "rule-based (no LLM)", key_env=key,
                mode="llm" if ok else "offline", configured_provider=p)


def get_assistant():
    """The configured LLM assistant, or the rule-based offline assistant when its API key is missing."""
    global _singleton
    want = provider() if available() else "offline"
    if _singleton is None or _singleton.provider != want:
        if want == "offline":
            from skyops.assistant.offline import OfflineAssistant

            _singleton = OfflineAssistant()
        elif want == "claude":
            from skyops.assistant.agent import ClaudeAssistant

            _singleton = ClaudeAssistant()
        else:
            from skyops.assistant.gemini_agent import GeminiAssistant

            _singleton = GeminiAssistant()
    return _singleton
