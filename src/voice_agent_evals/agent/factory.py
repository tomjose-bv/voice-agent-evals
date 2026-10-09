from __future__ import annotations

from ..backend import MockBackend
from ..config import Settings
from .base import Agent


def make_agent(kind: str, backend: MockBackend, settings: Settings | None = None) -> Agent:
    """Create an agent backend: "scripted" (default, no LLM) or "llm"."""
    kind = (kind or "scripted").lower()
    if kind == "scripted":
        from .scripted import ScriptedAgent

        return ScriptedAgent(backend)
    if kind == "llm":
        from .llm import LLMAgent

        return LLMAgent(backend, settings or Settings.load())
    raise ValueError(f"unknown agent backend {kind!r} (use 'scripted' or 'llm')")
