from __future__ import annotations

from ..backend import MockBackend
from ..config import Settings
from .base import Agent

#: Agent variants, from no model at all to a governed LLM graph.
AGENT_KINDS = ("scripted", "prompt", "context", "graph", "graph+grounding")
LLM_KINDS = ("prompt", "context", "graph", "graph+grounding", "llm")


def make_agent(kind: str, backend: MockBackend, settings: Settings | None = None) -> Agent:
    """Create an agent.

    - ``scripted``: deterministic rules, no model (default).
    - ``prompt`` (alias ``llm``): system prompt + tools + full raw history.
    - ``context``: context engineering (facts, state summary, trimmed history, relevant policy).
    - ``graph``: ``context`` plus a governed state machine that gates tools in code.
    - ``graph+grounding``: ``graph`` plus a pre-send grounding check on every reply.
    """
    kind = (kind or "scripted").lower()
    if kind == "scripted":
        from .scripted import ScriptedAgent

        return ScriptedAgent(backend)
    settings = settings or Settings.load()
    if kind in ("prompt", "llm"):
        from .llm import LLMAgent

        return LLMAgent(backend, settings)
    if kind == "context":
        from .context_agent import ContextAgent

        return ContextAgent(backend, settings)
    if kind == "graph":
        from .graph_agent import GraphAgent

        return GraphAgent(backend, settings)
    if kind == "graph+grounding":
        from .graph_agent import GroundedGraphAgent

        return GroundedGraphAgent(backend, settings)
    raise ValueError(f"unknown agent {kind!r} (use one of: {', '.join(AGENT_KINDS)})")
