"""Agent backends. All share the same interface: greet() and respond(text)."""

from .base import Agent, AgentTurn
from .factory import make_agent

__all__ = ["Agent", "AgentTurn", "make_agent"]
