from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..tools import ToolCallRecord, ToolExecutor


@dataclass
class AgentTurn:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    escalated: bool = False
    ended: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    blocked: list[dict] = field(default_factory=list)  # tool calls refused by a guard
    nodes: list[str] = field(default_factory=list)  # graph nodes visited (graph variant)
    grounding: list[dict] = field(default_factory=list)  # drafts rejected by a grounding check


class Agent(Protocol):
    name: str
    tools: ToolExecutor
    ended: bool
    escalated: bool

    def greet(self) -> AgentTurn: ...

    def respond(self, user_text: str) -> AgentTurn: ...
