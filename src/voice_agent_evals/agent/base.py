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


class Agent(Protocol):
    name: str
    tools: ToolExecutor
    ended: bool
    escalated: bool

    def greet(self) -> AgentTurn: ...

    def respond(self, user_text: str) -> AgentTurn: ...
