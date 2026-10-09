"""`graph` variant: the `context` agent run as a governed state machine (see ``graph.py``)."""

from __future__ import annotations

from typing import Any

from ..backend import MockBackend
from ..config import Settings
from ..tools import TOOL_SCHEMAS
from .base import AgentTurn
from .context import Facts, extract_facts
from .context_agent import ContextAgent
from .graph import ESCALATE, NODES, check_tool, escalation_trigger, pending_intents, route
from .llm import ChatClient

TRANSFER_LINE = "I'm transferring you to a colleague now."


class GraphAgent(ContextAgent):
    name = "graph"

    def __init__(self, backend: MockBackend, settings: Settings, client: ChatClient | None = None) -> None:
        super().__init__(backend, settings, client)
        self.node = "greet"
        self.trail: list[str] = ["greet"]  # every node visited, in order
        self.turn_nodes: list[str] = []
        self.forced: list[dict[str, Any]] = []  # actions the graph took itself
        self._facts_at_turn_start = Facts()

    # ---------------------------------------------------------------- routing
    def _route(self) -> str:
        facts = extract_facts(self.tools.log)
        node = route(self.state, facts, self._facts_at_turn_start, self.backend.credit_limit)
        if node != self.node:
            self.node = node
            self.trail.append(node)
        if not self.turn_nodes or self.turn_nodes[-1] != node:
            self.turn_nodes.append(node)
        return node

    def on_turn_start(self, user_text: str) -> None:
        super().on_turn_start(user_text)
        self._facts_at_turn_start = extract_facts(self.tools.log)
        self.turn_nodes = []

    def build_messages(self) -> list[dict[str, Any]]:
        self._route()  # re-evaluated before every model call, so tool results can move the node
        return super().build_messages()

    def tools_for_call(self) -> list[dict[str, Any]]:
        allowed = NODES[self.node].tools
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed]

    def extra_instructions(self) -> str:
        node = NODES[self.node]
        facts = extract_facts(self.tools.log)
        lines = [f"CURRENT STEP: {node.name}. {node.instructions}",
                 f"Tools available in this step: {', '.join(sorted(node.tools)) or 'none'}."]
        trigger = escalation_trigger(self.state, facts, self.backend.credit_limit)
        if trigger and node.name == "escalate":
            lines.append(f'Escalation reason: "{trigger}".')
        others = pending_intents(self.state, facts, self._facts_at_turn_start)[1:]
        if others:
            lines.append(f"Still to handle after this: {', '.join(others)}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ guard
    def guard(self, name: str, args: dict[str, Any]) -> str | None:
        return check_tool(NODES[self.node], name, args, self.state, extract_facts(self.tools.log),
                          self.backend.credit_limit)

    def on_turn_end(self, text: str) -> str:
        if self.node == "escalate" and not self.escalated:
            # The trigger is policy, not a suggestion: if the model did not escalate, the graph does.
            reason = escalation_trigger(self.state, extract_facts(self.tools.log), self.backend.credit_limit)
            acc = (self.account or {}).get("account_id")
            rec = self.tools.call(ESCALATE, {"reason": reason or "escalation trigger", **({"account_id": acc} if acc else {})})
            self.forced.append({"name": ESCALATE, "reason": reason})
            self.escalated = rec.ok
            text = TRANSFER_LINE
        return text

    def respond(self, user_text: str) -> AgentTurn:
        turn = super().respond(user_text)
        turn.nodes = list(self.turn_nodes)
        return turn
