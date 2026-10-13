"""`graph` variant: the `context` agent run as a governed state machine (see ``graph.py``).

`graph+grounding` adds a pre-send grounding check to the response step (see ``grounding.py``).
"""

from __future__ import annotations

from typing import Any

from ..backend import OUTAGE_CREDIT_AMOUNT, MockBackend
from ..config import Settings
from ..tools import TOOL_SCHEMAS
from .base import AgentTurn
from .context import Facts, extract_facts
from .context_agent import ContextAgent
from .graph import ESCALATE, NODES, check_tool, escalation_trigger, pending_intents, route
from .grounding import Evidence, Violation, check_reply, evidence_for, revision_request, strip_unsupported
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


GROUNDING_INSTRUCTIONS = (
    "GROUNDING: state amounts, dates, outage status, bookings, credits and references only if they appear "
    "under VERIFIED FACTS or in the caller's words. If a fact you need is missing, call this step's tool for it "
    "or say you cannot confirm it yet. Never fill in a value yourself."
)


class GroundedGraphAgent(GraphAgent):
    """`graph+grounding`: the `graph` agent plus a pre-send grounding check in the response step.

    Every draft reply is checked against the verified tool results and the caller's words
    (``grounding.check_reply``). An unsupported draft is sent back to the model once with the
    specific violations (tools stay available, so it can fetch the missing fact). If the rewrite
    still fails, the unsupported sentences are dropped and replaced by a plain "can't confirm".
    """

    name = "graph+grounding"

    def __init__(self, backend: MockBackend, settings: Settings, client: ChatClient | None = None) -> None:
        super().__init__(backend, settings, client)
        self.grounding: list[dict[str, Any]] = []  # one record per rejected draft
        self._turn_grounding: list[dict[str, Any]] = []

    def on_turn_start(self, user_text: str) -> None:
        super().on_turn_start(user_text)
        self._turn_grounding = []

    def extra_instructions(self) -> str:
        return f"{super().extra_instructions()}\n{GROUNDING_INSTRUCTIONS}"

    def _evidence(self) -> Evidence:
        return evidence_for(self.tools.log, self.transcript, self.backend.today, self.backend.credit_limit,
                            OUTAGE_CREDIT_AMOUNT)

    def _violations(self, text: str) -> list[Violation]:
        # In the escalate step the graph guarantees the transfer (see on_turn_end), so saying so is grounded.
        return check_reply(text, self._evidence(), self.backend.today, allow_transfer_claim=self.node == "escalate")

    def review_reply(self, text: str) -> str | None:
        violations = self._violations(text)
        if not violations:
            return None
        rec = {"node": self.node, "draft": text, "violations": [v.to_dict() for v in violations], "action": "revise"}
        self.grounding.append(rec)
        self._turn_grounding.append(rec)
        return revision_request(violations, NODES[self.node].tools)

    def fallback_reply(self, text: str) -> str:
        safe = strip_unsupported(text, self._evidence(), self.backend.today,
                                 allow_transfer_claim=self.node == "escalate")
        if self._turn_grounding:
            self._turn_grounding[-1]["action"] = "fallback"
            self._turn_grounding[-1]["sent"] = safe
        return safe

    def respond(self, user_text: str) -> AgentTurn:
        turn = super().respond(user_text)
        turn.grounding = list(self._turn_grounding)
        return turn
