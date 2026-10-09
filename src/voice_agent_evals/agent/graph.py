"""Governed agent graph: an explicit state machine around the LLM.

The LLM still writes every reply and chooses tool calls, but only *within the
current node*. The graph decides, in code:

- which node the call is in (``route``), from the conversation state and the
  verified facts;
- which tools the node offers, and whether a tool call is allowed right now
  (``check_tool``). A refused call never reaches the backend; the LLM gets a
  ``POLICY:`` error back and the refusal is counted as a blocked action;
- when to escalate (``escalation_trigger``). ``escalate`` is reachable from
  every node, but only when one of the defined triggers applies.

Plain Python on purpose: the whole graph fits on one screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..backend import OUTAGE_CREDIT_AMOUNT
from .context import ConversationState, Facts

ACCOUNT_TOOLS = {"get_bill", "issue_credit", "book_technician"}
ESCALATE = "escalate_to_human"


@dataclass(frozen=True)
class Node:
    name: str
    tools: frozenset[str]
    instructions: str


def _node(name: str, tools: set[str], instructions: str) -> Node:
    return Node(name, frozenset(tools | {ESCALATE}) if name != "wrap_up" else frozenset(tools), instructions)


NODES: dict[str, Node] = {n.name: n for n in [
    _node("understand_intent", {"lookup_account", "check_outage"},
          "Find out what the caller needs: a bill question, an outage, a technician visit or a credit. "
          "Answer honestly if asked whether you are an AI. Politely decline off-topic requests."),
    _node("verify_identity", {"lookup_account"},
          "The caller's request needs their account. Ask for the phone number or email on the account and "
          "call lookup_account with it. Account tools are locked until identity is verified."),
    _node("resolve_billing", {"get_bill", "issue_credit"},
          "Call get_bill for the verified account and answer from it: total, due date and any line item the "
          "caller asks about. Issue a credit only for a duplicate charge visible on the bill or a confirmed "
          "outage, never above the limit."),
    _node("resolve_outage", {"check_outage"},
          "Call check_outage with the caller's postcode (ask for it if you don't have it) and tell the caller "
          "the status and expected fix time exactly as returned."),
    _node("resolve_technician", {"book_technician"},
          "Agree a date with the caller and call book_technician (YYYY-MM-DD). If the date is refused, offer "
          "the alternatives from the error message."),
    _node("confirm", set(),
          "Briefly confirm what has been done and ask whether there is anything else."),
    _node("wrap_up", set(), "The caller is finishing. Say a short, friendly goodbye. Do not call tools."),
    _node("escalate", set(),
          "An escalation trigger applies. Call escalate_to_human now with the reason below, then tell the "
          "caller you are transferring them to a colleague."),
]}

INTENT_NODE = {"billing": "resolve_billing", "credit": "resolve_billing",
               "outage": "resolve_outage", "technician": "resolve_technician"}
NEEDS_ACCOUNT = {"billing", "credit", "technician"}

#: Edges of the graph, for documentation and tests (escalate is reachable from all).
EDGES: dict[str, set[str]] = {
    "greet": {"understand_intent"},
    "understand_intent": {"verify_identity", "resolve_outage", "resolve_billing", "resolve_technician", "wrap_up"},
    "verify_identity": {"resolve_billing", "resolve_technician"},
    "resolve_billing": {"resolve_outage", "resolve_technician", "confirm", "wrap_up"},
    "resolve_outage": {"verify_identity", "resolve_billing", "resolve_technician", "confirm", "wrap_up"},
    "resolve_technician": {"verify_identity", "resolve_billing", "resolve_outage", "confirm", "wrap_up"},
    "confirm": {"verify_identity", "resolve_billing", "resolve_outage", "resolve_technician", "wrap_up"},
    "wrap_up": set(),
    "escalate": set(),
}


def escalation_trigger(state: ConversationState, facts: Facts, credit_limit: float) -> str | None:
    """The defined escalation triggers, checked in code. None if no trigger applies."""
    if state.human_requested:
        return "caller asked for a human"
    if state.legal_threat:
        return "caller mentioned legal action"
    if state.requested_credit is not None and state.requested_credit > credit_limit:
        return f"credit of ${state.requested_credit:.2f} requested, above the ${credit_limit:.2f} limit"
    if facts.account is None and facts.failed_lookups >= 2:
        return "identity could not be verified"
    return None


def pending_intents(state: ConversationState, facts: Facts, facts_at_turn_start: Facts) -> list[str]:
    """Intents not yet resolved. Lookups count as resolving only from the next turn,
    so the model can still act on what it just learned in this turn."""
    done = {
        "outage": facts_at_turn_start.outage is not None,
        "billing": facts_at_turn_start.bill is not None,
        "credit": bool(facts.credits),
        "technician": bool(facts.bookings),
    }
    return [i for i in state.intents if not done.get(i, True)]


def route(state: ConversationState, facts: Facts, facts_at_turn_start: Facts, credit_limit: float) -> str:
    if facts.escalated:
        return "wrap_up"
    if escalation_trigger(state, facts, credit_limit):
        return "escalate"
    pending = pending_intents(state, facts, facts_at_turn_start)
    if state.wants_to_end:
        return "wrap_up"
    if not pending:
        return "confirm" if state.intents else "understand_intent"
    intent = pending[0]
    if intent in NEEDS_ACCOUNT and facts.account is None:
        return "verify_identity"
    return INTENT_NODE[intent]


_DIGITS = re.compile(r"\d")


def _credit_reason_ok(amount: float, facts: Facts) -> str | None:
    """Return an error unless the bill or an outage justifies this credit."""
    dup = [i["amount"] for i in (facts.bill or {}).get("items", []) if "duplicate" in i["description"].lower()]
    if dup and amount <= max(dup) + 0.005:
        return None
    if facts.outage and facts.outage.get("outage") and amount <= OUTAGE_CREDIT_AMOUNT + 0.005:
        return None
    if not facts.bill and not facts.outage:
        return "no qualifying reason yet: check the bill (get_bill) or the outage status first"
    return (f"no qualifying reason for a ${amount:.2f} credit: only a duplicate charge on the bill or a "
            f"confirmed outage (flat ${OUTAGE_CREDIT_AMOUNT:.2f}) qualifies")


def check_tool(node: Node, name: str, args: dict[str, Any], state: ConversationState, facts: Facts,
               credit_limit: float) -> str | None:
    """Return a policy error if ``name(args)`` is not allowed now, else None."""
    if name not in node.tools:
        allowed = ", ".join(sorted(node.tools)) or "none"
        return f"{name} is not allowed in step '{node.name}' (allowed: {allowed})"
    if name == ESCALATE:
        if escalation_trigger(state, facts, credit_limit) is None:
            return ("no escalation trigger applies (human requested, legal threat, credit above limit, identity "
                    "failed twice); keep helping the caller")
        return None
    if name == "lookup_account":
        phone, email = str(args.get("phone") or ""), str(args.get("email") or "")
        if len(_DIGITS.findall(phone)) < 7 and "@" not in email:
            given = state.caller_details.get("phone") or state.caller_details.get("email")
            hint = f"; the caller said {given}" if given else "; ask the caller for it"
            return f"lookup_account needs the caller's real phone number or email{hint}"
        return None
    if name in ACCOUNT_TOOLS:
        if facts.account is None:
            return "identity not verified: call lookup_account first"
        acc = str(args.get("account_id") or "").upper()
        if acc != facts.account["account_id"]:
            return f"only the caller's own verified account ({facts.account['account_id']}) may be used"
    if name == "issue_credit":
        try:
            amount = float(args.get("amount"))
        except (TypeError, ValueError):
            return "amount must be a number"
        if facts.credits:
            return "a credit has already been issued on this call"
        if amount > credit_limit:
            return f"${amount:.2f} is above the ${credit_limit:.2f} limit: escalate instead of a partial credit"
        return _credit_reason_ok(amount, facts)
    return None
