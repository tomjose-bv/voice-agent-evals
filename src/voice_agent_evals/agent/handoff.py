"""Handoff summary passed to a human agent on escalation."""

from __future__ import annotations

from typing import Any

from ..tools import ToolCallRecord


def _describe_call(rec: ToolCallRecord) -> str:
    if not rec.ok:
        return f"{rec.name} failed: {rec.error}"
    r = rec.result or {}
    if rec.name == "lookup_account":
        return f"verified account {r.get('account_id')}" if r.get("found") else "account lookup: not found"
    if rec.name == "check_outage":
        return f"outage check {r.get('postcode')}: {'ACTIVE' if r.get('outage') else 'none'}"
    if rec.name == "get_bill":
        return f"viewed bill {r.get('bill_id')} total ${r.get('total', 0):.2f}"
    if rec.name == "issue_credit":
        return f"issued credit ${r.get('amount', 0):.2f} ({r.get('reason')})"
    if rec.name == "book_technician":
        return f"booked technician {r.get('date')} {r.get('slot')} ({r.get('booking_id')})"
    if rec.name == "escalate_to_human":
        return "escalated to human"
    return rec.name


def build_handoff_summary(
    reason: str,
    account: dict[str, Any] | None,
    transcript: list[tuple[str, str]],
    tool_log: list[ToolCallRecord],
) -> dict[str, Any]:
    caller_lines = [t for role, t in transcript if role == "caller"]
    return {
        "reason": reason,
        "account_id": (account or {}).get("account_id"),
        "customer_name": (account or {}).get("name"),
        "verified": bool(account),
        "actions": [_describe_call(r) for r in tool_log if r.name != "escalate_to_human"],
        "recent_caller_statements": caller_lines[-3:],
        "turns": len(caller_lines),
    }


def handoff_text(summary: dict[str, Any]) -> str:
    who = summary.get("customer_name") or "Unverified caller"
    acc = summary.get("account_id") or "no account"
    lines = [f"Escalation: {summary.get('reason')}", f"Customer: {who} ({acc})"]
    if summary.get("actions"):
        lines.append("Done so far: " + "; ".join(summary["actions"]))
    if summary.get("recent_caller_statements"):
        lines.append("Caller said: " + " | ".join(summary["recent_caller_statements"]))
    return "\n".join(lines)
