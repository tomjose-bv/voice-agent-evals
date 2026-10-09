"""Tool definitions (OpenAI function-calling format) and a logging executor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .backend import MockBackend, ToolError

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_account",
            "description": "Find and verify the caller's account by the phone number or email address on file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Phone number, e.g. 555-0101"},
                    "email": {"type": "string", "description": "Email address on the account"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_outage",
            "description": "Check whether there is a known network outage for a postcode.",
            "parameters": {
                "type": "object",
                "properties": {"postcode": {"type": "string"}},
                "required": ["postcode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bill",
            "description": "Get the latest bill (total, due date, line items) for a verified account.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_credit",
            "description": "Apply a credit to a verified account. Amounts above the agent limit are rejected; escalate instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["account_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_technician",
            "description": "Book a technician visit (morning slot) for a verified account on a date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["account_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand the call to a human agent. Ends the automated conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "account_id": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "ok": self.ok, "result": self.result, "error": self.error}


@dataclass
class ToolExecutor:
    """Executes tool calls against a backend and keeps an ordered log."""

    backend: MockBackend
    log: list[ToolCallRecord] = field(default_factory=list)

    def call(self, name: str, args: dict[str, Any] | str | None = None) -> ToolCallRecord:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                rec = ToolCallRecord(name, {"_raw": args}, False, error="arguments were not valid JSON")
                self.log.append(rec)
                return rec
        args = dict(args or {})
        b = self.backend
        try:
            if name == "lookup_account":
                result = b.lookup_account(phone=args.get("phone"), email=args.get("email"))
            elif name == "check_outage":
                result = b.check_outage(args.get("postcode", ""))
            elif name == "get_bill":
                result = b.get_bill(args.get("account_id", ""))
            elif name == "issue_credit":
                result = b.issue_credit(args.get("account_id", ""), args.get("amount"), args.get("reason", ""))
            elif name == "book_technician":
                result = b.book_technician(args.get("account_id", ""), args.get("date", ""))
            elif name == "escalate_to_human":
                result = b.escalate_to_human(args.get("reason", ""), args.get("account_id"))
            else:
                raise ToolError(f"unknown tool {name!r}")
            rec = ToolCallRecord(name, args, True, result=result)
        except ToolError as exc:
            rec = ToolCallRecord(name, args, False, error=str(exc))
        self.log.append(rec)
        return rec
