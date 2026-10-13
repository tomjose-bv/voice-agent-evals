"""Pre-send grounding check for agent replies.

Before a reply is spoken, its factual statements are compared with what the
agent actually knows: the verified tool results (the same data as the
VERIFIED FACTS block), tool error messages (they can carry alternatives, e.g.
free technician dates), the caller's own words, today's date and the policy
constants. A statement that none of these support is a *violation*.

What counts as a factual statement (deliberately narrow, all rule-based):

- money amounts ("$45.00");
- explicit calendar dates ("16 March", "2026-03-16");
- outage status asserted as fact ("there's an outage", "no known outage").
  Conditional phrasing ("let me check if there's an outage") is not a claim;
- completed actions: a credit applied, a visit booked, a transfer started;
- account, booking and bill references (ACC-1234, TB-1234, B-2026-03).

This is the agent's own check, independent of the eval harness: it uses only
what the agent has seen in this call, never the scenario's ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from ..nlp import extract_amounts, extract_dates, fmt_money
from ..tools import ToolCallRecord
from .context import Facts, extract_facts, trim_tool_result

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_OUTAGE_POS_RE = re.compile(
    r"\b(there'?s|there is|we have|we'?re aware of|i can see|i see|i found|i've found) "
    r"(a |an )?(known |current |ongoing |active |confirmed )?outage"
    r"|\boutage (is )?(affecting|active|ongoing|in progress)",
    re.IGNORECASE,
)
_OUTAGE_NEG_RE = re.compile(
    r"\bno (known |current |reported |active )?outages?"
    r"|\b(isn'?t|is not|aren'?t|are not) (a |an |any )?(known |current )?outages?"
    r"|\bnot aware of any outage",
    re.IGNORECASE,
)
#: A status phrase preceded by one of these (in the same sentence) is a question, not a claim.
_CONDITIONAL_RE = re.compile(r"\b(if|whether|check|checking|see|find out|confirm)\b[^.!?]*$", re.IGNORECASE)
_CREDIT_RE = re.compile(
    r"\b(i'?ve|i have|we'?ve|we have) (now |just )?(applied|issued|added|credited|refunded)"
    r"|\bcredit (has been|was|is now) (applied|issued|added)",
    re.IGNORECASE,
)
_BOOKING_RE = re.compile(
    r"\b(you'?re|you are) (all )?booked|\b(i'?ve|i have) (now |just )?booked|\bbooking (is )?confirmed"
    r"|\bvisit (is|has been) (booked|scheduled|confirmed)",
    re.IGNORECASE,
)
_TRANSFER_RE = re.compile(
    r"\b(i'?m|i am) (now )?(transferring|connecting|putting) you|\b(i'?ll|i will|let me) (now )?(transfer|connect|put) you"
    r"|\bpassing you (over )?to",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(r"\b(ACC-\d{4}|TB-\d{4}|B-\d{4}-\d{2})\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Violation:
    kind: str  # amount | date | outage | action | reference
    claim: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "claim": self.claim, "detail": self.detail}


@dataclass
class Evidence:
    """Everything the agent may state in this call."""

    numbers: set[float]
    dates: set[date]
    text: str
    facts: Facts


def _numbers(text: str) -> set[float]:
    return {round(float(m), 2) for m in _NUM_RE.findall(text)}


def gather_evidence(log: Iterable[ToolCallRecord], caller_texts: Iterable[str], today: date,
                    constants: Iterable[float]) -> Evidence:
    log = list(log)
    numbers = {round(float(c), 2) for c in constants}
    dates = {today}
    parts: list[str] = []
    for rec in log:
        if rec.ok:
            blob = json.dumps(trim_tool_result(rec.name, rec.result))
        else:
            blob = str(rec.error or "")
        parts.append(blob)
        numbers |= _numbers(blob)
        dates |= set(extract_dates(blob, today))
    for text in caller_texts:
        parts.append(text)
        numbers |= _numbers(text) | {round(a, 2) for a in extract_amounts(text)}
        dates |= set(extract_dates(text, today, relative=True))
    return Evidence(numbers, dates, " ".join(parts).upper(), extract_facts(log))


def _asserted(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match of ``pattern`` that is stated as fact, not asked or conditional."""
    for m in pattern.finditer(text):
        start = max(text.rfind(".", 0, m.start()), text.rfind("?", 0, m.start()), text.rfind("!", 0, m.start()))
        if not _CONDITIONAL_RE.search(text[start + 1:m.start()]):
            return m
    return None


def check_reply(text: str, ev: Evidence, today: date, allow_transfer_claim: bool = False) -> list[Violation]:
    """Return the statements in ``text`` that the evidence does not support."""
    out: list[Violation] = []
    for amt in extract_amounts(text):
        if not any(abs(amt - n) < 0.005 for n in ev.numbers):
            out.append(Violation("amount", fmt_money(amt), "no tool result or caller statement contains this amount"))
    for d in extract_dates(text, today):
        if d not in ev.dates:
            out.append(Violation("date", d.isoformat(), "no tool result or caller statement contains this date"))
    pos, neg = _asserted(_OUTAGE_POS_RE, text), _asserted(_OUTAGE_NEG_RE, text)
    if pos or neg:
        said = (pos or neg).group(0)  # type: ignore[union-attr]
        if ev.facts.outage is None:
            out.append(Violation("outage", said, "outage status stated before check_outage was called"))
        elif bool(pos) and not neg and not ev.facts.outage.get("outage"):
            out.append(Violation("outage", said, "check_outage found no active outage"))
        elif neg and not pos and ev.facts.outage.get("outage"):
            out.append(Violation("outage", said, "check_outage found an active outage"))
    if (m := _CREDIT_RE.search(text)) and not ev.facts.credits:
        out.append(Violation("action", m.group(0), "no credit has been issued (issue_credit did not succeed)"))
    if (m := _BOOKING_RE.search(text)) and not ev.facts.bookings:
        out.append(Violation("action", m.group(0), "nothing has been booked (book_technician did not succeed)"))
    if not allow_transfer_claim and (m := _TRANSFER_RE.search(text)) and not ev.facts.escalated:
        out.append(Violation("action", m.group(0), "the call has not been escalated (escalate_to_human not called)"))
    for ident in _IDENT_RE.findall(text):
        if ident.upper() not in ev.text:
            out.append(Violation("reference", ident, "reference not found in tool results or caller statements"))
    return out


def revision_request(violations: list[Violation], tools: Iterable[str]) -> str:
    """Instruction for the model to rewrite a draft that failed the check."""
    lines = ["GROUNDING CHECK (automatic, not from the caller). Your draft reply stated things that are not in "
             "VERIFIED FACTS or the caller's words:"]
    lines += [f'- {v.kind} "{v.claim}": {v.detail}.' for v in violations]
    available = ", ".join(sorted(t for t in tools)) or "none"
    lines.append("Write the reply again without those statements. If you need one of those facts, call the tool "
                 f"that returns it (tools in this step: {available}) or tell the caller you cannot confirm it yet. "
                 "Do not mention this check.")
    return "\n".join(lines)


UNCONFIRMED = "I'm sorry, I can't confirm that detail right now."
SAFE_REPLY = f"{UNCONFIRMED} Is there anything else I can help with?"


def strip_unsupported(text: str, ev: Evidence, today: date, allow_transfer_claim: bool = False) -> str:
    """Safe fallback: drop every sentence that fails the check, and say what can't be confirmed."""
    sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s]
    kept = [s for s in sentences if not check_reply(s, ev, today, allow_transfer_claim)]
    if not kept:
        return SAFE_REPLY
    if len(kept) == len(sentences):
        return text
    return " ".join([*kept, UNCONFIRMED])


def _caller_texts(transcript: list[tuple[str, str]]) -> list[str]:
    return [t for role, t in transcript if role == "caller"]


def evidence_for(log: list[ToolCallRecord], transcript: list[tuple[str, str]], today: date,
                 credit_limit: float, outage_credit: float) -> Evidence:
    """Evidence for an agent: its tool log, the caller's turns and the policy constants."""
    return gather_evidence(log, _caller_texts(transcript), today, (credit_limit, outage_credit))

