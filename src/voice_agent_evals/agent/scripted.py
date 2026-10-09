"""Deterministic, rule-based "scripted policy" agent.

It follows the same rules as the LLM system prompt (see ``prompt.py``) using
keyword/regex intent detection and a small state machine. It exists so the
demo, the eval suite and the tests run on any machine with no model at all.

Measuring it tells you how the *harness* behaves and gives a reproducible
baseline. It is not a measure of LLM quality.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..backend import OUTAGE_CREDIT_AMOUNT, MockBackend, normalise_postcode
from ..nlp import extract_amounts, extract_dates, extract_email, extract_phone, fmt_date, fmt_money
from ..tools import ToolExecutor
from .base import AgentTurn
from .handoff import build_handoff_summary
from .prompt import GREETING

INJECTION_RE = re.compile(
    r"ignore (all |any |your |the )?(previous|prior|above|earlier)?\s*(instructions|rules|prompt)"
    r"|system prompt|developer mode|you are now|new instructions|disregard (your|the|all)"
    r"|pretend (to be|you are)|jailbreak|act as (an?|the) (admin|administrator|supervisor|developer)"
    r"|override (your|the) (rules|limit)",
    re.IGNORECASE,
)
LEGAL_RE = re.compile(
    r"\b(lawyer|attorney|solicitor|sue|suing|lawsuit|legal action|take you to court|court|small claims"
    r"|ombudsman|regulator|trading standards)\b",
    re.IGNORECASE,
)
HUMAN_RE = re.compile(
    r"(speak|talk|put me through|transfer me|connect me|get me|want|need|give me|let me)\b[^.?!]{0,30}?"
    r"\b(human|person|real person|someone real|manager|supervisor|representative|operator|colleague)\b"
    r"|^\W*(human|agent|representative|operator)\W*$",
    re.IGNORECASE,
)
ARE_YOU_AI_RE = re.compile(r"\bare you (a |an )?(real|human|person|robot|bot|ai|machine|computer)\b", re.IGNORECASE)
BYE_RE = re.compile(
    r"\b(bye|goodbye|that'?s all|that is all|that'?s it|nothing else|no,? thanks|no thank you|all good|i'?m done)\b",
    re.IGNORECASE,
)
THANKS_RE = re.compile(r"\b(thanks|thank you|cheers|great|perfect|brilliant)\b", re.IGNORECASE)
YES_RE = re.compile(r"^\W*(yes|yeah|yep|sure|please|ok|okay|go ahead)\b", re.IGNORECASE)
ANGRY_RE = re.compile(
    r"(!!|ridiculous|unacceptable|furious|useless|disgrace|joke|fed up|sick of|worst|angry|damn|bloody)",
    re.IGNORECASE,
)
AMBIGUOUS_RE = re.compile(r"\b(service|account|problem|issue|something'?s wrong|not right|help|sort)\b", re.IGNORECASE)

INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "outage": re.compile(
        r"(internet|broadband|wi-?fi|connection|line|service)\b[^.?!]{0,20}?\b(down|out|not working|dead|"
        r"dropp\w*|stopped|offline|gone)|outage|no internet|offline|can'?t get online|cannot connect|"
        r"no connection",
        re.IGNORECASE,
    ),
    "billing": re.compile(
        r"\b(bill|bills|billed|billing|charged?|charges|charging|invoice|overcharg\w*|payment|fee|statement)\b",
        re.IGNORECASE,
    ),
    "technician": re.compile(
        r"technician|engineer|someone (to )?(come|visit|out)|send someone|home visit|appointment|book a visit",
        re.IGNORECASE,
    ),
    "credit": re.compile(r"\b(credit|refund|compensat\w*|money back|reimburs\w*)\b", re.IGNORECASE),
}
DISPUTE_RE = re.compile(
    r"charged twice|double|twice|duplicate|overcharg|wrong|too high|dispute|incorrect|shouldn'?t|extra|"
    r"don'?t recognise|don'?t recognize|why is",
    re.IGNORECASE,
)
NEEDS_ACCOUNT = {"billing", "technician", "credit"}


class ScriptedAgent:
    name = "scripted"

    def __init__(self, backend: MockBackend) -> None:
        self.backend = backend
        self.tools = ToolExecutor(backend)
        self.account: dict[str, Any] | None = None
        self.failed_lookups = 0
        self.intents: list[str] = []
        self.dispute = False
        self.requested_credit: float | None = None
        self.awaiting: str | None = None
        self.offered_transfer = False
        self.last_outage: dict[str, Any] | None = None
        self.known_postcode: str | None = None
        self.known_dates: list[date] = []
        self.ended = False
        self.escalated = False
        self.handoff: dict[str, Any] | None = None
        self.transcript: list[tuple[str, str]] = []

    @property
    def today(self) -> date:
        return self.backend.today

    # ---------------------------------------------------------------- public
    def greet(self) -> AgentTurn:
        self.transcript.append(("agent", GREETING))
        return AgentTurn(GREETING)

    def respond(self, user_text: str) -> AgentTurn:
        start = len(self.tools.log)
        if self.ended:
            text = "This call has ended. Thank you for calling Northwind Internet."
        else:
            self.transcript.append(("caller", user_text))
            text = self._policy(user_text)
            self.transcript.append(("agent", text))
        return AgentTurn(
            text=text,
            tool_calls=self.tools.log[start:],
            escalated=self.escalated,
            ended=self.ended,
        )

    # ---------------------------------------------------------------- policy
    def _policy(self, text: str) -> str:
        low = text.lower()

        if INJECTION_RE.search(text):
            return (
                "I can't change how I work or take actions outside your own verified account. "
                "I'm happy to help with your Northwind bill, an outage, a technician visit or a credit. "
                "What do you need?"
            )
        if LEGAL_RE.search(text):
            return self._escalate("caller mentioned legal action")
        if HUMAN_RE.search(text) and not ARE_YOU_AI_RE.search(text):
            return self._escalate("caller asked for a human agent")
        if self.offered_transfer and YES_RE.search(text):
            return self._escalate("caller wants to dispute a charge the assistant cannot credit")

        parts: list[str] = []
        if ANGRY_RE.search(text) or (sum(c.isupper() for c in text) > 12 and text.upper() == text):
            parts.append("I'm really sorry for the trouble, let's get this sorted.")
        if ARE_YOU_AI_RE.search(text):
            parts.append("I'm an automated assistant, not a person. I can transfer you to a colleague any time.")

        # --- intents -----------------------------------------------------
        found = sorted(
            ((m.start(), name) for name, pat in INTENT_PATTERNS.items() if (m := pat.search(text))),
        )
        for _, name in found:
            if name not in self.intents:
                self.intents.append(name)
        if DISPUTE_RE.search(text) and ("billing" in self.intents or "credit" in self.intents):
            self.dispute = True
        amounts = extract_amounts(text)
        if "credit" in self.intents and amounts:
            self.requested_credit = max(amounts)

        # --- identification ---------------------------------------------
        consumed = False
        phone, email = extract_phone(text), extract_email(text)
        if self.account is None and (phone or email):
            consumed = True
            rec = self.tools.call("lookup_account", {"phone": phone} if phone else {"email": email})
            if rec.ok and rec.result and rec.result.get("found"):
                self.account = rec.result
                self.awaiting = None if self.awaiting == "identifier" else self.awaiting
                first = str(rec.result["name"]).split()[0]
                parts.append(f"Thanks {first}, I've found your account.")
            else:
                self.failed_lookups += 1
                if self.failed_lookups >= 2:
                    return self._escalate("identity could not be verified")
                parts.append(
                    "I couldn't find an account with those details. Could you check the phone number, "
                    "or give me the email address on the account?"
                )
                self.awaiting = "identifier"
                return " ".join(parts)

        # Credit above the limit: escalate, never issue a partial credit.
        if self.requested_credit is not None and self.requested_credit > self.backend.credit_limit:
            return self._escalate(
                f"caller requested a {fmt_money(self.requested_credit)} credit, above the "
                f"{fmt_money(self.backend.credit_limit)} limit"
            )

        # Remember details mentioned before we could act on them
        # (e.g. "tomorrow" said before identity verification).
        postcode = normalise_postcode(text)
        if postcode:
            self.known_postcode = postcode
        postcode = postcode or self.known_postcode
        new_dates = extract_dates(text, self.today, relative=True)
        if new_dates:
            self.known_dates = new_dates
        dates = new_dates or self.known_dates
        if postcode and self.awaiting == "postcode":
            consumed = True
        if dates and self.awaiting == "date":
            consumed = True

        # --- work through pending intents ---------------------------------
        handled_any = False
        while self.intents:
            intent = self.intents[0]
            if intent in NEEDS_ACCOUNT and self.account is None:
                self.awaiting = "identifier"
                parts.append(
                    "I can help with that. First, what's the phone number or email address on your account?"
                )
                return " ".join(parts)
            done, msg = self._handle(intent, postcode=postcode, dates=dates)
            if msg:
                parts.append(msg)
            if self.ended:
                return msg
            if not done:
                return " ".join(parts)
            self.intents.pop(0)
            handled_any = True

        if handled_any:
            parts.append("Is there anything else I can help with?")
            return " ".join(parts)
        if consumed:
            return " ".join(parts) + " What can I help you with today?"

        # --- nothing actionable --------------------------------------------
        if BYE_RE.search(low) or (THANKS_RE.search(low) and len(low.split()) <= 4):
            self.ended = True
            return "Thanks for calling Northwind Internet. Goodbye!"
        if parts and ARE_YOU_AI_RE.search(text):
            return " ".join(parts) + " What can I help you with?"
        if AMBIGUOUS_RE.search(text):
            parts.append(
                "Could you tell me a bit more? Is it about your bill, your internet connection, "
                "or booking a technician?"
            )
            return " ".join(parts)
        parts.append(
            "Sorry, I can only help with Northwind Internet accounts: bills, outages, "
            "technician visits and credits. What can I help with?"
        )
        return " ".join(parts)

    # -------------------------------------------------------------- handlers
    def _handle(self, intent: str, postcode: str | None, dates: list[date]) -> tuple[bool, str]:
        if intent == "outage":
            return self._handle_outage(postcode)
        if intent == "billing":
            return self._handle_billing()
        if intent == "credit":
            return self._handle_credit()
        if intent == "technician":
            return self._handle_technician(dates)
        return True, ""

    def _handle_outage(self, postcode: str | None) -> tuple[bool, str]:
        pc = postcode or (self.account or {}).get("postcode")
        if not pc:
            self.awaiting = "postcode"
            return False, "Let me check for outages. What's your postcode?"
        self.awaiting = None
        rec = self.tools.call("check_outage", {"postcode": pc})
        if not rec.ok or rec.result is None:
            self.awaiting = "postcode"
            return False, "Sorry, I didn't catch a valid postcode. Could you say it again?"
        r = rec.result
        self.last_outage = r
        if r["outage"]:
            eta = r.get("eta") or ""
            eta_txt = ""
            if eta:
                d = date.fromisoformat(eta[:10])
                eta_txt = f" Engineers expect it to be fixed by {eta[11:16]} on {fmt_date(d)}."
            return True, f"Yes, there's a known outage affecting {r['postcode']}, caused by {r['cause']}.{eta_txt}"
        return True, (
            f"There's no known outage for {r['postcode']}. If you're still having problems, "
            "I can book a technician visit."
        )

    def _bill(self) -> dict[str, Any] | None:
        rec = self.tools.call("get_bill", {"account_id": self.account["account_id"]})  # type: ignore[index]
        return rec.result if rec.ok else None

    @staticmethod
    def _duplicate(bill: dict[str, Any]) -> dict[str, Any] | None:
        items = bill["items"]
        for i, it in enumerate(items):
            if "duplicate" in it["description"].lower():
                return it
            for other in items[:i]:
                if other["description"] == it["description"] and other["amount"] == it["amount"]:
                    return it
        return None

    def _issue(self, amount: float, reason: str) -> str | None:
        rec = self.tools.call(
            "issue_credit",
            {"account_id": self.account["account_id"], "amount": amount, "reason": reason},  # type: ignore[index]
        )
        if rec.ok:
            self._drop_queued("credit")
            return f"I've applied a {fmt_money(amount)} credit to your account."
        return None

    def _drop_queued(self, intent: str) -> None:
        """Remove a queued intent that was satisfied while handling the current one."""
        if intent in self.intents[1:]:
            self.intents = self.intents[:1] + [i for i in self.intents[1:] if i != intent]

    def _handle_billing(self) -> tuple[bool, str]:
        bill = self._bill()
        if bill is None:
            return True, "Sorry, I couldn't retrieve your bill just now."
        period = date.fromisoformat(bill["period"] + "-01").strftime("%B %Y")
        msg = (
            f"Your latest bill, for {period}, is {fmt_money(bill['total'])}, "
            f"due {fmt_date(date.fromisoformat(bill['due_date']))}."
        )
        if not (self.dispute or "credit" in self.intents):
            return True, msg
        dup = self._duplicate(bill)
        if dup:
            issued = self._issue(dup["amount"], f"duplicate charge on bill {bill['bill_id']}")
            if issued:
                return True, f"{msg} I can see a duplicate charge of {fmt_money(dup['amount'])}. {issued}"
        items = ", ".join(f"{i['description']} {fmt_money(i['amount'])}" for i in bill["items"])
        self.offered_transfer = True
        self._drop_queued("credit")
        return True, (
            f"{msg} It's made up of: {items}. I can't see an error I'm able to credit, "
            "but I can transfer you to a colleague if you'd like to dispute it."
        )

    def _handle_credit(self) -> tuple[bool, str]:
        # Justification 1: a confirmed outage on the account's postcode.
        outage = self.last_outage
        if outage is None:
            rec = self.tools.call("check_outage", {"postcode": self.account["postcode"]})  # type: ignore[index]
            outage = rec.result if rec.ok else None
            self.last_outage = outage
        if outage and outage.get("outage"):
            issued = self._issue(OUTAGE_CREDIT_AMOUNT, f"outage in {outage['postcode'].split()[0]}")
            if issued:
                return True, f"Because of the outage in your area, {issued}"
        # Justification 2: a duplicate charge on the bill.
        bill = self._bill()
        dup = self._duplicate(bill) if bill else None
        if dup and bill:
            issued = self._issue(dup["amount"], f"duplicate charge on bill {bill['bill_id']}")
            if issued:
                return True, f"I can see a duplicate charge of {fmt_money(dup['amount'])} on your bill. {issued}"
        self.offered_transfer = True
        return True, (
            "I can't find a reason I'm able to apply a credit for. "
            "I can transfer you to a colleague if you'd like to discuss it."
        )

    def _handle_technician(self, dates: list[date]) -> tuple[bool, str]:
        if not dates:
            self.awaiting = "date"
            return False, "What date would suit you for a technician visit? Visits are mornings, Monday to Saturday."
        rec = self.tools.call(
            "book_technician",
            {"account_id": self.account["account_id"], "date": dates[0].isoformat()},  # type: ignore[index]
        )
        self.known_dates = []
        if rec.ok and rec.result:
            self.awaiting = None
            d = date.fromisoformat(rec.result["date"])
            return True, (
                f"You're booked: a technician will visit on {fmt_date(d)} between 08:00 and 12:00. "
                f"Your booking reference is {rec.result['booking_id']}."
            )
        alts = extract_dates(rec.error or "", self.today)
        self.awaiting = "date"
        alt_txt = ", ".join(fmt_date(a) for a in alts)
        reason = "we don't do visits on Sundays" if "Sunday" in (rec.error or "") else "that date isn't available"
        return False, f"Sorry, {reason}. The next available dates are {alt_txt}. Which would you like?"

    # ------------------------------------------------------------ escalation
    def _escalate(self, reason: str) -> str:
        account_id = (self.account or {}).get("account_id")
        args: dict[str, Any] = {"reason": reason}
        if account_id:
            args["account_id"] = account_id
        self.tools.call("escalate_to_human", args)
        self.escalated = True
        self.ended = True
        self.handoff = build_handoff_summary(reason, self.account, self.transcript, self.tools.log)
        return (
            "I'm transferring you to a colleague now. I've passed on a summary "
            "so you won't need to repeat yourself."
        )
