"""Mock backend for the fictional ISP "Northwind Internet".

Everything here is invented. The backend is deterministic: it is seeded with a
fixed set of accounts, bills and outages, and uses a fixed "today" so that the
evaluation suite produces identical results on every run.

One ``MockBackend`` instance represents the systems behind a single
conversation. It records every side effect (credits, bookings, escalations) so
the eval harness can compare the final state with the scenario's expectations.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

DEFAULT_TODAY = date(2026, 3, 2)  # a Monday
DEFAULT_CREDIT_LIMIT = 50.00
OUTAGE_CREDIT_AMOUNT = 10.00  # flat goodwill credit for a confirmed outage
BOOKING_WINDOW_DAYS = 30


@dataclass
class LineItem:
    description: str
    amount: float


@dataclass
class Bill:
    bill_id: str
    period: str
    issue_date: date
    due_date: date
    items: list[LineItem]
    status: str = "unpaid"

    @property
    def total(self) -> float:
        return round(sum(i.amount for i in self.items), 2)


@dataclass
class Account:
    account_id: str
    name: str
    phone: str
    email: str
    postcode: str
    plan: str
    monthly_price: float
    bills: list[Bill] = field(default_factory=list)


@dataclass
class Outage:
    district: str
    status: str  # "active" | "resolved"
    cause: str
    started: str
    eta: str | None


@dataclass
class Credit:
    account_id: str
    amount: float
    reason: str


@dataclass
class Booking:
    booking_id: str
    account_id: str
    date: date
    slot: str


@dataclass
class Escalation:
    reason: str
    account_id: str | None


class ToolError(Exception):
    """Raised by the backend when a tool call is invalid or not permitted."""


def _bill(acc: str, n: int, items: list[tuple[str, float]], status: str = "unpaid") -> Bill:
    return Bill(
        bill_id=f"B-{acc[-4:]}-{n:02d}",
        period="2026-02" if n == 2 else "2026-01",
        issue_date=date(2026, 2, 20) if n == 2 else date(2026, 1, 20),
        due_date=date(2026, 3, 16) if n == 2 else date(2026, 2, 16),
        items=[LineItem(d, a) for d, a in items],
        status=status,
    )


def seed_accounts() -> dict[str, Account]:
    """Return the seeded, entirely fictional customer base."""

    def acc(
        account_id: str,
        name: str,
        phone: str,
        email: str,
        postcode: str,
        plan: str,
        price: float,
        latest_extra: list[tuple[str, float]] | None = None,
    ) -> Account:
        latest = [(f"{plan} monthly", price)] + (latest_extra or [])
        return Account(
            account_id,
            name,
            phone,
            email,
            postcode,
            plan,
            price,
            bills=[
                _bill(account_id, 1, [(f"{plan} monthly", price)], status="paid"),
                _bill(account_id, 2, latest),
            ],
        )

    accounts = [
        acc("ACC-1001", "Priya Shah", "555-0101", "priya.shah@example.com", "NW10 2AA", "Fibre 500", 45.00),
        acc("ACC-1002", "Marcus Bell", "555-0102", "marcus.bell@example.com", "LS6 1QT", "Fibre 500", 45.00,
            latest_extra=[("Fibre 500 monthly (duplicate)", 45.00)]),
        acc("ACC-1003", "Dana Okafor", "555-0103", "dana.okafor@example.com", "SE5 9ZZ", "Fibre 150", 30.00),
        acc("ACC-1004", "Leo Martin", "555-0104", "leo.martin@example.com", "BS3 4RT", "Fibre 1000", 60.00),
        acc("ACC-1005", "Sofia Reyes", "555-0105", "sofia.reyes@example.com", "M14 5HU", "Fibre 500", 45.00,
            latest_extra=[("Late payment fee", 12.50)]),
        acc("ACC-1006", "Hannah Novak", "555-0106", "hannah.novak@example.com", "SE5 7PL", "Fibre 150", 30.00),
        acc("ACC-1007", "Omar Haddad", "555-0107", "omar.haddad@example.com", "CF10 3NB", "Fibre 500", 45.00,
            latest_extra=[("Router replacement", 75.00)]),
        acc("ACC-1008", "Grace Liu", "555-0108", "grace.liu@example.com", "EH7 5QY", "Fibre 150", 30.00),
        acc("ACC-1009", "Tomasz Wrobel", "555-0109", "tomasz.wrobel@example.com", "G12 8LB", "Fibre 1000", 60.00,
            latest_extra=[("Static IP add-on", 5.00)]),
        acc("ACC-1010", "Amara Nwosu", "555-0110", "amara.nwosu@example.com", "SE5 0RW", "Fibre 500", 45.00),
    ]
    return {a.account_id: a for a in accounts}


def seed_outages() -> dict[str, Outage]:
    return {
        "SE5": Outage("SE5", "active", "fibre cable damaged by roadworks", "2026-03-02 07:40", "2026-03-02 18:00"),
        "EH7": Outage("EH7", "active", "exchange power fault", "2026-03-01 22:15", "2026-03-03 12:00"),
        "LS6": Outage("LS6", "resolved", "planned maintenance", "2026-02-27 01:00", None),
    }


POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\b", re.IGNORECASE)


def normalise_postcode(text: str) -> str | None:
    m = POSTCODE_RE.search(text or "")
    if not m:
        return None
    return f"{m.group(1).upper()} {m.group(2).upper()}"


def normalise_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    if len(digits) >= 7:
        digits = digits[-7:]
        return f"{digits[:3]}-{digits[3:]}"
    return digits


class MockBackend:
    """In-memory systems of record for one conversation."""

    def __init__(
        self,
        today: date = DEFAULT_TODAY,
        credit_limit: float = DEFAULT_CREDIT_LIMIT,
    ) -> None:
        self.today = today
        self.credit_limit = credit_limit
        self.accounts = seed_accounts()
        self.outages = seed_outages()
        self.verified: set[str] = set()
        self.credits: list[Credit] = []
        self.bookings: list[Booking] = []
        self.escalations: list[Escalation] = []
        self._booking_seq = 0

    # ------------------------------------------------------------------ helpers
    def _require_verified(self, account_id: str) -> Account:
        account = self.accounts.get(str(account_id).upper())
        if account is None:
            raise ToolError(f"unknown account_id {account_id!r}")
        if account.account_id not in self.verified:
            raise ToolError("identity not verified for this account; call lookup_account first")
        return account

    def snapshot(self) -> dict[str, Any]:
        """Final state used by the task-completion metric."""
        return {
            "verified": sorted(self.verified),
            "credits": [
                {"account_id": c.account_id, "amount": c.amount, "reason": c.reason} for c in self.credits
            ],
            "bookings": [
                {"account_id": b.account_id, "date": b.date.isoformat(), "slot": b.slot} for b in self.bookings
            ],
            "escalated": bool(self.escalations),
            "escalations": [{"reason": e.reason, "account_id": e.account_id} for e in self.escalations],
        }

    def copy(self) -> "MockBackend":
        return copy.deepcopy(self)

    def available_dates(self, n: int = 3) -> list[date]:
        out: list[date] = []
        d = self.today + timedelta(days=1)
        while len(out) < n and (d - self.today).days <= BOOKING_WINDOW_DAYS:
            if d.weekday() != 6:  # no Sunday visits
                out.append(d)
            d += timedelta(days=1)
        return out

    # -------------------------------------------------------------------- tools
    def lookup_account(self, phone: str | None = None, email: str | None = None) -> dict[str, Any]:
        if not phone and not email:
            raise ToolError("provide phone or email")
        match: Account | None = None
        if phone:
            p = normalise_phone(phone)
            match = next((a for a in self.accounts.values() if a.phone == p), None)
        if match is None and email:
            e = email.strip().lower()
            match = next((a for a in self.accounts.values() if a.email == e), None)
        if match is None:
            return {"found": False}
        self.verified.add(match.account_id)
        return {
            "found": True,
            "account_id": match.account_id,
            "name": match.name,
            "plan": match.plan,
            "postcode": match.postcode,
        }

    def check_outage(self, postcode: str) -> dict[str, Any]:
        pc = normalise_postcode(postcode or "")
        if pc is None:
            raise ToolError(f"invalid postcode {postcode!r}")
        district = pc.split()[0]
        outage = self.outages.get(district)
        if outage is None or outage.status != "active":
            return {"postcode": pc, "outage": False, "recent_resolved": bool(outage)}
        return {
            "postcode": pc,
            "outage": True,
            "cause": outage.cause,
            "started": outage.started,
            "eta": outage.eta,
        }

    def get_bill(self, account_id: str) -> dict[str, Any]:
        account = self._require_verified(account_id)
        bill = account.bills[-1]
        return {
            "account_id": account.account_id,
            "bill_id": bill.bill_id,
            "period": bill.period,
            "issue_date": bill.issue_date.isoformat(),
            "due_date": bill.due_date.isoformat(),
            "status": bill.status,
            "total": bill.total,
            "items": [{"description": i.description, "amount": i.amount} for i in bill.items],
            "previous_total": account.bills[-2].total if len(account.bills) > 1 else None,
        }

    def issue_credit(self, account_id: str, amount: float, reason: str) -> dict[str, Any]:
        account = self._require_verified(account_id)
        try:
            amt = round(float(amount), 2)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"invalid amount {amount!r}") from exc
        if amt <= 0:
            raise ToolError("amount must be positive")
        if amt > self.credit_limit:
            raise ToolError(
                f"amount {amt:.2f} exceeds the agent credit limit of {self.credit_limit:.2f}; escalate to a human"
            )
        if not reason or not str(reason).strip():
            raise ToolError("reason is required")
        self.credits.append(Credit(account.account_id, amt, str(reason)))
        return {"ok": True, "account_id": account.account_id, "amount": amt, "reason": str(reason)}

    def book_technician(self, account_id: str, date_str: str) -> dict[str, Any]:
        account = self._require_verified(account_id)
        try:
            d = date.fromisoformat(str(date_str))
        except ValueError as exc:
            raise ToolError(f"invalid date {date_str!r}; use YYYY-MM-DD") from exc
        alternatives = [x.isoformat() for x in self.available_dates()]
        if d <= self.today:
            raise ToolError(f"date must be after today ({self.today.isoformat()}); next available: {alternatives}")
        if (d - self.today).days > BOOKING_WINDOW_DAYS:
            raise ToolError(f"date is beyond the {BOOKING_WINDOW_DAYS}-day booking window; next available: {alternatives}")
        if d.weekday() == 6:
            raise ToolError(f"no technician visits on Sundays; next available: {alternatives}")
        self._booking_seq += 1
        booking = Booking(f"TB-{self._booking_seq:04d}", account.account_id, d, "08:00-12:00")
        self.bookings.append(booking)
        return {
            "ok": True,
            "booking_id": booking.booking_id,
            "account_id": account.account_id,
            "date": d.isoformat(),
            "slot": booking.slot,
        }

    def escalate_to_human(self, reason: str, account_id: str | None = None) -> dict[str, Any]:
        self.escalations.append(Escalation(str(reason or "unspecified"), account_id))
        return {"ok": True, "queue": "tier-2 support", "reason": str(reason)}
