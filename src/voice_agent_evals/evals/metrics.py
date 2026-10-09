"""Per-conversation and aggregate metrics.

All metrics are computed from data the harness records itself: the ordered
tool-call log, the final mock-backend state and the transcript. Nothing is
estimated or sampled.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from ..backend import OUTAGE_CREDIT_AMOUNT
from ..nlp import extract_amounts, extract_dates
from ..tools import ToolCallRecord
from .scenario import Expected, ExpectedToolCall

# --------------------------------------------------------------------------
# Task completion
# --------------------------------------------------------------------------


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(expected) - float(actual)) < 0.005
        except (TypeError, ValueError):
            return False
    return _norm(expected) == _norm(actual)


def _subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(k in actual and _value_matches(v, actual[k]) for k, v in expected.items())


def _match_list(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    """Same length and every expected item matches a distinct actual item (subset of keys)."""
    if len(expected) != len(actual):
        return False
    remaining = list(actual)
    for e in expected:
        hit = next((a for a in remaining if _subset(e, a)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def task_completion(expected: Expected, final_state: dict[str, Any], agent_texts: list[str]) -> tuple[bool, list[str]]:
    """Did the conversation reach the expected end state in the mock backend?"""
    problems: list[str] = []
    es = expected.end_state
    if es.credits is not None and not _match_list(es.credits, final_state["credits"]):
        problems.append(f"credits expected {es.credits}, got {final_state['credits']}")
    if es.bookings is not None and not _match_list(es.bookings, final_state["bookings"]):
        problems.append(f"bookings expected {es.bookings}, got {final_state['bookings']}")
    if es.escalated is not None and es.escalated != final_state["escalated"]:
        problems.append(f"escalated expected {es.escalated}, got {final_state['escalated']}")
    if es.verified is not None and sorted(es.verified) != sorted(final_state["verified"]):
        problems.append(f"verified expected {es.verified}, got {final_state['verified']}")
    said = " ".join(agent_texts).lower()
    for phrase in expected.must_say:
        if phrase.lower() not in said:
            problems.append(f"agent never said {phrase!r}")
    for phrase in expected.must_not_say:
        if phrase.lower() in said:
            problems.append(f"agent said forbidden phrase {phrase!r}")
    return (not problems), problems


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------


def escalation_label(expected: bool, actual: bool) -> str:
    """Confusion-matrix cell: TP, FP, FN or TN (positive = escalated)."""
    if expected and actual:
        return "TP"
    if not expected and actual:
        return "FP"
    if expected and not actual:
        return "FN"
    return "TN"


def precision_recall(labels: Iterable[str]) -> dict[str, Any]:
    labels = list(labels)
    tp, fp, fn, tn = (labels.count(x) for x in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (tp + tn) / len(labels) if labels else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy}


# --------------------------------------------------------------------------
# Tool-call correctness
# --------------------------------------------------------------------------


def tool_call_correctness(
    expected_calls: list[ExpectedToolCall], forbidden: list[str], log: list[ToolCallRecord]
) -> dict[str, Any]:
    """Expected calls must appear, in order, among successful calls (args are a subset match).

    Forbidden tools must never be called (even unsuccessfully). Failed calls are
    reported separately as ``errors`` and do not count towards matching.
    """
    ok_calls = [c for c in log if c.ok]
    i = 0
    matched: list[str] = []
    missing: list[dict[str, Any]] = []
    for exp in expected_calls:
        j = i
        while j < len(ok_calls) and not (ok_calls[j].name == exp.name and _subset(exp.args, ok_calls[j].args)):
            j += 1
        if j < len(ok_calls):
            matched.append(exp.name)
            i = j + 1
        else:
            missing.append({"name": exp.name, "args": exp.args})
    forbidden_called = [c.name for c in log if c.name in set(forbidden)]
    n = len(expected_calls)
    return {
        "expected": n,
        "matched": len(matched),
        "recall": (len(matched) / n) if n else 1.0,
        "missing": missing,
        "forbidden_called": forbidden_called,
        "errors": [{"name": c.name, "error": c.error} for c in log if not c.ok],
        "total_calls": len(log),
        "correct": not missing and not forbidden_called,
    }


# --------------------------------------------------------------------------
# Hallucination (rule-based)
# --------------------------------------------------------------------------

OUTAGE_POS_RE = re.compile(
    r"(there'?s|there is|we have|we'?re aware of|i can see) (a |an )?(known |current |ongoing )?outage"
    r"|outage (is )?affecting|engineers (are|expect)",
    re.IGNORECASE,
)
OUTAGE_NEG_RE = re.compile(
    r"no (known |current |reported )?outage|(isn'?t|is not|aren'?t|are not) (a |any )?(known )?outages?"
    r"|not aware of any outage|no problems? (reported )?in your area",
    re.IGNORECASE,
)
CREDIT_CLAIM_RE = re.compile(
    r"(i'?ve|i have|we'?ve|we have) (now )?(applied|issued|added|credited|refunded)"
    r"|credit (has been|was|is now) (applied|issued|added)",
    re.IGNORECASE,
)
BOOKING_CLAIM_RE = re.compile(
    r"(you'?re|you are) (all )?booked|(i'?ve|i have) booked|booking (is )?confirmed|visit is (booked|scheduled)",
    re.IGNORECASE,
)
TRANSFER_CLAIM_RE = re.compile(
    r"(i'?m|i am) (now )?(transferring|connecting|putting) you|(i'?ll|i will|let me) (now )?(transfer|connect|put) you"
    r"|passing you (over )?to",
    re.IGNORECASE,
)
IDENT_RE = re.compile(r"\b(ACC-\d{4}|TB-\d{4}|B-\d{4}-\d{2})\b", re.IGNORECASE)


@dataclass
class Claim:
    turn: int
    kind: str
    claim: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "kind": self.kind, "claim": self.claim, "detail": self.detail}


def _numbers_in(obj: Any) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 2))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, str):
        for m in re.findall(r"\d+(?:\.\d+)?", obj):
            out.add(round(float(m), 2))
    return out


@dataclass
class _Grounding:
    numbers: set[float] = field(default_factory=set)
    dates: set[date] = field(default_factory=set)
    text: str = ""
    outage: bool | None = None
    credit_ok: bool = False
    booking_ok: bool = False
    escalated: bool = False


def hallucination_check(
    turns: list[dict[str, Any]],
    today: date,
    credit_limit: float,
) -> list[Claim]:
    """Flag agent statements not supported by tool results or the caller.

    ``turns`` is the transcript: dicts with ``role`` ("agent"/"caller"), ``text``
    and, for agent turns, ``tool_calls`` (list of ToolCallRecord dicts) made
    while producing that reply. Checks (all rule-based, deliberately narrow):

    - money amounts must match a number in a tool result, the caller's words or policy constants;
    - explicit dates must appear in a tool result, the caller's words, or be today;
    - outage yes/no statements must match the latest check_outage result;
    - "I've applied a credit" / "you're booked" / "I'm transferring you" need a
      successful issue_credit / book_technician / escalate_to_human call;
    - account, booking and bill references must appear in tool results.
    """
    g = _Grounding(numbers={round(credit_limit, 2), round(OUTAGE_CREDIT_AMOUNT, 2)}, dates={today})
    claims: list[Claim] = []
    for idx, t in enumerate(turns):
        text = t.get("text") or ""
        if t["role"] == "caller":
            g.numbers |= set(round(x, 2) for x in extract_amounts(text)) | _numbers_in(text)
            g.dates |= set(extract_dates(text, today, relative=True))
            g.text += " " + text
            continue
        for call in t.get("tool_calls") or []:
            if not call.get("ok"):
                g.text += " " + (call.get("error") or "")
                g.dates |= set(extract_dates(call.get("error") or "", today))
                continue
            res = call.get("result") or {}
            blob = json.dumps(res)
            g.numbers |= _numbers_in(res)
            g.dates |= set(extract_dates(blob, today))
            g.text += " " + blob
            name = call.get("name")
            if name == "check_outage":
                g.outage = bool(res.get("outage"))
            elif name == "issue_credit":
                g.credit_ok = True
            elif name == "book_technician":
                g.booking_ok = True
            elif name == "escalate_to_human":
                g.escalated = True

        for amt in extract_amounts(text):
            if not any(abs(amt - n) < 0.005 for n in g.numbers):
                claims.append(Claim(idx, "amount", f"${amt:.2f}", "amount not found in any tool result or caller statement"))
        for d in extract_dates(text, today):
            if d not in g.dates:
                claims.append(Claim(idx, "date", d.isoformat(), "date not found in any tool result or caller statement"))
        pos, neg = OUTAGE_POS_RE.search(text), OUTAGE_NEG_RE.search(text)
        if pos or neg:
            said = bool(pos) and not neg
            if g.outage is None:
                claims.append(Claim(idx, "outage", (pos or neg).group(0), "outage status stated without check_outage"))  # type: ignore[union-attr]
            elif said != g.outage:
                claims.append(Claim(idx, "outage", (pos or neg).group(0), f"backend says outage={g.outage}"))  # type: ignore[union-attr]
        if CREDIT_CLAIM_RE.search(text) and not g.credit_ok:
            claims.append(Claim(idx, "action", CREDIT_CLAIM_RE.search(text).group(0), "claims a credit without a successful issue_credit"))  # type: ignore[union-attr]
        if BOOKING_CLAIM_RE.search(text) and not g.booking_ok:
            claims.append(Claim(idx, "action", BOOKING_CLAIM_RE.search(text).group(0), "claims a booking without a successful book_technician"))  # type: ignore[union-attr]
        if TRANSFER_CLAIM_RE.search(text) and not g.escalated:
            claims.append(Claim(idx, "action", TRANSFER_CLAIM_RE.search(text).group(0), "claims a transfer without escalate_to_human"))  # type: ignore[union-attr]
        for ident in IDENT_RE.findall(text):
            if ident.upper() not in g.text.upper():
                claims.append(Claim(idx, "reference", ident, "reference not found in tool results or caller statements"))
    return claims


# --------------------------------------------------------------------------
# Latency & cost
# --------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency_stats(latencies_ms: list[float]) -> dict[str, Any]:
    return {
        "n": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "max_ms": max(latencies_ms) if latencies_ms else None,
        "mean_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
    }


def cost_summary(prompt_tokens: int, completion_tokens: int, wall_clock_s: float, local: bool = True) -> dict[str, Any]:
    """Cost of one conversation.

    Local backends have no per-token API bill, so ``api_cost_usd`` is 0.0; the
    real costs are compute time (``wall_clock_s``) and tokens processed.
    """
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "wall_clock_s": round(wall_clock_s, 4),
        "api_cost_usd": 0.0 if local else None,
    }
