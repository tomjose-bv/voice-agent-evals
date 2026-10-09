"""Small, dependency-free text helpers: money, dates, phones, emails.

Used by the scripted agent (to understand the caller) and by the rule-based
hallucination checks (to pull factual claims out of agent replies).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
MONEY_WORD_RE = re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|bucks|usd)\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH_ALT})\b\.?(?:,?\s+(\d{{4}}))?", re.IGNORECASE
)
MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.IGNORECASE
)
PHONE_RE = re.compile(r"(?<![\d.$])(?:\+?1[\s-]?)?(\d{3})[\s.-]?(\d{4})(?![\d.])")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def spoken_digits_to_numbers(text: str) -> str:
    """Turn runs of spoken digits ("five five five oh one oh two") into "5550102".

    Speech-to-text often spells phone numbers out; this keeps the scripted agent
    usable over voice. Only runs of 3+ digit words are converted.
    """
    tokens = re.split(r"(\s+|-)", text)
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 3:
            out.append("".join(_DIGIT_WORDS[r] for r in run))
        else:
            out.extend(run_raw)
        run.clear()
        run_raw.clear()

    run_raw: list[str] = []
    for tok in tokens:
        word = tok.strip(",.").lower()
        if word in _DIGIT_WORDS and tok.strip():
            run.append(word)
            run_raw.append(tok)
        elif not tok.strip() or tok == "-":
            if run:
                run_raw.append(tok)
            else:
                out.append(tok)
        else:
            if run:
                flush()
                out.append(" ")
            out.append(tok)
    if run:
        flush()
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def spoken_email_to_text(text: str) -> str:
    """'marcus dot bell at example dot com' -> 'marcus.bell@example.com'."""
    t = re.sub(r"\s+at\s+(?=[\w-]+\s+dot\s+)", "@", text, flags=re.IGNORECASE)
    t = re.sub(r"\s+dot\s+", ".", t, flags=re.IGNORECASE)
    return t


def extract_amounts(text: str) -> list[float]:
    vals = [float(m.replace(",", "")) for m in MONEY_RE.findall(text)]
    vals += [float(m) for m in MONEY_WORD_RE.findall(text)]
    return vals


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _infer_year(month: int, day: int, today: date) -> date | None:
    cand = _safe_date(today.year, month, day)
    if cand and cand < today - timedelta(days=180):
        cand = _safe_date(today.year + 1, month, day)
    return cand


def extract_dates(text: str, today: date, relative: bool = False) -> list[date]:
    """Find explicit dates (ISO, "5 March", "March 5th, 2026").

    With ``relative=True`` also understands "today", "tomorrow" and weekday
    names ("Thursday", "next Friday") relative to ``today``.
    """
    found: list[date] = []
    for y, m, d in ISO_DATE_RE.findall(text):
        dt = _safe_date(int(y), int(m), int(d))
        if dt:
            found.append(dt)
    for d, mon, y in DAY_MONTH_RE.findall(text):
        m = MONTHS[mon.lower()]
        dt = _safe_date(int(y), m, int(d)) if y else _infer_year(m, int(d), today)
        if dt:
            found.append(dt)
    for mon, d, y in MONTH_DAY_RE.findall(text):
        # "may" is also a verb: require a following day number (already in regex)
        m = MONTHS[mon.lower()]
        dt = _safe_date(int(y), m, int(d)) if y else _infer_year(m, int(d), today)
        if dt and dt not in found:
            found.append(dt)
    if relative:
        low = text.lower()
        if re.search(r"\bday after tomorrow\b", low):
            found.append(today + timedelta(days=2))
        elif re.search(r"\btomorrow\b", low):
            found.append(today + timedelta(days=1))
        elif re.search(r"\btoday\b", low) and not found:
            found.append(today)
        for i, wd in enumerate(WEEKDAYS):
            # "Thursday" and "next Thursday" both mean the coming Thursday here;
            # usage varies, so the agent always reads the concrete date back.
            if re.search(rf"\b{wd}\b", low):
                delta = (i - today.weekday()) % 7 or 7
                found.append(today + timedelta(days=delta))
    # de-duplicate, keep order
    seen: set[date] = set()
    uniq = []
    for dt in found:
        if dt not in seen:
            seen.add(dt)
            uniq.append(dt)
    return uniq


def extract_phone(text: str) -> str | None:
    t = spoken_digits_to_numbers(text)
    m = PHONE_RE.search(t)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text) or EMAIL_RE.search(spoken_email_to_text(text))
    return m.group(0).lower().rstrip(".") if m else None


def fmt_money(amount: float) -> str:
    return f"${amount:,.2f}"


def fmt_date(d: date) -> str:
    """Speech-friendly and unambiguous: 'Thursday 5 March 2026'."""
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"
