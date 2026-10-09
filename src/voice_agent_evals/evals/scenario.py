"""Scenario schema, loader and validation.

A scenario is one simulated phone call: who is calling (persona), what they
want (goal), what the caller knows (caller_facts), what the backend says is
true (ground_truth), what the caller says (caller_script, used by the scripted
caller) and what a correct agent should do (expected).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..backend import MockBackend
from ..tools import TOOL_NAMES

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Persona(_Strict):
    name: str
    description: str
    mood: str = "neutral"


class ExpectedToolCall(_Strict):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_tool(cls, v: str) -> str:
        if v not in TOOL_NAMES:
            raise ValueError(f"unknown tool {v!r}; expected one of {TOOL_NAMES}")
        return v


class EndState(_Strict):
    """Expected final backend state. ``None`` means "don't check"."""

    credits: list[dict[str, Any]] | None = None
    bookings: list[dict[str, Any]] | None = None
    escalated: bool | None = None
    verified: list[str] | None = None


class Expected(_Strict):
    escalation: bool
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    end_state: EndState = Field(default_factory=EndState)
    must_say: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)

    @field_validator("forbidden_tools")
    @classmethod
    def _known_forbidden(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if t not in TOOL_NAMES]
        if bad:
            raise ValueError(f"unknown forbidden tools {bad}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Expected":
        if self.end_state.escalated is None:
            self.end_state.escalated = self.escalation
        elif self.end_state.escalated != self.escalation:
            raise ValueError("end_state.escalated contradicts expected.escalation")
        expected_names = {t.name for t in self.tool_calls}
        clash = expected_names & set(self.forbidden_tools)
        if clash:
            raise ValueError(f"tools both expected and forbidden: {sorted(clash)}")
        if self.escalation and "escalate_to_human" in self.forbidden_tools:
            raise ValueError("escalation expected but escalate_to_human is forbidden")
        return self


class Scenario(_Strict):
    id: str
    title: str
    category: str
    description: str = ""
    persona: Persona
    goal: str
    caller_facts: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    caller_script: list[str] = Field(min_length=1)
    expected: Expected
    max_turns: int = Field(default=12, ge=1, le=50)
    tags: list[str] = Field(default_factory=list)
    source_path: str | None = None

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError("id must be a lowercase slug (a-z, 0-9, _ or -), 3-64 chars")
        return v


class ScenarioError(Exception):
    pass


GROUND_TRUTH_KEYS = {
    "account_id", "customer_name", "bill_total", "bill_due_date", "postcode", "outage",
    "outage_eta", "credit_limit", "notes",
}


def check_ground_truth(s: Scenario, backend: MockBackend | None = None) -> list[str]:
    """Cross-check a scenario's ground_truth against the seeded mock backend.

    Keeps scenarios honest: if someone edits the seed data, scenarios that
    relied on the old facts fail validation instead of silently drifting.
    """
    b = backend or MockBackend()
    gt = s.ground_truth
    errors: list[str] = []
    unknown = set(gt) - GROUND_TRUTH_KEYS
    if unknown:
        errors.append(f"unknown ground_truth keys {sorted(unknown)} (allowed: {sorted(GROUND_TRUTH_KEYS)})")
    acc = None
    if "account_id" in gt and gt["account_id"] is not None:
        acc = b.accounts.get(str(gt["account_id"]))
        if acc is None:
            errors.append(f"ground_truth.account_id {gt['account_id']} not in mock backend")
    if acc is not None:
        if "customer_name" in gt and gt["customer_name"] != acc.name:
            errors.append(f"customer_name {gt['customer_name']!r} != backend {acc.name!r}")
        if "bill_total" in gt and abs(float(gt["bill_total"]) - acc.bills[-1].total) > 0.005:
            errors.append(f"bill_total {gt['bill_total']} != backend {acc.bills[-1].total}")
        if "bill_due_date" in gt and str(gt["bill_due_date"]) != acc.bills[-1].due_date.isoformat():
            errors.append(f"bill_due_date {gt['bill_due_date']} != backend {acc.bills[-1].due_date}")
    if "outage" in gt:
        pc = gt.get("postcode") or (acc.postcode if acc else None)
        if not pc:
            errors.append("ground_truth.outage needs postcode or account_id")
        else:
            res = b.check_outage(pc)
            if bool(res["outage"]) != bool(gt["outage"]):
                errors.append(f"outage={gt['outage']} for {pc} but backend says {res['outage']}")
            if "outage_eta" in gt and res.get("eta") != gt["outage_eta"]:
                errors.append(f"outage_eta {gt['outage_eta']} != backend {res.get('eta')}")
    if "credit_limit" in gt and abs(float(gt["credit_limit"]) - b.credit_limit) > 0.005:
        errors.append(f"credit_limit {gt['credit_limit']} != backend {b.credit_limit}")
    for c in s.expected.end_state.credits or []:
        if "account_id" in c and c["account_id"] not in b.accounts:
            errors.append(f"end_state credit for unknown account {c['account_id']}")
        if "amount" in c and float(c["amount"]) > b.credit_limit:
            errors.append(f"end_state credit {c['amount']} exceeds credit limit; it can never be reached")
    return errors


def load_scenario(path: str | Path) -> Scenario:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{p}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{p}: top level must be a mapping")
    try:
        s = Scenario.model_validate({**data, "source_path": str(p)})
    except ValidationError as exc:
        raise ScenarioError(f"{p}: {exc}") from exc
    gt_errors = check_ground_truth(s)
    if gt_errors:
        raise ScenarioError(f"{p}: ground truth mismatch: " + "; ".join(gt_errors))
    return s


def load_scenarios(directory: str | Path, only: list[str] | None = None) -> list[Scenario]:
    d = Path(directory)
    if not d.is_dir():
        raise ScenarioError(f"scenario directory not found: {d}")
    files = sorted([*d.glob("*.yaml"), *d.glob("*.yml")])
    scenarios = [load_scenario(f) for f in files]
    ids = [s.id for s in scenarios]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ScenarioError(f"duplicate scenario ids: {dupes}")
    if only:
        wanted = set(only)
        scenarios = [s for s in scenarios if s.id in wanted or s.category in wanted or wanted & set(s.tags)]
    return scenarios
