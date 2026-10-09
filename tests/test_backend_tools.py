from datetime import date

import pytest

from voice_agent_evals.backend import MockBackend, ToolError, normalise_phone, normalise_postcode
from voice_agent_evals.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolExecutor


@pytest.fixture
def b() -> MockBackend:
    return MockBackend()


def test_seed_is_deterministic():
    a, c = MockBackend(), MockBackend()
    assert a.snapshot() == c.snapshot()
    assert a.accounts["ACC-1002"].bills[-1].total == 90.00
    assert a.today == date(2026, 3, 2)


def test_lookup_by_phone_and_email_verifies(b):
    r = b.lookup_account(phone="555 0102")
    assert r["found"] and r["account_id"] == "ACC-1002"
    assert "ACC-1002" in b.verified
    r2 = b.lookup_account(email="PRIYA.SHAH@example.com")
    assert r2["account_id"] == "ACC-1001"


def test_lookup_not_found_and_requires_identifier(b):
    assert b.lookup_account(phone="555-0199") == {"found": False}
    with pytest.raises(ToolError):
        b.lookup_account()


def test_account_tools_require_verification(b):
    with pytest.raises(ToolError, match="not verified"):
        b.get_bill("ACC-1002")
    with pytest.raises(ToolError, match="not verified"):
        b.issue_credit("ACC-1002", 10, "x")
    with pytest.raises(ToolError, match="unknown account"):
        b.get_bill("ACC-9999")


def test_get_bill(b):
    b.lookup_account(phone="555-0102")
    bill = b.get_bill("ACC-1002")
    assert bill["total"] == 90.00 and bill["due_date"] == "2026-03-16"
    assert len(bill["items"]) == 2


def test_credit_limit_enforced(b):
    b.lookup_account(phone="555-0107")
    with pytest.raises(ToolError, match="exceeds"):
        b.issue_credit("ACC-1007", 75, "router")
    assert b.credits == []
    ok = b.issue_credit("ACC-1007", 50, "router")
    assert ok["amount"] == 50.0
    with pytest.raises(ToolError):
        b.issue_credit("ACC-1007", -1, "x")
    with pytest.raises(ToolError):
        b.issue_credit("ACC-1007", 5, " ")


def test_outage(b):
    assert b.check_outage("se5 9zz")["outage"] is True
    assert b.check_outage("LS6 1QT")["outage"] is False  # resolved
    assert b.check_outage("NW10 2AA")["outage"] is False
    with pytest.raises(ToolError):
        b.check_outage("nowhere")


def test_booking_rules(b):
    b.lookup_account(phone="555-0104")
    with pytest.raises(ToolError, match="Sunday"):
        b.book_technician("ACC-1004", "2026-03-08")
    with pytest.raises(ToolError, match="after today"):
        b.book_technician("ACC-1004", "2026-03-02")
    with pytest.raises(ToolError, match="window"):
        b.book_technician("ACC-1004", "2026-05-01")
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        b.book_technician("ACC-1004", "next week")
    r = b.book_technician("ACC-1004", "2026-03-05")
    assert r["booking_id"] == "TB-0001"
    assert b.snapshot()["bookings"] == [{"account_id": "ACC-1004", "date": "2026-03-05", "slot": "08:00-12:00"}]


def test_escalation_recorded(b):
    b.escalate_to_human("asked for human")
    assert b.snapshot()["escalated"] is True


def test_normalisers():
    assert normalise_postcode("my code is se5 9zz ok") == "SE5 9ZZ"
    assert normalise_postcode("nothing") is None
    assert normalise_phone("+1 555 0102") == "555-0102"


def test_executor_logs_success_and_errors(b):
    ex = ToolExecutor(b)
    ex.call("lookup_account", {"phone": "555-0102"})
    ex.call("get_bill", '{"account_id": "ACC-1002"}')
    ex.call("issue_credit", {"account_id": "ACC-1002", "amount": 500, "reason": "x"})
    ex.call("nope", {})
    ex.call("get_bill", "{bad json")
    assert [r.ok for r in ex.log] == [True, True, False, False, False]
    assert "exceeds" in ex.log[2].error
    assert ex.log[4].error == "arguments were not valid JSON"


def test_schemas_cover_all_tools():
    assert set(TOOL_NAMES) == {
        "lookup_account", "check_outage", "get_bill", "issue_credit", "book_technician", "escalate_to_human",
    }
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function" and "parameters" in t["function"]
