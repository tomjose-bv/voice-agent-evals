from datetime import date

import pytest

from voice_agent_evals.evals import metrics as M
from voice_agent_evals.evals.scenario import EndState, Expected, ExpectedToolCall
from voice_agent_evals.tools import ToolCallRecord

TODAY = date(2026, 3, 2)


def state(**kw):
    base = {"verified": [], "credits": [], "bookings": [], "escalated": False, "escalations": []}
    base.update(kw)
    return base


# ---- task completion ------------------------------------------------------

def test_task_completion_matches_subset_and_tolerance():
    exp = Expected(escalation=False, end_state=EndState(credits=[{"account_id": "ACC-1002", "amount": 45}]))
    ok, probs = M.task_completion(exp, state(credits=[{"account_id": "ACC-1002", "amount": 45.0, "reason": "dup"}]), [])
    assert ok and probs == []


def test_task_completion_detects_wrong_amount_extra_items_and_phrases():
    exp = Expected(escalation=False, end_state=EndState(credits=[{"amount": 45}]), must_say=["$45.00"],
                   must_not_say=["secret"])
    ok, probs = M.task_completion(exp, state(credits=[{"amount": 40.0}, {"amount": 5.0}]), ["this is secret"])
    assert not ok
    assert len(probs) == 3


def test_task_completion_checks_escalation_by_default():
    exp = Expected(escalation=True)
    ok, probs = M.task_completion(exp, state(escalated=False), [])
    assert not ok and "escalated" in probs[0]


# ---- escalation -------------------------------------------------------------

def test_escalation_labels_and_precision_recall():
    labels = [M.escalation_label(e, a) for e, a in [(True, True), (True, False), (False, True), (False, False)]]
    assert labels == ["TP", "FN", "FP", "TN"]
    pr = M.precision_recall(labels + ["TP"])
    assert pr["precision"] == pytest.approx(2 / 3)
    assert pr["recall"] == pytest.approx(2 / 3)
    assert pr["accuracy"] == pytest.approx(3 / 5)


def test_precision_undefined_without_positives():
    pr = M.precision_recall(["TN", "TN"])
    assert pr["precision"] is None and pr["recall"] is None and pr["f1"] is None


# ---- tool calls ---------------------------------------------------------------

def rec(name, ok=True, **args):
    return ToolCallRecord(name, args, ok, result={} if ok else None, error=None if ok else "boom")


def test_tool_calls_in_order_subset_args():
    expected = [ExpectedToolCall(name="lookup_account"), ExpectedToolCall(name="get_bill", args={"account_id": "acc-1002"})]
    log = [rec("lookup_account", phone="555-0102"), rec("get_bill", account_id="ACC-1002")]
    r = M.tool_call_correctness(expected, [], log)
    assert r["correct"] and r["recall"] == 1.0


def test_tool_calls_order_matters_and_failed_calls_dont_count():
    expected = [ExpectedToolCall(name="get_bill"), ExpectedToolCall(name="lookup_account")]
    log = [rec("lookup_account"), rec("get_bill", ok=False)]
    r = M.tool_call_correctness(expected, [], log)
    assert not r["correct"] and r["matched"] == 1
    assert r["missing"] == [{"name": "get_bill", "args": {}}]
    assert r["errors"] == [{"name": "get_bill", "error": "boom"}]


def test_forbidden_tools_flagged_even_if_failed():
    r = M.tool_call_correctness([], ["issue_credit"], [rec("issue_credit", ok=False)])
    assert r["forbidden_called"] == ["issue_credit"] and not r["correct"]


# ---- hallucination ------------------------------------------------------------

def agent(text, calls=()):
    return {"role": "agent", "text": text, "tool_calls": [c.to_dict() for c in calls]}


def caller(text):
    return {"role": "caller", "text": text}


BILL = ToolCallRecord("get_bill", {"account_id": "ACC-1002"}, True,
                      result={"total": 90.0, "due_date": "2026-03-16", "items": [{"description": "x", "amount": 45.0}]})


def test_grounded_reply_has_no_claims():
    turns = [caller("my bill?"), agent("Your bill is $90.00, due Monday 16 March 2026.", [BILL])]
    assert M.hallucination_check(turns, TODAY, 50) == []


def test_invented_amount_and_date_flagged():
    turns = [caller("my bill?"), agent("Your bill is $95.00, due 20 March 2026.", [BILL])]
    kinds = sorted(c.kind for c in M.hallucination_check(turns, TODAY, 50))
    assert kinds == ["amount", "date"]


def test_amount_from_caller_is_grounded():
    turns = [caller("I want $75 back"), agent("I can't refund $75.00 myself.")]
    assert M.hallucination_check(turns, TODAY, 50) == []


def test_outage_claim_without_check_and_contradiction():
    assert M.hallucination_check([agent("There's no known outage in your area.")], TODAY, 50)[0].kind == "outage"
    chk = ToolCallRecord("check_outage", {"postcode": "SE5 9ZZ"}, True, result={"outage": True, "eta": "2026-03-02 18:00"})
    claims = M.hallucination_check([agent("There's no known outage for SE5 9ZZ.", [chk])], TODAY, 50)
    assert claims and "outage=True" in claims[0].detail


def test_action_claims_require_successful_tools():
    claims = M.hallucination_check([agent("I've applied a credit. You're booked. I'm transferring you now.")], TODAY, 50)
    assert sorted(c.kind for c in claims) == ["action", "action", "action"]
    failed = ToolCallRecord("issue_credit", {}, False, error="exceeds")
    assert M.hallucination_check([agent("I've applied a credit.", [failed])], TODAY, 50)[0].kind == "action"


def test_offer_to_transfer_is_not_a_claim():
    assert M.hallucination_check([agent("I can transfer you to a colleague if you like.")], TODAY, 50) == []


def test_unknown_reference_flagged():
    claims = M.hallucination_check([agent("Your booking reference is TB-0042.")], TODAY, 50)
    assert claims[0].kind == "reference"


# ---- latency & cost -----------------------------------------------------------

def test_percentile_matches_linear_interpolation():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert M.percentile(xs, 50) == pytest.approx(5.5)
    assert M.percentile(xs, 95) == pytest.approx(9.55)
    assert M.percentile([], 50) is None
    assert M.percentile([3.0], 95) == 3.0


def test_latency_and_cost():
    s = M.latency_stats([10.0, 20.0, 30.0])
    assert s["p50_ms"] == 20.0 and s["max_ms"] == 30.0 and s["n"] == 3
    c = M.cost_summary(100, 20, 1.23456)
    assert c["total_tokens"] == 120 and c["api_cost_usd"] == 0.0 and c["wall_clock_s"] == 1.2346
