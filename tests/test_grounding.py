"""Pre-send grounding check and the `graph+grounding` variant (fake model, no Ollama)."""

import json
from datetime import date

from voice_agent_evals.agent import make_agent
from voice_agent_evals.agent.graph_agent import GraphAgent, GroundedGraphAgent
from voice_agent_evals.agent.grounding import (
    SAFE_REPLY, UNCONFIRMED, check_reply, gather_evidence, revision_request, strip_unsupported,
)
from voice_agent_evals.backend import MockBackend
from voice_agent_evals.config import Settings
from voice_agent_evals.tools import ToolExecutor

TODAY = date(2026, 3, 2)


def _ev(*calls, caller=()):
    ex = ToolExecutor(MockBackend())
    for name, args in calls:
        ex.call(name, args)
    return gather_evidence(ex.log, caller, TODAY, (50.0, 10.0))


def _kinds(text, ev, **kw):
    return [v.kind for v in check_reply(text, ev, TODAY, **kw)]


LOOKUP = ("lookup_account", {"phone": "555-0102"})
BILL = ("get_bill", {"account_id": "ACC-1002"})


def test_bill_facts_need_get_bill():
    only_lookup = _ev(LOOKUP)
    text = "Your latest bill is $99.00, due on 20 March."
    assert _kinds(text, only_lookup) == ["amount", "date"]
    billed = _ev(LOOKUP, BILL)
    assert _kinds("Your bill is $90.00, due Monday 16 March.", billed) == []
    assert _kinds(text, billed) == ["amount", "date"]


def test_caller_words_policy_constants_and_error_alternatives_count_as_evidence():
    ev = _ev(LOOKUP, ("book_technician", {"account_id": "ACC-1002", "date": "2026-03-08"}),
             caller=["I was charged $37.50 twice"])
    assert _kinds("You mentioned $37.50. Credits are capped at $50.00.", ev) == []
    assert _kinds("Sundays aren't available, but 3 March or 4 March are free.", ev) == []
    assert _kinds("How about 9 March?", ev) == ["date"]


def test_outage_status_must_come_from_check_outage():
    none = _ev()
    assert _kinds("Let me check if there's a known outage in your area.", none) == []
    assert _kinds("I'll find out whether there is an outage.", none) == []
    assert _kinds("There's a known outage in your area.", none) == ["outage"]
    clear = _ev(("check_outage", {"postcode": "LS6 1QT"}))
    assert _kinds("There is no known outage at LS6 1QT.", clear) == []
    assert _kinds("Yes, there is an outage affecting LS6 1QT.", clear) == ["outage"]
    active = _ev(("check_outage", {"postcode": "SE5 0RW"}))
    assert _kinds("There is no outage there.", active) == ["outage"]
    assert _kinds("There's an outage; expected fix 2 March at 18:00.", active) == []


def test_completed_actions_and_references_need_successful_tools():
    ev = _ev(LOOKUP, BILL)
    assert _kinds("I've applied a $45.00 credit.", ev) == ["action"]
    assert _kinds("You're booked for Tuesday.", ev) == ["action"]
    assert _kinds("I'm transferring you to a colleague.", ev) == ["action"]
    assert _kinds("I'm transferring you to a colleague.", ev, allow_transfer_claim=True) == []
    assert _kinds("Your account is ACC-1002.", ev) == []
    assert _kinds("Your bill is B-1002-02.", ev) == ["reference"]  # bill_id is trimmed from what the model sees
    assert _kinds("Your booking TB-4321 is set.", ev) == ["reference"]
    credited = _ev(LOOKUP, BILL, ("issue_credit", {"account_id": "ACC-1002", "amount": 45, "reason": "duplicate"}))
    assert _kinds("I've applied a $45.00 credit.", credited) == []


def test_revision_request_names_violations_and_tools():
    v = check_reply("Your bill is $99.00.", _ev(LOOKUP), TODAY)
    msg = revision_request(v, {"get_bill", "issue_credit"})
    assert '"$99.00"' in msg and "get_bill, issue_credit" in msg and "GROUNDING CHECK" in msg


def test_strip_unsupported_keeps_supported_sentences():
    ev = _ev(LOOKUP)
    text = "Thanks Marcus, I found your account. Your bill is $99.00. Anything else?"
    assert strip_unsupported(text, ev, TODAY) == f"Thanks Marcus, I found your account. Anything else? {UNCONFIRMED}"
    assert strip_unsupported("Your bill is $99.00.", ev, TODAY) == SAFE_REPLY
    assert strip_unsupported("Anything else?", ev, TODAY) == "Anything else?"


# ------------------------------------------------------------------- agent
class _Fake:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def chat(self, messages, tools=None, temperature=0.0):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)


def _call(name, args):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": name, "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


def _say(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


FABRICATED = "I found your account. Your bill is $99.00, due on 20 March."


def test_plain_graph_sends_ungrounded_reply_unchanged():
    client = _Fake([_call("lookup_account", {"phone": "555-0102"}), _say(FABRICATED)])
    agent = GraphAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    assert agent.respond("What's my bill? 555-0102").text == FABRICATED


def test_grounded_graph_revises_and_model_fetches_the_fact():
    client = _Fake([_call("lookup_account", {"phone": "555-0102"}), _say(FABRICATED),
                    _call("get_bill", {"account_id": "ACC-1002"}), _say("Your bill is $90.00, due Monday 16 March.")])
    agent = GroundedGraphAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    turn = agent.respond("What's my bill? 555-0102")
    assert turn.text == "Your bill is $90.00, due Monday 16 March."
    assert [c.name for c in turn.tool_calls] == ["lookup_account", "get_bill"]
    assert len(turn.grounding) == 1 and turn.grounding[0]["action"] == "revise"
    assert {v["kind"] for v in turn.grounding[0]["violations"]} == {"amount", "date"}
    retry = client.requests[2]
    assert retry["messages"][-2] == {"role": "assistant", "content": FABRICATED}
    assert "GROUNDING CHECK" in retry["messages"][-1]["content"]
    assert "get_bill" in {t["function"]["name"] for t in retry["tools"]}  # tools stay available
    assert "GROUNDING:" in client.requests[0]["messages"][0]["content"]
    # The rejected draft never reaches the transcript or the next turn's context.
    assert all(FABRICATED not in t for _, t in agent.transcript)
    assert sum(t.prompt_tokens for t in [turn]) == 40


def test_grounded_graph_falls_back_when_rewrite_still_ungrounded():
    client = _Fake([_call("lookup_account", {"phone": "555-0102"}), _say(FABRICATED),
                    _say("Sorry. Your bill is $99.00.")])
    agent = GroundedGraphAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    turn = agent.respond("What's my bill? 555-0102")
    assert "$99.00" not in turn.text and UNCONFIRMED in turn.text and turn.text.startswith("Sorry.")
    assert [g["action"] for g in turn.grounding] == ["revise", "fallback"] and agent.grounding == turn.grounding
    assert len(client.requests) == 3  # one rewrite only


def test_grounded_graph_accepts_grounded_reply_without_extra_call():
    client = _Fake([_say("Sure, what's the postcode? I'll check if there's an outage.")])
    agent = GroundedGraphAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    turn = agent.respond("My internet is down")
    assert turn.grounding == [] and len(client.requests) == 1


def test_grounded_graph_still_forces_escalation():
    agent = GroundedGraphAgent(MockBackend(), Settings(), client=_Fake([_say("I'm transferring you now.")]))
    agent.greet()
    turn = agent.respond("I want to speak to a real person, NOW.")
    assert turn.escalated and turn.grounding == [] and agent.forced


def test_factory_builds_grounded_variant():
    agent = make_agent("graph+grounding", MockBackend(), Settings())
    assert isinstance(agent, GroundedGraphAgent) and agent.name == "graph+grounding"


def test_compare_reports_grounding_and_reads_old_runs():
    from voice_agent_evals.evals import load_scenarios, run_suite
    from voice_agent_evals.evals.compare import compact_run, comparison_rows
    from voice_agent_evals.evals.compare_run import run_label
    from voice_agent_evals.evals.runner import aggregate

    run = run_suite(load_scenarios("scenarios", only=["off_topic_requests"]), "scripted", "scripted", Settings())
    old = dict(run["summary"])
    old.pop("grounding_revisions"), old.pop("grounding_fallbacks")  # runs saved before this metric existed
    row = next(r for r in comparison_rows({"old": old}) if r[0].startswith("Drafts rejected"))
    assert row[1] == "0 (0)"
    r = run["results"][0]
    r["turns"][-1]["grounding"] = [{"action": "revise", "violations": [{"kind": "amount", "claim": "$99.00"}]}]
    r["metrics"]["grounding_revisions"] = 1
    assert aggregate(run["results"])["grounding_revisions"] == 1
    assert "  GROUNDING revise: amount $99.00" in compact_run(run)["results"][0]["transcript"]
    assert run_label("graph+grounding", 7) == "graph+grounding@seed7"
