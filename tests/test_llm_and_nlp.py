import json
from datetime import date

from voice_agent_evals.agent.llm import LLMAgent
from voice_agent_evals.backend import MockBackend
from voice_agent_evals.config import Settings
from voice_agent_evals.evals.caller import END_TOKEN, LLMCaller
from voice_agent_evals.evals.judge import llm_judge
from voice_agent_evals.evals.scenario import load_scenarios
from voice_agent_evals.nlp import (
    extract_amounts, extract_dates, extract_email, extract_phone, spoken_digits_to_numbers,
)

TODAY = date(2026, 3, 2)


class FakeClient:
    """Replays canned OpenAI-style responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)


def tool_msg(name, args, call_id="c1"):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def text_msg(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 15}}


def test_llm_agent_tool_loop_tokens_and_escalation():
    client = FakeClient([
        tool_msg("lookup_account", {"phone": "555-0106"}),
        tool_msg("escalate_to_human", {"reason": "asked for human", "account_id": "ACC-1006"}, "c2"),
        text_msg("I'm transferring you to a colleague now."),
    ])
    agent = LLMAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    turn = agent.respond("555-0106, get me a human")
    assert [c.name for c in turn.tool_calls] == ["lookup_account", "escalate_to_human"]
    assert turn.escalated and turn.ended
    assert turn.prompt_tokens == 320 and turn.completion_tokens == 35
    assert agent.handoff["account_id"] == "ACC-1006"
    sys_prompt = client.requests[0]["messages"][0]["content"]
    assert "escalate_to_human" in sys_prompt and "2026-03-02" in sys_prompt
    assert client.requests[0]["tools"]
    # tool results are fed back to the model
    assert client.requests[1]["messages"][-1]["role"] == "tool"


def test_llm_caller_ends_on_token():
    s = load_scenarios("scenarios", only=["billing_duplicate_charge"])[0]
    c = LLMCaller(s, FakeClient([text_msg("I was charged twice."), text_msg(END_TOKEN)]))
    assert c.next("Hi") == "I was charged twice."
    assert c.next("Done.") is None
    assert "555-0102" in c.system


def test_llm_judge_parses_and_handles_garbage():
    turns = [{"role": "agent", "text": "Your bill is $99.", "tool_calls": []}]
    ok = llm_judge(turns, FakeClient([text_msg('Sure: {"unsupported_claims": [{"turn": 0, "claim": "$99", "why": "no tool"}]}')]))
    assert ok["ok"] and ok["unsupported_claims"][0]["claim"] == "$99"
    bad = llm_judge(turns, FakeClient([text_msg("no idea")]))
    assert bad["ok"] is False


def test_nlp_helpers():
    assert spoken_digits_to_numbers("five five five oh one oh two") == "5550102"
    assert extract_phone("it's 555-0102.") == "555-0102"
    assert extract_phone("my bill is $45.00") is None
    assert extract_email("marcus dot bell at example dot com") == "marcus.bell@example.com"
    assert extract_amounts("$1,045.50 and 20 dollars") == [1045.5, 20.0]
    assert extract_dates("Thursday 5 March 2026", TODAY) == [date(2026, 3, 5)]
    assert extract_dates("the 10th of March", TODAY) == [date(2026, 3, 10)]
    assert extract_dates("March 7th", TODAY) == [date(2026, 3, 7)]
    assert extract_dates("tomorrow", TODAY, relative=True) == [date(2026, 3, 3)]
    assert extract_dates("next Wednesday", TODAY, relative=True) == [date(2026, 3, 4)]
    assert extract_dates("Monday", TODAY, relative=True) == [date(2026, 3, 9)]
    assert extract_dates("tomorrow", TODAY) == []
