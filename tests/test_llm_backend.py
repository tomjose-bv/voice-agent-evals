"""LLM backend (`prompt` variant) against a fake OpenAI-compatible client. No Ollama needed."""

import json

import httpx

from voice_agent_evals.agent.llm import ChatClient, LLMAgent, parse_tool_calls
from voice_agent_evals.backend import MockBackend
from voice_agent_evals.config import Settings


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)


def _text(t, p=50, c=5):
    return {"choices": [{"message": {"role": "assistant", "content": t}}],
            "usage": {"prompt_tokens": p, "completion_tokens": c}}


def test_parse_tool_calls_structured_and_text_fallback():
    structured = {"tool_calls": [{"id": "1", "function": {"name": "get_bill", "arguments": "{}"}}]}
    assert parse_tool_calls(structured)[0]["function"]["name"] == "get_bill"
    text = {"content": '<tool_call>\n{"name": "lookup_account", "arguments": {"phone": "555-0102"}}\n</tool_call>'}
    calls = parse_tool_calls(text)
    assert calls[0]["function"]["name"] == "lookup_account"
    assert json.loads(calls[0]["function"]["arguments"]) == {"phone": "555-0102"}
    # unknown tool names and ordinary JSON-ish replies are not treated as actions
    assert parse_tool_calls({"content": '{"name": "delete_everything", "arguments": {}}'}) == []
    assert parse_tool_calls({"content": "Your total is $45.00."}) == []


def test_prompt_variant_sends_full_history_and_handles_dict_args():
    client = FakeClient([
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "lookup_account", "arguments": {"phone": "555-0101"}}}]}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        _text("Thanks Priya, you're verified."),
        _text("Bye!"),
    ])
    agent = LLMAgent(MockBackend(), Settings(), client=client)
    agent.greet()
    t1 = agent.respond("My number is 555-0101")
    assert [c.name for c in t1.tool_calls] == ["lookup_account"] and t1.tool_calls[0].ok
    assert t1.prompt_tokens == 60 and t1.completion_tokens == 7
    agent.respond("bye")
    last = client.requests[-1]["messages"]
    roles = [m["role"] for m in last]
    assert roles[:3] == ["system", "assistant", "user"] and "tool" in roles
    assert last[-1] == {"role": "user", "content": "bye"}
    assert agent.name == "prompt" and agent.blocked == []


def test_empty_reply_falls_back_to_text():
    agent = LLMAgent(MockBackend(), Settings(), client=FakeClient([_text("")]))
    agent.greet()
    assert agent.respond("hello").text


def test_chat_client_sends_seed_and_temperature():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_text("ok"))

    client = ChatClient("http://fake/v1", "k", "m", seed=11)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert seen["seed"] == 11 and seen["temperature"] == 0.0 and seen["tool_choice"] == "auto"
