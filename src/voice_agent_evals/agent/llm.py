"""LLM agent backend over any OpenAI-compatible chat-completions endpoint.

Defaults to a local Ollama server (http://localhost:11434/v1). Nothing here
talks to a cloud API unless you point VAE_LLM_BASE_URL at one yourself.

``LLMAgent`` is the ``prompt`` variant: system prompt + tool schemas + the raw,
full conversation history, nothing else. The ``context`` and ``graph`` variants
subclass it and override the hooks marked "hook" below.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx

from ..backend import OUTAGE_CREDIT_AMOUNT, MockBackend
from ..config import Settings
from ..tools import TOOL_NAMES, TOOL_SCHEMAS, ToolCallRecord, ToolExecutor
from .base import AgentTurn
from .handoff import build_handoff_summary
from .prompt import GREETING, build_system_prompt

MAX_TOOL_ROUNDS = 6
FALLBACK_TEXT = "Sorry, I'm having trouble with that. Let me transfer you to a colleague."


class LLMUnavailable(RuntimeError):
    pass


class ChatClient:
    """Minimal OpenAI-compatible client (no SDK dependency)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 120.0,
                 seed: int | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self._http = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.0) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"cannot reach LLM endpoint {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMUnavailable(f"LLM endpoint error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def ping(self) -> bool:
        try:
            r = self._http.get(f"{self.base_url}/models", timeout=3.0)
            return r.status_code < 400
        except httpx.HTTPError:
            return False


_TEXT_TOOL_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return OpenAI-style tool calls from a message.

    Small models sometimes write the call as JSON in ``content`` (for example
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``) instead of
    using the structured field. We accept that only when the JSON names a
    known tool, so ordinary replies are never misread as actions.
    """
    calls = msg.get("tool_calls") or []
    if calls:
        return list(calls)
    content = msg.get("content") or ""
    m = _TEXT_TOOL_RE.search(content)
    if not m or '"name"' not in m.group(0):
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict) or obj.get("name") not in TOOL_NAMES:
        return []
    args = obj.get("arguments", obj.get("parameters", {}))
    return [{"id": f"text-{uuid.uuid4().hex[:8]}", "type": "function",
             "function": {"name": obj["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}}]


class LLMAgent:
    """`prompt` variant: system prompt + tools + full raw history."""

    name = "prompt"

    def __init__(self, backend: MockBackend, settings: Settings, client: ChatClient | None = None) -> None:
        self.backend = backend
        self.settings = settings
        self.client = client or ChatClient(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s,
            seed=settings.llm_seed,
        )
        self.tools = ToolExecutor(backend)
        self.system_prompt = build_system_prompt(backend.today, backend.credit_limit, OUTAGE_CREDIT_AMOUNT)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        self.ended = False
        self.escalated = False
        self.handoff: dict[str, Any] | None = None
        self.account: dict[str, Any] | None = None
        self.transcript: list[tuple[str, str]] = []
        self.blocked: list[dict[str, Any]] = []  # guard refusals (graph variant only)
        self.scratch: list[dict[str, Any]] = []  # assistant/tool messages of the current turn

    # ------------------------------------------------------------------ hooks
    def build_messages(self) -> list[dict[str, Any]]:
        """hook: what the model sees on this call. Here: the full raw history."""
        return self.messages

    def tools_for_call(self) -> list[dict[str, Any]]:
        """hook: tool schemas offered on this call."""
        return TOOL_SCHEMAS

    def guard(self, name: str, args: dict[str, Any]) -> str | None:
        """hook: return a policy error to refuse a tool call before it runs."""
        return None

    def on_turn_start(self, user_text: str) -> None:
        """hook: called once per caller turn before the first model call."""

    def on_tool_result(self, rec: ToolCallRecord) -> None:
        """hook: called after each executed tool call."""

    def on_turn_end(self, text: str) -> str:
        """hook: may post-process the reply text."""
        return text

    # ------------------------------------------------------------------ loop
    def greet(self) -> AgentTurn:
        self.messages.append({"role": "assistant", "content": GREETING})
        self.transcript.append(("agent", GREETING))
        return AgentTurn(GREETING)

    def _execute(self, tc: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw = fn.get("arguments") or {}
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else dict(raw or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            args = None
        refusal = self.guard(name, args) if isinstance(args, dict) else None
        if refusal:
            self.blocked.append({"name": name, "args": args, "reason": refusal})
            return json.dumps({"error": f"POLICY: {refusal}"}), []
        rec = self.tools.call(name, args if isinstance(args, dict) else raw)
        if rec.name == "lookup_account" and rec.ok and (rec.result or {}).get("found"):
            self.account = rec.result
        if rec.name == "escalate_to_human" and rec.ok:
            self.escalated = True
        self.on_tool_result(rec)
        return json.dumps(rec.result if rec.ok else {"error": rec.error}), [rec.to_dict()]

    def respond(self, user_text: str) -> AgentTurn:
        start, blocked_start = len(self.tools.log), len(self.blocked)
        self.messages.append({"role": "user", "content": user_text})
        self.transcript.append(("caller", user_text))
        self.scratch = []
        self.on_turn_start(user_text)
        p_tok = c_tok = 0
        text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            data = self.client.chat(self.build_messages(), self.tools_for_call())
            usage = data.get("usage") or {}
            p_tok += int(usage.get("prompt_tokens") or 0)
            c_tok += int(usage.get("completion_tokens") or 0)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            tool_calls = parse_tool_calls(msg)
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                break
            for tc in tool_calls:
                tc.setdefault("id", f"call-{uuid.uuid4().hex[:8]}")
                tc.setdefault("type", "function")
            assistant = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
            self.messages.append(assistant)
            self.scratch.append(assistant)
            for tc in tool_calls:
                content, _ = self._execute(tc)
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": content}
                self.messages.append(tool_msg)
                self.scratch.append(tool_msg)
        else:
            text = text or FALLBACK_TEXT
        text = self.on_turn_end(text or FALLBACK_TEXT)
        self.messages.append({"role": "assistant", "content": text})
        if self.escalated and not self.ended:
            self.ended = True
            self.handoff = build_handoff_summary(
                self.backend.escalations[-1].reason if self.backend.escalations else "escalated",
                self.account, self.transcript, self.tools.log,
            )
        self.transcript.append(("agent", text))
        return AgentTurn(
            text=text,
            tool_calls=self.tools.log[start:],
            escalated=self.escalated,
            ended=self.ended,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            blocked=self.blocked[blocked_start:],
        )
