"""LLM agent backend over any OpenAI-compatible chat-completions endpoint.

Defaults to a local Ollama server (http://localhost:11434/v1). Nothing here
talks to a cloud API unless you point VAE_LLM_BASE_URL at one yourself.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..backend import OUTAGE_CREDIT_AMOUNT, MockBackend
from ..config import Settings
from ..tools import TOOL_SCHEMAS, ToolExecutor
from .base import AgentTurn
from .handoff import build_handoff_summary
from .prompt import GREETING, build_system_prompt

MAX_TOOL_ROUNDS = 6


class LLMUnavailable(RuntimeError):
    pass


class ChatClient:
    """Minimal OpenAI-compatible client (no SDK dependency)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.0) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
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


class LLMAgent:
    name = "llm"

    def __init__(self, backend: MockBackend, settings: Settings, client: ChatClient | None = None) -> None:
        self.backend = backend
        self.settings = settings
        self.client = client or ChatClient(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s
        )
        self.tools = ToolExecutor(backend)
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(backend.today, backend.credit_limit, OUTAGE_CREDIT_AMOUNT),
            }
        ]
        self.ended = False
        self.escalated = False
        self.handoff: dict[str, Any] | None = None
        self.account: dict[str, Any] | None = None
        self.transcript: list[tuple[str, str]] = []

    def greet(self) -> AgentTurn:
        self.messages.append({"role": "assistant", "content": GREETING})
        self.transcript.append(("agent", GREETING))
        return AgentTurn(GREETING)

    def respond(self, user_text: str) -> AgentTurn:
        start = len(self.tools.log)
        self.messages.append({"role": "user", "content": user_text})
        self.transcript.append(("caller", user_text))
        p_tok = c_tok = 0
        text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            data = self.client.chat(self.messages, TOOL_SCHEMAS)
            usage = data.get("usage") or {}
            p_tok += int(usage.get("prompt_tokens") or 0)
            c_tok += int(usage.get("completion_tokens") or 0)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
                                 | {"role": "assistant"})
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                break
            for tc in tool_calls:
                fn = tc.get("function") or {}
                rec = self.tools.call(fn.get("name", ""), fn.get("arguments") or {})
                if rec.name == "lookup_account" and rec.ok and (rec.result or {}).get("found"):
                    self.account = rec.result
                if rec.name == "escalate_to_human" and rec.ok:
                    self.escalated = True
                content = json.dumps(rec.result if rec.ok else {"error": rec.error})
                self.messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": content})
        else:
            text = text or "Sorry, I'm having trouble with that. Let me transfer you to a colleague."
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
        )
