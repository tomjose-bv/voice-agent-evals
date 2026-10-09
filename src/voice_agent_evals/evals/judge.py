"""Optional LLM judge for hallucinations the rule-based checks can't see.

Only used with ``vae run --judge`` and a reachable LLM endpoint. Its output is
reported separately from the rule-based metric so the two are never mixed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..agent.llm import ChatClient

JUDGE_PROMPT = """You are auditing a customer-service phone agent for unsupported claims.
Below is the conversation. Lines starting TOOL show exactly what the agent's tools returned.
An unsupported claim is any factual statement by the AGENT (amounts, dates, times, outage status,
account details, actions taken, policies) that is not supported by a TOOL line or by the CALLER.
Return ONLY JSON: {{"unsupported_claims": [{{"turn": <int>, "claim": "<text>", "why": "<short>"}}]}}

CONVERSATION:
{conversation}"""


def render_conversation(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, t in enumerate(turns):
        if t["role"] == "agent":
            for c in t.get("tool_calls") or []:
                payload = c.get("result") if c.get("ok") else {"error": c.get("error")}
                lines.append(f"TOOL {c['name']}({json.dumps(c.get('args'))}) -> {json.dumps(payload)}")
            lines.append(f"[{i}] AGENT: {t['text']}")
        else:
            lines.append(f"[{i}] CALLER: {t['text']}")
    return "\n".join(lines)


def llm_judge(turns: list[dict[str, Any]], client: ChatClient) -> dict[str, Any]:
    data = client.chat([{"role": "user", "content": JUDGE_PROMPT.format(conversation=render_conversation(turns))}])
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    try:
        parsed = json.loads(m.group(0)) if m else {}
        claims = parsed.get("unsupported_claims", [])
        if not isinstance(claims, list):
            raise ValueError
        return {"ok": True, "unsupported_claims": claims}
    except (ValueError, json.JSONDecodeError):
        return {"ok": False, "error": "judge returned unparseable output", "raw": content[:500]}
