"""`context` variant: the `prompt` agent plus context engineering.

Same model, same tools, same rules wording. What changes is what the model sees
on each call: see ``context.py``.
"""

from __future__ import annotations

from typing import Any

from ..backend import OUTAGE_CREDIT_AMOUNT, MockBackend
from ..config import Settings
from .context import (
    ContextBuilder,
    ContextStats,
    ConversationState,
    extract_facts,
    render_facts,
    render_state,
    select_rules,
    update_state,
)
from .llm import ChatClient, LLMAgent
from .prompt import build_prompt_with_rules


class ContextAgent(LLMAgent):
    name = "context"

    def __init__(self, backend: MockBackend, settings: Settings, client: ChatClient | None = None,
                 builder: ContextBuilder | None = None) -> None:
        super().__init__(backend, settings, client)
        self.state = ConversationState()
        self.builder = builder or ContextBuilder()
        self.context_stats: list[ContextStats] = []

    def on_turn_start(self, user_text: str) -> None:
        update_state(self.state, user_text, self.backend.today)

    def extra_instructions(self) -> str:
        """Hook for subclasses (the graph adds the current node's instructions)."""
        return ""

    def system_message(self) -> str:
        facts = extract_facts(self.tools.log)
        rules = build_prompt_with_rules(
            self.backend.today, self.backend.credit_limit, OUTAGE_CREDIT_AMOUNT, select_rules(self.state)
        )
        parts = [rules, render_facts(facts), render_state(self.state, facts)]
        extra = self.extra_instructions()
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    def build_messages(self) -> list[dict[str, Any]]:
        history = self.transcript[:-1]  # the current caller turn is passed separately
        msgs, stats = self.builder.build(self.system_message(), history, self.transcript[-1][1], self.scratch)
        self.context_stats.append(stats)
        return msgs
