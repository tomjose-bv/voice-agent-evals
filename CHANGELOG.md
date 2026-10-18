# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-18

First public release. Everything below runs locally; no cloud API is used by default.

### Agent

- Mock backend for the fictional ISP "Northwind Internet" (accounts, bills,
  outages, technician slots, credits, escalations) with tool schemas and a
  tool executor.
- `scripted` agent: deterministic policy agent, no model. It follows the same
  rules as the LLM system prompt.
- `prompt` agent: an LLM over any OpenAI-compatible endpoint (default: local
  Ollama), with a fixed seed, a fallback for tool calls written as text, named
  rule snippets and a reply-length cap.
- `context` agent: context engineering on top of `prompt`. Each call gets a
  verified-facts block, a conversation-state summary, trimmed tool results and
  only the policy rules relevant to the caller's intent, within a token budget.
- `graph` agent: `context` inside a governed state machine. Each step offers
  only its own tools, a code-enforced guard refuses disallowed calls, and four
  escalation triggers are checked in code (the graph escalates itself if the
  model does not).
- `graph+grounding` agent: `graph` plus a pre-send grounding check. Amounts,
  dates, outage status, completed actions and references in a draft reply must
  be backed by tool results or the caller's words. Otherwise the model rewrites
  the reply once, and if it still fails, unsupported sentences are dropped.
- Handoff summary for escalated calls.

### Evaluation harness

- 25 YAML scenarios with ground truth cross-checked against the mock backend
  (`vae validate`), simulated callers (scripted or LLM), and a held-out subset
  of six scenarios fixed before agent iteration (`scenarios/holdout.txt`).
- Metrics: task completion, escalation precision/recall, tool-call
  correctness, rule-based unsupported-claim detection, an optional LLM judge,
  latency percentiles, tokens and wall-clock. HTML, JSON and Markdown reports.
- `vae compare`: runs several variants and seeds on the same scenarios and
  writes a comparison table with separate held-out results, run-to-run
  variance, per-scenario results and compact JSON. Runs are saved per scenario
  as they finish and can be resumed (`--resume`). On Windows, compare runs keep
  the machine and display awake so Modern Standby can't pause them.
- Compare tables report blocked actions, escalations forced by the graph and
  drafts rejected by the grounding check.

### Voice

- Local browser voice demo (`vae serve`): streaming speech-to-text
  (faster-whisper), text-to-speech (Piper or Windows SAPI), energy-based voice
  activity detection and barge-in over a websocket.

### Project

- `vae` CLI: `run`, `compare`, `validate`, `list`, `chat`, `serve`.
- One-command Windows setup script (uses `truststore` behind TLS inspection).
- Private-terms checker (`scripts/check_terms.py`).
- Tests use fakes and need no model, network or audio hardware. CI runs them
  on Ubuntu and Windows with Python 3.11 and 3.12.
- README with architecture, measured three-seed results on `qwen2.5:3b` for all five agents, metric
  definitions, limitations and responsible-AI notes. MIT licence.

[0.1.0]: https://github.com/tomjose-bv/voice-agent-evals/releases/tag/v0.1.0
