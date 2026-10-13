"""Compare several agent variants run on the same scenarios with the same caller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import aggregate


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _secs(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.2f} ms" if ms < 100 else f"{ms / 1000:.2f} s"


def load_holdout(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


def subset_summary(run: dict[str, Any], ids: set[str], include: bool = True) -> dict[str, Any]:
    """Aggregate only the scenarios in ``ids`` (or all others if ``include`` is False)."""
    return aggregate([r for r in run["results"] if (r["scenario_id"] in ids) == include])


METRICS: list[tuple[str, Any]] = [
    ("Passed (all checks)", lambda s: f"{s['passed']}/{s['n']} ({_pct(s['pass_rate'])})"),
    ("Task completion", lambda s: _pct(s["task_completion_rate"])),
    ("Escalation precision", lambda s: _pct(s["escalation"]["precision"])),
    ("Escalation recall", lambda s: _pct(s["escalation"]["recall"])),
    ("Escalation TP/FP/FN/TN", lambda s: "{tp}/{fp}/{fn}/{tn}".format(**s["escalation"])),
    ("Tool-call correctness", lambda s: _pct(s["tool_call_correct_rate"])),
    ("Hallucination-free (rule-based)", lambda s: _pct(s["hallucination_free_rate"])),
    ("Unsupported claims flagged", lambda s: str(s["unsupported_claims"])),
    ("Blocked actions (convs affected)",
     lambda s: f"{s.get('blocked_actions', 0)} ({s.get('conversations_with_blocked_actions', 0)})"),
    ("Escalations forced by the graph", lambda s: str(s.get("forced_actions", 0))),
    ("Drafts rejected by grounding check (fallbacks)",
     lambda s: f"{s.get('grounding_revisions', 0)} ({s.get('grounding_fallbacks', 0)})"),
    ("Mean caller turns", lambda s: f"{s['mean_caller_turns']:.2f}"),
    ("Latency per turn p50 / p95",
     lambda s: f"{_secs(s['latency']['p50_ms'])} / {_secs(s['latency']['p95_ms'])}"),
    ("Tokens per conversation (prompt + completion)",
     lambda s: f"{s['mean_tokens_per_conversation']:.0f} ({s.get('mean_prompt_tokens_per_conversation', 0):.0f}"
               f" + {s.get('mean_completion_tokens_per_conversation', 0):.0f})"),
    ("Wall-clock per conversation", lambda s: _secs(s["mean_wall_clock_s"] * 1000)),
    ("Runtime errors", lambda s: str(s["errors"])),
]


def comparison_rows(summaries: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Rows of [metric, value for each label]."""
    return [[name, *(fn(s) if s.get("n") else "n/a" for s in summaries.values())] for name, fn in METRICS]


def markdown_table(summaries: dict[str, dict[str, Any]]) -> str:
    labels = list(summaries)
    lines = ["| Metric | " + " | ".join(f"`{x}`" for x in labels) + " |", "|---|" + "---|" * len(labels)]
    lines += ["| " + " | ".join(row) + " |" for row in comparison_rows(summaries)]
    return "\n".join(lines)


def variance_table(runs: dict[str, dict[str, Any]]) -> str:
    """Pass rate across repeated runs of the same agent (label 'agent@seedN')."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for label, run in runs.items():
        by_agent.setdefault(label.split("@")[0], []).append(run["summary"])
    rows = ["| Agent | Runs | Pass rate per run | Task completion per run | Tool-call correctness per run |",
            "|---|---|---|---|---|"]
    for agent, ss in by_agent.items():
        if len(ss) < 2:
            continue
        rows.append(f"| `{agent}` | {len(ss)} | " + " / ".join(_pct(s["pass_rate"]) for s in ss) + " | "
                    + " / ".join(_pct(s["task_completion_rate"]) for s in ss) + " | "
                    + " / ".join(_pct(s["tool_call_correct_rate"]) for s in ss) + " |")
    return "\n".join(rows) if len(rows) > 2 else ""


def scenario_matrix(runs: dict[str, dict[str, Any]], holdout: set[str]) -> str:
    labels = list(runs)
    ids = [r["scenario_id"] for r in next(iter(runs.values()))["results"]]
    lines = ["| Scenario | " + " | ".join(f"`{x}`" for x in labels) + " |", "|---|" + "---|" * len(labels)]
    for sid in ids:
        cells = []
        for run in runs.values():
            r = next((x for x in run["results"] if x["scenario_id"] == sid), None)
            cells.append("-" if r is None else ("PASS" if r["passed"] else "fail"))
        lines.append(f"| {sid}{' (held-out)' if sid in holdout else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_markdown(runs: dict[str, dict[str, Any]], holdout: set[str] | None = None) -> str:
    holdout = holdout or set()
    first = next(iter(runs.values()))["meta"]
    mach = first["machine"]
    models = sorted({r["meta"].get("agent_model") for r in runs.values() if r["meta"].get("agent_model")})
    out = [
        f"Run: {first['run_at']} | caller: `{first['caller']}` (fixed) | model: {', '.join(models) or 'none'} "
        f"| commit: `{first.get('git_commit') or 'n/a'}`",
        f"Machine: {mach['cpu']}, {mach['ram_gb']} GB RAM, {mach['os']}, Python {mach['python']}",
        "API cost: $0.00 for every variant (local model); tokens are shown instead.",
        "",
        "### All scenarios",
        "",
        markdown_table({k: v["summary"] for k, v in runs.items()}),
    ]
    if holdout:
        n_dev = sum(1 for r in next(iter(runs.values()))["results"] if r["scenario_id"] not in holdout)
        out += ["", f"### Development scenarios ({n_dev}, excluding held-out)", "",
            markdown_table({k: subset_summary(v, holdout, include=False) for k, v in runs.items()}),
            "", f"### Held-out scenarios ({len(holdout)}, never used to tune the agents)", "",
            markdown_table({k: subset_summary(v, holdout) for k, v in runs.items()})]
    var = variance_table(runs)
    if var:
        out += ["", "### Run-to-run variance", "", var]
    out += ["", "### Per-scenario results", "", scenario_matrix(runs, holdout)]
    return "\n".join(out) + "\n"


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    """A small JSON record of a run: summary plus per-scenario metrics and a short transcript."""
    rows = []
    for r in run["results"]:
        m = r["metrics"]
        transcript = []
        for t in r["turns"]:
            transcript.append(f"{t['role']}: {t['text']}")
            for c in t.get("tool_calls") or []:
                transcript.append(f"  tool {c['name']}({json.dumps(c['args'])}) -> "
                                  f"{'ok' if c['ok'] else 'ERROR ' + str(c['error'])}")
            for b in t.get("blocked") or []:
                transcript.append(f"  BLOCKED {b['name']}({json.dumps(b['args'])}): {b['reason']}")
            for g in t.get("grounding") or []:
                claims = "; ".join(f"{v['kind']} {v['claim']}" for v in g["violations"])
                transcript.append(f"  GROUNDING {g['action']}: {claims}")
            if t.get("nodes"):
                transcript.append("  graph: " + " -> ".join(t["nodes"]))
        rows.append({
            "scenario_id": r["scenario_id"], "passed": r["passed"], "error": r["error"],
            "task_completed": m["task_completed"], "task_problems": m["task_problems"],
            "escalation": m["escalation_label"], "tool_calls_correct": m["tool_calls"]["correct"],
            "tool_calls_missing": m["tool_calls"]["missing"], "forbidden_called": m["tool_calls"]["forbidden_called"],
            "unsupported_claims": m["hallucinations"], "blocked_actions": m.get("blocked_actions", 0),
            "forced_actions": m.get("forced_actions", 0), "grounding_revisions": m.get("grounding_revisions", 0),
            "caller_turns": r["caller_turns"],
            "latency": m["latency"], "cost": m["cost"], "transcript": transcript,
        })
    return {"meta": run["meta"], "summary": run["summary"], "results": rows}
