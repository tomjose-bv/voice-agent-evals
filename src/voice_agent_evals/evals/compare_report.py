"""Self-contained HTML page for `vae compare`."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .compare import comparison_rows, subset_summary

STYLE = """
:root { --bg:#fbfbfa; --fg:#1d1d1f; --muted:#666; --card:#fff; --line:#e3e3e0; --ok:#1b7f3b; --bad:#b42318; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececec; --muted:#a0a0a0; --card:#1f1f22;
  --line:#333; --ok:#4cc27a; --bad:#ff6b5e; } }
body { margin:0; font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 12px; }
.meta, .note { color:var(--muted); font-size:13px; }
.wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); font-size:14px; }
th { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.pass { color:var(--ok); font-weight:600; } .fail { color:var(--bad); }
"""


def _table(header: list[str], rows: list[list[str]], cls: dict[str, str] | None = None) -> str:
    cls = cls or {}
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f'<td class="{cls.get(c, "")}">{html.escape(c)}</td>' for c in row) + "</tr>"
        for row in rows
    )
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_compare_html(runs: dict[str, dict[str, Any]], holdout: set[str], links: dict[str, str]) -> str:
    labels = list(runs)
    first = next(iter(runs.values()))["meta"]
    mach = first["machine"]
    models = sorted({r["meta"].get("agent_model") for r in runs.values() if r["meta"].get("agent_model")})
    parts = [
        "<h1>From prompt to governed graph</h1>",
        f'<p class="meta">{html.escape(first["run_at"])} | caller: {html.escape(first["caller"])} (fixed) | '
        f"model: {html.escape(', '.join(models) or 'none')} | {html.escape(mach['cpu'])}, {mach['ram_gb']} GB RAM | "
        f"commit {html.escape(first.get('git_commit') or 'n/a')}</p>",
        '<p class="note">Same scenarios, same scripted caller, temperature 0. Only the agent changes. '
        "API cost is $0.00 for every variant (local model); tokens are shown instead.</p>",
        "<h2>All scenarios</h2>",
        _table(["Metric", *labels], comparison_rows({k: v["summary"] for k, v in runs.items()})),
    ]
    if holdout:
        parts += [
            "<h2>Development scenarios</h2>",
            _table(["Metric", *labels],
                   comparison_rows({k: subset_summary(v, holdout, include=False) for k, v in runs.items()})),
            f"<h2>Held-out scenarios ({len(holdout)})</h2>",
            _table(["Metric", *labels], comparison_rows({k: subset_summary(v, holdout) for k, v in runs.items()})),
        ]
    ids = [r["scenario_id"] for r in next(iter(runs.values()))["results"]]
    rows = []
    for sid in ids:
        row = [sid + (" (held-out)" if sid in holdout else "")]
        for run in runs.values():
            r = next((x for x in run["results"] if x["scenario_id"] == sid), None)
            row.append("-" if r is None else ("PASS" if r["passed"] else "fail"))
        rows.append(row)
    parts += ["<h2>Per-scenario results</h2>", _table(["Scenario", *labels], rows, {"PASS": "pass", "fail": "fail"})]
    if links:
        items = "".join(f'<li><a href="{html.escape(href)}">{html.escape(lbl)}</a></li>' for lbl, href in links.items())
        parts += ["<h2>Full reports (transcripts, tool calls, blocked actions)</h2>", f"<ul>{items}</ul>"]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Agent Variant Comparison</title><style>{STYLE}</style></head><body><main>"
            + "".join(parts) + "</main></body></html>")


def write_compare_html(runs: dict[str, dict[str, Any]], holdout: set[str], links: dict[str, str],
                       path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_compare_html(runs, holdout, links), encoding="utf-8")
    return p
