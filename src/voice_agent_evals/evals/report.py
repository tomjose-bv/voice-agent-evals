"""JSON, Markdown and self-contained HTML reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f} ms"


def summary_rows(run: dict[str, Any]) -> list[tuple[str, str]]:
    s = run["summary"]
    e = s["escalation"]
    lat = s["latency"]
    return [
        ("Scenarios", str(s["n"])),
        ("Passed (all checks)", f"{s['passed']}/{s['n']} ({_pct(s['pass_rate'])})"),
        ("Task completion", _pct(s["task_completion_rate"])),
        ("Escalation precision", _pct(e["precision"])),
        ("Escalation recall", _pct(e["recall"])),
        ("Escalation confusion (TP/FP/FN/TN)", f"{e['tp']}/{e['fp']}/{e['fn']}/{e['tn']}"),
        ("Tool-call correctness", _pct(s["tool_call_correct_rate"])),
        ("Expected tool calls matched (mean recall)", _pct(s["tool_call_mean_recall"])),
        ("Forbidden tool calls", str(s["forbidden_tool_calls"])),
        ("Hallucination-free conversations (rule-based)", _pct(s["hallucination_free_rate"])),
        ("Unsupported claims flagged", str(s["unsupported_claims"])),
        ("Blocked actions (guard refusals)", str(s.get("blocked_actions", 0))),
        ("Mean caller turns", f"{s['mean_caller_turns']:.2f}"),
        ("Agent latency per turn p50 / p95", f"{_ms(lat['p50_ms'])} / {_ms(lat['p95_ms'])}"),
        ("Mean wall-clock per conversation", f"{s['mean_wall_clock_s'] * 1000:.2f} ms"),
        ("Tokens per conversation (mean)", f"{s['mean_tokens_per_conversation']:.0f}"),
        ("API cost", f"${s['api_cost_usd']:.2f} (local, no per-token billing)"),
    ]


def summary_markdown(run: dict[str, Any]) -> str:
    m = run["meta"]
    mach = m["machine"]
    lines = [
        f"Run: {m['run_at']} | agent backend: `{m['agent_backend']}`"
        + (f" ({m['agent_model']})" if m.get("agent_model") else "")
        + f" | caller: `{m['caller']}` | commit: `{m.get('git_commit') or 'n/a'}`",
        f"Machine: {mach['cpu']}, {mach['ram_gb']} GB RAM, {mach['os']}, Python {mach['python']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary_rows(run)]
    lines += ["", "| Scenario | Category | Pass | Task | Escalation | Tools | Halluc. | Turns |", "|---|---|---|---|---|---|---|---|"]
    for r in run["results"]:
        mt = r["metrics"]
        lines.append(
            f"| {r['scenario_id']} | {r['category']} | {'PASS' if r['passed'] else 'FAIL'} | "
            f"{'ok' if mt['task_completed'] else 'no'} | {mt['escalation_label']} | "
            f"{mt['tool_calls']['matched']}/{mt['tool_calls']['expected']}"
            f"{' +forbidden' if mt['tool_calls']['forbidden_called'] else ''} | "
            f"{len(mt['hallucinations'])} | {r['caller_turns']} |"
        )
    return "\n".join(lines) + "\n"


def write_json(run: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return p


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Agent Eval Report</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1f; --muted:#666; --card:#fff; --line:#e3e3e0; --ok:#1b7f3b; --bad:#b42318; --accent:#3557c6; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececec; --muted:#a0a0a0; --card:#1f1f22; --line:#333; --ok:#4cc27a; --bad:#ff6b5e; --accent:#8fa7ff; } }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 12px; }
.meta { color:var(--muted); font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:12px; margin-top:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.card .k { color:var(--muted); font-size:12px; } .card .v { font-size:20px; font-weight:600; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); font-size:14px; vertical-align:top; }
th { font-weight:600; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
tr.row { cursor:pointer; } tr.row:hover { background:rgba(127,127,127,.08); }
.pass { color:var(--ok); font-weight:600; } .fail { color:var(--bad); font-weight:600; }
.detail td { background:rgba(127,127,127,.05); }
.turn { margin:6px 0; padding:8px 10px; border-radius:8px; border:1px solid var(--line); background:var(--card); }
.turn.caller { border-left:3px solid var(--accent); } .turn.agent { border-left:3px solid var(--ok); }
.role { font-size:11px; text-transform:uppercase; color:var(--muted); }
.tool { font-family:ui-monospace, Consolas, monospace; font-size:12px; color:var(--muted); white-space:pre-wrap; word-break:break-word; }
.issues li { color:var(--bad); }
.wrap { overflow-x:auto; }
.note { color:var(--muted); font-size:13px; max-width:75ch; }
</style>
</head>
<body>
<main>
<h1>Voice Agent Eval Report</h1>
<div class="meta" id="meta"></div>
<div class="cards" id="cards"></div>
<p class="note">API cost is $0.00 because every model runs locally; compute cost shows up as wall-clock time and tokens.
Hallucination checks are rule-based (amounts, dates, outage status, action claims, references) and deliberately narrow.</p>
<h2>Scenarios</h2>
<div class="wrap"><table id="tbl"><thead><tr><th>Scenario</th><th>Category</th><th>Result</th><th>Task</th><th>Escalation</th><th>Tools</th><th>Unsupported claims</th><th>Turns</th><th>p50 latency</th></tr></thead><tbody></tbody></table></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const run = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => x == null ? 'n/a' : (x*100).toFixed(1) + '%';
const m = run.meta, s = run.summary, e = s.escalation;
document.getElementById('meta').textContent =
  `${m.run_at} | agent: ${m.agent_backend}${m.agent_model ? ' ('+m.agent_model+')' : ''} | caller: ${m.caller} | ` +
  `${m.machine.cpu}, ${m.machine.ram_gb} GB RAM, Python ${m.machine.python} | commit ${m.git_commit || 'n/a'}`;
const cards = [
  ['Passed', `${s.passed}/${s.n}`], ['Task completion', pct(s.task_completion_rate)],
  ['Escalation precision', pct(e.precision)], ['Escalation recall', pct(e.recall)],
  ['Tool-call correctness', pct(s.tool_call_correct_rate)], ['Hallucination-free', pct(s.hallucination_free_rate)],
  ['Latency p50', s.latency.p50_ms == null ? 'n/a' : s.latency.p50_ms.toFixed(2)+' ms'],
  ['Latency p95', s.latency.p95_ms == null ? 'n/a' : s.latency.p95_ms.toFixed(2)+' ms'],
  ['Mean turns', s.mean_caller_turns.toFixed(2)], ['API cost', '$' + s.api_cost_usd.toFixed(2)],
];
document.getElementById('cards').innerHTML = cards.map(([k,v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join('');
const tb = document.querySelector('#tbl tbody');
run.results.forEach((r, i) => {
  const mt = r.metrics, tc = mt.tool_calls;
  const tr = document.createElement('tr'); tr.className = 'row';
  tr.innerHTML = `<td>${esc(r.scenario_id)}<br><span class="meta">${esc(r.title)}</span></td><td>${esc(r.category)}</td>
    <td class="${r.passed ? 'pass' : 'fail'}">${r.passed ? 'PASS' : 'FAIL'}</td><td>${mt.task_completed ? 'ok' : 'no'}</td>
    <td>${esc(mt.escalation_label)}</td><td>${tc.matched}/${tc.expected}${tc.forbidden_called.length ? ' +forbidden' : ''}</td>
    <td>${mt.hallucinations.length}</td><td>${r.caller_turns}</td><td>${mt.latency.p50_ms == null ? 'n/a' : mt.latency.p50_ms.toFixed(2)+' ms'}</td>`;
  const dt = document.createElement('tr'); dt.className = 'detail'; dt.hidden = true;
  const issues = [...mt.task_problems, ...tc.missing.map(x => 'missing tool call ' + x.name + ' ' + JSON.stringify(x.args)),
    ...tc.forbidden_called.map(x => 'forbidden tool called: ' + x), ...mt.hallucinations.map(h => `turn ${h.turn}: ${h.kind} "${h.claim}" (${h.detail})`)];
  if (r.error) issues.unshift('error: ' + r.error);
  const turns = r.turns.map(t => `<div class="turn ${t.role}"><div class="role">${esc(t.role)}${t.latency_ms != null ? ' &middot; ' + t.latency_ms.toFixed(2) + ' ms' : ''}</div>${esc(t.text)}` +
    (t.tool_calls || []).map(c => `<div class="tool">&rarr; ${esc(c.name)}(${esc(JSON.stringify(c.args))}) ${c.ok ? '= ' + esc(JSON.stringify(c.result)) : 'ERROR ' + esc(c.error)}</div>`).join('') +
    (t.blocked || []).map(b => `<div class="tool fail">&#8856; BLOCKED ${esc(b.name)}(${esc(JSON.stringify(b.args))}): ${esc(b.reason)}</div>`).join('') +
    ((t.nodes || []).length ? `<div class="tool">graph: ${t.nodes.map(esc).join(' &rarr; ')}</div>` : '') + `</div>`).join('');
  const handoff = r.handoff ? `<p><b>Handoff summary:</b></p><pre class="tool">${esc(JSON.stringify(r.handoff, null, 2))}</pre>` : '';
  dt.innerHTML = `<td colspan="9">${issues.length ? '<ul class="issues">' + issues.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul>' : '<p class="pass">All checks passed.</p>'}${turns}${handoff}
    <p class="tool">final state: ${esc(JSON.stringify(r.final_state))}</p></td>`;
  tr.onclick = () => { dt.hidden = !dt.hidden; };
  tb.appendChild(tr); tb.appendChild(dt);
});
</script>
</body>
</html>
"""


def render_html(run: dict[str, Any]) -> str:
    data = json.dumps(run, default=str).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data)


def write_html(run: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(run), encoding="utf-8")
    return p
