"""`vae compare` report generation (no model needed: uses the scripted agent and fake LLM runs)."""

import copy
import json

from voice_agent_evals.config import Settings
from voice_agent_evals.evals import load_scenarios, run_suite
from voice_agent_evals.evals.compare import (
    compact_run, compare_markdown, comparison_rows, load_holdout, subset_summary, variance_table,
)
from voice_agent_evals.evals.compare_report import render_compare_html
from voice_agent_evals.evals.compare_run import run_label, run_variants, write_outputs

IDS = ["billing_duplicate_charge", "human_request_angry", "off_topic_requests"]


def _run():
    return run_suite(load_scenarios("scenarios", only=IDS), "scripted", "scripted", Settings())


def _fake_llm_run(base, passed, blocked):
    run = copy.deepcopy(base)
    run["meta"]["agent_backend"] = "graph"
    run["meta"]["agent_model"] = "tiny-model"
    for r, p in zip(run["results"], passed):
        r["passed"] = p
        r["metrics"]["blocked_actions"] = blocked
    from voice_agent_evals.evals.runner import aggregate

    run["summary"] = aggregate(run["results"])
    return run


def test_holdout_file_ignores_comments(tmp_path):
    p = tmp_path / "h.txt"
    p.write_text("# comment\nlegal_threat\n\n  human_request_angry \n", encoding="utf-8")
    assert load_holdout(p) == {"legal_threat", "human_request_angry"}
    assert load_holdout(tmp_path / "missing.txt") == set()
    ids = load_holdout("scenarios/holdout.txt")
    assert len(ids) >= 5 and all((p := f"scenarios/{i}.yaml") and __import__("os").path.exists(p) for i in ids)


def test_comparison_table_and_subsets():
    base = _run()
    runs = {"scripted": base, "graph@seed7": _fake_llm_run(base, [True, False, True], 2),
            "graph@seed8": _fake_llm_run(base, [True, True, True], 0)}
    rows = comparison_rows({k: v["summary"] for k, v in runs.items()})
    names = [r[0] for r in rows]
    for metric in ("Passed (all checks)", "Escalation precision", "Blocked actions (convs affected)",
                   "Latency per turn p50 / p95", "Tokens per conversation (prompt + completion)",
                   "Wall-clock per conversation", "Hallucination-free (rule-based)"):
        assert metric in names
    passed = rows[0]
    assert passed[1] == "3/3 (100.0%)" and passed[2] == "2/3 (66.7%)"
    blocked = rows[names.index("Blocked actions (convs affected)")]
    assert blocked[2] == "6 (3)"
    held = subset_summary(runs["graph@seed7"], {"human_request_angry"})
    assert held["n"] == 1 and held["passed"] == 0
    md = compare_markdown(runs, {"human_request_angry"})
    assert "### Held-out scenarios (1" in md and "### Development scenarios (2" in md
    assert "human_request_angry (held-out)" in md and "tiny-model" in md
    assert "| `graph` | 2 | 66.7% / 100.0%" in variance_table(runs)
    html = render_compare_html(runs, {"human_request_angry"}, {"scripted": "reports/scripted.html"})
    assert "<table>" in html and "Held-out scenarios (1)" in html and "reports/scripted.html" in html


def test_compact_run_is_small_and_has_transcript():
    c = compact_run(_run())
    assert set(c) == {"meta", "summary", "results"}
    first = c["results"][0]
    assert first["transcript"][0].startswith("agent: ") and any("tool lookup_account" in t for t in first["transcript"])
    assert "turns" not in first and len(json.dumps(c)) < 30000


def test_run_variants_saves_resumes_and_writes_outputs(tmp_path):
    scenarios = load_scenarios("scenarios", only=IDS)
    out = tmp_path / "cmp"
    log = tmp_path / "progress.log"
    runs = run_variants(scenarios, ["scripted"], [7], Settings(), out, log)
    assert list(runs) == ["scripted"] and (out / "runs" / "scripted.json").exists()
    assert "scripted: done 3/3" in log.read_text(encoding="utf-8")
    again = run_variants(scenarios, ["scripted"], [7], Settings(), out, log, resume=True)
    assert again["scripted"]["summary"]["n"] == 3 and "loaded saved run" in log.read_text(encoding="utf-8")
    paths = write_outputs(runs, {"human_request_angry"}, out, tmp_path / "compact")
    assert paths["markdown"].exists() and paths["html"].exists()
    assert (out / "reports" / "scripted.html").exists() and (tmp_path / "compact" / "scripted.json").exists()
    assert run_label("graph", 8) == "graph@seed8" and run_label("scripted", None) == "scripted"
