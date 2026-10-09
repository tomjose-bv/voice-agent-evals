import json
from pathlib import Path

import pytest
import yaml

from voice_agent_evals.cli import main
from voice_agent_evals.evals import ScenarioError, load_scenario, load_scenarios, run_suite
from voice_agent_evals.evals.caller import ScriptedCaller
from voice_agent_evals.evals.report import render_html, summary_markdown, write_json

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios"

MINIMAL = {
    "id": "tmp_case",
    "title": "t",
    "category": "billing",
    "persona": {"name": "A", "description": "B"},
    "goal": "g",
    "caller_script": ["hello"],
    "expected": {"escalation": False},
}


def write(tmp_path, data, name="s.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_bundled_scenarios_load_and_cover_categories():
    scenarios = load_scenarios(SCEN)
    assert len(scenarios) >= 20
    cats = {s.category for s in scenarios}
    assert {"billing", "outage", "technician", "identity", "escalation", "safety", "off-topic",
            "ambiguous", "multi-intent"} <= cats
    assert any(s.expected.escalation for s in scenarios)
    assert any(not s.expected.escalation for s in scenarios)


def test_minimal_scenario_defaults(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL))
    assert s.max_turns == 12 and s.expected.end_state.escalated is False


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"id": "Bad Id"}, "slug"),
        ({"expected": {"escalation": False, "tool_calls": [{"name": "delete_everything"}]}}, "unknown tool"),
        ({"expected": {"escalation": True, "forbidden_tools": ["escalate_to_human"]}}, "forbidden"),
        ({"expected": {"escalation": False, "end_state": {"escalated": True}}}, "contradicts"),
        ({"caller_script": []}, "caller_script"),
        ({"surprise": 1}, "surprise"),
        ({"ground_truth": {"account_id": "ACC-1002", "bill_total": 12.0}}, "bill_total"),
        ({"ground_truth": {"postcode": "SE5 9ZZ", "outage": False}}, "outage"),
        ({"ground_truth": {"account_id": "ACC-0000"}}, "not in mock backend"),
        ({"expected": {"escalation": False, "end_state": {"credits": [{"amount": 75}]}}}, "exceeds credit limit"),
    ],
)
def test_invalid_scenarios_rejected(tmp_path, patch, msg):
    with pytest.raises(ScenarioError, match=msg):
        load_scenario(write(tmp_path, {**MINIMAL, **patch}))


def test_invalid_yaml_and_duplicates(tmp_path):
    (tmp_path / "bad.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(ScenarioError, match="invalid YAML"):
        load_scenario(tmp_path / "bad.yaml")
    d = tmp_path / "dupes"
    d.mkdir()
    write(d, MINIMAL, "a.yaml")
    write(d, MINIMAL, "b.yaml")
    with pytest.raises(ScenarioError, match="duplicate"):
        load_scenarios(d)


def test_filter_by_id_category_tag():
    assert {s.id for s in load_scenarios(SCEN, only=["legal_threat"])} == {"legal_threat"}
    assert all(s.category == "outage" for s in load_scenarios(SCEN, only=["outage"]))
    assert all("injection" in s.tags for s in load_scenarios(SCEN, only=["injection"]))


def test_scripted_caller_replays_then_hangs_up():
    s = load_scenarios(SCEN, only=["legal_threat"])[0]
    c = ScriptedCaller(s)
    lines = [c.next("x") for _ in range(len(s.caller_script) + 1)]
    assert lines[:-1] == s.caller_script and lines[-1] is None


@pytest.fixture(scope="module")
def run():
    return run_suite(load_scenarios(SCEN), "scripted", "scripted")


def test_suite_is_deterministic(run):
    again = run_suite(load_scenarios(SCEN), "scripted", "scripted")
    strip = lambda r: [(x["scenario_id"], x["passed"], x["final_state"]) for x in r["results"]]  # noqa: E731
    assert strip(run) == strip(again)


def test_summary_fields(run):
    s = run["summary"]
    assert s["n"] == len(run["results"])
    assert 0 <= s["pass_rate"] <= 1
    assert s["api_cost_usd"] == 0.0
    assert s["escalation"]["tp"] + s["escalation"]["fn"] == sum(r["metrics"]["escalation_expected"] for r in run["results"])
    assert run["meta"]["agent_backend"] == "scripted"


def test_reports(run, tmp_path):
    html = render_html(run)
    assert html.startswith("<!doctype html>") and "<script src" not in html and "http" not in html.split("<script id=")[0].replace("http-equiv", "")
    data = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(data.replace("<\\/", "</"))["summary"]["n"] == run["summary"]["n"]
    md = summary_markdown(run)
    assert "| Task completion |" in md and md.count("| PASS |") + md.count("| FAIL |") == run["summary"]["n"]
    p = write_json(run, tmp_path / "out" / "r.json")
    assert json.loads(p.read_text(encoding="utf-8"))["meta"]["tool"] == "voice-agent-evals"


def test_html_escapes_script_breakout(run):
    import copy

    evil = copy.deepcopy(run)
    evil["results"][0]["turns"][0]["text"] = "</script><script>alert(1)</script>"
    assert "</script><script>alert(1)" not in render_html(evil)


def test_cli_run_validate_list(tmp_path, capsys):
    assert main(["validate", "--scenarios", str(SCEN)]) == 0
    assert main(["list", "--scenarios", str(SCEN)]) == 0
    out = tmp_path / "res"
    assert main(["run", "--scenarios", str(SCEN), "--only", "outage", "--out", str(out)]) == 0
    assert (out / "results.json").exists() and (out / "report.html").exists() and (out / "summary.md").exists()
    assert main(["run", "--scenarios", str(SCEN), "--only", "no_such_thing", "--out", str(out)]) == 2
