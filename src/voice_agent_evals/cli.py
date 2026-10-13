"""`vae` command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, enable_system_trust_store

DEFAULT_SCENARIOS = "scenarios"
AGENT_CHOICES = ["scripted", "prompt", "context", "graph", "llm"]


def _cmd_run(args: argparse.Namespace) -> int:
    from .evals import ScenarioError, load_scenarios, run_suite
    from .evals.report import summary_markdown, write_html, write_json

    settings = Settings.load()
    if args.agent != "scripted" or args.caller == "llm" or args.judge:
        from .agent.llm import ChatClient

        if not ChatClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model).ping():
            print(f"error: no LLM endpoint reachable at {settings.llm_base_url}. Start Ollama (or another "
                  "OpenAI-compatible server) or use the scripted backend: vae run --agent scripted", file=sys.stderr)
            return 2
    try:
        scenarios = load_scenarios(args.scenarios, only=args.only or None)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not scenarios:
        print("error: no scenarios matched", file=sys.stderr)
        return 2

    def progress(r: dict) -> None:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['scenario_id']}")

    print(f"Running {len(scenarios)} scenarios (agent={args.agent}, caller={args.caller})")
    run = run_suite(scenarios, args.agent, args.caller, settings, judge=args.judge, progress=progress)
    out = Path(args.out)
    jp = write_json(run, out / "results.json")
    hp = write_html(run, out / "report.html")
    md = summary_markdown(run)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"JSON:   {jp}\nHTML:   {hp}\nMarkdown: {out / 'summary.md'}")
    if args.fail_under is not None and run["summary"]["pass_rate"] < args.fail_under:
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .evals import ScenarioError, load_scenarios
    from .evals.compare import load_holdout
    from .evals.compare_run import run_variants, write_outputs

    settings = Settings.load()
    if args.model:
        settings.llm_model = args.model
    if any(a != "scripted" for a in args.agents):
        from .agent.llm import ChatClient

        if not ChatClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model).ping():
            print(f"error: no LLM endpoint reachable at {settings.llm_base_url}", file=sys.stderr)
            return 2
    try:
        scenarios = load_scenarios(args.scenarios, only=args.only or None)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    seeds = args.seeds or [settings.llm_seed]
    progress = Path(args.progress) if args.progress else out / "progress.log"
    progress.parent.mkdir(parents=True, exist_ok=True)
    from .evals.keep_awake import keep_awake

    with keep_awake():
        runs = run_variants(scenarios, args.agents, seeds, settings, out, progress, resume=args.resume)
    holdout = load_holdout(args.holdout)
    paths = write_outputs(runs, holdout, out, Path(args.save_compact) if args.save_compact else None)
    print()
    print(paths["markdown"].read_text(encoding="utf-8"))
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .evals import ScenarioError, load_scenarios

    try:
        scenarios = load_scenarios(args.scenarios)
    except ScenarioError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(scenarios)} scenarios valid")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .evals import load_scenarios

    for s in load_scenarios(args.scenarios):
        esc = "escalate" if s.expected.escalation else "-"
        print(f"{s.id:38} {s.category:22} {esc:9} {s.title}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .agent import make_agent
    from .agent.handoff import handoff_text
    from .backend import MockBackend

    settings = Settings.load()
    agent = make_agent(args.agent, MockBackend(credit_limit=settings.credit_limit), settings)
    print("Text chat with the Northwind agent. Ctrl+C or empty line to quit.")
    print("Try phone 555-0102 (duplicate charge) or postcode SE5 9ZZ (outage).\n")
    print(f"agent> {agent.greet().text}")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        turn = agent.respond(line)
        for c in turn.tool_calls:
            print(f"   [tool] {c.name}({c.args}) -> {'ok' if c.ok else 'ERROR ' + str(c.error)}")
        print(f"agent> {turn.text}")
        if turn.escalated and getattr(agent, "handoff", None):
            print("\n=== ESCALATED TO HUMAN ===\n" + handoff_text(agent.handoff))  # type: ignore[attr-defined]
        if turn.ended:
            break
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .voice.server import create_app
    except ImportError as exc:
        print(f"error: voice extras not installed ({exc}). Run: pip install -e .[voice]", file=sys.stderr)
        return 2
    app = create_app(agent_kind=args.agent, stt=args.stt, tts=args.tts)
    print(f"Voice demo on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vae", description="Local customer-service voice agent + evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the evaluation suite")
    r.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="scenario directory (default: ./scenarios)")
    r.add_argument("--agent", choices=AGENT_CHOICES, default="scripted", help="agent backend")
    r.add_argument("--caller", choices=["scripted", "llm"], default="scripted", help="simulated caller")
    r.add_argument("--judge", action="store_true", help="also run the optional LLM hallucination judge")
    r.add_argument("--only", nargs="*", help="scenario ids, categories or tags to run")
    r.add_argument("--out", default="results", help="output directory (default: ./results)")
    r.add_argument("--fail-under", type=float, default=None, help="exit 1 if pass rate is below this (0-1)")
    r.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser("compare", help="run several agent variants on the same scenarios and compare them")
    cmp_.add_argument("--agents", nargs="+", choices=AGENT_CHOICES, default=["scripted", "prompt", "context", "graph"])
    cmp_.add_argument("--seeds", nargs="*", type=int, help="repeat each LLM variant once per seed")
    cmp_.add_argument("--model", default=None, help="override VAE_LLM_MODEL")
    cmp_.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    cmp_.add_argument("--only", nargs="*", help="scenario ids, categories or tags to run")
    cmp_.add_argument("--holdout", default="scenarios/holdout.txt", help="held-out scenario list (reported separately)")
    cmp_.add_argument("--out", default="results/compare", help="output directory")
    cmp_.add_argument("--progress", default=None, help="progress log file (default: <out>/progress.log)")
    cmp_.add_argument("--resume", action="store_true", help="reuse runs already saved in <out>/runs/")
    cmp_.add_argument("--save-compact", default=None, help="also write small per-run JSON + compare.md here")
    cmp_.set_defaults(func=_cmd_compare)

    v = sub.add_parser("validate", help="validate scenario files against the schema and mock backend")
    v.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    v.set_defaults(func=_cmd_validate)

    ls = sub.add_parser("list", help="list scenarios")
    ls.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    ls.set_defaults(func=_cmd_list)

    c = sub.add_parser("chat", help="text chat with the agent in the terminal")
    c.add_argument("--agent", choices=AGENT_CHOICES, default="scripted")
    c.set_defaults(func=_cmd_chat)

    s = sub.add_parser("serve", help="start the local voice demo web app")
    s.add_argument("--agent", choices=AGENT_CHOICES, default="scripted")
    s.add_argument("--stt", choices=["whisper", "fake"], default="whisper")
    s.add_argument("--tts", choices=["auto", "piper", "sapi", "none"], default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    enable_system_trust_store()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
