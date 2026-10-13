"""Run several agent variants (and seeds) and write the comparison outputs."""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from .compare import compact_run, compare_markdown
from .compare_report import write_compare_html
from .report import write_html, write_json
from .runner import run_suite
from .scenario import Scenario


def run_label(agent: str, seed: int | None) -> str:
    return agent if agent == "scripted" or seed is None else f"{agent}@seed{seed}"


def _log(progress_path: Path | None, line: str) -> None:
    stamped = f"{datetime.now().strftime('%H:%M:%S')} {line}"
    print(stamped, flush=True)
    if progress_path:
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


def _load_checkpoints(folder: Path) -> dict[str, dict[str, Any]]:
    """Per-scenario results saved by an interrupted run. Errored results are skipped so they rerun."""
    cached: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("error"):
            cached[r["scenario_id"]] = r
    return cached


def run_variants(
    scenarios: list[Scenario],
    agents: list[str],
    seeds: list[int | None],
    settings: Settings,
    out: Path,
    progress_path: Path | None = None,
    resume: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run each agent (each LLM agent once per seed). Each run is saved as soon as it finishes,
    so an interrupted comparison can be resumed with ``resume=True``."""
    runs: dict[str, dict[str, Any]] = {}
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for agent in agents:
        for seed in ([None] if agent == "scripted" else seeds):
            label = run_label(agent, seed)
            path = out / "runs" / f"{label.replace('@', '-')}.json"
            if resume and path.exists():
                runs[label] = json.loads(path.read_text(encoding="utf-8"))
                _log(progress_path, f"{label}: loaded saved run")
                continue
            s = settings if seed is None else dataclasses.replace(settings, llm_seed=seed)
            partial = out / "runs" / f"{label.replace('@', '-')}.partial"
            cached = _load_checkpoints(partial) if resume else {}
            partial.mkdir(parents=True, exist_ok=True)
            _log(progress_path, f"{label}: start ({len(scenarios)} scenarios"
                                + (f", {len(cached)} reused from an interrupted run)" if cached else ")"))
            t0 = time.perf_counter()
            done = [0]

            def progress(r: dict[str, Any], label: str = label, partial: Path = partial) -> None:
                done[0] += 1
                if not r["error"]:
                    (partial / f"{r['scenario_id']}.json").write_text(json.dumps(r, default=str), encoding="utf-8")
                c = r["metrics"]["cost"]
                _log(progress_path, f"{label}: [{done[0]}/{len(scenarios)}] {'PASS' if r['passed'] else 'FAIL'} "
                                    f"{r['scenario_id']} ({c['wall_clock_s']:.1f} s, {c['total_tokens']} tok)"
                                    + (f" ERROR {r['error']}" if r["error"] else ""))

            run = run_suite(scenarios, agent, "scripted", s, progress=progress, cached=cached)
            write_json(run, path)
            runs[label] = run
            sm = run["summary"]
            _log(progress_path, f"{label}: done {sm['passed']}/{sm['n']} in {time.perf_counter() - t0:.0f} s")
    return runs


def write_outputs(runs: dict[str, dict[str, Any]], holdout: set[str], out: Path,
                  compact_dir: Path | None = None) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    links = {}
    for label, run in runs.items():
        name = label.replace("@", "-")
        write_html(run, out / "reports" / f"{name}.html")
        links[label] = f"reports/{name}.html"
    md = compare_markdown(runs, holdout)
    paths = {"markdown": out / "compare.md", "html": out / "compare.html"}
    paths["markdown"].write_text(md, encoding="utf-8")
    write_compare_html(runs, holdout, links, paths["html"])
    if compact_dir:
        compact_dir.mkdir(parents=True, exist_ok=True)
        for label, run in runs.items():
            p = compact_dir / f"{label.replace('@', '-')}.json"
            p.write_text(json.dumps(compact_run(run), indent=1, default=str), encoding="utf-8")
        (compact_dir / "compare.md").write_text(md, encoding="utf-8")
        paths["compact"] = compact_dir
    return paths
