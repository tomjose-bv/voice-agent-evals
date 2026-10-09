"""Run scenarios: agent <-> simulated caller over the mock backend, then score."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..agent import make_agent
from ..agent.llm import ChatClient
from ..backend import MockBackend
from ..config import Settings
from . import metrics as M
from .caller import Caller, LLMCaller, ScriptedCaller
from .scenario import Scenario


def run_scenario(
    scenario: Scenario,
    agent_kind: str = "scripted",
    caller_kind: str = "scripted",
    settings: Settings | None = None,
    judge_client: ChatClient | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.load()
    backend = MockBackend(credit_limit=settings.credit_limit)
    agent = make_agent(agent_kind, backend, settings)
    caller: Caller
    if caller_kind == "llm":
        caller = LLMCaller(
            scenario,
            ChatClient(settings.llm_base_url, settings.llm_api_key, settings.caller_model, settings.llm_timeout_s),
        )
    else:
        caller = ScriptedCaller(scenario)

    turns: list[dict[str, Any]] = []
    latencies: list[float] = []
    p_tok = c_tok = 0
    error: str | None = None
    t_start = time.perf_counter()

    greet = agent.greet()
    turns.append({"role": "agent", "text": greet.text, "tool_calls": [], "latency_ms": None})
    agent_text = greet.text
    try:
        for _ in range(scenario.max_turns):
            user = caller.next(agent_text)
            if user is None:
                break
            turns.append({"role": "caller", "text": user})
            t0 = time.perf_counter()
            reply = agent.respond(user)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            p_tok += reply.prompt_tokens
            c_tok += reply.completion_tokens
            turns.append({
                "role": "agent",
                "text": reply.text,
                "tool_calls": [c.to_dict() for c in reply.tool_calls],
                "latency_ms": round(dt_ms, 3),
            })
            agent_text = reply.text
            if reply.ended:
                break
    except Exception as exc:  # keep the suite running; record the failure
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t_start

    final_state = backend.snapshot()
    agent_texts = [t["text"] for t in turns if t["role"] == "agent"]
    task_ok, task_problems = M.task_completion(scenario.expected, final_state, agent_texts)
    esc_label = M.escalation_label(scenario.expected.escalation, final_state["escalated"])
    tools = M.tool_call_correctness(scenario.expected.tool_calls, scenario.expected.forbidden_tools, agent.tools.log)
    claims = M.hallucination_check(turns, backend.today, backend.credit_limit)
    judge: dict[str, Any] | None = None
    if judge_client is not None:
        from .judge import llm_judge

        try:
            judge = llm_judge(turns, judge_client)
        except Exception as exc:
            judge = {"ok": False, "error": str(exc)}

    escalation_ok = esc_label in ("TP", "TN")
    passed = error is None and task_ok and escalation_ok and tools["correct"] and not claims
    caller_tokens = (getattr(caller, "prompt_tokens", 0), getattr(caller, "completion_tokens", 0))
    return {
        "scenario_id": scenario.id,
        "title": scenario.title,
        "category": scenario.category,
        "passed": passed,
        "error": error,
        "turns": turns,
        "caller_turns": sum(1 for t in turns if t["role"] == "caller"),
        "final_state": final_state,
        "handoff": getattr(agent, "handoff", None),
        "metrics": {
            "task_completed": task_ok,
            "task_problems": task_problems,
            "escalation_expected": scenario.expected.escalation,
            "escalation_actual": final_state["escalated"],
            "escalation_label": esc_label,
            "escalation_correct": escalation_ok,
            "tool_calls": tools,
            "hallucinations": [c.to_dict() for c in claims],
            "hallucination_free": not claims,
            "llm_judge": judge,
            "latency": M.latency_stats(latencies),
            "cost": M.cost_summary(p_tok, c_tok, wall),
            "caller_tokens": {"prompt": caller_tokens[0], "completion": caller_tokens[1]},
        },
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}

    def rate(key: Callable[[dict[str, Any]], bool]) -> float:
        return sum(1 for r in results if key(r)) / n

    all_lat = [t["latency_ms"] for r in results for t in r["turns"] if t.get("latency_ms") is not None]
    esc = M.precision_recall(r["metrics"]["escalation_label"] for r in results)
    costs = [r["metrics"]["cost"] for r in results]
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += int(r["passed"])
    return {
        "n": n,
        "passed": sum(1 for r in results if r["passed"]),
        "pass_rate": rate(lambda r: r["passed"]),
        "task_completion_rate": rate(lambda r: r["metrics"]["task_completed"]),
        "escalation": esc,
        "tool_call_correct_rate": rate(lambda r: r["metrics"]["tool_calls"]["correct"]),
        "tool_call_mean_recall": sum(r["metrics"]["tool_calls"]["recall"] for r in results) / n,
        "forbidden_tool_calls": sum(len(r["metrics"]["tool_calls"]["forbidden_called"]) for r in results),
        "tool_errors": sum(len(r["metrics"]["tool_calls"]["errors"]) for r in results),
        "hallucination_free_rate": rate(lambda r: r["metrics"]["hallucination_free"]),
        "unsupported_claims": sum(len(r["metrics"]["hallucinations"]) for r in results),
        "mean_caller_turns": sum(r["caller_turns"] for r in results) / n,
        "latency": M.latency_stats(all_lat),
        "mean_wall_clock_s": sum(c["wall_clock_s"] for c in costs) / n,
        "total_tokens": sum(c["total_tokens"] for c in costs),
        "mean_tokens_per_conversation": sum(c["total_tokens"] for c in costs) / n,
        "api_cost_usd": 0.0,
        "errors": sum(1 for r in results if r["error"]),
        "by_category": by_cat,
    }


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def machine_info() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
    }


def run_suite(
    scenarios: list[Scenario],
    agent_kind: str = "scripted",
    caller_kind: str = "scripted",
    settings: Settings | None = None,
    judge: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.load()
    judge_client = (
        ChatClient(settings.llm_base_url, settings.llm_api_key, settings.judge_model, settings.llm_timeout_s)
        if judge else None
    )
    results = []
    for s in scenarios:
        r = run_scenario(s, agent_kind, caller_kind, settings, judge_client)
        results.append(r)
        if progress:
            progress(r)
    meta = {
        "tool": "voice-agent-evals",
        "version": __version__,
        "git_commit": _git_commit(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_backend": agent_kind,
        "agent_model": settings.llm_model if agent_kind == "llm" else None,
        "caller": caller_kind,
        "caller_model": settings.caller_model if caller_kind == "llm" else None,
        "llm_judge": settings.judge_model if judge else None,
        "credit_limit": settings.credit_limit,
        "machine": machine_info(),
    }
    return {"meta": meta, "summary": aggregate(results), "results": results}
