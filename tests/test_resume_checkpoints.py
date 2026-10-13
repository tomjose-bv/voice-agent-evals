import json

from voice_agent_evals.evals.compare_run import _load_checkpoints


def test_checkpoints_skip_errored_results(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"scenario_id": "a", "error": None, "passed": True}))
    (tmp_path / "b.json").write_text(json.dumps({"scenario_id": "b", "error": "LLMUnavailable", "passed": False}))
    cached = _load_checkpoints(tmp_path)
    assert set(cached) == {"a"}


def test_checkpoints_missing_folder(tmp_path):
    assert _load_checkpoints(tmp_path / "nope") == {}


def test_keep_awake_is_safe_to_enter_and_exit():
    from voice_agent_evals.evals.keep_awake import keep_awake

    with keep_awake():
        pass
