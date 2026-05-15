import json
import os
from pathlib import Path

import pytest

import hf_training_step_trace as st


def test_train_step_trace_enabled():
    prev = dict(os.environ)
    try:
        os.environ.pop("AZR_TRAIN_STEP_LOG", None)
        assert st.train_step_trace_enabled() is False
        for v in ("1", "true", "YES", "on"):
            os.environ["AZR_TRAIN_STEP_LOG"] = v
            assert st.train_step_trace_enabled() is True
        os.environ["AZR_TRAIN_STEP_LOG"] = "0"
        assert st.train_step_trace_enabled() is False
    finally:
        os.environ.clear()
        os.environ.update(prev)


def test_train_step_trace_max_chars():
    prev = dict(os.environ)
    try:
        os.environ.pop("AZR_TRAIN_STEP_LOG_MAX_CHARS", None)
        assert st.train_step_trace_max_chars() == 2000
        os.environ["AZR_TRAIN_STEP_LOG_MAX_CHARS"] = "bad"
        assert st.train_step_trace_max_chars() == 2000
        os.environ["AZR_TRAIN_STEP_LOG_MAX_CHARS"] = "5000000"
        assert st.train_step_trace_max_chars() == 500_000
    finally:
        os.environ.clear()
        os.environ.update(prev)


def test_truncate_utf8():
    assert st.truncate_utf8(None, 10) == ""
    assert st.truncate_utf8("hello", 10) == "hello"
    assert st.truncate_utf8("abcdefghijkl", 5) == "abcde...[truncated]"
    s = "日本語test"
    assert len(st.truncate_utf8(s, 3)) < len(s) + 15


def test_resolve_step_trace_log_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AZR_TRAIN_STEP_LOG_FILE", raising=False)
    monkeypatch.delenv("AZR_RUN_LOG_DIR", raising=False)
    p = st.resolve_step_trace_log_path("")
    assert p == Path("training_metrics") / "step_trace.log"

    rd = tmp_path / "run1"
    assert st.resolve_step_trace_log_path(str(rd)) == rd / "step_trace.log"

    f = tmp_path / "custom.log"
    monkeypatch.setenv("AZR_TRAIN_STEP_LOG_FILE", str(f))
    assert st.resolve_step_trace_log_path("ignored") == f


@pytest.mark.parametrize("enabled", [True, False])
def test_log_training_step_detail_writes_when_enabled(tmp_path, monkeypatch, enabled):
    logf = tmp_path / "step_trace.log"
    prev = dict(os.environ)
    try:
        if enabled:
            os.environ["AZR_TRAIN_STEP_LOG"] = "1"
        else:
            os.environ.pop("AZR_TRAIN_STEP_LOG", None)
        os.environ["AZR_TRAIN_STEP_LOG_MAX_CHARS"] = "80"
        st.log_training_step_detail(
            log_path=logf,
            epoch=1,
            step=2,
            role="proposer",
            problem_type="deduction",
            prompt_text="x" * 200,
            model_raw_output="y" * 200,
            parse_ok=False,
            proposer_reward=-0.5,
            reward_components={"fmt": -0.1},
        )
    finally:
        os.environ.clear()
        os.environ.update(prev)

    if enabled:
        assert logf.is_file()
        line = logf.read_text(encoding="utf-8").strip().splitlines()[0]
        row = json.loads(line)
        assert row["role"] == "proposer"
        assert row["epoch"] == 1
        assert row["step"] == 2
        assert len(row["prompt_truncated"]) <= 80 + 20
    else:
        assert not logf.exists()
