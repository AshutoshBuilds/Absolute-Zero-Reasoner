"""Tests for local-first benchmark dataset resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from datasets import Dataset

from hf_benchmark_data import (
    allow_online_fetch,
    canonical_local_split_dir,
    load_azr_benchmark_split,
    resolve_benchmark_data_root,
)


def test_canonical_local_split_dir_paths():
    root = Path("/data")
    assert canonical_local_split_dir(root, "openai_humaneval", None, "test") == Path(
        "/data/openai_humaneval/test"
    )
    assert canonical_local_split_dir(root, "mbpp", "sanitized", "test") == Path("/data/mbpp/sanitized/test")
    assert canonical_local_split_dir(root, "gsm8k", "main", "test") == Path("/data/gsm8k/main/test")
    assert canonical_local_split_dir(root, "hendrycks/competition_math", None, "test") == Path(
        "/data/hendrycks__competition_math/test"
    )


def test_resolve_benchmark_data_root_absolute(tmp_path, monkeypatch):
    sub = tmp_path / "bench_root"
    monkeypatch.setenv("AZR_BENCHMARK_DATA_ROOT", str(sub))
    assert resolve_benchmark_data_root() == sub.resolve()


def test_load_prefers_local_snapshot_without_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("AZR_BENCHMARK_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AZR_BENCHMARK_ALLOW_ONLINE", "0")

    root = resolve_benchmark_data_root()
    local_dir = canonical_local_split_dir(root, "openai_humaneval", None, "test")
    local_dir.mkdir(parents=True)
    ds = Dataset.from_dict(
        {
            "task_id": ["t0"],
            "prompt": ["p"],
            "test": ["assert True"],
            "entry_point": ["f"],
        }
    )
    ds.save_to_disk(str(local_dir))

    manifest = {
        "version": 1,
        "datasets": [
            {
                "dataset": "openai_humaneval",
                "config": None,
                "split": "test",
                "split_dir": "openai_humaneval/test",
                "revision": "abc123",
                "prefetched_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    log = logging.getLogger("test_hf_benchmark_data")
    loaded = load_azr_benchmark_split("openai_humaneval", "test", None, logger=log)
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "t0"


def test_load_missing_local_raises_when_offline_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("AZR_BENCHMARK_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AZR_BENCHMARK_ALLOW_ONLINE", "0")
    with pytest.raises(FileNotFoundError, match="prefetch"):
        load_azr_benchmark_split("openai_humaneval", "test", None)


def test_allow_online_fetch_default_false(monkeypatch):
    monkeypatch.delenv("AZR_BENCHMARK_ALLOW_ONLINE", raising=False)
    monkeypatch.delenv("AZR_BENCHMARK_ALLOW_ONLINE_LOAD", raising=False)
    assert allow_online_fetch() is False
    monkeypatch.setenv("AZR_BENCHMARK_ALLOW_ONLINE", "1")
    assert allow_online_fetch() is True
