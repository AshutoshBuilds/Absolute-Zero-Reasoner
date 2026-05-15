# hf_training_step_trace.py
"""Optional structured per-step training traces (proposer / solver) for HF RL.

Enabled only when ``AZR_TRAIN_STEP_LOG`` is truthy. Default off: no file I/O and
no string truncation work in :func:`log_training_step_detail`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import threading
_LOG = logging.getLogger("AZR-HF-RL.step_trace")
_FILE_WARNED = False
_FILE_WARNED_LOCK = threading.Lock()


def train_step_trace_enabled() -> bool:
    v = os.environ.get("AZR_TRAIN_STEP_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def train_step_trace_max_chars() -> int:
    raw = os.environ.get("AZR_TRAIN_STEP_LOG_MAX_CHARS", "").strip()
    if not raw:
        return 2000
    try:
        return max(64, min(500_000, int(raw)))
    except ValueError:
        return 2000


def truncate_utf8(text: Optional[str], max_chars: int) -> str:
    """Truncate by Unicode codepoints; UTF-8 safe on write (str is unicode)."""
    if text is None:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def resolve_step_trace_log_path(
    config_run_log_dir: Optional[str] = None,
) -> Path:
    """Resolve log file path without creating directories.

    Precedence:
    1. ``AZR_TRAIN_STEP_LOG_FILE`` — explicit file path
    2. ``(config_run_log_dir or AZR_RUN_LOG_DIR) / step_trace.log``
    3. ``training_metrics/step_trace.log`` under cwd
    """
    explicit = os.environ.get("AZR_TRAIN_STEP_LOG_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    run_dir = (config_run_log_dir or "").strip() or os.environ.get("AZR_RUN_LOG_DIR", "").strip()
    if run_dir:
        return Path(run_dir).expanduser() / "step_trace.log"

    return Path("training_metrics") / "step_trace.log"


def _safe_json_value(obj: Any, max_chars: int) -> Any:
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return truncate_utf8(obj, max_chars)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 32:
                out["..."] = f"+{len(obj) - 32} more keys"
                break
            key = str(k)[:128]
            out[key] = _safe_json_value(v, min(max_chars, 512))
        return out
    if isinstance(obj, (list, tuple)):
        lim = min(16, len(obj))
        return [_safe_json_value(x, min(max_chars, 512)) for x in obj[:lim]]
    try:
        return truncate_utf8(json.dumps(obj, default=str, ensure_ascii=False), max_chars)
    except Exception:
        return truncate_utf8(repr(obj), max_chars)


def summarize_execution_for_trace(execution_result: Any, max_each: int) -> Optional[Dict[str, Any]]:
    if not isinstance(execution_result, dict):
        return None
    d: Dict[str, Any] = {}
    for k in ("valid", "success", "reason", "similarity"):
        if k in execution_result:
            d[k] = execution_result[k]
    err = execution_result.get("error")
    if err:
        d["error"] = truncate_utf8(str(err), max_each)
    if "output" in execution_result:
        d["output_preview"] = truncate_utf8(
            repr(execution_result.get("output")), max_each
        )
    return d


def log_training_step_detail(
    *,
    log_path: Union[str, Path],
    epoch: int,
    step: int,
    role: str,
    problem_type: str,
    prompt_text: str,
    model_raw_output: str,
    parse_ok: Optional[bool] = None,
    parsed_task_count: Optional[int] = None,
    first_task_keys: Optional[List[str]] = None,
    proposer_reward: Optional[float] = None,
    solver_reward: Optional[float] = None,
    execution_summary: Optional[Dict[str, Any]] = None,
    reward_components: Optional[Dict[str, Any]] = None,
    elapsed_seconds: Optional[float] = None,
    iso_timestamp: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON line to ``log_path`` when tracing is enabled."""
    if not train_step_trace_enabled():
        return

    max_c = train_step_trace_max_chars()
    record: Dict[str, Any] = {
        "ts": iso_timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch": int(epoch),
        "step": int(step),
        "role": str(role),
        "problem_type": str(problem_type),
        "prompt_truncated": truncate_utf8(prompt_text, max_c),
        "model_output_truncated": truncate_utf8(model_raw_output, max_c),
    }
    if parse_ok is not None:
        record["parse_ok"] = bool(parse_ok)
    if parsed_task_count is not None:
        record["parsed_task_count"] = int(parsed_task_count)
    if first_task_keys is not None:
        record["first_task_keys"] = [str(k) for k in first_task_keys[:24]]
    if proposer_reward is not None:
        record["proposer_reward"] = float(proposer_reward)
    if solver_reward is not None:
        record["solver_reward"] = float(solver_reward)
    if execution_summary is not None:
        record["execution_summary"] = _safe_json_value(execution_summary, min(max_c, 800))
    if reward_components is not None:
        record["reward_components"] = _safe_json_value(dict(reward_components), min(max_c, 800))
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    if extra:
        record["extra"] = _safe_json_value(extra, min(max_c, 600))

    path = Path(log_path)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    global _FILE_WARNED
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
    except OSError as e:
        with _FILE_WARNED_LOCK:
            if not _FILE_WARNED:
                _LOG.info(
                    "AZR_TRAIN_STEP: file append failed (%s); falling back to logger for further traces.",
                    e,
                )
                _FILE_WARNED = True
        _LOG.info("AZR_TRAIN_STEP_TRACE %s", json.dumps(record, ensure_ascii=False))
