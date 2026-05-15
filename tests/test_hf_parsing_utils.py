"""Regression tests for proposer output parsing (fenced JSON vs other fences)."""

import json

import pytest

from hf_parsing_utils import find_json_objects, parse_generated_tasks, strip_leading_trailing_code_fences


def test_find_json_objects_ignores_python_fence_after_json():
    text = """Here is the task:
```json
{"code": "def f(): return 1", "input": "", "output": "1"}
```
And some code:
```python
def f():
    return 1
```
"""
    segments = find_json_objects(text)
    assert len(segments) == 1
    data = json.loads(segments[0])
    assert data["code"] == "def f(): return 1"


def test_find_json_objects_multiple_json_fences():
    text = """```json
{"code": "a", "input": "", "output": "1"}
```
```json
{"code": "b", "input": "", "output": "2"}
```
"""
    segments = find_json_objects(text)
    assert len(segments) == 2


def test_find_json_objects_malformed_duplicate_json_fence_then_python():
    """Model sometimes closes with ```json ```python; first ``` must end the json fence."""
    text = """Proposal:
```json
{"code": "ok", "input": "", "output": "42"}
```json ```python
def ok():
    return 42
```
"""
    segments = find_json_objects(text)
    assert len(segments) == 1
    data = json.loads(segments[0])
    assert data["code"] == "ok"


def test_parse_generated_tasks_json_then_python():
    raw = """```json
{"code": "x", "input": "1", "output": "2"}
```
```python
print("hi")
```
"""
    tasks = parse_generated_tasks(None, raw, task_idx=0)
    assert len(tasks) == 1
    assert tasks[0]["code"] == "x"


def test_parse_generated_tasks_json_list_mixed_types_strict_parse():
    """Model may emit a JSON array with scalars mixed in; those entries must not crash parsing."""
    raw = json.dumps(
        [
            {"code": "def f(): return 1", "input": "", "output": "1"},
            1,
            "not-a-task",
            None,
        ]
    )
    tasks = parse_generated_tasks(None, raw, task_idx=0)
    assert len(tasks) == 1
    assert "f" in tasks[0]["code"]


def test_parse_generated_tasks_fenced_json_list_mixed_types_recovery():
    """Same as strict path but via ```json``` segment + list iteration."""
    inner = json.dumps(
        [
            {"code": "g", "input": "", "output": "2"},
            42,
        ]
    )
    raw = f"```json\n{inner}\n```"
    tasks = parse_generated_tasks(None, raw, task_idx=1)
    assert len(tasks) == 1
    assert tasks[0]["code"] == "g"


def test_strip_leading_trailing_code_fences_full_python_block():
    raw = """```python
def f(x):
    return x + 1
```
"""
    assert strip_leading_trailing_code_fences(raw).strip().startswith("def f")


def test_strip_leading_trailing_code_fences_json_tagged_python():
    raw = """```json
def g():
    return 2
```
"""
    body = strip_leading_trailing_code_fences(raw)
    assert "def g" in body
    assert not body.strip().startswith("```")


def test_strip_leading_trailing_code_fences_plain_fence():
    raw = """
```
x = 1
```
"""
    assert strip_leading_trailing_code_fences(raw).strip() == "x = 1"


def test_code_executor_strips_fenced_solver_code():
    from code_executor import CodeExecutor

    ex = CodeExecutor(timeout_seconds=3)
    fenced = '''```python
def f():
    return 42
```'''
    result = ex.execute_code(fenced, (), "f")
    assert result.get("error") is None, result
    assert result.get("return_value") == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
