"""
Smoke tests for the repo default **Qwen 3.5 @ 0.8B** checkpoint.

**Hugging Face repo ID used for weights:** ``Qwen/Qwen3.5-0.8B``

- Prefers a local snapshot at ``models/Qwen3.5-0.8B`` (layout compatible with ``snapshot_download`` /
  ``download_model.py``-style trees).
- Falls back to the Hub id ``Qwen/Qwen3.5-0.8B`` (downloads on first load if no local snapshot).
- The ``snapshot_download`` test is skipped when ``models/Qwen3.5-0.8B/config.json`` exists so you can
  validate the local copy only.

Run only these tests::

    pytest tests/test_qwen35_0_8b_model_smoke.py -m slow -v

Skip entirely (e.g. CI without GPU/network)::

    set SKIP_QWEN_SMOKE=1
    pytest tests/test_qwen35_0_8b_model_smoke.py -v

Print prompt and model continuation (use with ``pytest -s``)::

    set AZR_QWEN35_0_8B_SMOKE_VERBOSE=1
    pytest tests/test_qwen35_0_8b_model_smoke.py -m slow -s -v

Wall-clock time for tokenization + ``generate()`` is logged at INFO (see pytest ``--log-cli-level=INFO``)::

    pytest tests/test_qwen35_0_8b_model_smoke.py -m slow -v --log-cli-level=INFO

**Thinking vs non-thinking style comparison (template-level controls)**

The checkpoint’s ``chat_template.jinja`` branches on ``enable_thinking`` (passed through to
``tokenizer.apply_chat_template``):

- **Non-thinking (default):** ``enable_thinking=False`` — the template inserts ``<think>`` then
  immediately ``</think>`` (empty reasoning), so the model continues with the **visible answer**.
- **Thinking:** ``enable_thinking=True`` — the template opens ``<think>`` only; the model should emit
  reasoning, then ``</think>``, then the final answer.

The long-answer test **always** runs twice on the same prompt: first **without** thinking, then
**with** thinking, and logs timings plus answer previews for comparison (use
``--log-cli-level=INFO``). Full outputs: set ``AZR_QWEN35_0_8B_SMOKE_VERBOSE=1`` and ``pytest -s``.

For harder custom prompts, set ``AZR_QWEN35_0_8B_USER_QUESTION`` (PowerShell `set` or plain env value).
For full output, raise ``AZR_QWEN35_0_8B_SMOKE_PRINT_CAP`` and ``AZR_QWEN35_0_8B_SMOKE_PROMPT_PRINT_CAP``.

In your own code, pass ``enable_thinking=`` into ``apply_chat_template`` the same way (see
``_format_generation_prompt`` below). Other APIs (vLLM, etc.) may use different flags; this matches
Transformers + the model HF template.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pytest
from typing import Callable, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Official Hub weights at 0.8B (Qwen3.5).
DEFAULT_HUB_ID = "Qwen/Qwen3.5-0.8B"
LOCAL_RELATIVE = Path("models") / "Qwen3.5-0.8B"

# Demanding user turn: expects a long, structured answer (multi-paragraph).
# Formatted with the tokenizer chat template when available (important for instruction-tuned checkpoints).
HARD_LONG_ANSWER_USER = """You are a senior research exam candidate in theoretical computer science.

Write a rigorous, standalone essay in multiple paragraphs with explicit structure
(definition -> lemma chain -> theorem proof -> corollaries).

Task:
1) Define precisely the following languages over encodings of Turing machines and inputs:
   A_TM = {<M, w> | TM M accepts input w} and
   HALT_TM = {<M, w> | TM M halts on input w}.
2) Prove HALT_TM is undecidable by contradiction via a self-reference construction.
   Do not summarize; show the machine construction and why it implies a contradiction.
3) Use your argument to derive a Rice-style statement: show that any non-trivial semantic
   property of partial computable functions is undecidable.
4) Show a concrete reduction from HALT_TM to a source-code optimizer/sanitizer question:
   Given program P and input x, does an analyzer decide P always terminates with bounded output size on x?
   Keep the reduction precise.
5) Give one real-world consequence for static analysis in compilers and one for security/safety tooling.

Constraints:
- No chain-of-thought. No bullet-only answer. No slogans.
- Use notation, at least one theorem-like statement, and a formal contradiction line.
- End when all requested points are logically closed.
"""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def qwen35_user_question() -> str:
    """Allow overriding the smoke-test question through an environment variable."""
    return os.environ.get("AZR_QWEN35_0_8B_USER_QUESTION", "").strip() or HARD_LONG_ANSWER_USER


def qwen35_enable_thinking_from_env() -> bool:
    """Read ``AZR_QWEN35_0_8B_ENABLE_THINKING`` (default False). Used only when ``enable_thinking`` is omitted from ``_format_generation_prompt``."""
    return _env_bool("AZR_QWEN35_0_8B_ENABLE_THINKING", False)


def _format_generation_prompt(
    tokenizer,
    user_text: str,
    *,
    enable_thinking: Optional[bool] = None,
) -> str:
    if enable_thinking is None:
        enable_thinking = qwen35_enable_thinking_from_env()
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text.strip()}],
                **kwargs,
            )
        except TypeError:
            # Some chat templates (for non-Qwen models) may not accept ``enable_thinking``.
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text.strip()}],
                **kwargs,
            )
    return user_text.strip() + "\n"


def _quality_hint(text: str) -> tuple[int, int, dict[str, bool]]:
    """
    Lightweight heuristic for smoke diagnostics (not a correctness oracle).

    The goal is to catch trivially degenerate outputs before manual review.
    """
    if not text:
        keyword_labels = _quality_keywords()
        return 0, len(keyword_labels), {label: False for label in keyword_labels}
    checks = {label: key(text.lower()) for label, key in _quality_keyword_checks().items()}
    score = sum(1 for matched in checks.values() if matched)
    return score, len(checks), checks


def _quality_keywords() -> list[str]:
    return [
        "halting mention",
        "A_TM mention",
        "acceptance mention",
        "rice mention",
        "reduction mention",
        "security mention",
    ]


def _quality_keyword_checks() -> dict[str, Callable[[str], bool]]:
    return {
        "halting mention": lambda text: "halt" in text and "tm" in text,
        "A_TM mention": lambda text: "a_tm" in text,
        "acceptance mention": lambda text: "accept" in text and "tm" in text,
        "rice mention": lambda text: "rice" in text,
        "reduction mention": lambda text: "reduction" in text,
        "security mention": lambda text: "security" in text or "safety" in text,
    }


def _generate_continuation(
    adapter,
    *,
    user_text: str,
    enable_thinking: bool,
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, str, float]:
    """
    Build the chat prompt and run ``adapter.generate``.

    Returns:
        (formatted_prompt, decoded_continuation_only, wall_seconds)
    """
    prompt = _format_generation_prompt(
        adapter.tokenizer, user_text, enable_thinking=enable_thinking
    )
    t0 = time.perf_counter()
    outputs = adapter.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        max_prompt_length=max_prompt_length,
        repetition_penalty=repetition_penalty,
    )
    elapsed = time.perf_counter() - t0
    assert outputs, "generate() returned an empty list"
    text = (outputs[0] or "").strip()
    return prompt, text, elapsed


def _local_qwen3508b_ready() -> bool:
    """True when a full local snapshot exists (no Hub download smoke needed)."""
    p = REPO_ROOT / LOCAL_RELATIVE
    return p.is_dir() and (p / "config.json").is_file()


def _resolve_model_name() -> str:
    """
    Prefer local ``models/Qwen3.5-0.8B`` when it looks complete (same layout as Hub snapshot).
    Override with env ``AZR_QWEN35_0_8B_MODEL_PATH`` (absolute or relative to cwd).
    Hub override: ``AZR_QWEN35_0_8B_MODEL_ID`` (defaults to ``DEFAULT_HUB_ID``).
    """
    explicit = os.environ.get("AZR_QWEN35_0_8B_MODEL_PATH", "").strip()
    if explicit:
        return explicit
    hub = os.environ.get("AZR_QWEN35_0_8B_MODEL_ID", DEFAULT_HUB_ID).strip() or DEFAULT_HUB_ID
    local = REPO_ROOT / LOCAL_RELATIVE
    if local.is_dir() and (local / "config.json").is_file():
        return str(local)
    return hub


skip_qwen35_0_8b = pytest.mark.skipif(
    os.environ.get("SKIP_QWEN_SMOKE", "").strip().lower() in {"1", "true", "yes"},
    reason="SKIP_QWEN_SMOKE is set",
)


@skip_qwen35_0_8b
@pytest.mark.slow
@pytest.mark.skipif(
    _local_qwen3508b_ready(),
    reason="Local models/Qwen3.5-0.8B snapshot present; skipping Hub download smoke.",
)
def test_qwen35_0_8b_snapshot_download_writes_config(tmp_path) -> None:
    """Ensures ``huggingface_hub.snapshot_download`` can populate a directory (same primitive as ``download_model.py``)."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import snapshot_download

    hub_id = os.environ.get("AZR_QWEN35_0_8B_MODEL_ID", DEFAULT_HUB_ID).strip() or DEFAULT_HUB_ID
    dest = tmp_path / "qwen35_0_8b_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hub_id, local_dir=str(dest))
    assert (dest / "config.json").is_file(), f"Expected config.json under {dest}"


@skip_qwen35_0_8b
@pytest.mark.slow
def test_qwen35_0_8b_load_and_generates_long_answer() -> None:
    """Load once, then compare non-thinking vs thinking: timings, previews, and assertions on both continuations."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from azr_hf_adapter import HuggingFaceAdapter

    model_name = _resolve_model_name()
    dedicated_cache = REPO_ROOT / "models" / ".hf_cache"
    dedicated_cache.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    adapter = HuggingFaceAdapter(
        model_name=model_name,
        auth_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        use_separate_value_model=False,
        hf_cache_dir=str(dedicated_cache),
        load_in_4bit=False,
    )
    load_seconds = time.perf_counter() - t_load0
    logger.info(
        "Qwen3.5-0.8B smoke (weights=%s): model+tokenizer load wall time %.3fs",
        DEFAULT_HUB_ID,
        load_seconds,
    )

    max_new = int(os.environ.get("AZR_QWEN35_0_8B_MAX_NEW_TOKENS", "512"))
    max_prompt_len = 2048
    temperature = 0.55
    top_p = 0.92
    rep_pen = 1.12
    min_chars = int(os.environ.get("AZR_QWEN35_0_8B_MIN_ANSWER_CHARS", "48"))
    preview = int(os.environ.get("AZR_QWEN35_0_8B_PREVIEW_CHARS", "400"))

    user_question = qwen35_user_question()

    nt_prompt, nt_text, nt_sec = _generate_continuation(
        adapter,
        user_text=user_question,
        enable_thinking=False,
        max_new_tokens=max_new,
        max_prompt_length=max_prompt_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_pen,
    )
    th_prompt, th_text, th_sec = _generate_continuation(
        adapter,
        user_text=user_question,
        enable_thinking=True,
        max_new_tokens=max_new,
        max_prompt_length=max_prompt_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_pen,
    )

    logpfx = f"Qwen3.5-0.8B smoke (weights={DEFAULT_HUB_ID})"
    logger.info(
        "%s: non-thinking - generate %.3fs, continuation %d chars (max_new_tokens=%s)",
        logpfx,
        nt_sec,
        len(nt_text),
        max_new,
    )
    logger.info(
        "%s: thinking - generate %.3fs, continuation %d chars (max_new_tokens=%s)",
        logpfx,
        th_sec,
        len(th_text),
        max_new,
    )
    delta = th_sec - nt_sec
    ratio = th_sec / max(nt_sec, 1e-9)
    logger.info(
        "%s: comparison - thinking vs non-thinking: delta_time %+.3fs "
        "(thinking %.2fx non-thinking wall time)",
        logpfx,
        delta,
        ratio,
    )
    nt_prev = nt_text[:preview] + ("..." if len(nt_text) > preview else "")
    th_prev = th_text[:preview] + ("..." if len(th_text) > preview else "")
    logger.info("%s: non-thinking preview (%d chars max): %s", logpfx, preview, nt_prev)
    logger.info("%s: thinking preview (%d chars max): %s", logpfx, preview, th_prev)

    assert len(nt_text) > 0, "Non-thinking: empty continuation"
    assert len(th_text) > 0, "Thinking: empty continuation"
    assert len(nt_text) >= min_chars, (
        f"Non-thinking: expected ≥{min_chars} chars, got {len(nt_text)}"
    )
    assert len(th_text) >= min_chars, (
        f"Thinking: expected ≥{min_chars} chars, got {len(th_text)}"
    )

    if os.environ.get("AZR_QWEN35_0_8B_SMOKE_VERBOSE", "").strip().lower() in {"1", "true", "yes"}:
        cap = int(os.environ.get("AZR_QWEN35_0_8B_SMOKE_PRINT_CAP", "6000"))
        pcap = int(os.environ.get("AZR_QWEN35_0_8B_SMOKE_PROMPT_PRINT_CAP", "4000"))
        nt_score, nt_total, nt_checks = _quality_hint(nt_text)
        th_score, th_total, th_checks = _quality_hint(th_text)
        print("\n========== QWEN3.5-0.8B SMOKE (HF weights: Qwen/Qwen3.5-0.8B) ==========", flush=True)
        print(f"\nmodel+tokenizer load: {load_seconds:.3f}s", flush=True)
        print("\n--- SHARED USER QUESTION ---\n", user_question, sep="", flush=True)
        print("\n--- NON-THINKING ---", flush=True)
        print(f"generate wall time: {nt_sec:.3f}s | continuation chars: {len(nt_text)}", flush=True)
        print("\nformatted prompt (capped):\n", nt_prompt[:pcap], sep="", flush=True)
        print("\nfull continuation (capped):\n", nt_text[:cap], sep="", flush=True)
        print("\n--- THINKING ---", flush=True)
        print(f"generate wall time: {th_sec:.3f}s | continuation chars: {len(th_text)}", flush=True)
        print("\nformatted prompt (capped):\n", th_prompt[:pcap], sep="", flush=True)
        print("\nfull continuation (capped):\n", th_text[:cap], sep="", flush=True)
        print("\n--- HEURISTIC CORRECTNESS CHECK (not a proof) ---", flush=True)
        print(f"non-thinking concept coverage: {nt_score}/{nt_total} -> {nt_checks}", flush=True)
        print(f"thinking concept coverage: {th_score}/{th_total} -> {th_checks}", flush=True)
        print("\n--- COMPARISON ---", flush=True)
        print(f"delta generate time (thinking - non-thinking): {delta:+.3f}s", flush=True)
        print(f"ratio (thinking / non-thinking): {ratio:.2f}x", flush=True)
        print("===============================================================\n", flush=True)
