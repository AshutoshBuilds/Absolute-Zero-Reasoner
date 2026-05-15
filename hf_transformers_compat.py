"""
Transformers API compatibility (dtype kwarg vs torch_dtype, etc.).
Kept dependency-free of other hf_* modules to avoid import cycles.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional, Type

import torch

_AZR_CUDA_ATTN_ENV_APPLIED = False


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def apply_azr_attention_env_once() -> None:
    """
    When AZR_SDPA_DISABLED is set, best-effort disable CUDA SDP backends that can
    interact badly with some Windows + PyTorch builds during scaled_dot_product_attention.
    Safe to call multiple times (runs kernel toggles once per process).
    """
    global _AZR_CUDA_ATTN_ENV_APPLIED
    if _AZR_CUDA_ATTN_ENV_APPLIED:
        return
    _AZR_CUDA_ATTN_ENV_APPLIED = True
    if not _truthy_env("AZR_SDPA_DISABLED"):
        return
    if not torch.cuda.is_available():
        return
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass


def explicit_attn_implementation_from_azr_env() -> Optional[str]:
    """
    Optional attn_implementation for AutoModel*.from_pretrained.

    - AZR_ATTN_IMPLEMENTATION: non-empty string passed through (e.g. eager, sdpa, flash_attention_2).
    - AZR_SDPA_DISABLED: treated as eager when AZR_ATTN_IMPLEMENTATION is unset.

    Returns None when neither applies — callers keep legacy defaults (e.g. forced sdpa in setup utils).
    """
    raw = os.environ.get("AZR_ATTN_IMPLEMENTATION", "").strip()
    if raw:
        return raw
    if _truthy_env("AZR_SDPA_DISABLED"):
        return "eager"
    return None

try:
    from transformers.modeling_utils import PreTrainedModel
except Exception:  # pragma: no cover - optional import shape
    PreTrainedModel = None  # type: ignore[misc, assignment]


def _from_pretrained_accepts_dtype(fn: Any) -> bool:
    try:
        return "dtype" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def dtype_kwargs_for_from_pretrained(model_cls: Type[Any], torch_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    """
    Prefer `dtype` when the installed Transformers supports it (silences deprecation
    warnings for `torch_dtype`); otherwise fall back to `torch_dtype`.
    """
    if torch_dtype is None:
        return {}
    for fn in (
        PreTrainedModel.from_pretrained if PreTrainedModel is not None else None,
        getattr(model_cls, "from_pretrained", None),
    ):
        if fn is not None and _from_pretrained_accepts_dtype(fn):
            return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}
