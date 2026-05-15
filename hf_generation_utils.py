import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from hf_transformers_compat import apply_azr_attention_env_once

logger = logging.getLogger(__name__)

_genu_throttle_counts: Dict[str, int] = defaultdict(int)


def _strip_sampling_kwargs_for_greedy_or_beam(gen_kwargs: Dict[str, Any]) -> None:
    """
    When do_sample is False, Transformers may warn that temperature/top_p/top_k are ignored.
    Remove them so eval/benchmark greedy and beam paths stay quiet and configs stay valid.
    """
    if gen_kwargs.get("do_sample", False):
        return
    for k in ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
        gen_kwargs.pop(k, None)


def _genu_throttled_log(key: str, msg: str, *args: Any) -> None:
    """Emit GenU warnings at WARNING for the first N hits per process key, then DEBUG."""
    raw = (os.environ.get("AZR_GENU_LOG_WARN_CAP") or "3").strip()
    try:
        cap = max(0, int(raw))
    except ValueError:
        cap = 3
    _genu_throttle_counts[key] += 1
    n = _genu_throttle_counts[key]
    level = logging.WARNING if n <= cap else logging.DEBUG
    logger.log(level, msg, *args)


def _want_gen_logits_fp32() -> bool:
    """
    Prefer float32 matmul autocast during CUDA generation for logits/sampling stability.

    Default: on when CUDA is available and AZR_GEN_LOGITS_FP32 is unset; set AZR_GEN_LOGITS_FP32=0/false/off
    to disable. Explicit 1/true/yes/on always enables; explicit 0/false/no/off always disables.

    If logits are still non-finite while this is enabled, weights are likely pure fp16; prefer bf16-capable GPUs
    with ``--model-dtype auto`` (adapter matches ``hf_model_setup_utils`` bf16-first) or explicit ``bf16``.
    """
    raw = os.environ.get("AZR_GEN_LOGITS_FP32", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return torch.cuda.is_available()


def _cuda_generation_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if _want_gen_logits_fp32():
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float32)
        except (TypeError, ValueError):
            try:
                # Some builds expose only the cuda.amp alias for float32 cast dtype.
                return torch.cuda.amp.autocast(dtype=torch.float32)
            except (TypeError, ValueError, AttributeError):
                logger.warning(
                    "AZR_GEN_LOGITS_FP32: float32 autocast unavailable; using autocast disabled "
                    "(forward follows weight dtype—use bf16/auto or AZR_SDPA_DISABLED on Windows)."
                )
    return torch.autocast(device_type="cuda", enabled=False)


class _FiniteLogitsProcessor(LogitsProcessor):
    """
    Generation-time guard: sanitize non-finite logits before sampling.
    """

    def __init__(
        self,
        replacement: float = 0.0,
        max_abs: float = 50.0,
        stats: Optional[Dict[str, int]] = None,
    ):
        self.replacement = replacement
        self.max_abs = max_abs
        self.stats = stats

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        orig_dtype = scores.dtype
        probe = scores.float() if _want_gen_logits_fp32() else scores
        if torch.isfinite(probe).all():
            return scores

        if self.stats is not None:
            self.stats["sanitized_steps"] = int(self.stats.get("sanitized_steps", 0)) + 1
            self.stats["invalid_total"] = int(self.stats.get("invalid_total", 0)) + int(
                (~torch.isfinite(probe)).sum().item()
            )

        cleaned = torch.nan_to_num(
            probe,
            nan=self.replacement,
            posinf=self.max_abs,
            neginf=-self.max_abs,
        )
        cleaned = torch.clamp(cleaned, -self.max_abs, self.max_abs)
        return cleaned.to(dtype=orig_dtype)


def generate_text_with_model(
    model_to_use: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 8,
    max_prompt_length: Optional[int] = None,
    critic_model_for_mode_management: Optional[PreTrainedModel] = None,
    use_separate_value_model: bool = False,
    **kwargs: Dict[str, Any],
) -> List[str]:
    """
    Generates text using the provided Hugging Face model and tokenizer.

    Args:
        model_to_use: The Hugging Face model (actor model if separate, else main model).
        tokenizer: The Hugging Face tokenizer.
        device: The torch device to use.
        prompt (str): The input prompt for the model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling top_p.
        num_return_sequences (int): Number of sequences to generate.
        max_prompt_length (Optional[int]): If provided, input prompt will be truncated.
        critic_model_for_mode_management (Optional[PreTrainedModel]): The critic model, if separate, for eval/train mode management.
        use_separate_value_model (bool): Flag indicating if a separate value model architecture is in use.
        **kwargs: Additional keyword arguments to pass to the model's generate method.

    Returns:
        list[str]: A list of generated text sequences (excluding the prompt).
    """
    original_modes = {}

    apply_azr_attention_env_once()

    if model_to_use.training:
        original_modes["model_to_use"] = True
        model_to_use.eval()

    if (
        use_separate_value_model
        and critic_model_for_mode_management is not None
        and critic_model_for_mode_management.training
    ):
        original_modes["critic"] = True
        critic_model_for_mode_management.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)

    if max_prompt_length is not None:
        for key in inputs:
            inputs[key] = inputs[key][:, :max_prompt_length]

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    _gen_ctx = _cuda_generation_autocast(device)

    gen_stats: Dict[str, int] = {"sanitized_steps": 0, "invalid_total": 0}
    outputs: Optional[torch.Tensor] = None

    try:
        try:
            with torch.no_grad(), _gen_ctx:
                initial_outputs = model_to_use(input_ids=input_ids, attention_mask=attention_mask)
                initial_logits = initial_outputs.logits
                if initial_logits is not None and initial_logits.numel() > 0:
                    probe = initial_logits.float() if _want_gen_logits_fp32() else initial_logits
                    if not torch.isfinite(probe).all():
                        invalid_count = int((~torch.isfinite(probe)).sum().item())
                        _genu_throttled_log(
                            "initial_logits",
                            "GenU: Initial logits are non-finite (%d/%d); sanitizing for checks.",
                            invalid_count,
                            probe.numel(),
                        )
                        probe = torch.nan_to_num(
                            probe,
                            nan=0.0,
                            posinf=50.0,
                            neginf=-50.0,
                        )
                    logger.debug(
                        "GenU: Initial logits - Shape: %s, dtype: %s", initial_logits.shape, initial_logits.dtype
                    )
                    logger.debug(
                        "GenU: Initial logits - Has NaN: %s, Has Inf: %s, Has NegInf: %s",
                        torch.isnan(probe).any().item(),
                        torch.isinf(probe).any().item(),
                        torch.isneginf(probe).any().item(),
                    )
                    logger.debug(
                        "GenU: Initial logits - Min: %s, Max: %s, Mean: %s",
                        probe.min().item(),
                        probe.max().item(),
                        probe.mean().item(),
                    )
                    initial_probs_check = F.softmax(probe[:, -1, :], dim=-1)
                    logger.debug(
                        "GenU: Softmax check on last token logits - Has NaN: %s, Sum: %s",
                        torch.isnan(initial_probs_check).any().item(),
                        initial_probs_check.sum(dim=-1),
                    )
                else:
                    logger.debug("GenU: Initial logits are None or empty.")
        except Exception as e_log_logits:
            logger.error("GenU: Error during initial logit check: %s", e_log_logits)

        logits_processor = kwargs.pop("logits_processor", LogitsProcessorList())
        if not isinstance(logits_processor, LogitsProcessorList):
            logits_processor = LogitsProcessorList(logits_processor)
        logits_processor.append(_FiniteLogitsProcessor(stats=gen_stats))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "logits_processor": logits_processor,
        }
        gen_kwargs.update(kwargs)
        _strip_sampling_kwargs_for_greedy_or_beam(gen_kwargs)

        try:
            with torch.no_grad(), _gen_ctx:
                outputs = model_to_use.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        except Exception as e:
            logger.error("GenU: Error during model generation (sampled): %s", e)
            if "probability tensor contains either" in str(e).lower():
                _genu_throttled_log(
                    "retry_deterministic",
                    "GenU: Detected invalid sampling probabilities. Retrying generation in deterministic mode with sanitized logits.",
                )
                safe_gen_kwargs = dict(gen_kwargs)
                safe_gen_kwargs["do_sample"] = False
                safe_gen_kwargs.pop("top_p", None)
                safe_gen_kwargs.pop("temperature", None)
                safe_gen_kwargs.pop("logits_processor", None)
                _strip_sampling_kwargs_for_greedy_or_beam(safe_gen_kwargs)
                with torch.no_grad(), _gen_ctx:
                    outputs = model_to_use.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **safe_gen_kwargs,
                    )
            else:
                raise

        if gen_stats.get("sanitized_steps", 0) > 0:
            _genu_throttled_log(
                "sanitized_scores",
                "GenU: Sanitized non-finite generation scores in %d step(s); %d bad scalar(s) total (see AZR_GEN_LOGITS_FP32, --model-dtype).",
                gen_stats["sanitized_steps"],
                gen_stats.get("invalid_total", 0),
            )

        prompt_len = input_ids.shape[1]
        return [
            tokenizer.decode(output[prompt_len:], skip_special_tokens=True) for output in outputs
        ]
    finally:
        if "model_to_use" in original_modes:
            model_to_use.train()
        if "critic" in original_modes and critic_model_for_mode_management:
            critic_model_for_mode_management.train()


def generate_texts_batched_with_model(
    model_to_use: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    max_prompt_length: Optional[int] = None,
    critic_model_for_mode_management: Optional[PreTrainedModel] = None,
    use_separate_value_model: bool = False,
    skip_preflight_logits_check: bool = True,
    **kwargs,
) -> List[str]:
    """
    Batched greedy/single-sequence generation for equal max_new_tokens and shared kwargs.

    Requires ``num_return_sequences == 1`` (and no beam multi-return). Uses left padding
    for batched decoder-only generation, then restores ``tokenizer.padding_side``.
    """
    if not prompts:
        return []
    if num_return_sequences != 1:
        raise ValueError(
            "generate_texts_batched_with_model only supports num_return_sequences=1; "
            "use generate_text_with_model per prompt for pass@k / beam multi-return."
        )
    merged: Dict[str, Any] = dict(kwargs)

    original_modes: Dict[str, bool] = {}
    if model_to_use.training:
        original_modes["model_to_use"] = True
        model_to_use.eval()
    if (
        use_separate_value_model
        and critic_model_for_mode_management is not None
        and critic_model_for_mode_management.training
    ):
        original_modes["critic"] = True
        critic_model_for_mode_management.eval()

    prev_padding_side = getattr(tokenizer, "padding_side", "right")
    tok_kw: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
    if max_prompt_length is not None:
        tok_kw["truncation"] = True
        tok_kw["max_length"] = max_prompt_length
    else:
        tok_kw["truncation"] = False

    apply_azr_attention_env_once()

    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, **tok_kw)
    finally:
        tokenizer.padding_side = prev_padding_side

    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    _gen_ctx = _cuda_generation_autocast(device)
    gen_stats: Dict[str, int] = {"sanitized_steps": 0, "invalid_total": 0}

    try:
        if not skip_preflight_logits_check:
            try:
                with torch.no_grad(), _gen_ctx:
                    initial_outputs = model_to_use(input_ids=input_ids, attention_mask=attention_mask)
                    initial_logits = initial_outputs.logits
                    if initial_logits is not None and initial_logits.numel() > 0:
                        probe = initial_logits.float() if _want_gen_logits_fp32() else initial_logits
                        if not torch.isfinite(probe).all():
                            invalid_count = int((~torch.isfinite(probe)).sum().item())
                            _genu_throttled_log(
                                "batch_initial_logits",
                                "GenU: Initial logits are non-finite (%d/%d) in batch preflight; sanitizing for checks.",
                                invalid_count,
                                probe.numel(),
                            )
            except Exception as e_log_logits:
                logger.error("GenU: Error during batched initial logit check: %s", e_log_logits)

        logits_processor = merged.pop("logits_processor", LogitsProcessorList())
        if not isinstance(logits_processor, LogitsProcessorList):
            logits_processor = LogitsProcessorList(logits_processor)
        logits_processor.append(_FiniteLogitsProcessor(stats=gen_stats))

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "logits_processor": logits_processor,
        }
        gen_kwargs.update(merged)
        _strip_sampling_kwargs_for_greedy_or_beam(gen_kwargs)
        if int(gen_kwargs.get("num_return_sequences", 1) or 1) != 1:
            raise ValueError(
                "Batched generation requires num_return_sequences=1 after kwargs merge; "
                "lower --samples-per-task or disable AZR_BENCHMARK_BATCH_SIZE."
            )

        try:
            with torch.no_grad(), _gen_ctx:
                outputs = model_to_use.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        except Exception as e:
            logger.error("GenU: Error during batched model generation (sampled): %s", e)
            if "probability tensor contains either" in str(e).lower():
                _genu_throttled_log(
                    "batch_retry_deterministic",
                    "GenU: Detected invalid sampling probabilities (batch). Retrying in deterministic mode.",
                )
                safe_gen_kwargs = dict(gen_kwargs)
                safe_gen_kwargs["do_sample"] = False
                safe_gen_kwargs.pop("top_p", None)
                safe_gen_kwargs.pop("temperature", None)
                safe_gen_kwargs.pop("logits_processor", None)
                _strip_sampling_kwargs_for_greedy_or_beam(safe_gen_kwargs)
                with torch.no_grad(), _gen_ctx:
                    outputs = model_to_use.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **safe_gen_kwargs,
                    )
            else:
                raise

        if gen_stats.get("sanitized_steps", 0) > 0:
            _genu_throttled_log(
                "batch_sanitized_scores",
                "GenU (batch): Sanitized non-finite generation scores in %d step(s); %d bad scalar(s) total.",
                gen_stats["sanitized_steps"],
                gen_stats.get("invalid_total", 0),
            )

        prompt_len = input_ids.shape[1]
        decoded: List[str] = []
        for row in outputs:
            decoded.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        return decoded
    finally:
        if "model_to_use" in original_modes:
            model_to_use.train()
        if "critic" in original_modes and critic_model_for_mode_management:
            critic_model_for_mode_management.train()
