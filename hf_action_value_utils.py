from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import logging
from collections import defaultdict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_gavu_warn_counts: dict[str, int] = defaultdict(int)


def _cross_entropy_flat_chunked(
    logits_2d: torch.Tensor,
    targets_1d: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """
    F.cross_entropy with reduction='none' on flattened (N, V) logits, processing row blocks
    to cap peak softmax memory (large vocab + long unrolled sequences).
    """
    n = logits_2d.size(0)
    if n == 0:
        return logits_2d.new_zeros(0)
    if chunk_rows <= 0 or n <= chunk_rows:
        return F.cross_entropy(logits_2d, targets_1d, reduction="none")
    parts: list[torch.Tensor] = []
    for i in range(0, n, chunk_rows):
        parts.append(
            F.cross_entropy(
                logits_2d[i : i + chunk_rows],
                targets_1d[i : i + chunk_rows],
                reduction="none",
            )
        )
    return torch.cat(parts, dim=0)


def _ppo_ce_chunk_rows() -> int:
    raw = (os.environ.get("AZR_PPO_CE_CHUNK") or "").strip()
    if not raw:
        return 4096
    try:
        v = int(raw)
    except ValueError:
        logger.warning("Invalid AZR_PPO_CE_CHUNK=%r; using 4096.", raw)
        return 4096
    if v <= 0:
        return 4096
    return v


def _gavu_throttled_warning(key: str, msg: str, *args: object) -> None:
    """First N emissions per key at WARNING, then DEBUG (mirrors GenU throttling)."""
    raw = (os.environ.get("AZR_GAVU_LOG_WARN_CAP") or "3").strip()
    try:
        cap = max(0, int(raw))
    except ValueError:
        cap = 3
    _gavu_warn_counts[key] += 1
    n = _gavu_warn_counts[key]
    level = logging.WARNING if n <= cap else logging.DEBUG
    logger.log(level, msg, *args)


def _sanitize_finite_tensor(tensor: Optional[torch.Tensor], name: str, replacement: float = 0.0):
    if tensor is None:
        return tensor
    if torch.isfinite(tensor).all():
        return tensor

    finite_mask = torch.isfinite(tensor)
    invalid_mask = ~finite_mask
    invalid_count = int(invalid_mask.sum().item())
    total_count = tensor.numel()

    if invalid_count < total_count:
        logger.warning(
            f"GAVU: Sanitized non-finite tensor '{name}' with {invalid_count}/{total_count} invalid entries."
        )
    else:
        logger.debug(
            f"GAVU: Sanitized tensor '{name}' where all {total_count} entries are non-finite."
        )

    # Always use nan_to_num (never torch.full_like): full_like severs the graph and makes PPO loss
    # a constant tensor with requires_grad=False when an entire slice was non-finite.
    tensor = torch.nan_to_num(tensor, nan=replacement, posinf=replacement, neginf=-replacement)
    return tensor

def get_action_and_value_from_models(
    actor_model: torch.nn.Module,
    critic_model: Optional[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    device: torch.device = None,
    logit_clamping_value: Optional[float] = 30.0,
    use_separate_value_model: bool = False # Added to maintain consistency with adapter logic
):
    """
    Performs a forward pass to get action logits/log_probs and value estimate.
    Assumes relevant models are in train() mode if gradients are needed.
    Args:
        actor_model: The actor model.
        critic_model: The critic model (can be None if not use_separate_value_model, though this function is primarily for that case).
        actions (torch.Tensor, optional): The actions taken (token ids) for which to compute log_probs.
                                        Shape (batch_size, sequence_length_of_actions).
                                        If None, raw logits are returned.
        device: The torch device.
        logit_clamping_value: Value to clamp logits to.
        use_separate_value_model: Flag indicating if a separate value model is used.
    """
    if not use_separate_value_model:
        # This function is primarily designed for the separate model case.
        # If called without it, it might indicate a logic error elsewhere.
        logger.debug(
            "get_action_and_value_from_models called with use_separate_value_model=False (unified actor-critic)."
        )
        # Fallback or error if necessary, for now, proceed but critic path will be skipped if critic_model is None.

    # Actor forward pass: with a separate critic we skip hidden states to save memory; unified
    # PPO path needs them for the value head fallback.
    actor_output_hidden_states = not use_separate_value_model
    actor_output_attentions = bool(output_attentions) if output_attentions is not None else False
    critic_output_attentions = bool(output_attentions) if output_attentions is not None else False

    actor_outputs = actor_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        use_cache=False, 
        output_attentions=actor_output_attentions,
        output_hidden_states=actor_output_hidden_states,
        return_dict=True
    )
    
    action_logits = _sanitize_finite_tensor(actor_outputs.logits, "actor_outputs.logits")
    if action_logits is not None:
        logger.debug(f"GAVU: actor_outputs.logits (direct from model) - Shape: {action_logits.shape}, Dtype: {action_logits.dtype}")
        if actor_outputs.logits.numel() > 0:
            logger.debug(f"  actor_outputs.logits - Min: {action_logits.min().item():.4f}, Max: {action_logits.max().item():.4f}, Mean: {action_logits.mean():.4f}")
            logger.debug(f"  actor_outputs.logits - Has NaN: {torch.isnan(action_logits).any().item()}, Has Inf: {torch.isinf(action_logits).any().item()}")
    else:
        logger.warning("GAVU: actor_outputs.logits is None directly from model.")

    if actor_outputs.hidden_states is not None:
        last_actor_hidden_state = actor_outputs.hidden_states[-1]
        last_actor_hidden_state = _sanitize_finite_tensor(last_actor_hidden_state, "actor_outputs.hidden_states[-1]")
        logger.debug(f"GAVU: Actor's last hidden state (input to lm_head) - Shape: {last_actor_hidden_state.shape}, Dtype: {last_actor_hidden_state.dtype}")
        if last_actor_hidden_state.numel() > 0:
            logger.debug(f"  Actor LHS - Min: {last_actor_hidden_state.min().item()}, Max: {last_actor_hidden_state.max().item()}, Mean: {last_actor_hidden_state.mean().item()}")
            logger.debug(f"  Actor LHS - Has NaN: {torch.isnan(last_actor_hidden_state).any().item()}, Has Inf: {torch.isinf(last_actor_hidden_state).any().item()}")
    elif actor_output_hidden_states:
        # We asked for hidden states but the model did not return them (unusual).
        _gavu_throttled_warning(
            "hidden_missing_after_request",
            "GAVU: actor_outputs.hidden_states is None after output_hidden_states=True; cannot log lm_head input.",
        )
    else:
        logger.debug(
            "GAVU: skipping lm_head-input diagnostics (output_hidden_states=False for separate value model)."
        )

    if hasattr(actor_model, 'lm_head') and actor_model.lm_head is not None and hasattr(actor_model.lm_head, 'weight'):
        lm_head_weights = actor_model.lm_head.weight
        logger.debug(f"GAVU: Actor's lm_head weights - Shape: {lm_head_weights.shape}, Dtype: {lm_head_weights.dtype}")
        if lm_head_weights.numel() > 0:
            logger.debug(f"  LM Head Weights - Min: {lm_head_weights.min().item()}, Max: {lm_head_weights.max().item()}, Mean: {lm_head_weights.mean().item()}")
            logger.debug(f"  LM Head Weights - Has NaN: {torch.isnan(lm_head_weights).any().item()}, Has Inf: {torch.isinf(lm_head_weights).any().item()}")
    else:
        logger.warning("GAVU: Could not access actor's lm_head weights for logging.")

    if logit_clamping_value is not None and action_logits is not None:
        raw_min, raw_max = action_logits.min(), action_logits.max()
        action_logits = torch.clamp(action_logits, min=-logit_clamping_value, max=logit_clamping_value)
        if not torch.allclose(raw_min, action_logits.min()) or not torch.allclose(raw_max, action_logits.max()):
            _gavu_throttled_warning(
                "logit_clamp",
                "GAVU: Clamped action_logits from [%.2f, %.2f] to [%.2f, %.2f]",
                float(raw_min),
                float(raw_max),
                float(action_logits.min()),
                float(action_logits.max()),
            )
    # Ensure action logits remain finite after clamping.
    action_logits = _sanitize_finite_tensor(action_logits, "action_logits")
    if action_logits is None:
        action_logits = _sanitize_finite_tensor(actor_outputs.logits, "actor_outputs.logits fallback")

    log_probs_per_token = None
    if actions is not None:
        prompt_len = input_ids.shape[1] - actions.shape[1]
        if prompt_len < 0:
            logger.warning(
                f"GAVU: prompt_len computed as negative ({prompt_len}) with input_ids_len={input_ids.shape[1]}, "
                f"actions_len={actions.shape[1]}. Falling back to no-prompt alignment."
            )
            prompt_len = 0

        start_idx = max(prompt_len - 1, 0)
        end_idx = start_idx + actions.shape[1]
        logits_for_actions = action_logits[:, start_idx:end_idx, :]
        targets_for_actions = actions[:, :logits_for_actions.shape[1]]

        if logits_for_actions.shape[1] != targets_for_actions.shape[1]:
            logger.warning(
                f"GAVU: Shape mismatch for log_prob calculation: logits_for_actions.shape[1]={logits_for_actions.shape[1]} "
                f"vs targets.shape[1]={targets_for_actions.shape[1]}. Input_ids len: {input_ids.shape[1]}, "
                f"prompt_len: {prompt_len}. Action_logits shape: {action_logits.shape}. Using overlap."
            )
            overlap_len = min(logits_for_actions.shape[1], targets_for_actions.shape[1])
            logits_for_actions = logits_for_actions[:, :overlap_len, :]
            targets_for_actions = targets_for_actions[:, :overlap_len]

        if logits_for_actions.shape[1] > 0:
            logits_for_actions = _sanitize_finite_tensor(logits_for_actions, "logits_for_actions")
            ce_logits_input_for_loss = (
                logits_for_actions.float()
                if logits_for_actions.dtype in (torch.float16, torch.bfloat16)
                else logits_for_actions
            )
            ce_targets_input = targets_for_actions

            logger.debug(
                f"GAVU: CrossEntropy Logits for log_probs_per_token - Shape: {ce_logits_input_for_loss.shape}, "
                f"Dtype: {ce_logits_input_for_loss.dtype}"
            )
            logger.debug(f"GAVU: CrossEntropy Targets for log_probs_per_token - Shape: {ce_targets_input.shape}, Dtype: {ce_targets_input.dtype}")

            if torch.is_tensor(ce_targets_input) and ce_targets_input.numel() > 0 and actor_model and actor_model.config and hasattr(actor_model.config, 'vocab_size'):
                vocab_size = actor_model.config.vocab_size
                if ce_targets_input.max().item() >= vocab_size:
                    logger.error(
                        f"  !!! CRITICAL: Target ID {ce_targets_input.max().item()} for log_probs_per_token is >= vocab_size {vocab_size}"
                    )
                if ce_targets_input.min().item() < 0:
                    logger.error(f"  !!! CRITICAL: Target ID {ce_targets_input.min().item()} for log_probs_per_token is < 0")

            flat_logits = ce_logits_input_for_loss.reshape(-1, ce_logits_input_for_loss.size(-1))
            flat_targets = ce_targets_input.reshape(-1)
            _ce_chunk = _ppo_ce_chunk_rows()
            if flat_logits.size(0) > _ce_chunk:
                logger.debug(
                    "GAVU: chunked cross_entropy rows=%s chunk=%s vocab=%s",
                    flat_logits.size(0),
                    _ce_chunk,
                    flat_logits.size(-1),
                )
            _log_probs = _cross_entropy_flat_chunked(flat_logits, flat_targets, _ce_chunk)
            log_probs_per_token = -_log_probs.reshape(logits_for_actions.shape[0], logits_for_actions.shape[1])
            
    values = None
    if use_separate_value_model and critic_model is not None:
        if actor_outputs is not None:
            # Actor outputs are no longer needed before critic pass in the separate model path.
            # Freeing them explicitly helps reduce peak memory when the critic forward and backward pass runs.
            del actor_outputs
            torch.cuda.empty_cache()

        # Avoid passing cached past states to the critic pass to reduce GPU memory usage.
        logger.debug(f"GAVU: Dtype of critic_attention_mask passed to critic_model: {attention_mask.dtype if attention_mask is not None else 'None'}")
        critic_outputs = critic_model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            past_key_values=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=critic_output_attentions,
            return_dict=True
        )
        values = _sanitize_finite_tensor(critic_outputs.pooler_output, "critic_outputs.pooler_output", replacement=0.0)
    elif critic_model is None and use_separate_value_model:
        logger.warning("GAVU: Critic model is None, but use_separate_value_model is True. Cannot compute value estimate.")
        values = None
    elif not use_separate_value_model:
        # Single-model PPO path: derive a scalar value from actor hidden states.
        logger.debug("GAVU: Single model mode (use_separate_value_model=False). Deriving value from actor hidden states.")
        if actor_outputs.hidden_states is not None and actor_outputs.hidden_states[-1] is not None:
            hidden_states = actor_outputs.hidden_states[-1]
            hidden_states = _sanitize_finite_tensor(hidden_states, "actor_outputs.hidden_states[-1] (single-model)")
            if attention_mask is not None:
                last_token_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
                last_token_idx = last_token_idx.to(dtype=torch.long, device=hidden_states.device)
            else:
                last_token_idx = torch.full(
                    (hidden_states.shape[0],),
                    hidden_states.shape[1] - 1,
                    device=hidden_states.device,
                    dtype=torch.long
                )
            values = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_token_idx].mean(dim=-1)
            values = _sanitize_finite_tensor(values, "single_model values")
        else:
            _gavu_throttled_warning(
                "single_model_hidden_missing",
                "GAVU: Actor output hidden states unavailable for single-model fallback; using zeros.",
            )
            values = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=action_logits.dtype if action_logits is not None else torch.float32)
            values = _sanitize_finite_tensor(values, "zero_fallback values")


    return action_logits, log_probs_per_token, values 