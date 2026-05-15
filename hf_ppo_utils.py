from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

import torch.distributions as D

logger = logging.getLogger(__name__)


@contextmanager
def _ppo_train_mode_guard(adapter):
    """Force train() during PPO re-forward/backward; gradient checkpointing under eval() drops autograd."""
    saved_actor = saved_critic = saved_main = None
    if adapter.use_separate_value_model:
        saved_actor = adapter.actor_model.training
        saved_critic = adapter.critic_model.training
        adapter.actor_model.train()
        adapter.critic_model.train()
    elif getattr(adapter, "model", None) is not None:
        saved_main = adapter.model.training
        adapter.model.train()
    try:
        yield
    finally:
        if adapter.use_separate_value_model:
            adapter.actor_model.train(saved_actor)
            adapter.critic_model.train(saved_critic)
        elif getattr(adapter, "model", None) is not None:
            adapter.model.train(saved_main)


def _sanitize_finite_tensor(
    tensor_name: str,
    tensor: torch.Tensor | None,
    replacement: float = 0.0,
    max_abs: float | None = None,
):
    """Return a finite copy of a tensor with NaN/Inf values replaced."""
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
            f"PPO util sanitized non-finite values in {tensor_name}: {invalid_count}/{total_count} entries."
        )
    elif total_count > 0:
        logger.debug(
            f"PPO util encountered all-non-finite values in {tensor_name} with {total_count} entries."
        )

    tensor = torch.nan_to_num(tensor, nan=replacement, posinf=replacement, neginf=-replacement)
    if max_abs is not None:
        tensor = torch.clamp(tensor, min=-max_abs, max=max_abs)
    return tensor

def get_model_outputs_for_ppo(trainer_instance,
                               full_sequence_ids: torch.Tensor,
                               full_sequence_mask: torch.Tensor,
                               action_ids: torch.Tensor,
                               action_mask: torch.Tensor = None
                               ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Gets action logits, sequence log probabilities, and values from the adapter.
    """
    adapter = trainer_instance.adapter
    device = trainer_instance.device
    full_sequence_ids = full_sequence_ids.to(device)
    full_sequence_mask = full_sequence_mask.to(device)
    action_ids = action_ids.to(device)
    if action_mask is not None:
        action_mask = action_mask.to(device).bool()
    else:
        action_mask = torch.ones_like(action_ids, dtype=torch.bool, device=device)

    raw_full_logits, log_probs_per_token, values = adapter.get_action_and_value(
        input_ids=full_sequence_ids,
        attention_mask=full_sequence_mask,
        actions=action_ids
    )

    raw_full_logits = _sanitize_finite_tensor("raw_full_logits", raw_full_logits, replacement=0.0)
    log_probs_per_token = _sanitize_finite_tensor("log_probs_per_token", log_probs_per_token, replacement=0.0)
    values = _sanitize_finite_tensor("values", values, replacement=0.0)

    if raw_full_logits is None or values is None:
        logger.error("Adapter failed to return logits or values.")
        return None, None, None

    sequence_log_probs = None
    if log_probs_per_token is not None:
        overlap_len = min(log_probs_per_token.shape[1], action_ids.shape[1], action_mask.shape[1])
        if overlap_len == 0:
            logger.warning("No overlapping tokens for sequence log-prob alignment. Cannot compute sequence_log_probs.")
        else:
            if (
                log_probs_per_token.shape[1] != overlap_len
                or action_ids.shape[1] != overlap_len
                or action_mask.shape[1] != overlap_len
            ):
                logger.warning(
                    f"Alignment mismatch for sequence_log_probs: log_probs_len={log_probs_per_token.shape[1]}, "
                    f"action_ids_len={action_ids.shape[1]}, action_mask_len={action_mask.shape[1]}, overlap_len={overlap_len}"
                )
            masked_log_probs = log_probs_per_token[:, :overlap_len] * action_mask[:, :overlap_len]
            sequence_log_probs = masked_log_probs.sum(dim=1)
    else:
        logger.warning("Adapter returned None for log_probs_per_token. Cannot compute sequence_log_probs.")
    sequence_log_probs = _sanitize_finite_tensor("sequence_log_probs", sequence_log_probs, replacement=0.0)

    prompt_len = full_sequence_ids.shape[1] - action_ids.shape[1]
    start_idx = prompt_len - 1 if prompt_len > 0 else 0
    action_logits_for_actions = raw_full_logits[:, start_idx : start_idx + action_ids.shape[1], :]
    
    if action_logits_for_actions.shape[1] != action_ids.shape[1]:
        logger.warning(f"Shape mismatch for action_logits_for_actions ({action_logits_for_actions.shape[1]}) vs action_ids ({action_ids.shape[1]}). Entropy calculation might be incorrect.")
        overlap_len = min(action_logits_for_actions.shape[1], action_ids.shape[1])
        action_logits_for_actions = action_logits_for_actions[:, :overlap_len, :]
        if overlap_len == 0:
            logger.warning("No overlapping action logits available after alignment in get_model_outputs_for_ppo.")

    action_logits_for_actions = _sanitize_finite_tensor(
        "action_logits_for_actions",
        action_logits_for_actions,
        replacement=0.0,
    )

    return action_logits_for_actions, sequence_log_probs, values


def calculate_gae(trainer_instance, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    config = trainer_instance.config
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = rewards.size(0)
    detached_values = values.detach()

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            delta = rewards[t] - detached_values[t]
            last_gae_lam = delta
        else:
            delta = rewards[t] + config["gamma"] * detached_values[t+1] * (1.0 - dones[t+1].float()) - detached_values[t]
            last_gae_lam = delta + config["gamma"] * config["lambda_gae"] * last_gae_lam * (1.0 - dones[t+1].float())
        advantages[t] = last_gae_lam
    
    returns = advantages + detached_values
    return advantages, returns


def perform_ppo_update(trainer_instance) -> dict:
    device = trainer_instance.device
    adapter = trainer_instance.adapter
    config = trainer_instance.config
    optimizer = trainer_instance.optimizer
    scaler = trainer_instance.scaler
    ppo_buffer = trainer_instance.ppo_buffer

    pad_token_id = adapter.tokenizer.pad_token_id if adapter.tokenizer else 0

    all_prompts_ids_list = list(ppo_buffer["prompts_ids"])
    all_prompts_mask_list = list(ppo_buffer["prompts_mask"])
    all_generated_ids_list = list(ppo_buffer["generated_ids"])
    
    all_rewards = torch.stack(list(ppo_buffer["rewards"]))
    old_log_probs_tensor = torch.stack(list(ppo_buffer["log_probs"]))
    old_values_tensor = torch.stack(list(ppo_buffer["values"]))
    all_dones = torch.stack(list(ppo_buffer["dones"]))

    num_samples = len(all_rewards)
    if num_samples == 0: return {}

    advantages, returns = calculate_gae(trainer_instance, all_rewards, old_values_tensor, all_dones)
    advantages_std = advantages.std(unbiased=False)
    if not torch.isfinite(advantages_std) or advantages_std < 1e-12:
        logger.warning(
            "Advantages have near-zero/non-finite standard deviation. Skipping normalization to avoid instability."
        )
        advantages = advantages - advantages.mean()
    else:
        advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)

    if adapter.use_separate_value_model:
        clip_target_params = trainer_instance.actor_params + trainer_instance.critic_params
    else:
        clip_target_params = list(adapter.model.parameters()) if getattr(adapter, "model", None) is not None else []

    total_policy_loss = 0; total_value_loss = 0; total_entropy = 0
    valid_minibatch_updates = 0
    skipped_minibatches = 0
    
    logger.debug(f"PPO Update: old_values_tensor - Has NaN: {torch.isnan(old_values_tensor).any()}, Has Inf: {torch.isinf(old_values_tensor).any()}")
    logger.debug(f"PPO Update: all_rewards - Has NaN: {torch.isnan(all_rewards).any()}, Has Inf: {torch.isinf(all_rewards).any()}")
    logger.debug(f"PPO Update: advantages - Has NaN: {torch.isnan(advantages).any()}, Has Inf: {torch.isinf(advantages).any()}, Min: {advantages.min()}, Max: {advantages.max()}, Mean: {advantages.mean()}")
    logger.debug(f"PPO Update: returns - Has NaN: {torch.isnan(returns).any()}, Has Inf: {torch.isinf(returns).any()}")

    ppo_mb_stride = int(config["batch_size"])
    _raw_ppo_micro = os.environ.get("AZR_PPO_MICROBATCH_SIZE", "").strip()
    if _raw_ppo_micro:
        try:
            _v = int(_raw_ppo_micro)
            if _v > 0:
                ppo_mb_stride = min(ppo_mb_stride, _v)
            else:
                logger.warning("AZR_PPO_MICROBATCH_SIZE=%r is not positive; using batch_size=%s", _raw_ppo_micro, ppo_mb_stride)
        except ValueError:
            logger.warning("Invalid AZR_PPO_MICROBATCH_SIZE=%r; using batch_size=%s", _raw_ppo_micro, ppo_mb_stride)

    for ppo_epoch_num in range(config["ppo_epochs"]):
        permutation = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, ppo_mb_stride):
            end_idx = min(start_idx + ppo_mb_stride, num_samples)
            mb_indices = permutation[start_idx:end_idx]
            if len(mb_indices) == 0: continue

            optimizer.zero_grad()

            # Only enable autocast if scaler is present (mixed precision enabled)
            use_mixed_precision = (device.type == 'cuda' and scaler is not None)
            if adapter.use_separate_value_model:
                reference_model = adapter.actor_model
            else:
                reference_model = adapter.model
            if reference_model is None or not hasattr(reference_model, "parameters"):
                logger.error(
                    "PPO update expects a valid actor model but none was found on adapter. "
                    f"use_separate_value_model={adapter.use_separate_value_model}"
                )
                return {}
            model_dtype = torch.float16 if use_mixed_precision else next(reference_model.parameters()).dtype

            mb_prompts_ids_unpadded = [all_prompts_ids_list[i] for i in mb_indices]
            mb_prompts_mask_unpadded = [all_prompts_mask_list[i] for i in mb_indices]
            mb_generated_ids_unpadded = [all_generated_ids_list[i] for i in mb_indices]

            mb_old_log_probs = old_log_probs_tensor[mb_indices].to(device=device, dtype=model_dtype)
            mb_advantages = advantages[mb_indices].to(device=device, dtype=model_dtype)
            mb_returns = returns[mb_indices].to(device=device, dtype=model_dtype)
            mb_old_values = old_values_tensor[mb_indices].to(device=device, dtype=model_dtype)
            
            padded_mb_prompts_ids = pad_sequence(mb_prompts_ids_unpadded, batch_first=True, padding_value=pad_token_id).to(device)
            padded_mb_prompts_mask = pad_sequence(mb_prompts_mask_unpadded, batch_first=True, padding_value=0).to(device) 
            padded_mb_generated_ids = pad_sequence(mb_generated_ids_unpadded, batch_first=True, padding_value=pad_token_id).to(device)
            padded_mb_generated_mask = (padded_mb_generated_ids != pad_token_id).long().to(device)

            # CUDA autocast on top of fp16 weights often amplifies NaNs in PPO re-forward passes.
            # Default disable nested autocast on CUDA+fp16 when env is unset; set AZR_PPO_DISABLE_CUDA_AUTOCAST=0 to opt out.
            with _ppo_train_mode_guard(adapter):
                _ppo_env = os.environ.get("AZR_PPO_DISABLE_CUDA_AUTOCAST", "").strip().lower()
                if _ppo_env in ("0", "false", "no", "off"):
                    _ppo_disable_cuda_autocast = False
                elif _ppo_env in ("1", "true", "yes", "on"):
                    _ppo_disable_cuda_autocast = True
                else:
                    try:
                        _param_dtype = next(reference_model.parameters()).dtype
                    except StopIteration:
                        _param_dtype = torch.float32
                    _ppo_disable_cuda_autocast = device.type == "cuda" and _param_dtype == torch.float16
                _ppo_fwd_ctx = (
                    autocast(device_type="cuda", enabled=False)
                    if device.type == "cuda" and _ppo_disable_cuda_autocast
                    else nullcontext()
                )

                with _ppo_fwd_ctx:
                    current_action_logits_for_actions, current_sequence_log_probs, current_values = get_model_outputs_for_ppo(
                        trainer_instance,
                        full_sequence_ids=torch.cat((padded_mb_prompts_ids, padded_mb_generated_ids), dim=1),
                        full_sequence_mask=torch.cat((padded_mb_prompts_mask, padded_mb_generated_mask), dim=1),
                        action_ids=padded_mb_generated_ids,
                        action_mask=padded_mb_generated_mask
                    )

                if current_action_logits_for_actions is None or current_sequence_log_probs is None or current_values is None:
                    logger.warning(f"Skipping PPO minibatch due to None from get_model_outputs_for_ppo. Shapes: prompts {padded_mb_prompts_ids.shape}, generated {padded_mb_generated_ids.shape}")
                    continue
                
                mb_old_log_probs = _sanitize_finite_tensor("mb_old_log_probs", mb_old_log_probs, replacement=0.0, max_abs=50.0)
                mb_advantages = _sanitize_finite_tensor("mb_advantages", mb_advantages, replacement=0.0, max_abs=10.0)
                mb_returns = _sanitize_finite_tensor("mb_returns", mb_returns, replacement=0.0, max_abs=50.0)
                mb_old_values = _sanitize_finite_tensor("mb_old_values", mb_old_values, replacement=0.0, max_abs=50.0)
                current_sequence_log_probs = _sanitize_finite_tensor(
                    "current_sequence_log_probs",
                    current_sequence_log_probs,
                    replacement=0.0
                )
                current_values = _sanitize_finite_tensor(
                    "current_values",
                    current_values,
                    replacement=0.0,
                    max_abs=50.0,
                )
                current_action_logits_for_actions = _sanitize_finite_tensor(
                    "current_action_logits_for_actions",
                    current_action_logits_for_actions,
                    replacement=0.0,
                )
    
                # Skip minibatches with all-NaN/Inf advantages/returns as these create unstable PPO loss updates.
                if mb_advantages is None or not torch.isfinite(mb_advantages).any():
                    skipped_minibatches += 1
                    logger.warning(
                        f"Skipping PPO minibatch (size={len(mb_indices)}) due to non-finite mb_advantages."
                    )
                    continue
                if mb_returns is None or not torch.isfinite(mb_returns).any():
                    skipped_minibatches += 1
                    logger.warning(
                        f"Skipping PPO minibatch (size={len(mb_indices)}) due to non-finite mb_returns."
                    )
                    continue
                
                logger.debug(f"  MB PPO: mb_old_log_probs - Has NaN: {torch.isnan(mb_old_log_probs).any()}, Has Inf: {torch.isinf(mb_old_log_probs).any()}")
                logger.debug(f"  MB PPO: current_sequence_log_probs - Has NaN: {torch.isnan(current_sequence_log_probs).any()}, Has Inf: {torch.isinf(current_sequence_log_probs).any()}")
                logger.debug(f"  MB PPO: current_values - Has NaN: {torch.isnan(current_values).any()}, Has Inf: {torch.isinf(current_values).any()}")
                logger.debug(f"  MB PPO: mb_advantages - Has NaN: {torch.isnan(mb_advantages).any()}, Has Inf: {torch.isinf(mb_advantages).any()}, Min: {mb_advantages.min().item()}, Max: {mb_advantages.max().item()}, Mean: {mb_advantages.mean().item()}")
                logger.debug(f"  MB PPO: current_values (raw) - Min: {current_values.min().item()}, Max: {current_values.max().item()}, Mean: {current_values.mean().item()}")
                logger.debug(f"  MB PPO: mb_returns (target for critic) - Min: {mb_returns.min().item()}, Max: {mb_returns.max().item()}, Mean: {mb_returns.mean().item()}")
    
                entropy_mask = padded_mb_generated_mask
                if current_action_logits_for_actions.shape[1] != entropy_mask.shape[1]:
                    overlap_entropy_len = min(current_action_logits_for_actions.shape[1], entropy_mask.shape[1])
                    if overlap_entropy_len == 0:
                        logger.warning(
                            "No overlap between entropy logits and action mask after PPO alignment; entropy set to 0."
                        )
                    entropy_mask = entropy_mask[:, :overlap_entropy_len]
                    current_action_logits_for_actions = current_action_logits_for_actions[:, :overlap_entropy_len, :]
                    logger.debug(
                        f"Aligned entropy tensors after PPO utils slice: logits_shape={current_action_logits_for_actions.shape}, "
                        f"mask_shape={entropy_mask.shape}"
                    )
    
                if current_action_logits_for_actions.nelement() > 0:
                    logger.debug(f"  MB PPO: current_action_logits_for_actions - Shape: {current_action_logits_for_actions.shape}, Dtype: {current_action_logits_for_actions.dtype}")
                    if not torch.isnan(current_action_logits_for_actions).any() and not torch.isinf(current_action_logits_for_actions).any():
                        logger.debug(f"  MB PPO: current_action_logits_for_actions - Min: {current_action_logits_for_actions.min().item()}, Max: {current_action_logits_for_actions.max().item()}, Mean: {current_action_logits_for_actions.mean().item()}")
                    
                    try:
                        dist = D.Categorical(logits=current_action_logits_for_actions.float())
                        entropy_per_token = dist.entropy()
                    except ValueError as e_dist:
                        logger.error(f"  MB PPO: Error creating Categorical distribution for entropy: {e_dist}. Logits min: {current_action_logits_for_actions.min()}, max: {current_action_logits_for_actions.max()}. Setting entropy to 0.")
                        entropy_per_token = torch.zeros_like(padded_mb_generated_mask[:, 0].unsqueeze(1).expand(-1, current_action_logits_for_actions.shape[1]), device=device, dtype=model_dtype)
                    
                    masked_entropy = entropy_per_token * entropy_mask
                    sum_entropy_per_sequence = masked_entropy.sum(dim=1)
                    actual_tokens_per_sequence = entropy_mask.sum(dim=1).float()
                    actual_tokens_per_sequence = torch.max(actual_tokens_per_sequence, torch.ones_like(actual_tokens_per_sequence))
                    mean_entropy_per_sequence = sum_entropy_per_sequence / actual_tokens_per_sequence
                    entropy = mean_entropy_per_sequence.mean()
                else:
                    entropy = current_values.mean() * 0.0
    
                if mb_old_log_probs.numel() > 0:
                    logger.debug(f"  MB PPO: mb_old_log_probs          - Min: {mb_old_log_probs.min().item():.4f}, Max: {mb_old_log_probs.max().item():.4f}, Mean: {mb_old_log_probs.mean().item():.4f}, Shape: {mb_old_log_probs.shape}")
                if current_sequence_log_probs.numel() > 0:
                    logger.debug(f"  MB PPO: current_sequence_log_probs - Min: {current_sequence_log_probs.min().item():.4f}, Max: {current_sequence_log_probs.max().item():.4f}, Mean: {current_sequence_log_probs.mean().item():.4f}, Shape: {current_sequence_log_probs.shape}")
    
                log_prob_diff = current_sequence_log_probs - mb_old_log_probs.detach()
                log_prob_diff_min = config.get("log_prob_diff_clamp_min", -3.0)
                log_prob_diff_max = config.get("log_prob_diff_clamp_max", 3.0)
                log_prob_diff = torch.clamp(log_prob_diff, log_prob_diff_min, log_prob_diff_max)
    
                logger.debug(f"  MB PPO: log_prob_diff (POST-CLAMP) - Min: {log_prob_diff.min().item()}, Max: {log_prob_diff.max().item()}, Mean: {log_prob_diff.mean().item()}, Has NaN: {torch.isnan(log_prob_diff).any()}, Has Inf: {torch.isinf(log_prob_diff).any()}")
                ratio = torch.exp(log_prob_diff)
                logger.debug(f"  MB PPO: ratio - Min: {ratio.min().item()}, Max: {ratio.max().item()}, Mean: {ratio.mean().item()}, Has NaN: {torch.isnan(ratio).any()}, Has Inf: {torch.isinf(ratio).any()}")
    
                surr1 = ratio * mb_advantages.float()
                surr2 = (
                    torch.clamp(ratio, 1.0 - config["ppo_clip_epsilon"], 1.0 + config["ppo_clip_epsilon"])
                    * mb_advantages.float()
                )
                policy_loss = -torch.min(surr1, surr2).mean()
    
                value_pred_clipped = mb_old_values + torch.clamp(
                    current_values - mb_old_values,
                    -config["value_clip_epsilon"],
                    config["value_clip_epsilon"],
                )
                value_loss_unclipped = F.mse_loss(current_values.float(), mb_returns.float())
                value_loss_clipped = F.mse_loss(value_pred_clipped.float(), mb_returns.float())
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
                
                loss = policy_loss + config["value_loss_coef"] * value_loss - config["entropy_coef"] * entropy
                
                logger.debug(f"  MB PPO: policy_loss - Is NaN: {torch.isnan(policy_loss).any()}, Is Inf: {torch.isinf(policy_loss).any()}, Value: {policy_loss.item()}")
                logger.debug(f"  MB PPO: value_loss - Is NaN: {torch.isnan(value_loss).any()}, Is Inf: {torch.isinf(value_loss).any()}, Value: {value_loss.item()}")
                logger.debug(f"  MB PPO: entropy - Is NaN: {torch.isnan(entropy).any()}, Is Inf: {torch.isinf(entropy).any()}, Value: {entropy.item()}")
                logger.debug(f"  MB PPO: total_loss - Is NaN: {torch.isnan(loss).any()}, Is Inf: {torch.isinf(loss).any()}, Value: {loss.item()}")
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.error(
                        "CRITICAL: NaN or Inf detected in total PPO loss for this minibatch. "
                        "Skipping backward pass."
                    )
                    skipped_minibatches += 1
                    continue
    
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    if loss.dtype != torch.float32:
                        loss = loss.float()
                    loss.backward()
    
                if config.get("debug_log_scaled_gradients", True):
                    logger.debug("  MB PPO: Logging SCALED gradients (before unscale_):")
                    if adapter.use_separate_value_model:
                        for i, p in enumerate(trainer_instance.actor_params):
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                logger.debug(f"    Actor Param {i} SCALED Grad - Has NaN: {torch.isnan(p.grad).any()}, Has Inf: {torch.isinf(p.grad).any()}, Shape: {p.grad.shape}, Dtype: {p.grad.dtype}")
                        for i, p in enumerate(trainer_instance.critic_params):
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                logger.debug(f"    Critic Param {i} SCALED Grad - Has NaN: {torch.isnan(p.grad).any()}, Has Inf: {torch.isinf(p.grad).any()}, Shape: {p.grad.shape}, Dtype: {p.grad.dtype}")
                    elif hasattr(adapter, 'model') and adapter.model is not None:
                        for i, p in enumerate(adapter.model.parameters()):
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                 logger.debug(f"    Model Param {i} SCALED Grad - Has NaN: {torch.isnan(p.grad).any()}, Has Inf: {torch.isinf(p.grad).any()}, Shape: {p.grad.shape}, Dtype: {p.grad.dtype}")
                
                skip_unscale_grad_clip = False
                if scaler is not None:
                    try:
                        scaler.unscale_(optimizer, allow_fp16=True)
                    except TypeError:
                        try:
                            # Older torch versions don't support allow_fp16; fallback to default unscale.
                            scaler.unscale_(optimizer)
                        except ValueError as unscale_error:
                            # Some environments keep fp16 grads in optimizer params; skip explicit unscale/clip in this path.
                            skip_unscale_grad_clip = True
                            logger.warning(
                                "GradScaler.unscale_ failed with fp16 grads in current PyTorch build; "
                                f"skipping explicit gradient clipping for this minibatch: {unscale_error}"
                            )
                    except ValueError as unscale_error:
                        skip_unscale_grad_clip = True
                        logger.warning(
                            "GradScaler.unscale_ failed with fp16 grads in current PyTorch build; "
                            f"skipping explicit gradient clipping for this minibatch: {unscale_error}"
                        )
    
                gradient_clip_value = 1.0 
                if not skip_unscale_grad_clip:
                    if adapter.use_separate_value_model:
                        if trainer_instance.actor_params:
                            for p in trainer_instance.actor_params:
                                if p.grad is not None: torch.nn.utils.clip_grad_value_(p, gradient_clip_value)
                        if trainer_instance.critic_params:
                            for p in trainer_instance.critic_params:
                                if p.grad is not None: torch.nn.utils.clip_grad_value_(p, gradient_clip_value)
                    else:
                        if clip_target_params:
                            for p in clip_target_params:
                                if p.grad is not None: torch.nn.utils.clip_grad_value_(p, gradient_clip_value)
                    logger.debug(f"  MB PPO: Grads clipped by value to +/- {gradient_clip_value} (unscaled)")
                    
                    gradient_clip_norm = 1.0
                    if adapter.use_separate_value_model:
                        if trainer_instance.actor_params: torch.nn.utils.clip_grad_norm_(trainer_instance.actor_params, gradient_clip_norm)
                        if trainer_instance.critic_params: torch.nn.utils.clip_grad_norm_(trainer_instance.critic_params, gradient_clip_norm)
                    else:
                        if clip_target_params:
                            torch.nn.utils.clip_grad_norm_(clip_target_params, gradient_clip_norm)
                    logger.debug(f"  MB PPO: Grads clipped by norm to {gradient_clip_norm} (unscaled)")
    
                if config.get("log_gradients", False):
                    logger.debug(f"  MB PPO: Logging UNSCALED, CLIPPED gradients:")
                    # ... (gradient logging logic as in original trainer) ...
    
                if scaler is not None:
                    if skip_unscale_grad_clip:
                        # Native unscale_ failed (e.g. fp16 .grad); scaler.step would call unscale_ again and crash.
                        try:
                            scale = float(scaler.get_scale())
                            inv_scale = (1.0 / scale) if scale > 0.0 and math.isfinite(scale) else 1.0
                        except Exception:
                            inv_scale = 1.0
                        if inv_scale != 1.0:
                            for group in optimizer.param_groups:
                                for p in group.get("params", ()):
                                    g = getattr(p, "grad", None)
                                    if g is not None:
                                        g.mul_(inv_scale)
                        optimizer.step()
                        scaler.update()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    optimizer.step()
    
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                valid_minibatch_updates += 1
    
                if config.get("debug_log_scaled_gradients", False):
                    logger.debug(f"  MB PPO: completed valid update #{valid_minibatch_updates}.")
    
    if valid_minibatch_updates == 0:
        logger.warning(
            "PPO update completed with no valid minibatches after sanitizer checks. "
            f"skipped_minibatches={skipped_minibatches}, total_minibatches={len(all_prompts_ids_list)}."
        )
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
    else:
        avg_policy_loss = total_policy_loss / valid_minibatch_updates
        avg_value_loss = total_value_loss / valid_minibatch_updates
        avg_entropy = total_entropy / valid_minibatch_updates

    # Clear PPO buffer after update
    for k in ppo_buffer.keys():
        ppo_buffer[k].clear()
    trainer_instance.current_epoch_experiences_processed_in_ppo += num_samples


    return {
        "avg_policy_loss": avg_policy_loss,
        "avg_value_loss": avg_value_loss,
        "avg_entropy": avg_entropy,
        "valid_minibatch_updates": valid_minibatch_updates,
        "skipped_minibatches": skipped_minibatches,
    } 

