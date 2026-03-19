import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from contextlib import nullcontext

logger = logging.getLogger(__name__)

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
    use_separate_value_model: bool = True # Added to maintain consistency with adapter logic
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
        logger.warning("get_action_and_value_from_models called with use_separate_value_model=False. Ensure this is intended.")
        # Fallback or error if necessary, for now, proceed but critic path will be skipped if critic_model is None.

    is_autocast_currently_enabled = torch.is_autocast_enabled() and device.type == 'cuda'

    # Actor forward pass
    with torch.amp.autocast(device.type, enabled=False) if is_autocast_currently_enabled else nullcontext():
        current_input_ids_actor = input_ids
        current_attention_mask_actor = attention_mask
        current_past_key_values_actor = past_key_values
        current_position_ids_actor = position_ids
        current_token_type_ids_actor = token_type_ids

        if is_autocast_currently_enabled:
            if current_attention_mask_actor is not None and current_attention_mask_actor.dtype != torch.float32:
                current_attention_mask_actor = current_attention_mask_actor.to(torch.float32)
            if current_past_key_values_actor is not None:
                new_pkv_actor = []
                for layer_pkv in current_past_key_values_actor:
                    new_layer_pkv = []
                    for tensor in layer_pkv:
                        if tensor.is_floating_point() and tensor.dtype != torch.float32:
                            new_layer_pkv.append(tensor.to(torch.float32))
                        else:
                            new_layer_pkv.append(tensor)
                    new_pkv_actor.append(tuple(new_layer_pkv))
                current_past_key_values_actor = tuple(new_pkv_actor)
            if current_position_ids_actor is not None and current_position_ids_actor.is_floating_point() and current_position_ids_actor.dtype != torch.float32:
                    current_position_ids_actor = current_position_ids_actor.to(torch.float32)
            if current_token_type_ids_actor is not None and current_token_type_ids_actor.is_floating_point() and current_token_type_ids_actor.dtype != torch.float32:
                    current_token_type_ids_actor = current_token_type_ids_actor.to(torch.float32)
        
        actor_outputs = actor_model(
            input_ids=current_input_ids_actor,
            attention_mask=current_attention_mask_actor,
            past_key_values=current_past_key_values_actor,
            position_ids=current_position_ids_actor,
            token_type_ids=current_token_type_ids_actor,
            use_cache=False, 
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    if actor_outputs.logits is not None:
        logger.debug(f"GAVU: actor_outputs.logits (direct from model) - Shape: {actor_outputs.logits.shape}, Dtype: {actor_outputs.logits.dtype}")
        if actor_outputs.logits.numel() > 0:
            logger.debug(f"  actor_outputs.logits - Min: {actor_outputs.logits.min().item():.4f}, Max: {actor_outputs.logits.max().item():.4f}, Mean: {actor_outputs.logits.mean():.4f}")
            logger.debug(f"  actor_outputs.logits - Has NaN: {torch.isnan(actor_outputs.logits).any().item()}, Has Inf: {torch.isinf(actor_outputs.logits).any().item()}")
    else:
        logger.warning("GAVU: actor_outputs.logits is None directly from model.")

    if actor_outputs.hidden_states is not None:
        last_actor_hidden_state = actor_outputs.hidden_states[-1]
        logger.debug(f"GAVU: Actor's last hidden state (input to lm_head) - Shape: {last_actor_hidden_state.shape}, Dtype: {last_actor_hidden_state.dtype}")
        if last_actor_hidden_state.numel() > 0:
            logger.debug(f"  Actor LHS - Min: {last_actor_hidden_state.min().item()}, Max: {last_actor_hidden_state.max().item()}, Mean: {last_actor_hidden_state.mean().item()}")
            logger.debug(f"  Actor LHS - Has NaN: {torch.isnan(last_actor_hidden_state).any().item()}, Has Inf: {torch.isinf(last_actor_hidden_state).any().item()}")
    else:
        logger.warning("GAVU: actor_outputs.hidden_states is None, cannot log input to lm_head.")

    if hasattr(actor_model, 'lm_head') and actor_model.lm_head is not None and hasattr(actor_model.lm_head, 'weight'):
        lm_head_weights = actor_model.lm_head.weight
        logger.debug(f"GAVU: Actor's lm_head weights - Shape: {lm_head_weights.shape}, Dtype: {lm_head_weights.dtype}")
        if lm_head_weights.numel() > 0:
            logger.debug(f"  LM Head Weights - Min: {lm_head_weights.min().item()}, Max: {lm_head_weights.max().item()}, Mean: {lm_head_weights.mean().item()}")
            logger.debug(f"  LM Head Weights - Has NaN: {torch.isnan(lm_head_weights).any().item()}, Has Inf: {torch.isinf(lm_head_weights).any().item()}")
    else:
        logger.warning("GAVU: Could not access actor's lm_head weights for logging.")

    action_logits = actor_outputs.logits

    if logit_clamping_value is not None and actor_outputs.logits is not None:
        raw_min, raw_max = actor_outputs.logits.min(), actor_outputs.logits.max()
        action_logits = torch.clamp(actor_outputs.logits, min=-logit_clamping_value, max=logit_clamping_value)
        if not torch.allclose(raw_min, action_logits.min()) or not torch.allclose(raw_max, action_logits.max()):
            logger.warning(f"GAVU: Clamped action_logits from [{raw_min:.2f}, {raw_max:.2f}] to [{action_logits.min():.2f}, {action_logits.max():.2f}]")
    else:
        action_logits = actor_outputs.logits

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
            ce_logits_input_for_loss = logits_for_actions.float() if logits_for_actions.dtype == torch.float16 else logits_for_actions
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

            _log_probs = F.cross_entropy(
                ce_logits_input_for_loss.reshape(-1, ce_logits_input_for_loss.size(-1)),
                ce_targets_input.reshape(-1),
                reduction='none'
            )
            log_probs_per_token = -_log_probs.reshape(logits_for_actions.shape[0], logits_for_actions.shape[1])
            
    values = None
    if use_separate_value_model and critic_model is not None:
        with torch.amp.autocast(device.type, enabled=False) if is_autocast_currently_enabled else nullcontext():
            critic_input_ids = input_ids
            critic_attention_mask = attention_mask
            if critic_attention_mask is not None and critic_attention_mask.dtype != torch.float32:
                logger.debug(f"GAVU: Casting critic_attention_mask from {critic_attention_mask.dtype} to torch.float32 for critic_model.")
                critic_attention_mask = critic_attention_mask.to(torch.float32)
            
            critic_past_key_values = past_key_values
            if critic_past_key_values is not None:
                new_pkv_critic = []
                for layer_pkv in critic_past_key_values:
                    new_layer_pkv_inner_critic = [] 
                    for tensor_pkv in layer_pkv:
                        if tensor_pkv.is_floating_point() and tensor_pkv.dtype != torch.float32:
                            new_layer_pkv_inner_critic.append(tensor_pkv.to(torch.float32))
                        else:
                            new_layer_pkv_inner_critic.append(tensor_pkv)
                    new_pkv_critic.append(tuple(new_layer_pkv_inner_critic))
                critic_past_key_values = tuple(new_pkv_critic)

            critic_position_ids = position_ids
            critic_token_type_ids = token_type_ids
            critic_inputs_embeds = inputs_embeds
            if critic_inputs_embeds is not None and critic_inputs_embeds.is_floating_point() and critic_inputs_embeds.dtype != torch.float32:
                    logger.debug(f"GAVU: Casting critic_inputs_embeds from {critic_inputs_embeds.dtype} to torch.float32 for critic_model.")
                    critic_inputs_embeds = critic_inputs_embeds.to(torch.float32)
            
            output_attentions_for_critic = output_attentions if output_attentions is not None else False

            logger.debug(f"GAVU: Dtype of critic_attention_mask passed to critic_model: {critic_attention_mask.dtype if critic_attention_mask is not None else 'None'}")
            critic_outputs = critic_model(
                input_ids=critic_input_ids,
                attention_mask=critic_attention_mask, 
                past_key_values=critic_past_key_values,
                position_ids=critic_position_ids,
                token_type_ids=critic_token_type_ids,
                inputs_embeds=critic_inputs_embeds,
                output_attentions=output_attentions_for_critic,
                return_dict=True
            )
        values = critic_outputs.pooler_output
    elif critic_model is None and use_separate_value_model:
        logger.warning("GAVU: Critic model is None, but use_separate_value_model is True. Cannot compute value estimate.")
        values = None
    elif not use_separate_value_model:
        # Single-model PPO path: derive a scalar value from actor hidden states.
        logger.debug("GAVU: Single model mode (use_separate_value_model=False). Deriving value from actor hidden states.")
        if actor_outputs.hidden_states is not None and actor_outputs.hidden_states[-1] is not None:
            hidden_states = actor_outputs.hidden_states[-1]
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
        else:
            logger.warning("GAVU: Actor output hidden states unavailable for single-model fallback; using zeros.")
            values = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=action_logits.dtype if action_logits is not None else torch.float32)


    return action_logits, log_probs_per_token, values 