import torch
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

def generate_text_with_model(
    model_to_use: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompt: str, 
    max_new_tokens: int = 512, 
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    max_prompt_length: Optional[int] = None,
    # For separate actor/critic, critic might also need its mode managed
    critic_model_for_mode_management: Optional[PreTrainedModel] = None, 
    use_separate_value_model: bool = False, # To know if critic_model_for_mode_management is relevant
    **kwargs: Dict[str, Any]
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

    if model_to_use.training:
        original_modes['model_to_use'] = True
        model_to_use.eval()
    
    if use_separate_value_model and critic_model_for_mode_management is not None and critic_model_for_mode_management.training:
        original_modes['critic'] = True
        critic_model_for_mode_management.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False) # Truncation handled below
    
    if max_prompt_length is not None:
        for key in inputs:
            inputs[key] = inputs[key][:, :max_prompt_length]
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Log initial logits for stability check (copied from adapter's generate)
    try:
        with torch.no_grad():
            initial_outputs = model_to_use(input_ids=input_ids, attention_mask=attention_mask)
            initial_logits = initial_outputs.logits
            if initial_logits is not None and initial_logits.numel() > 0:
                logger.debug(f"GenU: Initial logits - Shape: {initial_logits.shape}, dtype: {initial_logits.dtype}")
                logger.debug(f"GenU: Initial logits - Has NaN: {torch.isnan(initial_logits).any().item()}, Has Inf: {torch.isinf(initial_logits).any().item()}, Has NegInf: {torch.isneginf(initial_logits).any().item()}")
                logger.debug(f"GenU: Initial logits - Min: {initial_logits.min().item()}, Max: {initial_logits.max().item()}, Mean: {initial_logits.mean().item()}")
                initial_probs_check = F.softmax(initial_logits[:, -1, :], dim=-1)
                logger.debug(f"GenU: Softmax check on last token logits - Has NaN: {torch.isnan(initial_probs_check).any().item()}, Sum: {initial_probs_check.sum(dim=-1)}")
            else:
                logger.debug("GenU: Initial logits are None or empty.")
    except Exception as e_log_logits:
        logger.error(f"GenU: Error during initial logit check: {e_log_logits}")

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
    }
    gen_kwargs.update(kwargs)

    try:
        with torch.no_grad():
            outputs = model_to_use.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
    except Exception as e:
        logger.error(f"GenU: Error during model generation: {e}")
        # Restore original training modes before raising
        if 'model_to_use' in original_modes: model_to_use.train()
        if 'critic' in original_modes and critic_model_for_mode_management: critic_model_for_mode_management.train()
        raise

    # Restore original training modes
    if 'model_to_use' in original_modes: model_to_use.train()
    if 'critic' in original_modes and critic_model_for_mode_management: critic_model_for_mode_management.train()

    prompt_len = input_ids.shape[1]
    generated_texts = [tokenizer.decode(output[prompt_len:], skip_special_tokens=True) for output in outputs]
    
    return generated_texts 