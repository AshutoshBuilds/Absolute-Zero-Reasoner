import torch
import torch.nn as nn # Added for nn.Linear
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, AutoConfig, PretrainedConfig # Added PreTrainedModel, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPoolingAndCrossAttentions # To ensure compatible output type
import logging
from typing import Optional, Tuple, Type, List, Dict, Any # Added List, Dict, Any for generate_text_with_model typing
import os
import shutil # Added import for shutil
import torch.nn.functional as F
from pathlib import Path
from contextlib import nullcontext # Add this import

# Import from the new hf_value_model.py
from hf_value_model import ValueModel, _modify_layernorm_eps

# Import from the new hf_model_io_utils.py
from hf_model_io_utils import (
    save_models_and_tokenizer,
    load_models_and_tokenizer,
    _handle_pad_token_for_models 
)

# Import the new model setup utility
from hf_model_setup_utils import initialize_models_and_tokenizer
from hf_transformers_compat import (
    apply_azr_attention_env_once,
    dtype_kwargs_for_from_pretrained,
    explicit_attn_implementation_from_azr_env,
)

# Import the new action/value utility
from hf_action_value_utils import get_action_and_value_from_models

# Import the new generation utility
from hf_generation_utils import generate_text_with_model, generate_texts_batched_with_model

logger = logging.getLogger(__name__)

class HuggingFaceAdapter:
    def __init__(
        self,
        model_name: str,
        device: str = None,
        auth_token: str = None,
        use_separate_value_model: bool = True,
        hf_cache_dir: Optional[str] = None,
        logit_clamping_value: Optional[float] = 30.0,
        load_in_4bit: bool = False,
        torch_dtype_for_actor_critic = None,
    ):
        """
        Initializes the Hugging Face model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to load (e.g., 'codellama/CodeLlama-7b-hf').
            device (str, optional): The device to load the model on ('cuda', 'cpu'). Auto-detects if None.
            auth_token (str, optional): Hugging Face API token, if required for the model.
            use_separate_value_model (bool): If True, loads a separate ValueModel for criticism alongside the actor model.
            hf_cache_dir (Optional[str]): Path to Hugging Face cache directory.
            logit_clamping_value (Optional[float]): Value to clamp logits to, e.g., +/-30.0. If None, no clamping.
            load_in_4bit (bool): Whether to load the model in 4-bit quantization (requires bitsandbytes).
            torch_dtype_for_actor_critic: Torch dtype for actor/critic model loading (default is auto-selected for device).
        """
        self.model_name = model_name
        self.logit_clamping_value = logit_clamping_value
        self.auth_token = auth_token # Store auth_token for use in load_model tokenizer fallback
        self.hf_cache_dir = hf_cache_dir # Store hf_cache_dir for use in load_model tokenizer fallback
        self.torch_dtype_for_actor_critic = torch_dtype_for_actor_critic
        self.tokenizer = None
        self.actor_model = None
        self.critic_model = None
        self.model = None
        
        if device is None:
            resolved_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device_str = device
        self.device = torch.device(resolved_device_str)
        
        self.use_separate_value_model = use_separate_value_model
        apply_azr_attention_env_once()

        logger.info(f"Initializing HuggingFaceAdapter with model: {model_name} on device: {self.device}")
        logger.info(f"Using separate ValueModel: {self.use_separate_value_model}")

        resolved_model_dtype = torch_dtype_for_actor_critic
        # Match hf_model_setup_utils / CLI --model-dtype auto: prefer bf16 on CUDA when supported
        # (pure fp16 on many consumer GPUs gives non-finite logits / unstable GenU sampling).
        if resolved_model_dtype is None:
            if resolved_device_str == "cuda":
                resolved_model_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                resolved_model_dtype = torch.float32
        elif isinstance(resolved_model_dtype, str):
            resolved_model_dtype = resolved_model_dtype.lower()
            if resolved_model_dtype in {"fp16", "float16", "half"}:
                resolved_model_dtype = torch.float16
            elif resolved_model_dtype in {"bf16", "bfloat16"}:
                resolved_model_dtype = torch.bfloat16
            elif resolved_model_dtype in {"fp32", "float32", "float"}:
                resolved_model_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported dtype '{torch_dtype_for_actor_critic}'.")

        try:
            is_checkpoint_dir = (
                os.path.isdir(self.model_name)
                and os.path.isfile(os.path.join(self.model_name, "actor_model", "config.json"))
                and os.path.isfile(os.path.join(self.model_name, "critic_model", "config.json"))
            )
            if is_checkpoint_dir:
                logger.info("Detected checkpoint directory format, loading via load_model().")
                try:
                    self.load_model(self.model_name)
                except Exception as checkpoint_error:
                    if not self.use_separate_value_model:
                        raise
                    logger.warning(
                        "Checkpoint-based load_model failed, attempting actor-only fallback. Error: %s",
                        checkpoint_error,
                    )
                    # Keep behavior stable for evaluation if ValueModel restoration hits compatibility issues.
                    self.use_separate_value_model = False
                    fallback_actor_path = os.path.join(self.model_name, "actor_model")
                    fallback_tokenizer_path = os.path.join(self.model_name, "tokenizer")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer_path,
                        token=self.auth_token,
                        trust_remote_code=True,
                        cache_dir=self.hf_cache_dir,
                    )
                    fallback_actor_kw = {
                        "token": self.auth_token,
                        "trust_remote_code": True,
                        "cache_dir": self.hf_cache_dir,
                    }
                    attn_fb = explicit_attn_implementation_from_azr_env()
                    if attn_fb is not None:
                        fallback_actor_kw["attn_implementation"] = attn_fb
                    fallback_actor_kw.update(
                        dtype_kwargs_for_from_pretrained(AutoModelForCausalLM, resolved_model_dtype)
                    )
                    self.actor_model = AutoModelForCausalLM.from_pretrained(
                        fallback_actor_path,
                        **fallback_actor_kw,
                    )
                    _modify_layernorm_eps(self.actor_model)
                    self.actor_model.to(self.device)
                    self.model = self.actor_model
                    _handle_pad_token_for_models(
                        tokenizer=self.tokenizer,
                        actor_model=self.actor_model,
                        critic_model=None,
                        main_model=None,
                        use_separate_value_model=False,
                    )
            else:
                # Call the standard utility function to load models and tokenizer
                self.tokenizer, self.actor_model, self.critic_model, self.model = initialize_models_and_tokenizer(
                    model_name=self.model_name,
                    device_str=resolved_device_str, # Pass the string representation of the device
                    auth_token=self.auth_token,
                    use_separate_value_model=self.use_separate_value_model,
                    hf_cache_dir=self.hf_cache_dir,
                    torch_dtype_for_actor_critic=resolved_model_dtype,
                    load_in_4bit=load_in_4bit # Pass quantization flag
                )

            # The _handle_pad_token_for_models is now called within initialize_models_and_tokenizer
            # So, no need to call it here explicitly.

        except Exception as e:
            logger.error(f"Error initializing HuggingFaceAdapter: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize HuggingFaceAdapter components.") from e

    # Add a method to get model outputs (logits and value) for PPO updates
    def get_action_and_value(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, token_type_ids=None, actions=None, inputs_embeds=None, output_attentions=None):
        """
        Performs a forward pass to get action logits/log_probs and value estimate.
        This method now calls the utility function from hf_action_value_utils.py.
        """
        if not self.use_separate_value_model:
            if self.model is None:
                raise RuntimeError("Adapter in single-model mode but shared model is None.")
            logger.debug("HFA.get_action_and_value called in single-model mode. Using shared actor/critic path in utility.")
            return get_action_and_value_from_models(
                actor_model=self.model, # Main model acts as actor
                critic_model=None,    # No separate critic
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                actions=actions,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                device=self.device,
                logit_clamping_value=self.logit_clamping_value,
                use_separate_value_model=False # use single-model path
            )
        if self.use_separate_value_model and (self.actor_model is None or self.critic_model is None):
            # If critic_model is None but use_separate_value_model is True, it implies an issue or a specific setup
            # where value might not be needed or comes from elsewhere. The utility handles critic_model being None.
            if self.actor_model is None:
                 raise RuntimeError("Adapter in separate_value_model mode but actor_model is None.")
            # If only critic_model is None, the utility function will handle it by returning None for values.
            # No specific error raise here for critic_model being None if actor_model is present.
        
        return get_action_and_value_from_models(
            actor_model=self.actor_model,
            critic_model=self.critic_model, # Can be None, utility handles it
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            actions=actions,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            device=self.device,
            logit_clamping_value=self.logit_clamping_value,
            use_separate_value_model=self.use_separate_value_model
        )

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 512, 
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_return_sequences: int = 8,
        max_prompt_length: Optional[int] = None,
        **kwargs: Any  # Use Dict[str, Any] for kwargs typing
    ) -> List[str]: # Use List[str] for return typing
        """
        Generates text using the Hugging Face model by calling the utility function.
        """
        model_for_generation = None
        critic_for_mode_management = None # For the utility function

        if self.use_separate_value_model:
            if self.actor_model is None:
                raise RuntimeError("Cannot generate: Adapter in separate_value_model mode but actor_model is None.")
            model_for_generation = self.actor_model
            critic_for_mode_management = self.critic_model # Pass critic for its mode management
        else:
            if self.model is None:
                raise RuntimeError("Cannot generate: Adapter in single-model mode but self.model is None.")
            model_for_generation = self.model
            # In single model mode, critic_for_mode_management remains None as there's no separate critic.

        return generate_text_with_model(
            model_to_use=model_for_generation,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            max_prompt_length=max_prompt_length,
            critic_model_for_mode_management=critic_for_mode_management,
            use_separate_value_model=self.use_separate_value_model,
            **kwargs
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        max_prompt_length: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Batched single-sequence generation (``num_return_sequences`` must be 1). See ``hf_generation_utils``."""
        if self.use_separate_value_model:
            if self.actor_model is None:
                raise RuntimeError("Cannot generate: Adapter in separate_value_model mode but actor_model is None.")
            model_for_generation = self.actor_model
            critic_for_mode_management = self.critic_model
        else:
            if self.model is None:
                raise RuntimeError("Cannot generate: Adapter in single-model mode but self.model is None.")
            model_for_generation = self.model
            critic_for_mode_management = None

        return generate_texts_batched_with_model(
            model_to_use=model_for_generation,
            tokenizer=self.tokenizer,
            device=self.device,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            max_prompt_length=max_prompt_length,
            critic_model_for_mode_management=critic_for_mode_management,
            use_separate_value_model=self.use_separate_value_model,
            **kwargs,
        )

    def save_model(self, save_directory: str):
        save_models_and_tokenizer(
            save_directory=save_directory,
            tokenizer=self.tokenizer,
            actor_model=self.actor_model,
            critic_model=self.critic_model,
            main_model=self.model,
            use_separate_value_model=self.use_separate_value_model
        )

    def load_model(self, load_directory: str):
        loaded_tokenizer, loaded_actor_model, loaded_critic_model, loaded_main_model = load_models_and_tokenizer(
            load_directory=load_directory,
            base_model_name_for_tokenizer=self.model_name, # Pass original model name for fallback
            device=self.device,
            use_separate_value_model=self.use_separate_value_model,
            hf_auth_token=self.auth_token, # Pass stored auth_token
            hf_cache_dir=self.hf_cache_dir    # Pass stored hf_cache_dir
        )
        self.tokenizer = loaded_tokenizer
        self.actor_model = loaded_actor_model
        self.critic_model = loaded_critic_model
        self.model = loaded_main_model
        # _handle_pad_token_for_models is called at the end of load_models_and_tokenizer utility

# The if __name__ == '__main__' block has been moved to test_azr_hf_adapter.py 