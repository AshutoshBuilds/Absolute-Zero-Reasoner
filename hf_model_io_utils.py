import torch
import os
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, Any

# Assuming ValueModel and _modify_layernorm_eps might be needed for loading,
# though ideally load_model should handle different model types gracefully
# or specific loader functions would be created for specific model wrapper types.
# For now, let's import them if they are in a known shared location.
from hf_value_model import ValueModel, _modify_layernorm_eps
from hf_transformers_compat import apply_azr_attention_env_once, explicit_attn_implementation_from_azr_env

logger = logging.getLogger(__name__)

def _handle_pad_token_for_models(tokenizer, actor_model, critic_model, main_model, use_separate_value_model):
    """
    Handles setting the pad token for the tokenizer and ensuring model configs are updated.
    This is a standalone version of the original _handle_pad_token method.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(f"Tokenizer has no pad_token. Setting pad_token to eos_token: {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning(f"Tokenizer has no pad_token and no eos_token. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize embeddings for all relevant models if a new token was added
            if use_separate_value_model:
                if actor_model: actor_model.resize_token_embeddings(len(tokenizer))
                if critic_model and hasattr(critic_model, 'core_model_for_value'):
                    critic_model.core_model_for_value.resize_token_embeddings(len(tokenizer))
                elif critic_model: # If critic is not ValueModel but some other PreTrainedModel
                    try:
                        critic_model.resize_token_embeddings(len(tokenizer))
                    except AttributeError:
                        logger.warning("Critic model does not have core_model_for_value and resize_token_embeddings failed directly.")
            else:
                if main_model: main_model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Tokenizer final pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    # Update model configs with the final pad_token_id
    models_to_update = []
    if use_separate_value_model:
        if actor_model: models_to_update.append(("actor_model", actor_model))
        if critic_model:
            models_to_update.append(("critic_model_wrapper", critic_model)) # The wrapper's config
            if hasattr(critic_model, 'core_model_for_value'): # The core model inside ValueModel
                models_to_update.append(("critic_model_core", critic_model.core_model_for_value))
    else:
        if main_model: models_to_update.append(("main_model", main_model))

    for model_name, model_obj in models_to_update:
        if model_obj and hasattr(model_obj, 'config'):
            model_obj.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Updated {model_name} config pad_token_id: {model_obj.config.pad_token_id}")
            if hasattr(model_obj, 'generation_config') and model_obj.generation_config is not None:
                model_obj.generation_config.pad_token_id = tokenizer.pad_token_id
                # Also ensure eos_token_id is consistent if possible
                if tokenizer.eos_token_id is not None:
                     model_obj.generation_config.eos_token_id = tokenizer.eos_token_id
            elif hasattr(model_obj, 'generation_config') and model_obj.generation_config is None:
                 logger.warning(f"{model_name}.generation_config is None. Cannot set pad_token_id/eos_token_id. May need to initialize it.")


def save_models_and_tokenizer(
    save_directory: str,
    tokenizer,
    actor_model,
    critic_model,
    main_model, # For single model case
    use_separate_value_model: bool
):
    os.makedirs(save_directory, exist_ok=True)
    if tokenizer:
        tokenizer.save_pretrained(save_directory)
        logger.info(f"Tokenizer saved to {save_directory}")
    else:
        logger.warning("No tokenizer provided to save_models_and_tokenizer.")

    if use_separate_value_model:
        if actor_model:
            actor_save_path = os.path.join(save_directory, "actor_model") # Changed to actor_model for consistency
            os.makedirs(actor_save_path, exist_ok=True)
            actor_model.save_pretrained(actor_save_path)
            logger.info(f"Actor model saved to {actor_save_path}")
        else:
            logger.warning("use_separate_value_model is True, but no actor_model provided to save.")

        if critic_model:
            critic_save_path = os.path.join(save_directory, "critic_model") # Changed to critic_model
            os.makedirs(critic_save_path, exist_ok=True)
            critic_model.save_pretrained(critic_save_path) # ValueModel is a PreTrainedModel
            logger.info(f"Critic model saved to {critic_save_path}")
        else:
            logger.warning("use_separate_value_model is True, but no critic_model provided to save.")
    else:
        if main_model:
            # For a single model, save it to the root of save_directory or a 'model' subfolder
            # To match common Hugging Face Hub structure, often it's saved directly.
            # Let's keep it simple and save to the root of save_directory.
            # The original code also saved a single model directly to save_directory.
            main_model_save_path = os.path.join(save_directory, "model") # Explicitly save to 'model' subfolder
            os.makedirs(main_model_save_path, exist_ok=True)
            main_model.save_pretrained(main_model_save_path)
            logger.info(f"Main model saved to {main_model_save_path}")
        else:
            logger.warning("use_separate_value_model is False, but no main_model provided to save.")

def _extract_quantization_config(model_path: Path) -> Optional[Any]:
    """
    Reads model configuration from the provided path and returns quantization_config if present.
    """
    if not Path(model_path).exists():
        return None
    config = AutoConfig.from_pretrained(str(model_path))
    raw_quantization_config = getattr(config, "quantization_config", None)
    if raw_quantization_config is None:
        return None

    if isinstance(raw_quantization_config, BitsAndBytesConfig):
        return raw_quantization_config

    if isinstance(raw_quantization_config, dict):
        logger.info("Converting serialized quantization_config dict to BitsAndBytesConfig for consistent reload.")
        try:
            return BitsAndBytesConfig.from_dict(raw_quantization_config)
        except Exception as convert_error:
            logger.warning(
                "Failed to convert quantization_config dict to BitsAndBytesConfig; "
                "falling back to None (non-quantized reload). Error: %s",
                convert_error,
            )
            return None

    if raw_quantization_config is not None:
        return raw_quantization_config

    return raw_quantization_config

def load_models_and_tokenizer(
    load_directory: str,
    base_model_name_for_tokenizer: str, # Fallback if tokenizer not in load_directory
    device,
    use_separate_value_model: bool,
    hf_auth_token: Optional[str] = None, # For loading tokenizer fallback
    hf_cache_dir: Optional[str] = None   # For loading tokenizer fallback & ValueModel
):
    load_directory_path = Path(load_directory)
    apply_azr_attention_env_once()
    logger.info(f"Loading models and tokenizer from directory: {load_directory_path}")

    # Tokenizer
    # Try to load tokenizer from a 'tokenizer' subdirectory first, then from the root.
    tokenizer = None
    tokenizer_sub_path = load_directory_path / "tokenizer" # Consistent with how _save_checkpoint in callbacks saves it
    if tokenizer_sub_path.exists() and (tokenizer_sub_path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_sub_path))
        logger.info(f"Tokenizer loaded from {tokenizer_sub_path}")
    elif (load_directory_path / "tokenizer_config.json").exists(): # Fallback to root if adapter saved it there.
        tokenizer = AutoTokenizer.from_pretrained(str(load_directory_path))
        logger.info(f"Tokenizer loaded from {load_directory_path} (root)")
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_sub_path} or {load_directory_path}. Attempting to load from base model name {base_model_name_for_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_for_tokenizer,
            token=hf_auth_token,
            trust_remote_code=True, # Assuming trust_remote_code from adapter's init
            cache_dir=hf_cache_dir
        )

    actor_model = None
    critic_model = None
    main_model = None

    if use_separate_value_model:
        actor_load_path = load_directory_path / "actor_model"
        critic_load_path = load_directory_path / "critic_model"
        actor_quantization_config = _extract_quantization_config(actor_load_path)
        actor_load_kwargs = {"trust_remote_code": True, "cache_dir": hf_cache_dir}
        attn_impl = explicit_attn_implementation_from_azr_env()
        if attn_impl is not None:
            actor_load_kwargs["attn_implementation"] = attn_impl
        # Local AZR checkpoints embed quantization in config.json; passing quantization_config
        # again makes Transformers warn about ambiguous duplicate quantization metadata.
        if actor_quantization_config is not None:
            logger.info(
                "Actor checkpoint includes quantization metadata in config.json; "
                "loading without an extra quantization_config kwarg."
            )

        if actor_load_path.exists():
            actor_model = AutoModelForCausalLM.from_pretrained(
                str(actor_load_path),
                **actor_load_kwargs
            )
            actor_model.to(device)
            _modify_layernorm_eps(actor_model) # Re-apply LayerNorm modification
            logger.info(f"Actor model loaded from {actor_load_path} on {device}")
        else:
            logger.error(f"Actor model directory not found at {actor_load_path}")
            # Not raising FileNotFoundError here, will return None for the model
            # raise FileNotFoundError(f"Actor model directory not found at {actor_load_path}")

        if critic_load_path.exists():
            critic_wrapper_config = AutoConfig.from_pretrained(str(critic_load_path))
            # ValueModel needs base_model_name_or_path for its core model.
            # We prefer base_model_name_for_tokenizer (the original model) to avoid loading from the checkpoint dir recursively
            # which causes "weights not initialized" warnings and is inefficient.
            base_for_critic_core = base_model_name_for_tokenizer
            if not base_for_critic_core:
                 base_for_critic_core = getattr(critic_wrapper_config, '_name_or_path', None)
            
            logger.info(f"Loading ValueModel (critic) from {critic_load_path}. Using '{base_for_critic_core}' as base_model_name_or_path for its core.")

            critic_model = ValueModel.from_pretrained(
                str(critic_load_path),
                config=critic_wrapper_config,
                # Prefer the actor checkpoint folder for ValueModel's base config when available
                # so we avoid using a generic checkpoint root that may not contain model_type.
                base_model_name_or_path=str(actor_load_path if actor_load_path.exists() else base_for_critic_core), 
                hf_auth_token=hf_auth_token, # Pass token if needed for core model
                hf_cache_dir=hf_cache_dir,   # Pass cache_dir for core model
                torch_dtype_for_core_model=actor_model.dtype if actor_model is not None else torch.float32,
                quantization_config=actor_quantization_config
            )
            # _modify_layernorm_eps is called inside ValueModel.__init__ for its core_model_for_value
            critic_model.to(device)
            logger.info(f"Critic model (ValueModel) loaded from {critic_load_path} on {device}")
        else:
            logger.error(f"Critic model directory not found at {critic_load_path}")
            # raise FileNotFoundError(f"Critic model directory not found at {critic_load_path}")
    else:
        # Single model case
        # Check for 'model' subdirectory first, then root of load_directory
        model_load_actual_path_str = None
        if (load_directory_path / "model").exists() and (load_directory_path / "model" / "config.json").exists():
             model_load_actual_path_str = str(load_directory_path / "model")
        elif (load_directory_path / "config.json").exists():
             model_load_actual_path_str = str(load_directory_path)
        
        if model_load_actual_path_str:
            main_quant = _extract_quantization_config(str(model_load_actual_path_str))
            main_kwargs: dict = {"trust_remote_code": True, "cache_dir": hf_cache_dir}
            attn_impl_main = explicit_attn_implementation_from_azr_env()
            if attn_impl_main is not None:
                main_kwargs["attn_implementation"] = attn_impl_main
            if main_quant is not None:
                logger.info(
                    "Main model checkpoint includes quantization in config.json; "
                    "loading without an extra quantization_config kwarg."
                )
            main_model = AutoModelForCausalLM.from_pretrained(
                model_load_actual_path_str,
                **main_kwargs,
            )
            _modify_layernorm_eps(main_model) # Re-apply LayerNorm modification
            main_model.to(device)
            logger.info(f"Main model loaded from {model_load_actual_path_str} on {device}")
        else:
            logger.error(f"Single model directory not found at {load_directory_path} or {load_directory_path / 'model'}")
            # raise FileNotFoundError(f"Single model files not found in {load_directory_path}")

    # Update pad token IDs after loading models and tokenizer
    _handle_pad_token_for_models(tokenizer, actor_model, critic_model, main_model, use_separate_value_model)
    logger.info("Model(s) and tokenizer loading process finished.")

    return tokenizer, actor_model, critic_model, main_model 