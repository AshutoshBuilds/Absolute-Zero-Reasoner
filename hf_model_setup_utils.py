import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, Tuple

# Import from hf_value_model.py and hf_model_io_utils.py as they contain necessary components
from hf_value_model import ValueModel, _modify_layernorm_eps
from hf_model_io_utils import _handle_pad_token_for_models # For pad token handling

logger = logging.getLogger(__name__)

def initialize_models_and_tokenizer(
    model_name: str, 
    device_str: str, # Changed from device: torch.device to device_str: str for easier passing
    auth_token: Optional[str] = None, 
    use_separate_value_model: bool = True, 
    hf_cache_dir: Optional[str] = None,
    torch_dtype_for_actor_critic = None, # Added for consistency
    load_in_4bit: bool = False # NEW: 4-bit quantization support
) -> Tuple[AutoTokenizer, Optional[AutoModelForCausalLM], Optional[ValueModel], Optional[AutoModelForCausalLM]]:
    """
    Initializes the Hugging Face model(s) and tokenizer.

    Args:
        model_name (str): The name of the Hugging Face model to load.
        device_str (str): The device string to load the model on ('cuda', 'cpu').
        auth_token (str, optional): Hugging Face API token.
        use_separate_value_model (bool): If True, loads a separate ValueModel.
        hf_cache_dir (Optional[str]): Path to Hugging Face cache directory.
        torch_dtype_for_actor_critic (torch.dtype): Dtype for actor and critic models.
        load_in_4bit (bool): Whether to load models in 4-bit quantization.

    Returns:
        Tuple containing:
            - tokenizer: The loaded tokenizer.
            - actor_model: The loaded actor model (or None if not use_separate_value_model).
            - critic_model: The loaded critic model (ValueModel, or None).
            - main_model: The loaded main model (if not use_separate_value_model, else None).
    """
    device = torch.device(device_str)
    logger.info(f"Initializing models and tokenizer for: {model_name} on device: {device}")
    logger.info(f"Using separate ValueModel: {use_separate_value_model}")
    logger.info(f"4-bit Quantization: {load_in_4bit}")

    config_kwargs = {"trust_remote_code": True}
    if hf_cache_dir:
        config_kwargs["cache_dir"] = hf_cache_dir

    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        if device_str == 'cpu':
            logger.warning("4-bit quantization requested but device is CPU. Ignoring quantization.")
        else:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("BitsAndBytesConfig created for 4-bit loading.")
            except Exception as e:
                logger.error(f"Failed to create BitsAndBytesConfig: {e}. Proceeding without quantization.")

    tokenizer_kwargs = {
        "token": auth_token,
        "trust_remote_code": True,
        "cache_dir": hf_cache_dir,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        logger.info("Loaded fast tokenizer successfully.")
    except Exception as fast_tokenizer_error:
        logger.warning(
            "Fast tokenizer load failed for '%s'. Falling back to slow tokenizer. "
            "This can happen when tokenizer extras (protobuf / sentencepiece) are unavailable: %s",
            model_name,
            fast_tokenizer_error,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, **tokenizer_kwargs)
        logger.info("Loaded slow tokenizer fallback successfully.")
    
    actor_model = None
    critic_model = None
    main_model = None

    # Helper to get model kwargs
    def get_model_kwargs():
        dtype_for_model = torch_dtype_for_actor_critic
        if dtype_for_model is None:
            dtype_for_model = torch.float16 if device_str == 'cuda' else torch.float32

        kwargs = {
            "token": auth_token,
            "torch_dtype": dtype_for_model,
            "trust_remote_code": True,
            "cache_dir": hf_cache_dir,
        }
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
            # When using quantization, we don't explicitly move to device with .to() usually, 
            # but transformers handles it.
        return kwargs

    if use_separate_value_model:
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **get_model_kwargs()
        )
        if not quantization_config:
            actor_model.to(device)
        
        logger.info(f"Actor model ({model_name}) loaded successfully.")
        # Prepare for LoRA if quantized? (Not implementing full LoRA here yet, just loading)
        
        _modify_layernorm_eps(actor_model) # Uses the one from hf_value_model
        logger.info(f"Finished LayerNorm eps modification for actor_model.")

        base_model_specific_config = AutoConfig.from_pretrained(
            model_name, 
            token=auth_token, 
            trust_remote_code=True, 
            cache_dir=hf_cache_dir,
            pad_token_id=tokenizer.pad_token_id # Ensure config has pad_token_id
        )
        logger.info(f"Loaded base model config for critic: {type(base_model_specific_config)}")

        critic_model = ValueModel(
            config=base_model_specific_config, 
            base_model_name_or_path=model_name, 
            hf_auth_token=auth_token,
            hf_cache_dir=hf_cache_dir,
            torch_dtype_for_core_model=torch_dtype_for_actor_critic,
            quantization_config=quantization_config
        )
        critic_model.to(device)
        logger.info(f"Critic model (ValueModel wrapping {model_name}) loaded on {device}")
    
    else: # Single model for both actor and critic
        main_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **get_model_kwargs()
        )
        _modify_layernorm_eps(main_model) # Uses the one from hf_value_model
        logger.info(f"Finished LayerNorm eps modification for single model.")
        if not quantization_config:
            main_model.to(device)
        logger.info(f"Standard AutoModelForCausalLM {model_name} loaded.")
            
    logger.info(f"Model(s) for {model_name} loaded successfully.")
    
    # Call the utility function for handling pad token
    _handle_pad_token_for_models(
        tokenizer=tokenizer, 
        actor_model=actor_model, 
        critic_model=critic_model, 
        main_model=main_model, 
        use_separate_value_model=use_separate_value_model
    )
    
    return tokenizer, actor_model, critic_model, main_model