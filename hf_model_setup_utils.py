import torch
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, Tuple
import torch.nn as nn

# Import from hf_value_model.py and hf_model_io_utils.py as they contain necessary components
from hf_value_model import ValueModel, _modify_layernorm_eps
from hf_model_io_utils import _handle_pad_token_for_models # For pad token handling
from hf_transformers_compat import (
    apply_azr_attention_env_once,
    dtype_kwargs_for_from_pretrained,
    explicit_attn_implementation_from_azr_env,
)

logger = logging.getLogger(__name__)

if not hasattr(nn.Module, "set_submodule"):
    def _module_set_submodule(self: nn.Module, target: str, module: nn.Module) -> None:
        """
        Compatibility shim for environments where nn.Module.set_submodule is unavailable.
        Some third-party quantization helpers (for example, bitsandbytes) call this
        method during replacement, so defining it keeps quantized model loading functional.
        """
        if "." in target:
            parent_path, child_name = target.rsplit(".", 1)
            parent = self.get_submodule(parent_path)
            setattr(parent, child_name, module)
        else:
            setattr(self, target, module)

    nn.Module.set_submodule = _module_set_submodule
    logger.info("Applied nn.Module.set_submodule compatibility shim for quantization workflows.")


def _layernorm_eps_from_env() -> float:
    raw = os.environ.get("AZR_LAYERNORM_EPS", "1e-4").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid AZR_LAYERNORM_EPS=%r; using 1e-4.", raw)
        return 1e-4


def initialize_models_and_tokenizer(
    model_name: str, 
    device_str: str, # Changed from device: torch.device to device_str: str for easier passing
    auth_token: Optional[str] = None, 
    use_separate_value_model: bool = True, 
    hf_cache_dir: Optional[str] = None,
    torch_dtype_for_actor_critic = None, # Added for consistency
    load_in_4bit: bool = False,
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
    apply_azr_attention_env_once()
    logger.info(f"Initializing models and tokenizer for: {model_name} on device: {device}")
    logger.info(f"Using separate ValueModel: {use_separate_value_model}")
    logger.info(f"4-bit Quantization: {load_in_4bit}")

    # Local checkpoints saved by this repo use `tokenizer/` and `model/` (or `actor_model/` /
    # `critic_model/`) subfolders. Hub-style trees have tokenizer + weights at the root.
    # Mirror the resolution rules in `hf_model_io_utils.load_models_and_tokenizer` so
    # `AutoTokenizer.from_pretrained` does not point at a directory with no tokenizer files.
    load_root = Path(model_name)
    tokenizer_pretrained_id = model_name
    causal_pretrained_id = model_name
    actor_pretrained_id = model_name
    if load_root.is_dir():
        tokenizer_sub = load_root / "tokenizer"
        if tokenizer_sub.is_dir() and (tokenizer_sub / "tokenizer_config.json").exists():
            tokenizer_pretrained_id = str(tokenizer_sub.resolve())
            logger.info("Resolved tokenizer path to nested checkpoint folder: %s", tokenizer_pretrained_id)
        model_sub = load_root / "model"
        if model_sub.is_dir() and (model_sub / "config.json").exists():
            causal_pretrained_id = str(model_sub.resolve())
            logger.info("Resolved causal LM weights path to nested checkpoint folder: %s", causal_pretrained_id)
        actor_sub = load_root / "actor_model"
        if actor_sub.is_dir() and (actor_sub / "config.json").exists():
            actor_pretrained_id = str(actor_sub.resolve())
            logger.info("Resolved actor weights path to nested checkpoint folder: %s", actor_pretrained_id)

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
                compute_dtype = (
                    torch.bfloat16
                    if device_str == "cuda" and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_id, **tokenizer_kwargs)
        logger.info("Loaded fast tokenizer successfully.")
    except Exception as fast_tokenizer_error:
        logger.warning(
            "Fast tokenizer load failed for '%s'. Falling back to slow tokenizer. "
            "This can happen when tokenizer extras (protobuf / sentencepiece) are unavailable: %s",
            tokenizer_pretrained_id,
            fast_tokenizer_error,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_id, use_fast=False, **tokenizer_kwargs)
        logger.info("Loaded slow tokenizer fallback successfully.")
    
    actor_model = None
    critic_model = None
    main_model = None

    # Helper to get model kwargs
    def get_model_kwargs():
        dtype_for_model = torch_dtype_for_actor_critic
        if dtype_for_model is None:
            if device_str == "cuda":
                dtype_for_model = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                dtype_for_model = torch.float32

        kwargs = {
            "token": auth_token,
            "trust_remote_code": True,
            "cache_dir": hf_cache_dir,
        }
        kwargs.update(dtype_kwargs_for_from_pretrained(AutoModelForCausalLM, dtype_for_model))
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
            # When using quantization, we don't explicitly move to device with .to() usually, 
            # but transformers handles it.
        return kwargs

    def _load_causal_model(
        model_name: str,
        model_kwargs: dict,
        model_kind: str,
    ):
        model_kwargs_sdpa = dict(model_kwargs)
        attn_override = explicit_attn_implementation_from_azr_env()
        model_kwargs_sdpa["attn_implementation"] = (
            attn_override if attn_override is not None else "sdpa"
        )
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs_sdpa)
        except (TypeError, ValueError) as e:
            message = str(e).lower()
            if "attn_implementation" in message or "unexpected keyword argument" in message:
                fallback_model_kwargs = dict(model_kwargs_sdpa)
                fallback_model_kwargs.pop("attn_implementation", None)
                logger.warning(
                    "Model loader does not accept attn_implementation for %s (%s). "
                    "Retrying without attn_implementation.",
                    model_name,
                    model_kind,
                )
                return AutoModelForCausalLM.from_pretrained(model_name, **fallback_model_kwargs)
            raise

    if use_separate_value_model:
        actor_model = _load_causal_model(
            model_name=actor_pretrained_id,
            model_kwargs=get_model_kwargs(),
            model_kind="actor"
        )
        if not quantization_config:
            actor_model.to(device)
        
        logger.info(f"Actor model ({model_name}) loaded successfully.")
        # Prepare for LoRA if quantized? (Not implementing full LoRA here yet, just loading)
        
        _modify_layernorm_eps(actor_model, new_eps=_layernorm_eps_from_env()) # Uses the one from hf_value_model
        logger.info(f"Finished LayerNorm eps modification for actor_model.")

        base_model_specific_config = AutoConfig.from_pretrained(
            actor_pretrained_id,
            token=auth_token, 
            trust_remote_code=True, 
            cache_dir=hf_cache_dir,
            pad_token_id=tokenizer.pad_token_id # Ensure config has pad_token_id
        )
        logger.info(f"Loaded base model config for critic: {type(base_model_specific_config)}")

        critic_model = ValueModel(
            config=base_model_specific_config, 
            base_model_name_or_path=actor_pretrained_id,
            hf_auth_token=auth_token,
            hf_cache_dir=hf_cache_dir,
            torch_dtype_for_core_model=torch_dtype_for_actor_critic,
            quantization_config=quantization_config
        )
        if not quantization_config:
            critic_model.to(device)
        if hasattr(critic_model, "gradient_checkpointing_enable"):
            try:
                critic_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for critic model.")
            except Exception as error:
                logger.warning(f"Critic gradient checkpointing is not supported: {error}. Continuing without it.")
        logger.info(f"Critic model (ValueModel wrapping {model_name}) loaded on {device}")
        target_dtype = torch_dtype_for_actor_critic
        if target_dtype is None:
            if device_str == "cuda":
                target_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                target_dtype = torch.float32

        if quantization_config:
            logger.info(
                "Quantized model loaded; skipping explicit dtype cast because bitsandbytes models "
                "do not support post-load dtype conversion."
            )
        else:
            if actor_model is not None:
                actor_model = actor_model.to(dtype=target_dtype)
            if critic_model is not None:
                critic_model = critic_model.to(dtype=target_dtype)
        logger.info(f"Aligned actor/critic model params to dtype={target_dtype}.")
    
    else: # Single model for both actor and critic
        main_model = _load_causal_model(
            model_name=causal_pretrained_id,
            model_kwargs=get_model_kwargs(),
            model_kind="single_model"
        )
        _modify_layernorm_eps(main_model, new_eps=_layernorm_eps_from_env()) # Uses the one from hf_value_model
        logger.info(f"Finished LayerNorm eps modification for single model.")
        if not quantization_config:
            main_model.to(device)
        logger.info(f"Standard AutoModelForCausalLM {causal_pretrained_id} loaded.")
            
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
