import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast # For ValueModel's forward typing
import logging
from typing import Optional, Type

logger = logging.getLogger(__name__)

def _modify_layernorm_eps(module: nn.Module, new_eps: float = 1e-4, layer_norm_type: Type[nn.LayerNorm] = nn.LayerNorm):
    """
    Recursively iterates through all modules and modifies the epsilon
    of nn.LayerNorm layers.
    """
    for child_module in module.children():
        if isinstance(child_module, layer_norm_type):
            original_eps = child_module.eps
            child_module.eps = new_eps
            logger.info(f"Modified epsilon of {child_module.__class__.__name__} from {original_eps} to {new_eps}")
        else:
            _modify_layernorm_eps(child_module, new_eps, layer_norm_type)


class ValueModel(PreTrainedModel):
    """A model that uses a base transformer and adds a value head on top."""
    def __init__(
        self,
        config: PretrainedConfig,
        base_model_name_or_path: str,
        hf_auth_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        torch_dtype_for_core_model = torch.float32,
        quantization_config = None,
    ):
        super().__init__(config)
        
        # Ensure pooling_strategy is set on the config
        if not hasattr(self.config, 'pooling_strategy'):
            logger.info("ValueModel's config does not have 'pooling_strategy'. Setting default to 'last_token'.")
            self.config.pooling_strategy = "last_token"

        # Some model configs omit pad_token_id in older checkpoints; default to eos token or 0 to avoid
        # initialization-time attribute errors during critic/value model construction.
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "eos_token_id", 0)
        if isinstance(pad_token_id, (tuple, list)):
            pad_token_id = pad_token_id[0] if len(pad_token_id) > 0 else 0
        if pad_token_id is None:
            pad_token_id = 0
        self.config.pad_token_id = int(pad_token_id)
        
        actual_base_model_config = AutoConfig.from_pretrained(
            base_model_name_or_path, 
            pad_token_id=self.config.pad_token_id, 
            trust_remote_code=getattr(self.config, 'trust_remote_code', True),
        )
        
        # Use an explicit device map to avoid meta-device initialization paths.
        # Load core critic model on CPU first for robust checkpoint reloads on Windows,
        # then move to the target device after wrapper initialization.
        preferred_device_map = {"": "cpu"}

        self.core_model_for_value = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            config=actual_base_model_config,
            token=hf_auth_token,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
            torch_dtype=torch_dtype_for_core_model,
            # Disable low-memory initialization with meta tensors for local checkpoint reloads.
            low_cpu_mem_usage=False,
            quantization_config=quantization_config,
            device_map=preferred_device_map,
            # Ignore checkpoint / config shape differences (e.g., lm_head in actor checkpoints) during reload.
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"Attempting to modify LayerNorm eps for ValueModel's core_model_for_value ({base_model_name_or_path})...")
        _modify_layernorm_eps(self.core_model_for_value)
        logger.info(f"Finished LayerNorm eps modification for ValueModel's core_model_for_value.")

        # Assign generation_config from core model to wrapper to suppress warnings
        if hasattr(self.core_model_for_value, 'generation_config'):
            self.generation_config = self.core_model_for_value.generation_config

        hidden_size = self.core_model_for_value.config.hidden_size
        if hidden_size is None:
            hidden_size = actual_base_model_config.hidden_size
            if hidden_size is None:
                logger.error("CRITICAL: hidden_size is None in ValueModel. self.config: %s, actual_base_model_config: %s", self.config, actual_base_model_config)
                raise ValueError("Could not determine hidden_size from base model config for ValueModel.")

        self.value_head = nn.Linear(hidden_size, 1)
        if hasattr(self.core_model_for_value, 'dtype'):
            self.value_head.to(dtype=self.core_model_for_value.dtype)
        if hasattr(self.core_model_for_value, 'device'):
             self.value_head.to(device=self.core_model_for_value.device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        is_autocast = torch.is_autocast_enabled()
        core_input_ids = input_ids
        core_attention_mask = attention_mask
        core_past_key_values = past_key_values
        core_position_ids = position_ids
        core_token_type_ids = token_type_ids
        core_inputs_embeds = inputs_embeds

        if not is_autocast:
            expected_dtype = self.core_model_for_value.dtype
            if core_attention_mask is not None and core_attention_mask.is_floating_point() and core_attention_mask.dtype != expected_dtype:
                core_attention_mask = core_attention_mask.to(expected_dtype)
            elif core_attention_mask is not None and core_attention_mask.dtype == torch.int64 and expected_dtype.is_floating_point:
                logger.warning(f"ValueModel.forward: attention_mask was int64. Casting to {expected_dtype}. This might indicate an upstream issue.")
                core_attention_mask = core_attention_mask.to(expected_dtype)

            if core_past_key_values is not None:
                new_pkv = []
                for layer_pkv in core_past_key_values:
                    new_layer_pkv_inner = []
                    for t_pkv in layer_pkv:
                        if t_pkv.is_floating_point() and t_pkv.dtype != expected_dtype:
                            new_layer_pkv_inner.append(t_pkv.to(expected_dtype))
                        else:
                            new_layer_pkv_inner.append(t_pkv)
                    new_pkv.append(tuple(new_layer_pkv_inner))
                core_past_key_values = tuple(new_pkv)
            
            if core_position_ids is not None and core_position_ids.is_floating_point() and core_position_ids.dtype != expected_dtype:
                core_position_ids = core_position_ids.to(expected_dtype)
            if core_token_type_ids is not None and core_token_type_ids.is_floating_point() and core_token_type_ids.dtype != expected_dtype:
                core_token_type_ids = core_token_type_ids.to(expected_dtype)
            if core_inputs_embeds is not None and core_inputs_embeds.is_floating_point() and core_inputs_embeds.dtype != expected_dtype:
                core_inputs_embeds = core_inputs_embeds.to(expected_dtype)

        transformer_outputs = self.core_model_for_value(
            input_ids=core_input_ids,
            attention_mask=core_attention_mask,
            past_key_values=core_past_key_values,
            position_ids=core_position_ids,
            token_type_ids=core_token_type_ids,
            inputs_embeds=core_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_states_before_ln_f = transformer_outputs.hidden_states[-1]
        
        pooled_output = None
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            if self.config.pad_token_id is None:
                sequence_lengths = torch.tensor([seq_len-1] * batch_size, device=last_hidden_states_before_ln_f.device)
            else:
                if attention_mask is not None:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                else:
                    sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1
            
            sequence_lengths = sequence_lengths.clamp(min=0).long()
            
            if self.config.pooling_strategy == "last_token":
                 pooled_output = last_hidden_states_before_ln_f[torch.arange(batch_size, device=last_hidden_states_before_ln_f.device), sequence_lengths]
            elif self.config.pooling_strategy == "mean_pooling":
                if attention_mask is None:
                    attention_mask_for_pool = torch.ones_like(input_ids, device=last_hidden_states_before_ln_f.device)
                else:
                    attention_mask_for_pool = attention_mask
                
                expanded_attention_mask = attention_mask_for_pool.unsqueeze(-1).expand(last_hidden_states_before_ln_f.size()).float()
                sum_hidden_states = torch.sum(last_hidden_states_before_ln_f * expanded_attention_mask, dim=1)
                sum_mask = expanded_attention_mask.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
                pooled_output = sum_hidden_states / sum_mask
            else: # Default to last token if strategy is unknown or not specified
                logger.warning(f"Unknown or unspecified pooling_strategy: {self.config.pooling_strategy}. Defaulting to 'last_token'.")
                pooled_output = last_hidden_states_before_ln_f[torch.arange(batch_size, device=last_hidden_states_before_ln_f.device), sequence_lengths]

        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            # If only embeds are provided, assume all tokens are valid (no padding)
            # or rely on attention_mask if provided.
            if attention_mask is not None:
                 sequence_lengths = attention_mask.sum(dim=1) -1
            else:
                 sequence_lengths = torch.tensor([seq_len-1] * batch_size, device=last_hidden_states_before_ln_f.device)

            sequence_lengths = sequence_lengths.clamp(min=0).long()

            if self.config.pooling_strategy == "last_token":
                pooled_output = last_hidden_states_before_ln_f[torch.arange(batch_size, device=last_hidden_states_before_ln_f.device), sequence_lengths]
            elif self.config.pooling_strategy == "mean_pooling":
                if attention_mask is None:
                    attention_mask_for_pool = torch.ones(batch_size, seq_len, device=last_hidden_states_before_ln_f.device)
                else:
                    attention_mask_for_pool = attention_mask
                
                expanded_attention_mask = attention_mask_for_pool.unsqueeze(-1).expand(last_hidden_states_before_ln_f.size()).float()
                sum_hidden_states = torch.sum(last_hidden_states_before_ln_f * expanded_attention_mask, dim=1)
                sum_mask = expanded_attention_mask.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled_output = sum_hidden_states / sum_mask
            else:
                logger.warning(f"Unknown or unspecified pooling_strategy: {self.config.pooling_strategy} with inputs_embeds. Defaulting to 'last_token'.")
                pooled_output = last_hidden_states_before_ln_f[torch.arange(batch_size, device=last_hidden_states_before_ln_f.device), sequence_lengths]
        else:
            # Fallback if neither input_ids nor inputs_embeds is provided.
            # This case is unlikely for standard transformer usage for value prediction.
            logger.warning("ValueModel.forward called without input_ids or inputs_embeds. Using global average pooling over sequence as a fallback.")
            pooled_output = last_hidden_states_before_ln_f.mean(dim=1) # Global average pool


        if pooled_output is None:
            logger.error("ValueModel.forward: pooled_output is None before value_head. This should not happen.")
            # As a desperate fallback, use the mean of the last hidden state. This is suboptimal.
            pooled_output = last_hidden_states_before_ln_f.mean(dim=1)


        value_head_input_dtype = pooled_output.dtype
        value_head_dtype = self.value_head.weight.dtype
        if value_head_input_dtype != value_head_dtype:
            logger.debug(f"ValueModel.forward: Casting pooled_output from {value_head_input_dtype} to {value_head_dtype} for value_head.")
            pooled_output = pooled_output.to(value_head_dtype)
            
        value = self.value_head(pooled_output).squeeze(-1) # Squeeze the last dim to get (batch_size)

        if not return_dict:
            # Mimic CausalLMOutputWithPast structure if not return_dict
            # For ValueModel, logits are not standard, loss is not computed here.
            # Return value and hidden_states (from core model).
            return (None, transformer_outputs.past_key_values, transformer_outputs.hidden_states, transformer_outputs.attentions, value)

        output_obj = CausalLMOutputWithPast(
            loss=None, # No loss computed here
            logits=None, # ValueModel doesn't produce sequence logits in the traditional sense
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, # from core model
            attentions=transformer_outputs.attentions,  # from core model
            # custom field for value
            # custom_fields={"value": value} 
        )

        # Dynamically attach the value as pooler_output to match adapter expectations
        if return_dict:
            # The adapter expects critic_outputs.pooler_output
            output_obj.pooler_output = value 
            return output_obj
        else:
            # Non-dict output already includes value as the last element in the tuple
            return (None, transformer_outputs.past_key_values, transformer_outputs.hidden_states, transformer_outputs.attentions, value)

    # Add a dummy config_class for from_pretrained to work if ValueModel is saved/loaded directly
    # This might need to be more specific if ValueModel had its own configuration parameters beyond PretrainedConfig
    config_class = PretrainedConfig 

    def get_input_embeddings(self):
        return self.core_model_for_value.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.core_model_for_value.set_input_embeddings(value) 