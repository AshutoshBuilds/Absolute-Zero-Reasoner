import torch
import logging
import os
import shutil
from azr_hf_adapter import HuggingFaceAdapter # Assuming azr_hf_adapter.py is in the same directory or PYTHONPATH

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Configure basic logging
    logger = logging.getLogger(__name__)  # Create logger instance
    
    # Example Usage (requires a model like CodeLlama or a smaller one for quick testing)
    # model_to_test = "codellama/CodeLlama-7b-Instruct-hf" # Needs auth and GPU
    # model_to_test = "Salesforce/codegen-350M-mono" # Smaller, no auth usually
    model_to_test = "gpt2" # For very basic CPU test
    hf_cache_dir_for_test = "./temp_hf_cache" # Define a cache dir for the test

    # Create the cache directory if it doesn't exist, as the adapter might try to write to it
    # and to avoid errors if it's cleaned up by a previous failed run.
    if not os.path.exists(hf_cache_dir_for_test):
        os.makedirs(hf_cache_dir_for_test)

    try:
        logger.info(f"Testing HuggingFaceAdapter with model: {model_to_test}")
        # Test with ValueModel enabled
        logger.info("--- Testing with ValueModel (use_separate_value_model=True) --- ")
        adapter_ac = HuggingFaceAdapter(model_name=model_to_test, use_separate_value_model=True, hf_cache_dir=hf_cache_dir_for_test, logit_clamping_value=30.0)
        
        test_prompt_ac = "def hello_world_ac():\n    # AC test\n"
        logger.info(f"Test prompt AC: {test_prompt_ac}")
        
        # Test generation (should use actor_model)
        generated_code_ac = adapter_ac.generate(test_prompt_ac, max_new_tokens=50)
        if generated_code_ac:
            logger.info(f"Generated code AC (first sequence):\n{generated_code_ac[0]}")
        else:
            logger.error("AC Generation failed or returned empty.")

        # Test get_action_and_value
        logger.info("Testing get_action_and_value with separate models...")
        inputs = adapter_ac.tokenizer(test_prompt_ac, return_tensors="pt", padding=True, truncation=True).to(adapter_ac.device)
        
        dummy_action_ids = inputs.input_ids 
        if dummy_action_ids.shape[1] == 0: 
            dummy_action_ids = torch.tensor([[adapter_ac.tokenizer.eos_token_id]], device=adapter_ac.device)


        if adapter_ac.actor_model: adapter_ac.actor_model.train()
        if adapter_ac.critic_model: adapter_ac.critic_model.train()
        
        action_logits, log_probs, value_estimate = adapter_ac.get_action_and_value(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            actions=dummy_action_ids, 
            inputs_embeds=None
        )
        
        logger.info(f"Returned from get_action_and_value:")
        if action_logits is not None:
            logger.info(f"  Action Logits shape: {action_logits.shape}")
            if action_logits.numel() > 0:
                logger.info(f"  Action Logits contains inf: {torch.isinf(action_logits).any().item()}")
                logger.info(f"  Action Logits contains -inf: {torch.isneginf(action_logits).any().item()}")
                logger.info(f"  Action Logits contains nan: {torch.isnan(action_logits).any().item()}")
                logger.info(f"  Action Logits min: {action_logits.min().item()}, max: {action_logits.max().item()}, sum: {action_logits.sum().item()}")
                if action_logits.shape[2] > 20:
                    logger.info(f"  Action Logits sample [0,0,10:15]: {action_logits[0,0,10:15]}")
                else:
                    logger.info(f"  Action Logits sample [0,0,:5]: {action_logits[0,0,:5]}")     
            else:
                logger.info(f"  Action Logits: None or empty")
        else:
            logger.info(f"  Action Logits: None")
            
        if log_probs is not None:
            logger.info(f"  Log Probs shape: {log_probs.shape}")
            logger.info(f"  Log Probs sample (sum): {log_probs.sum().item() if log_probs.numel() > 0 else 'N/A'}")
        else:
            logger.info(f"  Log Probs: None")

        if value_estimate is not None:
            logger.info(f"  Value Estimate shape: {value_estimate.shape}")
            logger.info(f"  Value Estimate sample: {value_estimate[0].item() if value_estimate.numel() > 0 else 'N/A'}")
        else:
            logger.info(f"  Value Estimate: None")

        if adapter_ac.actor_model: adapter_ac.actor_model.eval()
        if adapter_ac.critic_model: adapter_ac.critic_model.eval()

        # Test with standard model (ValueModel disabled)
        logger.info("--- Testing with Standard Model (use_separate_value_model=False) --- ")
        adapter_std = HuggingFaceAdapter(model_name=model_to_test, use_separate_value_model=False, hf_cache_dir=hf_cache_dir_for_test, logit_clamping_value=30.0)
        test_prompt_std = "def hello_world_std():\n    # Standard test\n"
        logger.info(f"Test prompt Standard: {test_prompt_std}")
        generated_code_std = adapter_std.generate(test_prompt_std, max_new_tokens=50)
        if generated_code_std:
            logger.info(f"Generated code Standard (first sequence):\n{generated_code_std[0]}")
        else:
            logger.error("Standard Generation failed or returned empty.")
        
        # Trying to call get_action_and_value on standard model should raise error if not implemented for it
        logger.info("Testing get_action_and_value on standard model...")
        inputs_std = adapter_std.tokenizer(test_prompt_std, return_tensors="pt").to(adapter_std.device)
        try:
            # The adapter's get_action_and_value has a path for single-model mode now.
            # It should return logits, log_probs, and value_estimate=None
            action_logits_std, log_probs_std, value_estimate_std = adapter_std.get_action_and_value(
                inputs_std.input_ids, 
                actions=inputs_std.input_ids, 
                inputs_embeds=None
            )
            logger.info(f"Standard model get_action_and_value results:")
            if action_logits_std is not None:
                logger.info(f"  Std Action Logits shape: {action_logits_std.shape}")
            else:
                logger.info(f"  Std Action Logits: None")
            if log_probs_std is not None:
                logger.info(f"  Std Log Probs shape: {log_probs_std.shape}")
            else:
                logger.info(f"  Std Log Probs: None")
            if value_estimate_std is not None:
                logger.info(f"  Std Value Estimate: {value_estimate_std}") # Should be None
            else:
                logger.info(f"  Std Value Estimate: None (as expected for single model without value head)")

        except RuntimeError as e:
            logger.error(f"Error during get_action_and_value for standard model: {e}")


        # Test saving and loading ValueModel
        logger.info("--- Testing Save/Load with separate Actor/Critic models ---")
        save_dir = "./temp_ac_model_test"
        adapter_ac.save_model(save_dir)
        
        adapter_ac_loaded = HuggingFaceAdapter(model_name=model_to_test, use_separate_value_model=True, hf_cache_dir=hf_cache_dir_for_test, logit_clamping_value=30.0)
        adapter_ac_loaded.load_model(save_dir)
        logger.info(f"Adapter loaded from {save_dir}")

        inputs_for_load_test = adapter_ac_loaded.tokenizer(test_prompt_ac, return_tensors="pt", padding=True, truncation=True).to(adapter_ac_loaded.device)
        dummy_actions_loaded = inputs_for_load_test.input_ids
        if dummy_actions_loaded.shape[1] == 0: # Ensure not empty
            dummy_actions_loaded = torch.tensor([[adapter_ac_loaded.tokenizer.eos_token_id]], device=adapter_ac_loaded.device)


        _, _, value_loaded = adapter_ac_loaded.get_action_and_value(
            inputs_for_load_test.input_ids, 
            attention_mask=inputs_for_load_test.attention_mask,
            actions=dummy_actions_loaded,
            inputs_embeds=None
        )
        if value_loaded is not None:
            logger.info(f"Loaded Separate Models - Value shape: {value_loaded.shape}, Value sample: {value_loaded.item() if value_loaded.numel() > 0 else 'N/A'}")
        else:
            logger.info(f"Loaded Separate Models - Value Estimate: None")

        generated_text_ac_loaded = adapter_ac_loaded.generate(test_prompt_ac, max_new_tokens=50)
        if generated_text_ac_loaded:
            logger.info(f"Loaded Separate Models - Generated Text: {generated_text_ac_loaded[0]}")
        else:
            logger.error("Loaded Separate Models - Generation failed.")
            
        # Clean up
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        # Optional: Clean up the test cache directory after successful run
        # Be cautious with this if other tests might use it or if you want to inspect its contents
        # if os.path.exists(hf_cache_dir_for_test):
        #     logger.info(f"Cleaning up test cache directory: {hf_cache_dir_for_test}")
        #     shutil.rmtree(hf_cache_dir_for_test)

    except Exception as e:
        logger.error(f"Error in HuggingFaceAdapter example usage: {e}", exc_info=True)
    finally:
        # Ensure cleanup of the main test cache directory if it was created by this script,
        # especially if the test fails midway.
        if os.path.exists(hf_cache_dir_for_test) and hf_cache_dir_for_test == "./temp_hf_cache": # Extra safety check
            logger.info(f"Ensuring cleanup of test cache directory: {hf_cache_dir_for_test}")
            shutil.rmtree(hf_cache_dir_for_test)
        if os.path.exists("./temp_ac_model_test"): # also ensure this is cleaned up
             shutil.rmtree("./temp_ac_model_test")
             logger.info("Cleaned up ./temp_ac_model_test directory.") 