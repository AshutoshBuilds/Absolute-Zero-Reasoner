import os
import json
import time
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Assuming ExperienceBuffer has save/load methods as used in the original trainer
# from experience_buffer import ExperienceBuffer # This might be needed if we type hint more strictly

logger = logging.getLogger(__name__) # Use a logger specific to this module

def update_problem_type_weights(trainer_instance):
    config = trainer_instance.config
    problem_type_success_rates = trainer_instance.problem_type_success_rates
    problem_types = config["problem_types"]
    problem_type_weights = config["problem_type_weights"]

    total_weight = sum(problem_type_weights.values())
    # Ensure there are success rates to average and problem types to iterate
    if not problem_type_success_rates or not problem_types:
        logger.warning("Cannot update problem type weights: success rates or problem types are empty.")
        return

    avg_success_rate = sum(problem_type_success_rates.values()) / len(problem_type_success_rates)
    
    for pt in problem_types:
        if pt not in problem_type_weights or pt not in problem_type_success_rates:
            logger.warning(f"Problem type {pt} not found in weights or success rates. Skipping.")
            continue
        current_weight = problem_type_weights[pt]
        success_rate = problem_type_success_rates[pt]
        
        # Avoid division by zero if current_weight is 0
        adjustment_factor = total_weight
        if current_weight > 1e-6: # Use a small epsilon to avoid division by zero
            adjustment_factor = total_weight / current_weight

        if success_rate < avg_success_rate - 0.1: 
            adjustment = 0.05 * adjustment_factor
            problem_type_weights[pt] = min(current_weight + adjustment, total_weight * 0.7 if total_weight > 0 else 0.7) 
        elif success_rate > avg_success_rate + 0.1: 
            adjustment = 0.05 * adjustment_factor
            problem_type_weights[pt] = max(current_weight - adjustment, total_weight * 0.05 if total_weight > 0 else 0.05) 

    current_total_weight = sum(problem_type_weights.values())
    if current_total_weight > 1e-6: # Use a small epsilon to avoid division by zero
        config["problem_type_weights"] = {
            pt: w / current_total_weight for pt, w in problem_type_weights.items()
        }
    else: 
        logger.warning("Total problem type weight is near zero. Resetting to uniform distribution.")
        config["problem_type_weights"] = {
            pt: 1.0 / len(problem_types) if problem_types else 1.0 for pt in problem_types
        }
    logger.debug(f"Updated problem type weights: {config['problem_type_weights']}")

def update_curriculum_difficulty(trainer_instance, epoch: int):
    config = trainer_instance.config
    experience_buffer = trainer_instance.experience_buffer
    current_difficulty = trainer_instance.current_difficulty
    reward_manager = trainer_instance.reward_manager # Added to update reward manager
    metrics = trainer_instance.metrics # Added to update metrics

    if not config["use_curriculum"]:
        return

    recent_experiences = experience_buffer.get_recent_experiences(limit=config["batch_size"] * 2)
    solver_successes = sum(1 for exp in recent_experiences 
                           if hasattr(exp, 'reward_components') and exp.reward_components is not None and exp.reward_components.get("r_correctness", -1.0) > 0.8)
    total_solver_attempts = len([exp for exp in recent_experiences 
                                 if hasattr(exp, 'reward_components') and exp.reward_components is not None and "r_correctness" in exp.reward_components])
    
    overall_solver_success_rate = (solver_successes / total_solver_attempts) if total_solver_attempts > 0 else 0.5 

    new_difficulty = current_difficulty
    if overall_solver_success_rate > 0.75 and new_difficulty < config["max_task_difficulty"]:
        new_difficulty += 1
        logger.info(f"Increasing difficulty to {new_difficulty} based on high solver success ({overall_solver_success_rate:.2f})")
    elif overall_solver_success_rate < 0.4 and new_difficulty > config["min_task_difficulty"]:
        new_difficulty -=1
        logger.info(f"Decreasing difficulty to {new_difficulty} based on low solver success ({overall_solver_success_rate:.2f})")
    
    if epoch > 0 and epoch % 10 == 0 and new_difficulty < config["max_task_difficulty"] and 0.6 <= overall_solver_success_rate <= 0.75:
        new_difficulty +=1
        logger.info(f"Gradually increasing difficulty to {new_difficulty} at epoch {epoch}")

    trainer_instance.current_difficulty = max(config["min_task_difficulty"], min(new_difficulty, config["max_task_difficulty"]))
    metrics["difficulty_level"] = trainer_instance.current_difficulty # Update metrics in trainer instance
    reward_manager.update_difficulty(trainer_instance.current_difficulty) # Update reward manager

def save_checkpoint(trainer_instance,
                     checkpoint_name: Optional[str] = None,
                     epoch: Optional[int] = None,
                     optimizer=None,
                     scaler=None,
                     metrics=None,
                     experience_buffer=None,
                     ppo_buffer=None):
    """
    Save a training checkpoint. Backwards-compatible with calls that pass extra args.
    If checkpoint_name is provided, it overrides the default directory name to integrate
    with external pruning logic.
    """
    config = trainer_instance.config
    adapter = trainer_instance.adapter
    experience_buffer = experience_buffer or getattr(trainer_instance, 'experience_buffer', None)
    optimizer = optimizer or getattr(trainer_instance, 'optimizer', None)
    metrics = metrics or getattr(trainer_instance, 'metrics', {})
    current_difficulty = getattr(trainer_instance, 'current_difficulty', None)
    problem_type_success_rates = getattr(trainer_instance, 'problem_type_success_rates', {})

    # Determine checkpoint directory
    base_dir = Path(config["checkpoint_dir"]) if "checkpoint_dir" in config else Path("hf_checkpoints")
    if checkpoint_name:
        checkpoint_path = base_dir / checkpoint_name
    else:
        # Fallback: use original naming if epoch is provided, else generic
        ep = (epoch + 1) if isinstance(epoch, int) else 0
        checkpoint_path = base_dir / f"hf_checkpoint_epoch_{ep}"

    os.makedirs(checkpoint_path, exist_ok=True)
    logger.info(f"Saving checkpoint to {checkpoint_path}")

    # Save models/tokenizer
    try:
        if getattr(adapter, 'use_separate_value_model', False):
            if hasattr(adapter, 'actor_model') and adapter.actor_model is not None:
                try:
                    adapter.actor_model.save_pretrained(str(checkpoint_path / "actor_model"))
                    logger.info(f"Actor model saved to {checkpoint_path / 'actor_model'}")
                except Exception as e:
                    logger.error(f"Could not save actor model: {e}")
            else:
                logger.warning("Adapter indicates separate models, but actor_model not found for saving.")

            if hasattr(adapter, 'critic_model') and adapter.critic_model is not None:
                try:
                    adapter.critic_model.save_pretrained(str(checkpoint_path / "critic_model"))
                    logger.info(f"Critic model saved to {checkpoint_path / 'critic_model'}")
                except Exception as e:
                    logger.error(f"Could not save critic model: {e}")
            else:
                logger.warning("Adapter indicates separate models, but critic_model not found for saving.")
            
            if hasattr(adapter, 'tokenizer') and adapter.tokenizer is not None:
                try:
                    adapter.tokenizer.save_pretrained(str(checkpoint_path / "tokenizer"))
                    logger.info(f"Tokenizer saved to {checkpoint_path / 'tokenizer'}")
                except Exception as e:
                    logger.error(f"Could not save tokenizer: {e}")
        elif hasattr(adapter, 'model') and adapter.model is not None and hasattr(adapter.model, 'save_pretrained'):
            try:
                adapter.model.save_pretrained(str(checkpoint_path / "model"))
                if hasattr(adapter, 'tokenizer') and adapter.tokenizer is not None:
                    adapter.tokenizer.save_pretrained(str(checkpoint_path / "tokenizer"))
                logger.info(f"Single model and tokenizer saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Could not save single model/tokenizer: {e}")
        else:
            logger.warning("Adapter model(s) or tokenizer not found or does not support save_pretrained. Models not saved.")
    except Exception as e:
        logger.error(f"Unexpected error while saving models/tokenizer: {e}")

    # Save buffers/state (if available)
    try:
        if experience_buffer is not None and hasattr(experience_buffer, 'save_to_disk'):
            experience_buffer.save_to_disk(checkpoint_path / "experience_buffer.pkl")
    except Exception as e:
        logger.error(f"Error saving experience buffer: {e}")

    try:
        trainer_state = {
            "config": config,
            "metrics": metrics,
            "current_difficulty": current_difficulty,
            "problem_type_weights": config.get("problem_type_weights", {}),
            "problem_type_success_rates": problem_type_success_rates,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=4, default=lambda o: f'<not serializable: {type(o)}>')
        logger.info(f"Checkpoint saved successfully to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving trainer state: {e}")

def load_checkpoint(trainer_instance, checkpoint_path: str) -> int:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_path_obj = Path(checkpoint_path)
    adapter = trainer_instance.adapter
    experience_buffer = trainer_instance.experience_buffer
    # config will be updated from trainer_state, but keep a local ref for initial use
    # config = trainer_instance.config 

    # Model loading is complex and often involves re-instantiating or calling specific adapter methods.
    # The original trainer called adapter.load_model(). We assume this is still the preferred way.
    # For simplicity, we assume the adapter is already initialized and we are loading weights into it.
    # This utility function will focus on loading trainer state and buffer.
    # Model and tokenizer loading should be handled by the adapter's own load_model or by re-init.

    # Call adapter's load_model method if it exists, assuming it handles its own loading logic
    if hasattr(adapter, 'load_model') and callable(getattr(adapter, 'load_model')):
        try:
            adapter.load_model(str(checkpoint_path_obj)) # Pass the main checkpoint directory
            logger.info(f"Adapter's load_model method called with path: {checkpoint_path_obj}")
        except Exception as e:
            logger.error(f"Error calling adapter.load_model: {e}. Model/tokenizer state might not be fully restored.")
    else:
        logger.warning("Adapter does not have a load_model method. Model and tokenizer state may need manual loading or re-initialization.")

    try:
        experience_buffer.load_from_disk(checkpoint_path_obj / "experience_buffer.pkl")
        logger.info("Experience buffer loaded.")
    except FileNotFoundError:
        logger.warning("Experience buffer file not found in checkpoint.")
    except Exception as e_buffer:
        logger.error(f"Error loading experience buffer: {e_buffer}")
    
    try:
        with open(checkpoint_path_obj / "trainer_state.json", "r") as f:
            trainer_state = json.load(f)
        
        # Update trainer_instance attributes directly
        # Be careful with config updates, as some might be sensitive (e.g., model names)
        # For now, only updating specific curriculum and state variables from the loaded config.
        loaded_config = trainer_state.get("config", {})
        trainer_instance.config["problem_type_weights"] = loaded_config.get("problem_type_weights", trainer_instance.config["problem_type_weights"])
        # Other config elements could be selectively updated if needed.
        
        trainer_instance.metrics = trainer_state.get("metrics", trainer_instance.metrics)
        trainer_instance.current_difficulty = trainer_state.get("current_difficulty", trainer_instance.current_difficulty)
        trainer_instance.problem_type_success_rates = trainer_state.get("problem_type_success_rates", trainer_instance.problem_type_success_rates)
        start_epoch = trainer_state.get("epoch", -1) + 1
        
        if hasattr(trainer_instance, 'optimizer') and 'optimizer_state_dict' in trainer_state:
            try:
                trainer_instance.optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
                logger.info("Optimizer state loaded.")
            except Exception as e_optim:
                logger.error(f"Error loading optimizer state: {e_optim}. Optimizer might be reinitialized.")
        
        if hasattr(trainer_instance, 'reward_manager'):
            trainer_instance.reward_manager.update_difficulty(trainer_instance.current_difficulty)
        
        logger.info(f"Trainer state loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    except FileNotFoundError:
        logger.warning("Trainer state file not found in checkpoint.")
        return 0
    except Exception as e:
        logger.error(f"Error loading trainer state: {e}")
        return 0 