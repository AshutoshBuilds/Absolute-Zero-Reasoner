import logging
import random
import json
from typing import List, Dict, Any, Optional
from collections import deque
import torch
from colorama import Fore, Style

# Import AZR checks
from azr_common_utils import contains_banned_imports

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages seed tasks and experience buffers for AZR.
    Mimics the role of the DatasetManager Actor in the original AZR repo.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.problem_types = config.get("problem_types", ["deduction", "abduction", "induction"])
        
        # Buffers for different types of data
        self.seed_buffer: Dict[str, List[Dict]] = {pt: [] for pt in self.problem_types}
        self.experience_buffer: Dict[str, deque] = {pt: deque(maxlen=1000) for pt in self.problem_types}
        self.error_buffer: deque = deque(maxlen=500) # Stores tasks that caused errors/failures
        
        self.min_seed_size = config.get("min_seed_size", 5)
        self.proposer_num_return_sequences = max(1, int(config.get("proposer_num_return_sequences", 8)))
        self.k_reference = max(0, int(config.get("k_reference", 6)))
        self.is_seeded = False

    def generate_seeds(self, adapter, num_seeds: int = 5):
        """
        Uses the provided model adapter to generate initial seed tasks.
        This is crucial for 'Zero Data' start.
        """
        logger.info(f"{Fore.CYAN} Seeding DatasetManager with {num_seeds} tasks per type using {adapter.model_name}...{Style.RESET_ALL}")
        
        for p_type in self.problem_types:
            logger.info(f"  Generatng seeds for {p_type}...")
            
            # Use the prompt utils from the trainer context (imported locally to avoid circular deps if possible)
            # or reimplement a simple seeder prompt here.
            from hf_prompt_utils import create_proposer_prompt
            
            # We need a 'fake' trainer instance or context to pass difficulty
            class MockTrainer:
                current_difficulty = 1
            
            prompt_text = create_proposer_prompt(
                MockTrainer(),
                p_type,
                seed_tasks=self.seed_buffer.get(p_type, []),
                k_reference=self.k_reference,
            )
            
            # Generate
            # We do this in a loop or batch. For simplicity, loop.
            valid_seeds = 0
            attempts = 0
            max_attempts = num_seeds * 3
            
            while valid_seeds < num_seeds and attempts < max_attempts:
                attempts += 1
                try:
                    # Simple generation call
                    try:
                        generated_samples = adapter.generate(
                            prompt=prompt_text,
                            max_new_tokens=256,
                            temperature=0.2,
                            top_p=0.95,
                            do_sample=True,
                            num_return_sequences=self.proposer_num_return_sequences
                        )
                    except Exception:
                        generated_samples = adapter.generate(
                            prompt=prompt_text,
                            max_new_tokens=256,
                            temperature=0.2,
                            top_p=0.95,
                            do_sample=True,
                            num_return_sequences=1,
                        )
                    if not generated_samples:
                        continue
                    generated = generated_samples[0]
                    
                    # Parse (using simple JSON extraction from utils)
                    from hf_parsing_utils import find_json_objects
                    json_objects = find_json_objects(generated)
                    
                    if json_objects:
                        try:
                            task_data = json.loads(json_objects[0])
                            # Handle list of tasks if present
                            if isinstance(task_data, list):
                                task_data = task_data[0] if task_data else {}
                            
                            if isinstance(task_data, dict):
                                task = task_data
                                # Basic validation
                                if all(k in task for k in ["code", "input", "output"]):
                                    # Safety check
                                    if not contains_banned_imports(task["code"])[0]:
                                        task["problem_type"] = p_type
                                        task["origin"] = "seed"
                                        self.seed_buffer[p_type].append(task)
                                        valid_seeds += 1
                                        print(f"     Seed {valid_seeds}/{num_seeds} generated.")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON from seed generation: {json_objects[0][:100]}...")
                except Exception as e:
                    logger.warning(f"Seed generation failed: {e}")
                    
        self.is_seeded = True
        logger.info(f"{Fore.GREEN}DatasetManager seeding complete.{Style.RESET_ALL}")

    def add_experience(self, task: Dict[str, Any]):
        """Adds a successfully solved task to the experience buffer."""
        p_type = task.get("problem_type", "deduction") # Default fallback
        if p_type in self.experience_buffer:
            self.experience_buffer[p_type].append(task)

    def get_seed_task(self, problem_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a seed task for few-shot prompting or replay."""
        if self.seed_buffer[problem_type]:
            return random.choice(self.seed_buffer[problem_type])
        return None

    def get_experience_task(self, problem_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a solved task from experience."""
        if self.experience_buffer[problem_type]:
            return random.choice(self.experience_buffer[problem_type])
        return None
        
    def get_stats(self):
        stats = {k: len(v) for k, v in self.seed_buffer.items()}
        stats.update({f"exp_{k}": len(v) for k, v in self.experience_buffer.items()})
        return stats


