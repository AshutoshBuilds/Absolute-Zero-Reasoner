import os
import json
import time
import random
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pickle
from collections import deque

@dataclass
class RLExperience:
    """Stores a single reinforcement learning experience"""
    problem_type: str
    prompt: str
    generation: str
    reward: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "problem_type": self.problem_type,
            "prompt": self.prompt,
            "generation": self.generation,
            "reward": self.reward,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            problem_type=data["problem_type"],
            prompt=data["prompt"],
            generation=data["generation"],
            reward=data["reward"],
            metadata=data.get("metadata", {})
        )

class ExperienceBuffer:
    """Manages a buffer of RL experiences for training"""
    
    def __init__(self, capacity: int = 10000, save_dir: str = "saved_experiences"):
        self.capacity = capacity
        self.buffer: deque[RLExperience] = deque(maxlen=capacity)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def add(self, experience: RLExperience):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)
    
    def add_many(self, experiences: List[RLExperience]):
        """Add multiple experiences to the buffer"""
        for exp in experiences:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[RLExperience]:
        """Sample a batch of experiences randomly"""
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def sample_by_problem_type(self, problem_type: str, batch_size: int) -> List[RLExperience]:
        """Sample experiences of a specific problem type"""
        filtered = [exp for exp in self.buffer if exp.problem_type == problem_type]
        if batch_size >= len(filtered):
            return filtered
        return random.sample(filtered, min(batch_size, len(filtered)))
    
    def get_high_reward_examples(self, threshold: float = 0.7, limit: int = 5) -> List[RLExperience]:
        """Get high-reward examples for in-context learning"""
        sorted_exps = sorted(self.buffer, key=lambda x: x.reward, reverse=True)
        return [exp for exp in sorted_exps[:limit*2] if exp.reward >= threshold][:limit]
    
    def save(self, filename: str = None):
        """Save the experience buffer to disk"""
        if filename is None:
            filename = f"experiences_{int(time.time())}.json"
        
        path = self.save_dir / filename if not os.path.isabs(filename) else Path(filename)
        # Ensure the parent directory exists
        path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(path, 'w') as f:
            json.dump([exp.to_dict() for exp in self.buffer], f, indent=2)
        
        logging.info(f"Saved {len(self.buffer)} experiences to {path}")
        return path
    
    def load(self, path: Union[str, Path]):
        """Load experiences from disk"""
        path = Path(path)
        if not path.exists():
            logging.warning(f"Experience file {path} doesn't exist")
            return False
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.buffer = deque(RLExperience.from_dict(item) for item in data)
        logging.info(f"Loaded {len(self.buffer)} experiences from {path}")
        return True
    
    def get_recent_experiences(self, limit: int) -> List[RLExperience]:
        """Get the most recent 'limit' experiences from the buffer."""
        if limit <= 0:
            return []
        # Return a copy of the last 'limit' experiences
        # If limit > len(self.buffer), it will return all experiences
        return list(self.buffer)[-limit:]
    
    def save_to_disk(self, path: str) -> bool:
        """Saves the experience buffer (internal deque) to a file using pickle."""
        try:
            with open(path, "wb") as f:
                pickle.dump(self.buffer, f)
            logging.info(f"Experience buffer saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save experience buffer to {path}: {e}")
            return False

    def load_from_disk(self, path: str) -> bool:
        """Loads the experience buffer (internal deque) from a file using pickle."""
        try:
            with open(path, "rb") as f:
                loaded_buffer = pickle.load(f)
            if isinstance(loaded_buffer, deque):
                self.buffer = loaded_buffer
                # Ensure the loaded deque respects the current capacity if it was different
                if self.capacity and len(self.buffer) > self.capacity:
                     logging.warning(f"Loaded buffer has {len(self.buffer)} items, exceeding capacity {self.capacity}. Truncating.")
                     # Convert to list, take last 'capacity' items, then back to deque
                     self.buffer = deque(list(self.buffer)[-self.capacity:], maxlen=self.capacity)
                elif self.capacity:
                    # If capacity is set, ensure the loaded deque also has this maxlen
                    new_deque = deque(maxlen=self.capacity)
                    new_deque.extend(self.buffer)
                    self.buffer = new_deque

                logging.info(f"Experience buffer loaded from {path}. Contains {len(self.buffer)} experiences.")
                return True
            else:
                logging.error(f"Failed to load experience buffer: data in {path} is not a deque.")
                return False
        except FileNotFoundError:
            logging.warning(f"Experience buffer file not found at {path}. Starting with an empty buffer.")
            return False # Or True if starting empty is acceptable
        except Exception as e:
            logging.error(f"Failed to load experience buffer from {path}: {e}")
            return False

    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict:
        """Get statistics about the experience buffer"""
        if not self.buffer:
            return {"count": 0}
        
        rewards = [exp.reward for exp in self.buffer]
        problem_types = {}
        for exp in self.buffer:
            problem_types[exp.problem_type] = problem_types.get(exp.problem_type, 0) + 1
        
        return {
            "count": len(self.buffer),
            "reward_mean": np.mean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "problem_types": problem_types
        } 