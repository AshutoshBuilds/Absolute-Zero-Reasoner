"""
Progressive Context Length Training for AZR HuggingFace Trainer
Implements progressive context length expansion during training to improve model's ability
to handle longer sequences while maintaining training stability.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class ContextSchedule:
    """Configuration for progressive context length expansion"""
    initial_context: int = 512
    target_context: int = 4096
    expansion_epochs: int = 20
    expansion_strategy: str = "linear"  # "linear", "exponential", "stepped"
    validation_threshold: float = 0.8  # Success rate needed before expansion
    stability_epochs: int = 3  # Epochs to wait after expansion before next
    max_expansion_per_step: int = 512  # Maximum tokens to add per expansion

class ProgressiveContextTrainer:
    """Manages progressive context length expansion during training"""
    
    def __init__(self, config: ContextSchedule, model_max_length: int = 8192):
        self.config = config
        self.model_max_length = model_max_length
        
        # Current state
        self.current_context_length = config.initial_context
        self.current_epoch = 0
        self.last_expansion_epoch = 0
        self.expansion_count = 0
        
        # Performance tracking
        self.performance_history = []
        self.context_history = []
        
        # Validation for reasonable settings
        if config.target_context > model_max_length:
            logger.warning(f"Target context {config.target_context} exceeds model max {model_max_length}, adjusting")
            self.config.target_context = model_max_length
        
        logger.info(f"Progressive context training initialized: {config.initial_context} → {config.target_context}")
    
    def should_expand_context(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if context should be expanded based on performance"""
        
        # Don't expand if we've reached target
        if self.current_context_length >= self.config.target_context:
            return False
        
        # Don't expand too frequently
        if self.current_epoch - self.last_expansion_epoch < self.config.stability_epochs:
            return False
        
        # Don't expand if we haven't trained enough epochs yet
        if self.current_epoch < 5:
            return False
        
        # Check if performance is stable and good enough
        success_rate = performance_metrics.get('task_success_rate', 0.0)
        recent_performance = self.performance_history[-5:] if len(self.performance_history) >= 5 else []
        
        if len(recent_performance) < 3:
            return False
        
        # Performance should be above threshold and stable
        avg_recent_performance = np.mean([p['task_success_rate'] for p in recent_performance])
        performance_stable = np.std([p['task_success_rate'] for p in recent_performance]) < 0.1
        
        should_expand = (avg_recent_performance >= self.config.validation_threshold and 
                        performance_stable and
                        success_rate >= self.config.validation_threshold * 0.9)
        
        if should_expand:
            logger.info(f"Context expansion criteria met: success_rate={success_rate:.3f}, "
                       f"avg_recent={avg_recent_performance:.3f}, stable={performance_stable}")
        
        return should_expand
    
    def calculate_next_context_length(self) -> int:
        """Calculate the next context length based on expansion strategy"""
        
        if self.config.expansion_strategy == "linear":
            # Linear expansion over specified epochs
            progress = min(1.0, self.expansion_count / self.config.expansion_epochs)
            target_length = self.config.initial_context + (
                self.config.target_context - self.config.initial_context
            ) * progress
            
        elif self.config.expansion_strategy == "exponential":
            # Exponential expansion - more aggressive early on
            progress = min(1.0, self.expansion_count / self.config.expansion_epochs)
            exponential_progress = progress ** 0.5  # Square root for smoother curve
            target_length = self.config.initial_context + (
                self.config.target_context - self.config.initial_context
            ) * exponential_progress
            
        elif self.config.expansion_strategy == "stepped":
            # Stepped expansion - discrete jumps
            steps = self.config.expansion_epochs
            step_size = (self.config.target_context - self.config.initial_context) / steps
            target_length = self.config.initial_context + (self.expansion_count * step_size)
            
        else:
            logger.error(f"Unknown expansion strategy: {self.config.expansion_strategy}")
            target_length = self.current_context_length
        
        # Ensure we don't exceed maximum expansion per step
        max_next_length = self.current_context_length + self.config.max_expansion_per_step
        target_length = min(target_length, max_next_length, self.config.target_context)
        
        # Round to nearest multiple of 64 for efficiency
        target_length = ((int(target_length) + 63) // 64) * 64
        
        return max(target_length, self.current_context_length)
    
    def expand_context(self, performance_metrics: Dict[str, float]) -> bool:
        """Expand context length if conditions are met"""
        
        if not self.should_expand_context(performance_metrics):
            return False
        
        new_context_length = self.calculate_next_context_length()
        
        if new_context_length <= self.current_context_length:
            return False
        
        old_length = self.current_context_length
        self.current_context_length = new_context_length
        self.last_expansion_epoch = self.current_epoch
        self.expansion_count += 1
        
        logger.info(f"Context expanded: {old_length} → {new_context_length} "
                   f"(expansion {self.expansion_count}/{self.config.expansion_epochs})")
        
        return True
    
    def update_epoch(self, epoch: int, performance_metrics: Dict[str, float]):
        """Update epoch and performance tracking"""
        self.current_epoch = epoch
        
        # Store performance metrics with context length
        metrics_with_context = performance_metrics.copy()
        metrics_with_context['context_length'] = self.current_context_length
        metrics_with_context['epoch'] = epoch
        
        self.performance_history.append(metrics_with_context)
        self.context_history.append({
            'epoch': epoch,
            'context_length': self.current_context_length,
            'expansion_count': self.expansion_count
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
    
    def get_current_context_config(self) -> Dict[str, Any]:
        """Get current context configuration for the model/tokenizer"""
        return {
            'max_length': self.current_context_length,
            'context_length': self.current_context_length,
            'model_max_length': self.current_context_length,
            'truncation': True,
            'padding': 'max_length'
        }
    
    def get_adaptive_batch_size(self, base_batch_size: int) -> int:
        """Adjust batch size based on current context length to manage memory"""
        
        # Reduce batch size as context increases to maintain memory usage
        context_ratio = self.current_context_length / self.config.initial_context
        
        if context_ratio <= 1.5:
            return base_batch_size
        elif context_ratio <= 2.0:
            return max(1, base_batch_size // 2)
        elif context_ratio <= 3.0:
            return max(1, base_batch_size // 3)
        else:
            return max(1, base_batch_size // 4)
    
    def prepare_inputs_with_context(self, texts: List[str], tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
        """Prepare inputs with current context length"""
        
        # Tokenize with current context length
        inputs = tokenizer(
            texts,
            max_length=self.current_context_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of context expansion progress"""
        
        progress_pct = (self.current_context_length - self.config.initial_context) / (
            self.config.target_context - self.config.initial_context
        ) * 100
        
        return {
            'current_context_length': self.current_context_length,
            'target_context_length': self.config.target_context,
            'progress_percentage': min(100, progress_pct),
            'expansion_count': self.expansion_count,
            'last_expansion_epoch': self.last_expansion_epoch,
            'epochs_since_expansion': self.current_epoch - self.last_expansion_epoch,
            'context_history': self.context_history[-10:],  # Last 10 records
            'is_complete': self.current_context_length >= self.config.target_context
        }
    
    def should_adjust_learning_rate(self) -> bool:
        """Check if learning rate should be adjusted after context expansion"""
        # Typically want to reduce LR slightly after context expansion
        return self.current_epoch == self.last_expansion_epoch
    
    def get_lr_adjustment_factor(self) -> float:
        """Get learning rate adjustment factor after context expansion"""
        # Reduce LR by 10% after each expansion to maintain stability
        return 0.9
    
    def save_context_schedule(self, filepath: str):
        """Save context expansion history"""
        import json
        
        data = {
            'config': {
                'initial_context': self.config.initial_context,
                'target_context': self.config.target_context,
                'expansion_epochs': self.config.expansion_epochs,
                'expansion_strategy': self.config.expansion_strategy,
                'validation_threshold': self.config.validation_threshold
            },
            'progress': self.get_progress_summary(),
            'full_history': self.context_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Context schedule saved to {filepath}")

# Convenience function for creating common schedules
def create_context_schedule(
    initial: int = 512,
    target: int = 4096,
    epochs: int = 20,
    strategy: str = "linear"
) -> ContextSchedule:
    """Create a context schedule with common settings"""
    return ContextSchedule(
        initial_context=initial,
        target_context=target,
        expansion_epochs=epochs,
        expansion_strategy=strategy,
        validation_threshold=0.75,  # 75% success rate needed
        stability_epochs=3,         # Wait 3 epochs between expansions
        max_expansion_per_step=512  # Add max 512 tokens per step
    )