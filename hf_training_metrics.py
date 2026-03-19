"""
Advanced Training Metrics for AZR HuggingFace Trainer
Tracks comprehensive metrics for PPO training, model performance, and learning progress.
"""

import logging
import time
import json
import torch
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EpochMetrics:
    """Metrics for a single epoch"""
    epoch: int
    timestamp: float
    
    # PPO Training Metrics
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    
    # Learning Statistics
    mean_reward_proposer: float
    mean_reward_solver: float
    mean_advantage: float
    kl_divergence: float
    
    # Performance Metrics
    task_success_rate: float
    code_execution_success_rate: float
    json_parsing_success_rate: float
    
    # Model Statistics
    gradient_norm: float
    learning_rate: float
    tokens_generated: int
    context_length_used: int
    
    # Memory and Timing
    memory_usage_gb: float
    training_time_seconds: float
    tokens_per_second: float

class AdvancedTrainingMetrics:
    """Comprehensive training metrics tracking and analysis"""
    
    def __init__(self, save_dir: str = "./training_metrics", history_window: int = 100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history_window = history_window
        
        # Metrics storage
        self.epoch_metrics: List[EpochMetrics] = []
        self.metric_history = defaultdict(lambda: deque(maxlen=history_window))
        
        # Timing
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # Convergence tracking
        self.best_reward = float('-inf')
        self.best_epoch = 0
        self.plateau_count = 0
        self.improvement_threshold = 0.01
        
        logger.info(f"AdvancedTrainingMetrics initialized. Saving to {self.save_dir}")
    
    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
    
    def end_epoch(self, epoch: int, metrics_dict: Dict[str, Any]) -> EpochMetrics:
        """Record metrics for completed epoch"""
        if self.epoch_start_time is None:
            logger.warning("end_epoch called without start_epoch")
            epoch_time = 0.0
        else:
            epoch_time = time.time() - self.epoch_start_time
        
        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            timestamp=time.time(),
            
            # PPO Metrics
            policy_loss=metrics_dict.get('policy_loss', 0.0),
            value_loss=metrics_dict.get('value_loss', 0.0),
            entropy=metrics_dict.get('entropy', 0.0),
            total_loss=metrics_dict.get('total_loss', 0.0),
            
            # Learning Statistics
            mean_reward_proposer=metrics_dict.get('mean_reward_proposer', 0.0),
            mean_reward_solver=metrics_dict.get('mean_reward_solver', 0.0),
            mean_advantage=metrics_dict.get('mean_advantage', 0.0),
            kl_divergence=metrics_dict.get('kl_divergence', 0.0),
            
            # Performance Metrics
            task_success_rate=metrics_dict.get('task_success_rate', 0.0),
            code_execution_success_rate=metrics_dict.get('code_execution_success_rate', 0.0),
            json_parsing_success_rate=metrics_dict.get('json_parsing_success_rate', 0.0),
            
            # Model Statistics
            gradient_norm=metrics_dict.get('gradient_norm', 0.0),
            learning_rate=metrics_dict.get('learning_rate', 0.0),
            tokens_generated=metrics_dict.get('tokens_generated', 0),
            context_length_used=metrics_dict.get('context_length_used', 0),
            
            # Memory and Timing
            memory_usage_gb=self._get_memory_usage(),
            training_time_seconds=epoch_time,
            tokens_per_second=metrics_dict.get('tokens_generated', 0) / max(epoch_time, 0.001)
        )
        
        # Store metrics
        self.epoch_metrics.append(epoch_metrics)
        self._update_metric_history(epoch_metrics)
        
        # Check for improvement
        current_reward = (epoch_metrics.mean_reward_proposer + epoch_metrics.mean_reward_solver) / 2
        if current_reward > self.best_reward + self.improvement_threshold:
            self.best_reward = current_reward
            self.best_epoch = epoch
            self.plateau_count = 0
            logger.info(f"New best reward: {self.best_reward:.4f} at epoch {epoch}")
        else:
            self.plateau_count += 1
        
        # Save metrics periodically
        if epoch % 5 == 0:
            self.save_metrics()
        
        logger.info(f"Epoch {epoch} metrics: Loss={epoch_metrics.total_loss:.4f}, Reward={current_reward:.4f}, Time={epoch_time:.2f}s")
        
        return epoch_metrics
    
    def _update_metric_history(self, metrics: EpochMetrics):
        """Update rolling history of metrics"""
        for key, value in asdict(metrics).items():
            if isinstance(value, (int, float)):
                self.metric_history[key].append(value)
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def get_recent_average(self, metric_name: str, window: int = 10) -> float:
        """Get recent average of a metric"""
        recent_values = list(self.metric_history[metric_name])[-window:]
        return np.mean(recent_values) if recent_values else 0.0
    
    def detect_convergence(self) -> bool:
        """Detect if training has converged"""
        if len(self.epoch_metrics) < 20:
            return False
        
        # Simple convergence: plateau for many epochs
        return self.plateau_count > 15
    
    def save_metrics(self):
        """Save all metrics to files"""
        try:
            # Save raw metrics
            metrics_file = self.save_dir / "epoch_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump([asdict(m) for m in self.epoch_metrics], f, indent=2)
            
            # Save summary
            summary_file = self.save_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.get_training_summary(), f, indent=2)
            
            logger.info(f"Metrics saved to {self.save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.epoch_metrics:
            return {"status": "no_data"}
        
        latest = self.epoch_metrics[-1]
        total_time = time.time() - self.training_start_time
        
        return {
            "training_progress": {
                "total_epochs": len(self.epoch_metrics),
                "total_time_hours": total_time / 3600,
                "best_reward": self.best_reward,
                "best_epoch": self.best_epoch,
                "convergence_detected": self.detect_convergence(),
                "plateau_count": self.plateau_count
            },
            "current_performance": {
                "latest_reward": (latest.mean_reward_proposer + latest.mean_reward_solver) / 2,
                "latest_loss": latest.total_loss,
                "success_rate": latest.task_success_rate,
                "tokens_per_second": latest.tokens_per_second,
                "memory_usage_gb": latest.memory_usage_gb
            }
        }
    
    def should_stop_training(self) -> bool:
        """Determine if training should be stopped"""
        if len(self.epoch_metrics) < 10:
            return False
        
        # Stop if converged
        if self.detect_convergence():
            logger.info("Training stopped: Convergence detected")
            return True
        
        # Stop if loss is exploding
        recent_losses = [m.total_loss for m in self.epoch_metrics[-5:]]
        if any(loss > 100 for loss in recent_losses):
            logger.warning("Training stopped: Loss explosion detected")
            return True
        
        return False
