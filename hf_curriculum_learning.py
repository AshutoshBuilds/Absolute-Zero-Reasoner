"""
Curriculum Learning for AZR HuggingFace Trainer
Implements adaptive curriculum learning to progressively increase task difficulty
and improve learning efficiency through structured task progression.
Combines both discrete-level and continuous curriculum learning approaches.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import math
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """Task difficulty levels"""
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    initial_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    target_difficulty: DifficultyLevel = DifficultyLevel.EXPERT
    progression_epochs: int = 15
    success_threshold: float = 0.75  # Success rate needed to advance
    stability_window: int = 5  # Epochs to check for stability
    fallback_threshold: float = 0.5  # If performance drops below this, step back
    mix_ratio: float = 0.3  # Ratio of easier tasks to include
    auto_progression: bool = True

@dataclass
class CurriculumState:
    """Represents the current state of curriculum learning"""
    current_difficulty: float = 0.3
    success_rate: float = 0.0
    recent_performance: List[float] = field(default_factory=list)
    task_type_weights: Dict[str, float] = field(default_factory=lambda: {'deduction': 0.4, 'abduction': 0.3, 'induction': 0.3})
    adaptive_threshold: float = 0.05
    progression_rate: float = 0.02

class TaskDifficultyClassifier:
    """Classifies tasks by difficulty based on various criteria"""
    
    def __init__(self):
        self.difficulty_criteria = {
            'code_complexity': self._assess_code_complexity,
            'problem_type': self._assess_problem_type,
            'context_length': self._assess_context_length,
            'logic_depth': self._assess_logic_depth
        }
    
    def classify_task(self, task: Dict[str, Any]) -> DifficultyLevel:
        """Classify a task's difficulty level"""
        code = task.get('code', '')
        problem_type = task.get('type', 'deduction')
        input_data = task.get('input', '')
        output_data = task.get('output', '')
        
        scores = []
        
        # Assess different aspects
        for criterion_name, criterion_func in self.difficulty_criteria.items():
            score = criterion_func(code, problem_type, input_data, output_data)
            scores.append(score)
        
        # Average the scores
        avg_score = np.mean(scores)
        
        # Map to difficulty levels
        if avg_score <= 1.5:
            return DifficultyLevel.BEGINNER
        elif avg_score <= 2.5:
            return DifficultyLevel.EASY
        elif avg_score <= 3.5:
            return DifficultyLevel.MEDIUM
        elif avg_score <= 4.5:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT
    
    def _assess_code_complexity(self, code: str, problem_type: str, input_data: str, output_data: str) -> float:
        """Assess code complexity (1-5 scale)"""
        if not code:
            return 1.0
        
        complexity_indicators = {
            'lines': len(code.split('\n')),
            'loops': code.count('for ') + code.count('while '),
            'conditions': code.count('if ') + code.count('elif '),
            'functions': code.count('def '),
            'imports': code.count('import '),
            'list_comprehensions': code.count('[') + code.count('{'),
            'nested_structures': code.count('    ') // 4  # Rough indentation measure
        }
        
        # Simple scoring based on complexity indicators
        score = 1.0
        if complexity_indicators['lines'] > 5:
            score += 0.5
        if complexity_indicators['lines'] > 10:
            score += 0.5
        if complexity_indicators['loops'] > 0:
            score += 0.5
        if complexity_indicators['loops'] > 1:
            score += 0.5
        if complexity_indicators['conditions'] > 1:
            score += 0.5
        if complexity_indicators['functions'] > 0:
            score += 0.5
        if complexity_indicators['nested_structures'] > 2:
            score += 0.5
        
        return min(5.0, score)
    
    def _assess_problem_type(self, code: str, problem_type: str, input_data: str, output_data: str) -> float:
        """Assess difficulty based on problem type"""
        type_difficulty = {
            'deduction': 2.0,   # Given code + input, predict output
            'abduction': 3.5,   # Given code + output, predict input
            'induction': 4.5    # Given input + output, predict code
        }
        return type_difficulty.get(problem_type, 3.0)
    
    def _assess_context_length(self, code: str, problem_type: str, input_data: str, output_data: str) -> float:
        """Assess difficulty based on context length"""
        total_length = len(code) + len(str(input_data)) + len(str(output_data))
        
        if total_length < 100:
            return 1.0
        elif total_length < 300:
            return 2.0
        elif total_length < 600:
            return 3.0
        elif total_length < 1000:
            return 4.0
        else:
            return 5.0
    
    def _assess_logic_depth(self, code: str, problem_type: str, input_data: str, output_data: str) -> float:
        """Assess logical reasoning depth required"""
        depth_indicators = {
            'mathematical': any(op in code for op in ['**', 'math.', 'numpy', 'sum(', 'max(', 'min(']),
            'string_processing': any(func in code for func in ['.split(', '.join(', '.replace(', 're.']),
            'data_structures': any(struct in code for struct in ['list(', 'dict(', 'set(', 'tuple(']),
            'algorithms': any(alg in code for alg in ['sort', 'binary', 'recursive', 'dynamic']),
            'complex_logic': code.count('and') + code.count('or') + code.count('not') > 2
        }
        
        base_score = 2.0
        for indicator, present in depth_indicators.items():
            if present:
                base_score += 0.6
        
        return min(5.0, base_score)

class CurriculumLearningManager:
    """Manages curriculum learning progression with discrete difficulty levels"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.classifier = TaskDifficultyClassifier()
        
        # Current state
        self.current_difficulty = config.initial_difficulty
        self.current_epoch = 0
        
        # Performance tracking
        self.performance_history = []
        self.difficulty_history = []
        
        # Task pools by difficulty
        self.task_pools = {level: [] for level in DifficultyLevel}
        
        logger.info(f"Curriculum learning initialized: {config.initial_difficulty.name} → {config.target_difficulty.name}")
    
    @property
    def current_level(self) -> str:
        """Get current difficulty level name"""
        return self.current_difficulty.name

    def current_difficulty_numeric(self) -> int:
        """Get current difficulty as numeric value"""
        return self.current_difficulty.value

    def add_tasks_to_pools(self, tasks: List[Dict[str, Any]]):
        """Classify and add tasks to appropriate difficulty pools"""
        for task in tasks:
            difficulty = self.classifier.classify_task(task)
            task['difficulty_level'] = difficulty
            self.task_pools[difficulty].append(task)
        
        # Log pool sizes
        for level in DifficultyLevel:
            count = len(self.task_pools[level])
            logger.info(f"Task pool {level.name}: {count} tasks")
    
    def get_curriculum_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of tasks according to current curriculum"""
        
        # Determine task distribution
        primary_tasks = int(batch_size * (1 - self.config.mix_ratio))
        support_tasks = batch_size - primary_tasks
        
        batch = []
        
        # Get primary difficulty tasks
        primary_pool = self.task_pools[self.current_difficulty]
        if primary_pool:
            batch.extend(random.sample(primary_pool, min(primary_tasks, len(primary_pool))))
        
        # Fill remaining with easier tasks for support
        if support_tasks > 0 and self.current_difficulty.value > 1:
            easier_levels = [level for level in DifficultyLevel if level.value < self.current_difficulty.value]
            for level in easier_levels:
                if len(batch) >= batch_size:
                    break
                pool = self.task_pools[level]
                if pool:
                    needed = min(support_tasks, len(pool), batch_size - len(batch))
                    batch.extend(random.sample(pool, needed))
        
        # Fill any remaining slots with available tasks
        while len(batch) < batch_size:
            all_available = []
            for level_tasks in self.task_pools.values():
                all_available.extend(level_tasks)
            
            if not all_available:
                break
            
            batch.append(random.choice(all_available))
        
        return batch[:batch_size]
    
    def should_advance_difficulty(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if difficulty should be advanced"""
        
        if not self.config.auto_progression:
            return False
        
        if self.current_difficulty.value >= self.config.target_difficulty.value:
            return False
        
        # Need enough performance history
        if len(self.performance_history) < self.config.stability_window:
            return False
        
        # Check recent performance
        recent_performance = self.performance_history[-self.config.stability_window:]
        avg_success_rate = np.mean([p['task_success_rate'] for p in recent_performance])
        performance_stable = np.std([p['task_success_rate'] for p in recent_performance]) < 0.1
        
        should_advance = (avg_success_rate >= self.config.success_threshold and performance_stable)
        
        if should_advance:
            logger.info(f"Curriculum advancement criteria met: avg_success={avg_success_rate:.3f}, stable={performance_stable}")
        
        return should_advance
    
    def should_fallback_difficulty(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if difficulty should be reduced"""
        
        if self.current_difficulty.value <= self.config.initial_difficulty.value:
            return False
        
        # Check if recent performance is too poor
        recent_performance = self.performance_history[-3:] if len(self.performance_history) >= 3 else []
        if not recent_performance:
            return False
        
        avg_success_rate = np.mean([p['task_success_rate'] for p in recent_performance])
        should_fallback = avg_success_rate < self.config.fallback_threshold
        
        if should_fallback:
            logger.warning(f"Curriculum fallback triggered: avg_success={avg_success_rate:.3f} < {self.config.fallback_threshold}")
        
        return should_fallback
    
    def advance_difficulty(self) -> bool:
        """Advance to next difficulty level"""
        if self.current_difficulty.value >= self.config.target_difficulty.value:
            return False
        
        old_difficulty = self.current_difficulty
        new_difficulty = DifficultyLevel(self.current_difficulty.value + 1)
        self.current_difficulty = new_difficulty
        
        logger.info(f"Curriculum advanced: {old_difficulty.name} → {new_difficulty.name}")
        return True
    
    def fallback_difficulty(self) -> bool:
        """Fall back to easier difficulty level"""
        if self.current_difficulty.value <= self.config.initial_difficulty.value:
            return False
        
        old_difficulty = self.current_difficulty
        new_difficulty = DifficultyLevel(self.current_difficulty.value - 1)
        self.current_difficulty = new_difficulty
        
        logger.info(f"Curriculum fallback: {old_difficulty.name} → {new_difficulty.name}")
        return True
    
    def update_epoch(self, epoch: int, performance_metrics: Dict[str, float]):
        """Update epoch and performance tracking"""
        self.current_epoch = epoch
        
        # Store performance with difficulty context
        metrics_with_difficulty = performance_metrics.copy()
        metrics_with_difficulty['difficulty_level'] = self.current_difficulty.value
        metrics_with_difficulty['epoch'] = epoch
        
        self.performance_history.append(metrics_with_difficulty)
        self.difficulty_history.append({
            'epoch': epoch,
            'difficulty': self.current_difficulty.name,
            'difficulty_value': self.current_difficulty.value
        })
        
        # Keep recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        if len(self.difficulty_history) > 50:
            self.difficulty_history = self.difficulty_history[-50:]
        
        # Auto-adjust difficulty
        if self.should_advance_difficulty(performance_metrics):
            self.advance_difficulty()
        elif self.should_fallback_difficulty(performance_metrics):
            self.fallback_difficulty()
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get curriculum learning progress summary"""
        
        progress_pct = ((self.current_difficulty.value - self.config.initial_difficulty.value) / 
                       (self.config.target_difficulty.value - self.config.initial_difficulty.value)) * 100
        
        pool_stats = {level.name: len(tasks) for level, tasks in self.task_pools.items()}
        
        return {
            'current_difficulty': self.current_difficulty.name,
            'difficulty_value': self.current_difficulty.value,
            'target_difficulty': self.config.target_difficulty.name,
            'progress_percentage': min(100, max(0, progress_pct)),
            'task_pool_sizes': pool_stats,
            'difficulty_history': self.difficulty_history[-10:],
            'is_complete': self.current_difficulty.value >= self.config.target_difficulty.value
        }
    
    def save_curriculum_state(self, filepath: str):
        """Save curriculum learning state"""
        data = {
            'config': {
                'initial_difficulty': self.config.initial_difficulty.name,
                'target_difficulty': self.config.target_difficulty.name,
                'progression_epochs': self.config.progression_epochs,
                'success_threshold': self.config.success_threshold,
                'mix_ratio': self.config.mix_ratio
            },
            'current_state': self.get_curriculum_summary(),
            'full_history': self.difficulty_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Curriculum state saved to {filepath}")

class EnhancedCurriculumManager:
    """Advanced curriculum learning with adaptive difficulty progression and continuous scaling"""
    
    def __init__(self,
                 initial_difficulty: float = 0.3,
                 target_success_rate: float = 0.7,
                 adaptation_window: int = 50,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 progression_patience: int = 10):
        
        self.initial_difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.adaptation_window = adaptation_window
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.progression_patience = progression_patience
        
        # Initialize curriculum state
        self.state = CurriculumState(
            current_difficulty=initial_difficulty,
            success_rate=0.0,
            recent_performance=[],
            task_type_weights={'deduction': 0.4, 'abduction': 0.3, 'induction': 0.3},
            adaptive_threshold=0.05,
            progression_rate=0.02
        )
        
        # Performance tracking
        self.performance_history = []
        self.difficulty_history = []
        self.task_type_performance = {
            'deduction': [],
            'abduction': [],
            'induction': []
        }
        
        # Advanced features
        self.knowledge_graph = {}  # Track concept mastery
        self.skill_dependencies = {
            'basic_operations': [],
            'control_flow': ['basic_operations'],
            'data_structures': ['basic_operations'],
            'algorithms': ['control_flow', 'data_structures'],
            'advanced_concepts': ['algorithms']
        }
        
        # Exploration vs exploitation
        self.exploration_rate = 0.2
        self.exploitation_boost = 1.5
        
    def update_performance(self, task_type: str, success: bool, difficulty: float) -> None:
        """Update performance metrics and adjust curriculum"""
        
        # Update recent performance
        performance_score = 1.0 if success else 0.0
        self.state.recent_performance.append(performance_score)
        
        # Maintain window size
        if len(self.state.recent_performance) > self.adaptation_window:
            self.state.recent_performance.pop(0)
        
        # Update task type specific performance
        self.task_type_performance[task_type].append(performance_score)
        if len(self.task_type_performance[task_type]) > self.adaptation_window:
            self.task_type_performance[task_type].pop(0)
        
        # Calculate current success rate
        if len(self.state.recent_performance) > 0:
            self.state.success_rate = float(np.mean(self.state.recent_performance))
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'task_type': task_type,
            'success': success,
            'difficulty': difficulty,
            'success_rate': self.state.success_rate
        })
        
        # Adaptive difficulty adjustment
        self._adjust_difficulty()
        
        # Update task type weights
        self._update_task_weights()
        
        # Update knowledge graph
        self._update_knowledge_graph(task_type, success, difficulty)
        
        logger.debug(f"Performance updated: {task_type}, success={success}, "
                    f"difficulty={difficulty:.3f}, success_rate={self.state.success_rate:.3f}")
    
    def _adjust_difficulty(self) -> None:
        """Dynamically adjust difficulty based on performance"""
        
        if len(self.state.recent_performance) < 10:  # Need minimum data
            return
        
        current_sr = self.state.success_rate
        target_sr = self.target_success_rate
        
        # Calculate difficulty adjustment
        performance_gap = current_sr - target_sr
        
        # Adaptive adjustment rate based on confidence
        confidence = min(len(self.state.recent_performance) / self.adaptation_window, 1.0)
        adjustment_rate = self.state.progression_rate * confidence
        
        # Apply different strategies based on performance
        if performance_gap > self.state.adaptive_threshold:
            # Performance is good, increase difficulty
            difficulty_increase = adjustment_rate * (1 + performance_gap)
            self.state.current_difficulty = min(
                self.state.current_difficulty + difficulty_increase,
                self.max_difficulty
            )
            
        elif performance_gap < -self.state.adaptive_threshold:
            # Performance is poor, decrease difficulty
            difficulty_decrease = adjustment_rate * abs(performance_gap)
            self.state.current_difficulty = max(
                self.state.current_difficulty - difficulty_decrease,
                self.min_difficulty
            )
        
        # Record difficulty change
        self.difficulty_history.append({
            'timestamp': datetime.now(),
            'difficulty': self.state.current_difficulty,
            'success_rate': current_sr,
            'adjustment': performance_gap
        })
    
    def _update_task_weights(self) -> None:
        """Update task type weights based on relative performance"""
        
        # Calculate success rates for each task type
        task_success_rates = {}
        for task_type, performances in self.task_type_performance.items():
            if len(performances) > 5:  # Minimum data requirement
                task_success_rates[task_type] = np.mean(performances)
            else:
                task_success_rates[task_type] = self.target_success_rate  # Default
        
        # Apply adaptive weighting strategy
        total_weight = 0.0
        new_weights = {}
        
        for task_type, success_rate in task_success_rates.items():
            # Lower performing task types get higher weight (more practice)
            # But not too extreme to maintain diversity
            performance_ratio = success_rate / self.target_success_rate
            
            if performance_ratio < 0.8:  # Poor performance
                weight = 1.0 + (0.8 - performance_ratio) * 2.0  # Boost weight
            elif performance_ratio > 1.2:  # Good performance
                weight = 0.5 + (1.0 / performance_ratio) * 0.5  # Reduce weight
            else:
                weight = 1.0  # Neutral weight
            
            new_weights[task_type] = weight
            total_weight += weight
        
        # Normalize weights
        for task_type in new_weights:
            self.state.task_type_weights[task_type] = new_weights[task_type] / total_weight
        
        logger.debug(f"Updated task weights: {self.state.task_type_weights}")
    
    def _update_knowledge_graph(self, task_type: str, success: bool, difficulty: float) -> None:
        """Update knowledge graph based on performance"""
        
        # Simple knowledge tracking - can be expanded
        if task_type not in self.knowledge_graph:
            self.knowledge_graph[task_type] = {
                'mastery_level': 0.0,
                'attempts': 0,
                'successes': 0,
                'difficulty_progress': []
            }
        
        kg_entry = self.knowledge_graph[task_type]
        kg_entry['attempts'] += 1
        
        if success:
            kg_entry['successes'] += 1
        
        # Update mastery level (exponential moving average)
        success_rate = kg_entry['successes'] / kg_entry['attempts']
        kg_entry['mastery_level'] = 0.9 * kg_entry['mastery_level'] + 0.1 * success_rate
        
        # Track difficulty progress
        kg_entry['difficulty_progress'].append(difficulty)
        if len(kg_entry['difficulty_progress']) > 20:
            kg_entry['difficulty_progress'].pop(0)
    
    def get_next_task_type(self) -> str:
        """Select next task type based on curriculum strategy"""
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Exploration: random selection
            return random.choice(['deduction', 'abduction', 'induction'])
        else:
            # Exploitation: weighted selection based on curriculum
            task_types = list(self.state.task_type_weights.keys())
            weights = list(self.state.task_type_weights.values())
            
            # Apply exploitation boost to struggling areas
            boosted_weights = []
            for i, (task_type, weight) in enumerate(zip(task_types, weights)):
                task_sr = np.mean(self.task_type_performance[task_type][-10:]) if \
                         len(self.task_type_performance[task_type]) >= 10 else self.target_success_rate
                
                if task_sr < self.target_success_rate * 0.8:
                    boosted_weights.append(weight * self.exploitation_boost)
                else:
                    boosted_weights.append(weight)
            
            # Normalize boosted weights
            total_boosted = sum(boosted_weights)
            normalized_weights = [w / total_boosted for w in boosted_weights]
            
            return np.random.choice(task_types, p=normalized_weights)
    
    def get_current_difficulty(self, task_type: Optional[str] = None) -> float:
        """Get current difficulty level, optionally adjusted for task type"""
        
        base_difficulty = self.state.current_difficulty
        
        if task_type and task_type in self.knowledge_graph:
            # Adjust based on task-specific mastery
            mastery = self.knowledge_graph[task_type]['mastery_level']
            
            if mastery > 0.8:  # High mastery
                # Increase difficulty for this task type
                adjustment = (mastery - 0.8) * 0.5
                return min(base_difficulty + adjustment, self.max_difficulty)
            elif mastery < 0.4:  # Low mastery
                # Decrease difficulty for this task type
                adjustment = (0.4 - mastery) * 0.3
                return max(base_difficulty - adjustment, self.min_difficulty)
        
        return base_difficulty
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics"""
        
        stats = {
            'current_state': {
                'difficulty': self.state.current_difficulty,
                'success_rate': self.state.success_rate,
                'task_weights': self.state.task_type_weights.copy()
            },
            'performance_trends': {},
            'knowledge_mastery': {},
            'recommendations': []
        }
        
        # Calculate performance trends
        for task_type, performances in self.task_type_performance.items():
            if len(performances) >= 10:
                recent_perf = np.mean(performances[-10:])
                older_perf = np.mean(performances[-20:-10]) if len(performances) >= 20 else recent_perf
                trend = recent_perf - older_perf
                
                stats['performance_trends'][task_type] = {
                    'recent_performance': recent_perf,
                    'trend': trend,
                    'status': 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
                }
        
        # Knowledge mastery summary
        for task_type, kg_data in self.knowledge_graph.items():
            stats['knowledge_mastery'][task_type] = {
                'mastery_level': kg_data['mastery_level'],
                'attempts': kg_data['attempts'],
                'success_rate': kg_data['successes'] / kg_data['attempts'] if kg_data['attempts'] > 0 else 0
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        stats['recommendations'] = recommendations
        
        return stats
    
    def _generate_recommendations(self) -> List[str]:
        """Generate curriculum recommendations based on current state"""
        
        recommendations = []
        
        # Overall performance recommendations
        if self.state.success_rate < 0.5:
            recommendations.append("Consider decreasing difficulty more aggressively - current success rate is low")
        elif self.state.success_rate > 0.9:
            recommendations.append("Consider increasing difficulty - model may be under-challenged")
        
        # Task-specific recommendations
        for task_type, performances in self.task_type_performance.items():
            if len(performances) >= 10:
                task_sr = np.mean(performances[-10:])
                if task_sr < 0.4:
                    recommendations.append(f"Focus more on {task_type} tasks - performance is significantly below target")
                elif task_sr > 0.9:
                    recommendations.append(f"Consider advanced {task_type} tasks - current mastery is high")
        
        # Exploration recommendations
        if self.exploration_rate < 0.1:
            recommendations.append("Consider increasing exploration rate to discover new learning opportunities")
        
        return recommendations
    
    def save_curriculum_state(self, filepath: str) -> None:
        """Save curriculum state to file"""
        
        state_data = {
            'state': {
                'current_difficulty': self.state.current_difficulty,
                'success_rate': self.state.success_rate,
                'task_type_weights': self.state.task_type_weights,
                'adaptive_threshold': self.state.adaptive_threshold,
                'progression_rate': self.state.progression_rate
            },
            'performance_history': self.performance_history[-1000:],  # Last 1000 entries
            'difficulty_history': self.difficulty_history[-1000:],
            'task_type_performance': {
                k: v[-100:] for k, v in self.task_type_performance.items()  # Last 100 per type
            },
            'knowledge_graph': self.knowledge_graph,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'total_updates': len(self.performance_history)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Curriculum state saved to {filepath}")
    
    def load_curriculum_state(self, filepath: str) -> None:
        """Load curriculum state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            state_dict = state_data['state']
            self.state.current_difficulty = state_dict['current_difficulty']
            self.state.success_rate = state_dict['success_rate']
            self.state.task_type_weights = state_dict['task_type_weights']
            self.state.adaptive_threshold = state_dict['adaptive_threshold']
            self.state.progression_rate = state_dict['progression_rate']
            
            # Restore history (convert timestamps back to datetime if needed)
            self.performance_history = state_data.get('performance_history', [])
            self.difficulty_history = state_data.get('difficulty_history', [])
            self.task_type_performance = state_data.get('task_type_performance', {
                'deduction': [], 'abduction': [], 'induction': []
            })
            self.knowledge_graph = state_data.get('knowledge_graph', {})
            
            # Rebuild recent performance from history
            recent_perf = [entry['success'] for entry in self.performance_history[-self.adaptation_window:]]
            self.state.recent_performance = [1.0 if success else 0.0 for success in recent_perf]
            
            logger.info(f"Curriculum state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load curriculum state: {e}")
            logger.info("Continuing with default initialization")

    @property
    def current_level(self) -> str:
        """Get current difficulty level name"""
        return self.current_difficulty.name

    def current_difficulty_numeric(self) -> int:
        """Get current difficulty as numeric value"""
        return self.current_difficulty.value

def create_curriculum_config(initial_level: str = "BEGINNER",
                           target_level: str = "EXPERT",
                           epochs: int = 15,
                           success_threshold: float = 0.75) -> CurriculumConfig:
    """Create curriculum config with common settings"""
    return CurriculumConfig(
        initial_difficulty=DifficultyLevel[initial_level],
        target_difficulty=DifficultyLevel[target_level],
        progression_epochs=epochs,
        success_threshold=success_threshold,
        stability_window=5,
        fallback_threshold=0.5,
        mix_ratio=0.3,
        auto_progression=True
    ) 