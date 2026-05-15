"""
Hyperparameter optimization for AZR system using Optuna.
"""

import os
import json
import optuna
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
import re
import math

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    RichHandler = None
    Table = None
    Panel = None
    _RICH_AVAILABLE = False

# Import our modules
from hf_trainer import HuggingFaceRLTrainer
from azr_hf_adapter import HuggingFaceAdapter
from code_executor import CodeExecutor
from hf_reward_manager import HFRewardManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
_USE_RICH_LOGGING = False


def _apply_cpu_cap(cpu_cap_percent: float) -> int:
    """Apply a soft CPU cap using thread pool limits."""
    cpu_count = os.cpu_count() or 1
    cap = max(1.0, min(100.0, float(cpu_cap_percent)))
    max_threads = max(1, math.floor(cpu_count * cap / 100.0))
    max_threads = min(cpu_count, max_threads)

    env_limits = {
        "OMP_NUM_THREADS": str(max_threads),
        "MKL_NUM_THREADS": str(max_threads),
        "OPENBLAS_NUM_THREADS": str(max_threads),
        "NUMEXPR_MAX_THREADS": str(max_threads),
        "VECLIB_MAXIMUM_THREADS": str(max_threads),
    }
    for key, value in env_limits.items():
        os.environ[key] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(max_threads)
    torch.set_num_interop_threads(max_threads)

    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(max_threads)))
    except Exception:
        pass
    return max_threads


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from message text."""
    return _ANSI_ESCAPE_RE.sub("", text) if isinstance(text, str) else str(text)


def _configure_logging(use_rich: bool = True) -> None:
    """Configure logging with optional Rich formatting."""
    global _USE_RICH_LOGGING
    _USE_RICH_LOGGING = bool(use_rich and _RICH_AVAILABLE)

    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if _USE_RICH_LOGGING:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)

        class _AnsiSafeFormatter(logging.Formatter):
            def format(self, record):
                merged = record.getMessage()
                record.msg = _strip_ansi(str(merged))
                record.args = ()
                return super().format(record)

        handler.setFormatter(_AnsiSafeFormatter("%(name)s - %(levelname)s - %(message)s"))
        if Console is not None:
            Console(highlight=False, force_terminal=True)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _resolve_model_path(model_name: str) -> str:
    """Prefer local cache under models/<basename> when full checkpoint files are present."""
    base_name = Path(model_name).name if (os.path.sep in model_name or '/' in model_name) else model_name.split('/')[-1]
    local_candidate = Path("models") / base_name
    if local_candidate.exists() and (local_candidate / "config.json").exists():
        logger.info(f"Using local model at {local_candidate} instead of remote '{model_name}'")
        return str(local_candidate)
    return model_name


class HyperparameterOptimizer:
    """Optimize hyperparameters for AZR training"""
    
    def __init__(self, 
                 model_name: str = "google/gemma-4-E4B",
                 n_trials: int = 50,
                 results_dir: str = "optimization_results"):
        self.model_name = _resolve_model_path(model_name)
        self.n_trials = n_trials
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Training configuration base
        self.base_config = {
            'model_name': self.model_name,
            'batch_size': 4,
            'buffer_size': 1000,
            'problem_types': ['deduction', 'abduction', 'induction'],
            'save_dir': str(self.results_dir / 'temp_checkpoints'),
            'log_dir': str(self.results_dir / 'temp_logs'),
            'enable_flash_attention': False,
            'gradient_checkpointing': True,
            'mixed_precision': True
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        try:
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
            ppo_epochs = trial.suggest_int('ppo_epochs', 1, 5)
            kl_coef = trial.suggest_float('kl_coef', 0.01, 0.5, log=True)
            entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
            value_loss_coef = trial.suggest_float('value_loss_coef', 0.1, 1.0)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0)
            
            # Reward manager hyperparameters
            accuracy_weight = trial.suggest_float('accuracy_weight', 0.5, 0.9)
            learnability_weight = 1.0 - accuracy_weight
            temperature = trial.suggest_float('temperature', 0.6, 1.0)
            n_samples = trial.suggest_int('n_samples', 2, 5)
            
            # Create configuration
            config = self.base_config.copy()
            config.update({
                'learning_rate': learning_rate,
                'ppo_epochs': ppo_epochs,
                'kl_coef': kl_coef,
                'entropy_coef': entropy_coef,
                'value_loss_coef': value_loss_coef,
                'clip_range': clip_range,
                'max_grad_norm': max_grad_norm
            })
            
            # Initialize components
            trainer = HuggingFaceRLTrainer(config)
            executor = CodeExecutor(timeout=5)
            reward_manager = HFRewardManager(
                executor=executor,
                accuracy_weight=accuracy_weight,
                learnability_weight=learnability_weight,
                n_samples=n_samples,
                temperature=temperature
            )
            
            # Run short training evaluation
            score = self._evaluate_config(trainer, executor, reward_manager, trial)
            
            # Clean up
            executor.cleanup()
            del trainer, executor, reward_manager
            torch.cuda.empty_cache()
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0  # Return worst score for failed trials
    
    def _evaluate_config(self, trainer: HuggingFaceRLTrainer, 
                        executor: CodeExecutor,
                        reward_manager: HFRewardManager,
                        trial: optuna.Trial) -> float:
        """Evaluate a configuration with limited training"""
        
        # Quick evaluation metrics
        total_score = 0.0
        n_evaluations = 5  # Small number for speed
        
        try:
            for i in range(n_evaluations):
                # Generate some tasks
                problem_type = ['deduction', 'abduction', 'induction'][i % 3]
                
                # Test task generation quality
                prompt = trainer._create_proposer_prompt(problem_type, difficulty=1)
                generated = trainer.adapter.generate(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.8
                )[0]
                
                # Parse and validate tasks
                from azr_common_utils import parse_generated_tasks
                tasks = parse_generated_tasks(generated)
                
                if tasks:
                    task = tasks[0]
                    # Check if task is executable
                    if 'program' in task and 'input' in task:
                        result = executor.execute(
                            task.get('program', ''),
                            task.get('input', ''),
                            timeout=3
                        )
                        if result['success']:
                            total_score += 1.0
                        
                        # Test reward calculation
                        try:
                            rewards = reward_manager.compute_rewards(
                                problem_types=[problem_type],
                                generated_tasks=[json.dumps(task)],
                                solutions=[task.get('program', '')],
                                ground_truth=[result.get('output', '')],
                                model=trainer.adapter
                            )
                            
                            if 'r_propose' in rewards and 'r_solve' in rewards:
                                # Reward quality (higher is better)
                                r_prop = rewards['r_propose'][0] if rewards['r_propose'] else 0
                                r_solve = rewards['r_solve'][0] if rewards['r_solve'] else 0
                                total_score += (r_prop + r_solve) / 2
                                
                        except Exception as e:
                            logger.warning(f"Reward calculation failed: {e}")
                
                # Report intermediate progress
                if i % 2 == 0:
                    intermediate_score = total_score / (i + 1)
                    trial.report(intermediate_score, i)
                    
                    # Prune unpromising trials
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            final_score = total_score / n_evaluations
            logger.info(f"Trial {trial.number} score: {final_score:.3f}")
            return final_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return 0.0
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2,
                interval_steps=1
            )
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=3600,  # 1 hour timeout
            show_progress_bar=True
        )
        
        # Save results
        results = self._save_results(study)
        
        return results
    
    def _save_results(self, study: optuna.Study) -> Dict:
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'timestamp': timestamp,
            'model_name': self.model_name
        }
        
        # Save as JSON
        json_path = self.results_dir / f"optimization_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed report
        report_path = self.results_dir / f"optimization_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(f"# Hyperparameter Optimization Report\n\n")
            f.write(f"**Model:** {self.model_name}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Trials:** {len(study.trials)}\n")
            f.write(f"**Best Score:** {best_value:.4f}\n\n")
            
            f.write("## Best Parameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for param, value in best_params.items():
                if isinstance(value, float):
                    f.write(f"| {param} | {value:.6f} |\n")
                else:
                    f.write(f"| {param} | {value} |\n")
            
            f.write("\n## Parameter Importance\n\n")
            try:
                importance = optuna.importance.get_param_importances(study)
                f.write("| Parameter | Importance |\n")
                f.write("|-----------|------------|\n")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {param} | {imp:.4f} |\n")
            except:
                f.write("Parameter importance could not be calculated.\n")
            
            f.write("\n## Trial History\n\n")
            f.write("| Trial | Score | Status |\n")
            f.write("|-------|-------|--------|\n")
            for trial in study.trials[-10:]:  # Last 10 trials
                status = "COMPLETE" if trial.state == optuna.trial.TrialState.COMPLETE else str(trial.state)
                value = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                f.write(f"| {trial.number} | {value} | {status} |\n")
        
        logger.info(f"Results saved to {json_path}")
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        self._print_summary(results, study)
        
        return results
    
    def _print_summary(self, results: Dict, study: optuna.Study) -> None:
        """Print optimization summary"""
        if _USE_RICH_LOGGING and Console is not None and Table is not None and Panel is not None:
            console = Console()
            console.print(Panel.fit(
                f"[bold cyan]Model:[/bold cyan] {self.model_name}\n"
                f"[bold cyan]Total Trials:[/bold cyan] {results['n_trials']}\n"
                f"[bold cyan]Best Score:[/bold cyan] {results['best_value']:.4f}",
                title="[bold]Hyperparameter Optimization Summary[/bold]"
            ))

            table = Table(show_lines=True)
            table.add_column("Rank")
            table.add_column("Trial")
            table.add_column("Score")
            table.add_column("Status")
            sorted_trials = sorted(study.trials, key=lambda x: x.value or 0, reverse=True)
            for i, trial in enumerate(sorted_trials[:5]):
                if trial.value is not None:
                    status = "COMPLETE" if trial.state == optuna.trial.TrialState.COMPLETE else str(trial.state)
                    table.add_row(str(i + 1), str(trial.number), f"{trial.value:.4f}", status)

            console.print(table)

            console.print("[bold]Best Parameters[/bold]")
            param_table = Table(show_lines=True)
            param_table.add_column("Parameter")
            param_table.add_column("Value")
            for param, value in results["best_params"].items():
                if isinstance(value, float):
                    value = f"{value:.6f}"
                param_table.add_row(str(param), str(value))
            console.print(param_table)
            return

        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Total Trials: {results['n_trials']}")
        print(f"Best Score: {results['best_value']:.4f}")
        print("\nBest Parameters:")
        print("-"*30)
        
        for param, value in results['best_params'].items():
            if isinstance(value, float):
                print(f"{param:20}: {value:.6f}")
            else:
                print(f"{param:20}: {value}")
        
        print("\nTop 5 Trials:")
        print("-"*30)
        sorted_trials = sorted(study.trials, key=lambda x: x.value or 0, reverse=True)
        for i, trial in enumerate(sorted_trials[:5]):
            if trial.value is not None:
                print(f"#{i+1}: Score {trial.value:.4f} (Trial {trial.number})")
        
        print("="*60)


def main():
    """Main optimization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize AZR hyperparameters")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B",
                        help="Model name or path")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of optimization trials")
    parser.add_argument("--results-dir", type=str, default="optimization_results",
                        help="Directory to save results")
    parser.add_argument(
        "--rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rich terminal rendering",
    )
    parser.add_argument(
        "--cpu-cap",
        type=float,
        default=20.0,
        help="CPU cap percentage (0-100) for optimization runs",
    )
    
    args = parser.parse_args()
    cpu_threads = _apply_cpu_cap(args.cpu_cap)
    _configure_logging(use_rich=args.rich)
    logger.info(f"CPU cap set to {args.cpu_cap:.1f}% (max threads={cpu_threads})")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        model_name=_resolve_model_path(args.model),
        n_trials=args.n_trials,
        results_dir=args.results_dir
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    print(f"\nOptimization completed! Check {args.results_dir} for detailed results.")


if __name__ == "__main__":
    main()
