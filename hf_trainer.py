# hf_trainer.py

import os
import json
import time
import random
import logging
import re # Added for regex in parsing
import ast # Added for literal_eval in parsing
import gc
import numpy as np
import math
import torch
import torch.nn.functional as F # For loss functions
from torch.optim import AdamW # Optimizer
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Back, Style, init # Enhanced colorama support
init(autoreset=True) # Initialize colorama with auto-reset
from typing import Dict, List, Tuple, Optional, Any, Union, Deque # Added Deque

def print_status_header():
    """Print a formatted status header with project information"""
    print(f"\n{Fore.CYAN}╔{'═'*78}╗")
    print(f"{Fore.CYAN}║{' '*25} ABSOLUTE ZERO MODEL - TRAINING STATUS{' '*16}║")
    print(f"{Fore.CYAN}╚{'═'*78}╝")
    print(f"{Fore.YELLOW} Based on: 'Absolute Zero: Reinforced Self-play Reasoning with Zero Data'")
    print(f"{Fore.BLUE} Goal: Train AI to reason without initial data using self-play RL")
    print(f"{Fore.MAGENTA} Architecture: Actor-Critic PPO with Progressive Context & Curriculum Learning")
    print(f"{Fore.CYAN}╚{'═'*78}╝\n")

def print_training_config(config: Dict):
    """Print formatted training configuration"""
    print(f"{Fore.GREEN}  TRAINING CONFIGURATION:")
    print(f"{Fore.WHITE}┌{'─'*48}┐")
    print(f"{Fore.CYAN}│  Epochs:{Fore.WHITE} {config['rl_epochs']:<41} │")
    print(f"{Fore.CYAN}│  Gen Steps/Epoch:{Fore.WHITE} {config['generation_steps_per_epoch']:<29} │")
    print(f"{Fore.CYAN}│  PPO Threshold:{Fore.WHITE} {config['ppo_update_threshold']:<32} │")
    print(f"{Fore.CYAN}│  Batch Size:{Fore.WHITE} {config['batch_size']:<39} │")
    print(f"{Fore.CYAN}│  Max Tokens:{Fore.WHITE} {config['max_new_tokens']:<38} │")
    print(f"{Fore.CYAN}│  Checkpoint Dir:{Fore.WHITE} {config['checkpoint_dir']:<32} │")
    print(f"{Fore.WHITE}└{'─'*48}┘\n")

def print_system_status(adapter, device, config=None):
    """Print formatted system and model status"""
    print(f"{Fore.BLUE} SYSTEM STATUS:")
    print(f"{Fore.WHITE}┌{'─'*48}┐")
    model_name = adapter.model_name if hasattr(adapter, 'model_name') else str(adapter.model)[:40]
    memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 'N/A'
    grad_scaler = 'Enabled' if hasattr(adapter, 'scaler') and adapter.scaler else 'Disabled'
    actor_critic = 'Separate' if hasattr(adapter, 'use_separate_value_model') and adapter.use_separate_value_model else 'Shared'

    print(f"{Fore.GREEN}│  Model:{Fore.WHITE} {model_name:<39} │")
    print(f"{Fore.GREEN}│  Device:{Fore.WHITE} {str(device):<39} │")
    print(f"{Fore.GREEN}│  Memory:{Fore.WHITE} {f'{memory_gb}GB':<39} │")
    print(f"{Fore.GREEN}│  GradScaler:{Fore.WHITE} {grad_scaler:<33} │")
    print(f"{Fore.GREEN}│  Actor-Critic:{Fore.WHITE} {actor_critic:<32} │")

    # Show memory optimization settings if config provided
    if config:
        grad_checkpoint = 'Enabled' if config.get('gradient_checkpointing', False) else 'Disabled'
        mixed_precision = 'Enabled' if config.get('mixed_precision', False) else 'Disabled'
        print(f"{Fore.CYAN}│  Gradient Checkpointing:{Fore.WHITE} {grad_checkpoint:<19} │")
        print(f"{Fore.CYAN}│  Mixed Precision:{Fore.WHITE} {mixed_precision:<26} │")

    print(f"{Fore.WHITE}└{'─'*48}┘\n")

def print_advanced_features_status():
    """Print advanced features status"""
    print(f"{Fore.MAGENTA} ADVANCED FEATURES:")
    print(f"{Fore.WHITE}┌{'─'*48}┐")
    context_range = "256 → 1024 tokens"
    curriculum = "BEGINNER → HARD"
    metrics_dir = "./training_metrics/"
    anomaly_mode = 'Enabled' if torch.is_anomaly_enabled() else 'Disabled'

    print(f"{Fore.CYAN}│  Progressive Context:{Fore.WHITE} {context_range:<29} │")
    print(f"{Fore.CYAN}│  Curriculum Learning:{Fore.WHITE} {curriculum:<30} │")
    print(f"{Fore.CYAN}│  Metrics Tracking:{Fore.WHITE} {metrics_dir:<32} │")
    print(f"{Fore.CYAN}│  Autograd Anomaly:{Fore.WHITE} {anomaly_mode:<31} │")
    print(f"{Fore.WHITE}└{'─'*48}┘\n")

def print_training_summary(trainer, start_time, total_epochs):
    """Print comprehensive training summary"""
    print(f"\n{Fore.CYAN}╔{'═'*78}╗")
    print(f"{Fore.CYAN}║{' '*25} TRAINING COMPLETED!{' '*32}║")
    print(f"{Fore.CYAN}╚{'═'*78}╝")

    # Training duration
    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600

    # Load final metrics
    try:
        with open("training_metrics/training_summary.json", "r") as f:
            summary = json.load(f)
        training_progress = summary.get("training_progress", {})
        current_performance = summary.get("current_performance", {})

        print(f"{Fore.GREEN} FINAL RESULTS:")
        print(f"{Fore.WHITE}┌{'─'*76}┐")
        print(f"{Fore.CYAN}│   Duration:{Fore.WHITE} {duration_hours:.2f} hours{' '*51} │")
        print(f"{Fore.CYAN}│  Total Epochs:{Fore.WHITE} {training_progress.get('total_epochs', 0):<51} │")
        print(f"{Fore.CYAN}│  Best Reward:{Fore.WHITE} {training_progress.get('best_reward', 0):.4f}{' '*48} │")
        print(f"{Fore.CYAN}│  Best Epoch:{Fore.WHITE} {training_progress.get('best_epoch', 0):<52} │")
        print(f"{Fore.CYAN}│  Latest Proposer Reward:{Fore.WHITE} {current_performance.get('latest_reward', 0):.4f}{' '*36} │")
        print(f"{Fore.CYAN}│  Memory Usage:{Fore.WHITE} {current_performance.get('memory_usage_gb', 0):.1f} GB{' '*48} │")

        # Convergence status
        convergence_status = " Converged" if training_progress.get("convergence_detected", False) else " Still Training"
        plateau_status = f"Plateau: {training_progress.get('plateau_count', 0)}"
        print(f"{Fore.CYAN}│  Status:{Fore.WHITE} {convergence_status} | {plateau_status}{' '*44} │")
        print(f"{Fore.WHITE}└{'─'*76}┘")

    except FileNotFoundError:
        print(f"{Fore.YELLOW}  Metrics file not found - training may have crashed")
        print(f"{Fore.WHITE}└{'─'*76}┘")

    # Model artifacts
    checkpoint_dir = trainer.config.get("checkpoint_dir", "./hf_checkpoints") if hasattr(trainer, "config") else "./hf_checkpoints"

    print(f"\n{Fore.BLUE} SAVED ARTIFACTS:")
    print(f"{Fore.WHITE}┌{'─'*76}┐")
    print(f"{Fore.CYAN}│  Checkpoints:{Fore.WHITE} {checkpoint_dir}{' ' * max(0, 44 - len(str(checkpoint_dir)))} │")
    print(f"{Fore.CYAN}│  Metrics:{Fore.WHITE} ./training_metrics/{' '*58} │")
    print(f"{Fore.CYAN}│  Logs:{Fore.WHITE} ./training_run.log{' '*61} │")
    print(f"{Fore.CYAN}│  Visualizations:{Fore.WHITE} ./visualizations/{' '*53} │")
    print(f"{Fore.WHITE}└{'─'*76}┘")

    print(f"\n{Fore.GREEN} NEXT STEPS:")
    print(f"{Fore.WHITE}┌{'─'*76}┐")
    print(f"{Fore.CYAN}│ 1.  Analyze metrics:{Fore.WHITE} python -m tensorboard --logdir training_metrics/{' '*27} │")
    print(f"{Fore.CYAN}│ 2.  Review logs:{Fore.WHITE} tail -f training_run.log{' '*47} │")
    print(f"{Fore.CYAN}│ 3.  Test model:{Fore.WHITE} python evaluate_benchmarks.py{' '*42} │")
    print(f"{Fore.CYAN}│ 4.  Visualize:{Fore.WHITE} python -m matplotlib.pyplot training_metrics/*.json{' '*22} │")
    print(f"{Fore.WHITE}└{'─'*76}┘")

    print(f"\n{Fore.CYAN}╔{'═'*78}╗")
    print(f"{Fore.CYAN}║{' '*20} Ready for evaluation and deployment!{' '*20}║")
    print(f"{Fore.CYAN}╚{'═'*78}╝\n")

    # Demo the enhanced display features
    print(f"{Fore.MAGENTA} DISPLAY ENHANCEMENT SUMMARY:")
    print(f"{Fore.WHITE}┌{'─'*76}┐")
    print(f"{Fore.CYAN}│  Enhanced Progress Bars:{Fore.WHITE} Color-coded with text tags and better formatting")
    print(f"{Fore.CYAN}│  Real-time Status:{Fore.WHITE} Live updates during training steps")
    print(f"{Fore.CYAN}│  Boxed Layouts:{Fore.WHITE} Professional terminal UI with borders")
    print(f"{Fore.CYAN}│  Smart Logging:{Fore.WHITE} Reduced clutter, color-coded messages")
    print(f"{Fore.CYAN}│  Visual Feedback:{Fore.WHITE} Clear status indicators and progress tracking")
    print(f"{Fore.WHITE}└{'─'*76}┘\n")
from collections import defaultdict, Counter, deque # Added deque
from torch.nn.utils.rnn import pad_sequence # Added for padding
from torch.amp import GradScaler, autocast # UPDATED for mixed precision
import torch.distributions as D # Add this import

import sys # Import sys
sys.setrecursionlimit(3000) # Increased recursion limit

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    RichHandler = None
    _RICH_AVAILABLE = False

_ANSI_COLOR_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from a log message."""
    return _ANSI_COLOR_RE.sub("", text) if isinstance(text, str) else str(text)


def _apply_cpu_cap(cpu_cap_percent: float) -> int:
    """Apply a soft CPU cap using thread limits and backend environment variables."""
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


def _parse_torch_dtype(dtype_name: str):
    """Convert user-facing dtype strings to torch dtypes."""
    if dtype_name is None:
        return None

    normalized = str(dtype_name).lower().strip()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    if normalized == "auto":
        return None
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use fp16, bf16, fp32, or auto.")


def _azr_optional_env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logging.getLogger("AZR-HF-RL").warning("Invalid %s=%r; ignoring.", name, os.environ.get(name))
    return None


def _azr_optional_env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logging.getLogger("AZR-HF-RL").warning("Invalid %s=%r; ignoring.", name, raw)
        return None


def _azr_optional_env_positive_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        logging.getLogger("AZR-HF-RL").warning("Invalid %s=%r; ignoring.", name, raw)
        return None
    if v <= 0:
        logging.getLogger("AZR-HF-RL").warning("Invalid %s=%r (must be positive); ignoring.", name, raw)
        return None
    return v


def _azr_optional_env_int_allow_zero(name: str) -> Optional[int]:
    """Parse int from env when the variable is set; allows zero (e.g. seed task count). Unset -> None."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logging.getLogger("AZR-HF-RL").warning("Invalid %s=%r; ignoring.", name, raw)
        return None


def _paper_style_env_suppresses_preset_key(key: str) -> bool:
    """True when an explicit AZR_HF_* / seed env is set so the paper preset must not fill that key."""
    if key == "learning_rate":
        return _azr_optional_env_float("AZR_HF_LEARNING_RATE") is not None
    if key == "critic_learning_rate":
        return _azr_optional_env_float("AZR_HF_CRITIC_LEARNING_RATE") is not None
    if key == "generation_steps_per_epoch":
        return _azr_optional_env_positive_int("AZR_HF_GENERATION_STEPS_PER_EPOCH") is not None
    if key == "batch_size":
        return _azr_optional_env_positive_int("AZR_HF_BATCH_SIZE") is not None
    if key == "ppo_update_threshold":
        return _azr_optional_env_positive_int("AZR_HF_PPO_UPDATE_THRESHOLD") is not None
    return False


def _maybe_apply_azr_paper_style_defaults(config: Dict[str, Any], user_config_keys: set) -> None:
    """
    When AZR_PAPER_STYLE_DEFAULTS=1, apply modest paper-like LR and throughput defaults for keys
    the caller did not explicitly set (hyperparameters only; does not touch model paths).
    """
    if _azr_optional_env_bool("AZR_PAPER_STYLE_DEFAULTS") is not True:
        return
    log = logging.getLogger("AZR-HF-RL")
    preset = {
        "learning_rate": 5e-7,
        "critic_learning_rate": 5e-7,
        "generation_steps_per_epoch": 10,
        "batch_size": 16,
        "ppo_update_threshold": 64,
    }
    applied_parts = []
    for key, value in preset.items():
        if key not in user_config_keys and not _paper_style_env_suppresses_preset_key(key):
            config[key] = value
            applied_parts.append(f"{key}={value!r}")
    # Warm-start buffers like the protocol: small seed set unless caller set seed_tasks_per_type.
    seed_env = os.environ.get("AZR_SEED_TASKS_PER_TYPE", "").strip()
    if "seed_tasks_per_type" not in user_config_keys and not seed_env:
        cur = int(config.get("seed_tasks_per_type", 0) or 0)
        if cur == 0:
            config["seed_tasks_per_type"] = 6
            applied_parts.append("seed_tasks_per_type=6")
    log.info(
        "AZR_PAPER_STYLE_DEFAULTS=1: applied hyperparameters for keys not set by caller: %s",
        ", ".join(applied_parts) if applied_parts else "(none — caller set all preset keys)",
    )


def _apply_azr_hf_env_trainer_hyperparams(config: Dict[str, Any]) -> None:
    """
    Apply AZR_HF_* (and optional AZR_SEED_TASKS_PER_TYPE) from the environment after defaults
    and the paper preset so explicit .env values always win.
    """
    log = logging.getLogger("AZR-HF-RL")
    applied: List[str] = []
    lr = _azr_optional_env_float("AZR_HF_LEARNING_RATE")
    if lr is not None:
        config["learning_rate"] = lr
        applied.append(f"learning_rate={lr}")
    clr = _azr_optional_env_float("AZR_HF_CRITIC_LEARNING_RATE")
    if clr is not None:
        config["critic_learning_rate"] = clr
        applied.append(f"critic_learning_rate={clr}")
    elif lr is not None:
        config["critic_learning_rate"] = lr
        applied.append("critic_learning_rate=<matched AZR_HF_LEARNING_RATE>")
    gs = _azr_optional_env_positive_int("AZR_HF_GENERATION_STEPS_PER_EPOCH")
    if gs is not None:
        config["generation_steps_per_epoch"] = gs
        applied.append(f"generation_steps_per_epoch={gs}")
    bs = _azr_optional_env_positive_int("AZR_HF_BATCH_SIZE")
    if bs is not None:
        config["batch_size"] = bs
        applied.append(f"batch_size={bs}")
    ppo = _azr_optional_env_positive_int("AZR_HF_PPO_UPDATE_THRESHOLD")
    if ppo is not None:
        config["ppo_update_threshold"] = ppo
        applied.append(f"ppo_update_threshold={ppo}")
    st = _azr_optional_env_int_allow_zero("AZR_SEED_TASKS_PER_TYPE")
    if st is not None:
        config["seed_tasks_per_type"] = st
        applied.append(f"seed_tasks_per_type={st}")
    if applied:
        log.info("Applied AZR_HF_* / AZR_SEED_TASKS_PER_TYPE trainer hyperparameters: %s", ", ".join(applied))


def configure_training_logging(use_rich: bool = True) -> None:
    """Configure logging for the training process."""
    root_logger = logging.getLogger()

    # Remove any previously attached handlers to prevent duplicates.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    class _AnsiSafeFormatter(logging.Formatter):
        def format(self, record):
            # Expand msg % args once, then clear args. Otherwise Formatter.format
            # calls getMessage() again and applies % to an already-interpolated msg,
            # which raises TypeError when args remain non-empty.
            merged = record.getMessage()
            record.msg = _strip_ansi(str(merged))
            record.args = ()
            return super().format(record)

    if use_rich and _RICH_AVAILABLE:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(_AnsiSafeFormatter("%(name)s - %(levelname)s - %(message)s"))
        if Console is not None:
            Console(highlight=False, force_terminal=True)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    trainer_logger = logging.getLogger("AZR-HF-RL")
    trainer_logger.handlers = []
    trainer_logger.propagate = True
    trainer_logger.setLevel(logging.INFO)


def print_real_time_status(epoch, step, total_steps, exp_count, prop_reward, solv_reward, buf_size, buf_threshold, diff_level, elapsed_time=None):
    """Print a real-time training status with enhanced formatting"""
    # Clear previous line if in terminal
    print(f"\r{Fore.WHITE}┌─ {Fore.CYAN} Real-Time Status {Fore.WHITE}─" + "─" * 50 + "┐", end="")

    # Calculate progress percentage
    progress = (step / total_steps) * 100 if total_steps > 0 else 0

    # Format the status line
    status_line = (
        f"\r{Fore.CYAN}│{Fore.WHITE} Epoch: {epoch:<3} │ "
        f"{Fore.GREEN}Step: {step:<2}/{total_steps:<2} ({progress:>5.1f}%) │ "
        f"{Fore.YELLOW}Exp: {exp_count:<3} │ "
        f"{Fore.MAGENTA}Prop: {prop_reward:<5} │ "
        f"{Fore.BLUE}Solv: {solv_reward:<5} │ "
        f"{Fore.CYAN}Buf: {buf_size:<2}/{buf_threshold:<2} │ "
        f"{Fore.GREEN}Diff: {diff_level:<2} {Fore.CYAN}│"
    )

    # Add elapsed time if provided
    if elapsed_time:
        status_line += f" {Fore.WHITE}Elapsed {elapsed_time}s{Fore.CYAN} │"

    # Fill remaining space and close the box
    remaining_space = 80 - len(status_line.replace(Fore.WHITE, '').replace(Fore.CYAN, '').replace(Fore.GREEN, '').replace(Fore.YELLOW, '').replace(Fore.MAGENTA, '').replace(Fore.BLUE, '')) - 2
    if remaining_space > 0:
        status_line += " " * remaining_space

    print(status_line, end="", flush=True)

    # Print bottom border
    print(f"\r{Fore.WHITE}└" + "─" * 78 + "┘", flush=True)
# Enhanced logging setup with better formatting
logger = logging.getLogger("AZR-HF-RL")
logger.setLevel(logging.WARNING) # Reduced from INFO to WARNING to reduce clutter

# Create custom formatter with colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Reset color at start
        formatted = f"{Style.RESET_ALL}"

        # Add colors based on level
        if record.levelno >= logging.ERROR:
            formatted += f"{Fore.RED} ERROR{Style.RESET_ALL}"
        elif record.levelno >= logging.WARNING:
            formatted += f"{Fore.YELLOW}  WARNING{Style.RESET_ALL}"
        elif record.levelno >= logging.INFO:
            formatted += f"{Fore.BLUE}  INFO{Style.RESET_ALL}"
        else:
            formatted += f"{Fore.WHITE} DEBUG{Style.RESET_ALL}"

        # Add the actual message
        formatted += f" {record.getMessage()}"

        return formatted

# Apply custom formatter
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)
logger.propagate = False  # Prevent duplicate logs

# AZR Components - MOVED UP
from azr_hf_adapter import HuggingFaceAdapter, ValueModel # Ensure ValueModel is imported
from hf_reward_manager import HFRewardManager
from experience_buffer import ExperienceBuffer, RLExperience # Assuming this will be reused/adapted
from azr_common_utils import (
    _clean_input_string,
    _evaluate_input,
    _extract_code_from_solution,
    _compare_outputs,
    _calculate_ast_complexity,
    contains_banned_imports
)
# Removed PeftModel, LoraConfig etc. as adapter handles model loading.
# If PEFT/LoRA is applied *during* training by the trainer, these might be needed later.

# Import PPO utility functions
from hf_ppo_utils import get_model_outputs_for_ppo, calculate_gae, perform_ppo_update

# Import prompt utility functions
from hf_prompt_utils import create_proposer_prompt, create_solver_prompt

# Import parsing utility functions
from hf_parsing_utils import parse_generated_tasks, find_json_objects, is_valid_task_structure

# Import callback utility functions
from hf_trainer_callbacks import (
    update_problem_type_weights,
    update_curriculum_difficulty,
    save_checkpoint,
    load_checkpoint
)

# Import new advanced modules
from hf_training_metrics import AdvancedTrainingMetrics
from hf_context_progressive import ProgressiveContextTrainer, create_context_schedule
from hf_training_step_trace import (
    log_training_step_detail,
    resolve_step_trace_log_path,
    summarize_execution_for_trace,
    train_step_trace_enabled,
    train_step_trace_max_chars,
)
from hf_curriculum_learning import CurriculumLearningManager, create_curriculum_config
from hf_dataset_manager import DatasetManager # Import the sophisticated DatasetManager
from hf_training.checkpoint_state import score_checkpoint_from_metrics

# Import real CodeExecutor
from code_executor import CodeExecutor

# Add torch.autograd.set_detect_anomaly(True) for debugging NaN gradients
# torch.autograd.set_detect_anomaly(True) # Keep this commented out for now, enable if issue persists severely

# Initialize colorama
init(autoreset=True)

# Set up logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        if record.levelname != 'DEBUG':  # Only color non-debug messages
            log_color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
            record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Configure logging with colored formatter
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.handlers = [handler]
root_logger.setLevel(logging.INFO)  # Set root logger to INFO

# Keep debug level for specific loggers if needed
debug_loggers = ['hf_action_value_utils', 'hf_generation_utils', 'hf_ppo_utils', 'hf_parsing_utils']
for logger_name in debug_loggers:
    # Changed from DEBUG to WARNING to reduce clutter as requested by user
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Remove Regex for parsing tasks and find_json_objects helper
# They are now in hf_parsing_utils.py

class HuggingFaceRLTrainer:
    def __init__(
        self,
        hf_adapter: HuggingFaceAdapter,         
        experience_buffer: ExperienceBuffer,
        hf_reward_manager: HFRewardManager,   
        python_executor: Any,                 
        config: Dict = None
    ):
        config = config or {}
        user_config_keys = set(config.keys())
        seed_value = config.get("seed", 42)
        self.adapter = hf_adapter             
        self.experience_buffer = experience_buffer 
        self.reward_manager = hf_reward_manager 
        self.python_executor = python_executor
        
        # Add checkpoint management
        self.checkpoint_scores = {}  # Track checkpoint scores for pruning
        self.max_checkpoints = 3  # Keep only best N checkpoints
        
        self.config = {
            "rl_epochs": 52,
            "batch_size": 16, 
            "max_new_tokens": 512, 
            "temperature": 0.2, 
            "top_p": 0.95,       
            "temperature_range": (0.2, 0.2), 
            "top_p_range": (0.95, 0.95),      
            "learning_rate": 1e-8, # Drastically reduced from 5e-7         
            "critic_learning_rate": 1e-8, # Drastically reduced from 1e-7    
            "ppo_clip_epsilon": 0.2,         
            "value_clip_epsilon": 0.2,       
            "gamma": 1.0,                    
            "lambda_gae": 1.0,               
            "value_loss_coef": 0.5,          
            "entropy_coef": 0.001,           
            "max_grad_norm": 0.1, # Reduced from 1.0           
            "problem_types": ["deduction", "abduction", "induction"],
            "evaluation_interval": 5,
            "checkpoint_interval": 4, # Reduced from 10 to 1 for frequent saving
            "checkpoint_dir": "hf_checkpoints",
            "seed": seed_value,
            "current_difficulty": 3, 
            "use_curriculum": True, 
            "min_task_difficulty": 1,
            "max_task_difficulty": 5,
            "problem_type_weights": None,
            "generation_steps_per_epoch": 10, 
            "ppo_update_threshold": 64, 
            "proposer_num_return_sequences": 8,
            "solver_num_return_sequences": 1,
            "k_reference": 6,
            "ppo_epochs": 2,
            "log_probs_key": "log_probs", 
            "values_key": "values",       
            "advantages_key": "advantages", 
            "returns_key": "returns",        
            "save_buffers_in_checkpoint": False, # Keep checkpoints lighter by default
            "debug_log_scaled_gradients": False, # Disable by default for cleaner output
            "detect_anomaly": False, # Disable anomaly detection by default to reduce overhead
            "gpu_memory_fraction": 0.85, # Limit torch CUDA memory usage by default
            "cuda_alloc_config": "max_split_size_mb:128,garbage_collection_threshold:0.8",
            "log_gradients": False, # ADDED: Control unscaled, clipped gradient logging
            "log_prob_diff_clamp_min": -3.0,
            "log_prob_diff_clamp_max": 3.0,
            "save_frequency": 10,  # Save every N epochs instead of 5
            # Progressive context parameters
            "initial_context_length": 256,
            "target_context_length": 1024,
            "context_expansion_epochs": 10,
            "context_strategy": "performance_gated",
            # Curriculum learning parameters
            "initial_difficulty": "BEGINNER",
            "target_difficulty": "HARD",
            "curriculum_epochs": 15,
            "curriculum_threshold": 0.7,
            # Metrics tracking
            "metrics_dir": "./training_metrics",
            # Timestamped launcher run dir (optional); also read from AZR_RUN_LOG_DIR env
            "run_log_dir": "",
        }
        if config:
            self.config.update(config)

        _maybe_apply_azr_paper_style_defaults(self.config, user_config_keys)
        _apply_azr_hf_env_trainer_hyperparams(self.config)

        self.config.setdefault("run_log_dir", os.environ.get("AZR_RUN_LOG_DIR", "").strip())
        self._step_trace_log_path = resolve_step_trace_log_path(self.config.get("run_log_dir"))
        
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed"])
            
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        self.device = self.adapter.device 
        logger.info(f"{Fore.CYAN}Using device: {self.device} (from adapter){Style.RESET_ALL}")

        # Track training wall-clock start time for end-of-run summaries.
        self.start_time = time.time()

        # Initialize scaler for mixed precision (initially None)
        self.scaler = None
        
        # Parameter collection for optimizer
        self.actor_params = []
        self.critic_params = []

        if self.adapter.use_separate_value_model:
            logger.info("Adapter is using separate actor and critic models.")
            if not hasattr(self.adapter, 'actor_model') or self.adapter.actor_model is None:
                raise ValueError("Adapter indicates use_separate_value_model=True but actor_model is missing.")
            if not hasattr(self.adapter, 'critic_model') or self.adapter.critic_model is None:
                raise ValueError("Adapter indicates use_separate_value_model=True but critic_model is missing.")
            
            self.actor_params.extend(list(self.adapter.actor_model.parameters()))
            self.critic_params.extend(list(self.adapter.critic_model.parameters())) # Train all critic params for now
            
            optimizer_grouped_parameters = [
                {'params': self.actor_params, 'lr': self.config["learning_rate"]},
                {'params': self.critic_params, 'lr': self.config.get("critic_learning_rate", self.config["learning_rate"])}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters)
            logger.info(f"Optimizer initialized with AdamW for separate actor (LR: {self.config['learning_rate']}) and critic (LR: {self.config.get('critic_learning_rate', self.config['learning_rate'])}).")

        else: 
            logger.warning("Adapter is NOT using separate actor/critic. PPO typically requires this. Make sure this is intended.")
            if hasattr(self.adapter, 'model') and self.adapter.model is not None:
                # This path would be for a single, potentially unified actor-critic model, or non-PPO.
                self.optimizer = AdamW(self.adapter.model.parameters(), lr=self.config["learning_rate"])
                logger.info(f"Optimizer initialized with AdamW for a single model (LR: {self.config['learning_rate']}).")
            else:
                logger.error("Adapter not configured for separate models, but no general self.adapter.model found!")
                raise ValueError("Invalid adapter configuration for optimizer setup.")

        self.metrics = {
            "epoch": 0,
            "proposer_rewards": [], 
            "solver_rewards": [],   
            "policy_loss": [],      
            "value_loss": [],       
            "entropy": [],          
            "total_experiences": 0,
            "total_valid_tasks": 0,
            "problem_type_counts": {pt: 0 for pt in self.config["problem_types"]},
            "difficulty_level": self.config.get("current_difficulty", self.config["min_task_difficulty"]),
            "avg_proposer_reward_components": defaultdict(float),
            "avg_solver_reward_components": defaultdict(float),
            "training_start_time": time.time()
        }
        # For per-run epoch_summary.jsonl (valid tasks added this epoch vs cumulative).
        self._epoch_summary_prev_valid_tasks = 0
        if self.config["problem_type_weights"] is None:
            self.config["problem_type_weights"] = {
                pt: 1.0 / len(self.config["problem_types"]) for pt in self.config["problem_types"]
            }
        self.problem_type_success_rates = {pt: 0.5 for pt in self.config["problem_types"]}
        self.current_difficulty = self.config.get("current_difficulty", self.config["min_task_difficulty"])
        self.reward_manager.update_difficulty(self.current_difficulty)

        self.ppo_buffer: Dict[str, Deque[Any]] = defaultdict(lambda: deque(maxlen=self.config["ppo_update_threshold"] * 2)) 
        self.ppo_buffer_keys = ["prompts_ids", "prompts_mask", "generated_ids", "generated_mask", 
                                "rewards", "values", "log_probs", "dones", "advantages", "returns"]

        # Initialize advanced features
        self._initialize_advanced_features()
        
        logger.info(f"{Fore.GREEN}HuggingFaceRLTrainer initialized with adapter for model: {self.adapter.model_name}{Style.RESET_ALL}")

        self.num_ppo_epochs = config.get("num_ppo_epochs", 4)
        self.clip_param = config.get("clip_param", 0.2)
        self.vf_coeff = config.get("vf_coeff", 0.5)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.target_kl = config.get("target_kl", None) # 0.15 in original paper, but often None
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1) # TODO: Implement if needed

        self.ppo_buffer_ready_for_update = False
        self.total_ppo_updates = 0
        self.current_epoch_ppo_updates = 0
        self.step_counter = 0

        # Memory optimization settings
        self.empty_cache_frequency = self.config.get("empty_cache_frequency", 5)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        # Enable gradient checkpointing if requested (must target the module that actually runs forward).
        if self.config.get("gradient_checkpointing", False):
            gc_target = None
            if self.adapter.use_separate_value_model and self.adapter.actor_model is not None:
                gc_target = self.adapter.actor_model
            elif getattr(self.adapter, "model", None) is not None:
                gc_target = self.adapter.model
            if gc_target is not None and hasattr(gc_target, "gradient_checkpointing_enable"):
                gc_target.gradient_checkpointing_enable()
                if hasattr(gc_target, "enable_input_require_grads"):
                    gc_target.enable_input_require_grads()
                logger.info(f"{Fore.BLUE} Gradient checkpointing enabled on {gc_target.__class__.__name__}{Style.RESET_ALL}")
            elif self.config.get("gradient_checkpointing", False):
                logger.warning(
                    "gradient_checkpointing is set in config but no actor/main model was found to enable it."
                )

        # Setup scaler for mixed precision if requested
        self.scaler = None
        if self.config.get("mixed_precision", False) and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()
            logger.info(f"{Fore.BLUE} Mixed precision training enabled{Style.RESET_ALL}")
        self.current_epoch_experiences_processed_in_ppo = 0

        alloc_config = self.config.get("cuda_alloc_config", "")
        if alloc_config:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_config)
        self._configure_cuda_memory()
        self._cleanup_memory(force=True)

    def _cleanup_memory(self, force: bool = False):
        """Aggressive memory cleanup to prevent OOM errors"""
        self.step_counter += 1

        # Empty CUDA cache periodically or when forced
        if force or (self.step_counter % self.empty_cache_frequency == 0):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.debug(f"{Fore.YELLOW} CUDA cache emptied (step {self.step_counter}){Style.RESET_ALL}")

        # Force garbage collection
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _configure_cuda_memory(self):
        """Configure CUDA memory limits and allocator behavior."""
        if not torch.cuda.is_available():
            return

        requested_fraction = float(self.config.get("gpu_memory_fraction", 0.85))
        requested_fraction = min(1.0, max(0.05, requested_fraction))
        try:
            torch.cuda.set_per_process_memory_fraction(requested_fraction)
            logger.info(f"{Fore.BLUE} Per-process CUDA memory capped to {requested_fraction:.2f}{Style.RESET_ALL}")
        except Exception as e:
            logger.warning(f"{Fore.YELLOW} Could not set CUDA memory fraction: {e}{Style.RESET_ALL}")

    def _release_training_resources(self):
        """Release model resources and clear memory caches after training."""
        logger.info(f"{Fore.YELLOW}Releasing training resources...{Style.RESET_ALL}")

        actor_model = getattr(self.adapter, "actor_model", None)
        critic_model = getattr(self.adapter, "critic_model", None)
        main_model = getattr(self.adapter, "model", None)

        for model in [actor_model, critic_model, main_model]:
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                try:
                    del model
                except Exception:
                    pass

        self.adapter.actor_model = None
        self.adapter.critic_model = None
        self.adapter.model = None

        self.actor_params = []
        self.critic_params = []

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _get_model_max_position(self, config_obj: Optional[Any], default_len: int = 2048) -> int:
        """Return max position length for a model config, supporting multi-config wrappers."""
        if config_obj is None:
            return default_len

        candidate_keys = (
            "max_position_embeddings",
            "max_positions",
            "max_sequence_length",
            "n_positions",
            "seq_length",
            "max_len",
        )

        for key in candidate_keys:
            if hasattr(config_obj, key):
                value = getattr(config_obj, key)
                if isinstance(value, int) and value > 0:
                    return value

        # Gemma 4 and some wrapper configs store sequence limits on nested text_config.
        text_config = getattr(config_obj, "text_config", None)
        if text_config is not None:
            for key in candidate_keys:
                if hasattr(text_config, key):
                    value = getattr(text_config, key)
                    if isinstance(value, int) and value > 0:
                        return value

        return default_len

    def _initialize_advanced_features(self):
        """Initialize advanced training features"""
        
        # Progressive Context Length Training
        context_config = create_context_schedule(
            initial=self.config.get('initial_context_length', 256),
            target=self.config.get('target_context_length', 1024),
            epochs=self.config.get('context_expansion_epochs', 10),
            strategy=self.config.get('context_strategy', 'performance_gated')
        )
        # Get model max length
        if self.adapter.use_separate_value_model and hasattr(self.adapter, "actor_model") and self.adapter.actor_model is not None:
            model_max_len = self._get_model_max_position(getattr(self.adapter.actor_model, "config", None))
        elif hasattr(self.adapter, "model") and self.adapter.model is not None:
            model_max_len = self._get_model_max_position(getattr(self.adapter.model, "config", None))
        else:
            model_max_len = 2048  # Default
        
        self.context_trainer = ProgressiveContextTrainer(context_config, model_max_len)
        
        # Curriculum Learning
        curriculum_config = create_curriculum_config(
            initial_level=self.config.get('initial_difficulty', 'BEGINNER'),
            target_level=self.config.get('target_difficulty', 'HARD'),
            epochs=self.config.get('curriculum_epochs', 15),
            success_threshold=self.config.get('curriculum_threshold', 0.7)
        )
        self.curriculum_manager = CurriculumLearningManager(curriculum_config)
        
        # Advanced Training Metrics
        self.metrics_tracker = AdvancedTrainingMetrics(
            save_dir=self.config.get('metrics_dir', './training_metrics')
        )
        
        logger.info(f"{Fore.CYAN}Advanced features initialized:{Style.RESET_ALL}")
        logger.info(f"  - Progressive context: {context_config.initial_context} → {context_config.target_context}")
        logger.info(f"  - Curriculum learning: {curriculum_config.initial_difficulty.name} → {curriculum_config.target_difficulty.name}")
        logger.info(f"  - Metrics tracking to: {self.config.get('metrics_dir')}")

    def _get_temperature(self, epoch: int) -> float:
        """Get temperature based on current epoch (linear decay)"""
        temp_range = self.config.get("temperature_range", (0.7, 0.5)) # Default if not in config
        start_temp, end_temp = temp_range
        epochs = self.config["rl_epochs"]
        return start_temp - (start_temp - end_temp) * min(1.0, epoch / epochs)

    def _get_top_p(self, epoch: int) -> float:
        """Get top_p based on current epoch (linear decay)"""
        top_p_range = self.config.get("top_p_range", (0.95, 0.8)) # Default if not in config
        start_p, end_p = top_p_range
        epochs = self.config["rl_epochs"]
        return start_p - (start_p - end_p) * min(1.0, epoch / epochs)

    def _proposer_generation_kwargs(self, epoch: int) -> Dict[str, Any]:
        """
        Sampling hyperparameters for the task proposer only.

        Curriculum defaults use high temperatures (e.g. 0.7) that break structured ```json```
        proposals; when env overrides are unset, cap to moderate values (configurable).

        Env (optional): AZR_PROPOSER_TEMPERATURE, AZR_PROPOSER_TOP_P, AZR_PROPOSER_DO_SAMPLE
        (truthy/falsey for the latter).
        """
        curriculum_temp = self._get_temperature(epoch)
        curriculum_top_p = self._get_top_p(epoch)
        env_temp = _azr_optional_env_float("AZR_PROPOSER_TEMPERATURE")
        env_top_p = _azr_optional_env_float("AZR_PROPOSER_TOP_P")
        env_sample = _azr_optional_env_bool("AZR_PROPOSER_DO_SAMPLE")
        safe_temp_cap = float(self.config.get("proposer_temperature_cap", 0.25))
        safe_top_p_cap = float(self.config.get("proposer_top_p_cap", 0.92))

        temperature = curriculum_temp if env_temp is None else env_temp
        top_p = curriculum_top_p if env_top_p is None else env_top_p
        if env_temp is None:
            temperature = min(temperature, safe_temp_cap)
        if env_top_p is None:
            top_p = min(top_p, safe_top_p_cap)

        temperature = float(max(1e-6, temperature))
        top_p = float(max(1e-6, min(1.0, top_p)))
        do_sample = True if env_sample is None else env_sample

        return {"temperature": temperature, "top_p": top_p, "do_sample": do_sample}

    def _append_run_epoch_summary_jsonl(
        self,
        epoch: int,
        epoch_metrics: Dict[str, Any],
        num_generation_steps: int,
    ) -> None:
        """One compact JSON line per epoch under run_log_dir (e.g. launcher sets AZR_RUN_LOG_DIR)."""
        run_dir = (self.config.get("run_log_dir") or "").strip()
        if not run_dir:
            return
        valid_total = int(self.metrics.get("total_valid_tasks", 0))
        valid_delta = valid_total - self._epoch_summary_prev_valid_tasks
        self._epoch_summary_prev_valid_tasks = valid_total
        exp_total = int(self.metrics.get("total_experiences", 0))
        gen_steps = max(1, int(num_generation_steps))
        parse_rate = min(1.0, max(0.0, valid_delta / float(gen_steps)))
        row = {
            "epoch": epoch,
            "proposer_r": float(epoch_metrics.get("mean_reward_proposer", 0.0)),
            "solver_r": float(epoch_metrics.get("mean_reward_solver", 0.0)),
            "policy_loss": float(epoch_metrics.get("policy_loss", 0.0)),
            "value_loss": float(epoch_metrics.get("value_loss", 0.0)),
            "total_loss": float(epoch_metrics.get("total_loss", 0.0)),
            "experiences": exp_total,
            "valid_tasks_total": valid_total,
            "valid_tasks_epoch": valid_delta,
            "parse_rate": parse_rate,
            "plateau_count": int(getattr(self.metrics_tracker, "plateau_count", 0)),
            "best_reward": float(br) if (br := getattr(self.metrics_tracker, "best_reward", None)) is not None and not __import__('math').isnan(br) and not __import__('math').isinf(br) else None,
            "best_epoch": int(getattr(self.metrics_tracker, "best_epoch", 0)),
        }
        path = Path(run_dir) / "epoch_summary.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def train_epoch(self, epoch: int) -> Dict:
        # Set epoch-specific seed to ensure diversity across epochs and deterministic resumption
        # If we restart at epoch 10, we want the same behavior as if we reached it naturally,
        # but distinct from epoch 0.
        epoch_seed = self.config["seed"] + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        logger.info(f"Set random seed for epoch {epoch} to {epoch_seed}")

        # Start metrics tracking for this epoch
        self.metrics_tracker.start_epoch()
        
        self.metrics["epoch"] = epoch
        epoch_total_proposer_reward = 0
        epoch_total_solver_reward = 0
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        # Get adaptive batch size based on context length
        base_batch_size = self.config.get("batch_size", 16)
        current_batch_size = self.context_trainer.get_adaptive_batch_size(base_batch_size)
        
        # Update num_generation_steps based on adaptive batch size
        num_generation_steps = max(
            self.config.get("generation_steps_per_epoch", 10),
            self.config["ppo_update_threshold"] // current_batch_size
        )
        
        logger.info(f"{Fore.CYAN}Context length: {self.context_trainer.current_context_length}, "
                   f"Adaptive batch size: {current_batch_size}{Style.RESET_ALL}")
        
        if self.adapter.use_separate_value_model:
            self.adapter.actor_model.eval()
            self.adapter.critic_model.eval() 
        elif hasattr(self.adapter, 'model') and self.adapter.model is not None:
            self.adapter.model.eval()

        # Create enhanced progress bar with detailed formatting
        pbar_generation = tqdm(range(num_generation_steps),
                              desc=f"{Fore.CYAN} Epoch {epoch} - Generation{Style.RESET_ALL}",
                              position=0, leave=True,
                              bar_format=f"{Fore.CYAN}{{desc}}{Style.RESET_ALL}: {Fore.GREEN}{{percentage:3.0f}}%{Style.RESET_ALL}|{{bar}}| {Fore.YELLOW}{{n}}/{{total}}{Style.RESET_ALL} [{{elapsed}}<{{remaining}}] {{postfix}}")
        
        for step in pbar_generation:
            t_step_wall = time.perf_counter()
            current_temp = self._get_temperature(epoch)
            current_top_p = self._get_top_p(epoch)

            problem_types = self.config["problem_types"]
            problem_type_weights_list = [self.config["problem_type_weights"][pt] for pt in problem_types]
            chosen_problem_type = random.choices(problem_types, weights=problem_type_weights_list, k=1)[0]
            self.metrics["problem_type_counts"][chosen_problem_type] += 1
            
            # Update curriculum if enabled
            if self.config.get("use_curriculum", True):
                # Sample task from curriculum - just update difficulty level
                # task_params = self.curriculum_manager.get_task_for_generation(chosen_problem_type)
                # task_params could include difficulty hints, complexity targets, etc.
                # For now, just update the difficulty level
                self.current_difficulty = self.curriculum_manager.current_difficulty.value
                self.reward_manager.update_difficulty(self.current_difficulty)
            
            proposer_prompt_text = create_proposer_prompt(
                self,
                chosen_problem_type,
                seed_tasks=getattr(self.dataset_manager, "seed_buffer", {}).get(chosen_problem_type, []) if hasattr(self, "dataset_manager") else None,
                k_reference=self.config.get("k_reference", 6),
            )
            
            # Use progressive context length for tokenization
            max_prompt_length = min(
                self.config.get("max_prompt_length_proposer", 512),
                self.context_trainer.current_context_length
            )
            
            proposer_prompt_inputs = self.adapter.tokenizer(
                proposer_prompt_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_prompt_length
            )
            
            proposer_prompt_ids_batch = proposer_prompt_inputs.input_ids.to(self.device) # Shape (1, prompt_len)
            proposer_prompt_mask_batch = proposer_prompt_inputs.attention_mask.to(self.device)

            max_new_tokens_proposer = self.config.get("max_new_tokens_proposer", 256)
            
            # Get model_max_len safely
            if self.adapter.use_separate_value_model and hasattr(self.adapter, "actor_model") and self.adapter.actor_model is not None:
                model_max_len = self._get_model_max_position(getattr(self.adapter.actor_model, "config", None))
            elif hasattr(self.adapter, "model") and self.adapter.model is not None:
                model_max_len = self._get_model_max_position(getattr(self.adapter.model, "config", None))
            else:
                logger.warning("Cannot determine model_max_len from adapter. Defaulting to 2048.")
                model_max_len = 2048
            
            safety_margin = self.config.get("generation_safety_margin", 20) 
            max_prompt_len_for_generation = model_max_len - max_new_tokens_proposer - safety_margin
            if max_prompt_len_for_generation <= 0:
                max_prompt_len_for_generation = self.config.get("min_max_prompt_len_for_generation", 64)
                logger.warning(f"Calculated max_prompt_len_for_proposer_generation ({max_prompt_len_for_generation}) is low or negative. Using {max_prompt_len_for_generation}.")

            proposer_num_return_sequences = max(1, int(self.config.get("proposer_num_return_sequences", 8)))
            proposer_gen_kw = self._proposer_generation_kwargs(epoch)
            try:
                generated_task_texts = self.adapter.generate(
                    prompt=proposer_prompt_text,
                    max_new_tokens=max_new_tokens_proposer,
                    num_return_sequences=proposer_num_return_sequences,
                    max_prompt_length=max_prompt_len_for_generation,
                    **proposer_gen_kw,
                )
            except Exception as e:
                if proposer_num_return_sequences == 1:
                    raise
                logger.warning(
                    "Proposer generation with num_return_sequences=%s failed (%s); falling back to single sequence.",
                    proposer_num_return_sequences,
                    e,
                )
                generated_task_texts = self.adapter.generate(
                    prompt=proposer_prompt_text,
                    max_new_tokens=max_new_tokens_proposer,
                    num_return_sequences=1,
                    max_prompt_length=max_prompt_len_for_generation,
                    **proposer_gen_kw,
                )
            if not generated_task_texts:
                raise RuntimeError("No proposer samples returned")
            generated_task_text = generated_task_texts[0]
            
            # --- NEW: Log content periodically to show improvement/diversity ---
            if step % 5 == 0:  # Log every 5 steps
                logger.info(f"\n{Fore.MAGENTA} Step {step}: Generated Task Proposal:{Style.RESET_ALL}")
                logger.info(f"{Fore.WHITE}{generated_task_text[:500]}...{Style.RESET_ALL}") 
            # -------------------------------------------------------------------

            proposer_action_inputs = self.adapter.tokenizer(generated_task_text, return_tensors="pt", padding=True, truncation=True, max_length=max_new_tokens_proposer)
            proposer_action_ids_batch = proposer_action_inputs.input_ids.to(self.device) # Shape (1, action_len)
            proposer_action_mask_batch = proposer_action_inputs.attention_mask.to(self.device)

            proposer_full_sequence_ids = torch.cat([proposer_prompt_ids_batch, proposer_action_ids_batch], dim=1)
            proposer_full_sequence_mask = torch.cat([proposer_prompt_mask_batch, proposer_action_mask_batch], dim=1)
            
            # Truncate if exceeding model max length
            if proposer_full_sequence_ids.shape[1] > model_max_len:
                logger.warning(f"Proposer full sequence length {proposer_full_sequence_ids.shape[1]} exceeds model max {model_max_len}. Truncating.")
                proposer_full_sequence_ids = proposer_full_sequence_ids[:, :model_max_len]
                proposer_full_sequence_mask = proposer_full_sequence_mask[:, :model_max_len]
                # Adjust action_ids part if full sequence was truncated
                current_prompt_len = proposer_prompt_ids_batch.shape[1]
                new_action_len = max(0, model_max_len - current_prompt_len)
                # Ensure action_ids are taken from the original proposer_action_ids_batch before cat, then truncated
                proposer_action_ids_batch_truncated = proposer_action_ids_batch[:,:new_action_len]
                proposer_action_mask_batch_truncated = proposer_action_mask_batch[:,:new_action_len]
            else:
                proposer_action_ids_batch_truncated = proposer_action_ids_batch
                proposer_action_mask_batch_truncated = proposer_action_mask_batch


            parsed_tasks = parse_generated_tasks(self, generated_task_text, chosen_problem_type) # Call imported function
            proposer_reward = 0 # Initialize proposer reward for this step

            if not parsed_tasks:
                # Call calculate_proposer_reward with generated_task=None (or a dict indicating failure)
                # and empty solver_attempts_results
                proposer_experience_tuple = self.reward_manager.calculate_proposer_reward(
                    generated_task=None, # Or {"parsing_failed": True} if preferred by reward_manager
                    proposer_raw_output=generated_task_text,
                    solver_attempts_results=[] # No solver attempts if task parsing failed
                )
                proposer_reward = proposer_experience_tuple[0] # Reward is the first element
                proposer_reward_components = proposer_experience_tuple[1]

                with torch.no_grad():
                    _, proposer_old_log_prob_seq, proposer_old_value = get_model_outputs_for_ppo(
                        self, # Pass trainer instance
                        full_sequence_ids=proposer_full_sequence_ids,
                        full_sequence_mask=proposer_full_sequence_mask,
                        action_ids=proposer_action_ids_batch_truncated, 
                        action_mask=proposer_action_mask_batch_truncated
                    )
                if proposer_old_log_prob_seq is not None and proposer_old_value is not None:
                    self._store_experience_in_buffer(proposer_prompt_ids_batch, proposer_action_ids_batch_truncated, proposer_old_log_prob_seq, proposer_old_value, proposer_reward, proposer_full_sequence_ids)
                    # Use the components from the tuple
                    for k, v in proposer_reward_components.items(): self.metrics["avg_proposer_reward_components"][k] += v
                    self.metrics["proposer_rewards"].append(proposer_reward) # Use the actual reward
                epoch_total_proposer_reward += proposer_reward
                if train_step_trace_enabled():
                    log_training_step_detail(
                        log_path=self._step_trace_log_path,
                        epoch=epoch,
                        step=step,
                        role="proposer",
                        problem_type=chosen_problem_type,
                        prompt_text=proposer_prompt_text,
                        model_raw_output=generated_task_text,
                        parse_ok=False,
                        parsed_task_count=0,
                        proposer_reward=float(proposer_reward),
                        reward_components=dict(proposer_reward_components),
                        elapsed_seconds=time.perf_counter() - t_step_wall,
                    )
            else: # Parsed tasks found
                task_data = parsed_tasks[0] # Process first valid task
                self.metrics["total_valid_tasks"] += 1
                task_data["problem_type"] = chosen_problem_type
                
                # Initial proposer reward (can be updated after solver)
                # For now, we don't have solver_attempts_results yet for this path.
                # This implies calculate_proposer_reward might need to be called again later,
                # or its learnability component will be based on an empty list initially.
                # Let's assume for now learnability is zero if solver_attempts_results is empty.
                proposer_experience_initial_tuple = self.reward_manager.calculate_proposer_reward(
                    generated_task=task_data,
                    proposer_raw_output=generated_task_text, # Pass the raw output too
                    solver_attempts_results=[] # Initially empty, can be updated
                )
                proposer_reward = proposer_experience_initial_tuple[0]
                proposer_initial_reward_components = proposer_experience_initial_tuple[1]
                
                with torch.no_grad():
                     _, proposer_old_log_prob_seq, proposer_old_value = get_model_outputs_for_ppo(
                        self, # Pass trainer instance
                        full_sequence_ids=proposer_full_sequence_ids,
                        full_sequence_mask=proposer_full_sequence_mask,
                        action_ids=proposer_action_ids_batch_truncated,
                        action_mask=proposer_action_mask_batch_truncated
                    )
                if proposer_old_log_prob_seq is not None and proposer_old_value is not None:
                    self._store_experience_in_buffer(proposer_prompt_ids_batch, proposer_action_ids_batch_truncated, proposer_old_log_prob_seq, proposer_old_value, proposer_reward, proposer_full_sequence_ids)
                    for k, v in proposer_initial_reward_components.items(): self.metrics["avg_proposer_reward_components"][k] += v
                    self.metrics["proposer_rewards"].append(proposer_reward)
                epoch_total_proposer_reward += proposer_reward
                if train_step_trace_enabled():
                    log_training_step_detail(
                        log_path=self._step_trace_log_path,
                        epoch=epoch,
                        step=step,
                        role="proposer",
                        problem_type=chosen_problem_type,
                        prompt_text=proposer_prompt_text,
                        model_raw_output=generated_task_text,
                        parse_ok=True,
                        parsed_task_count=len(parsed_tasks),
                        first_task_keys=list(task_data.keys()),
                        proposer_reward=float(proposer_reward),
                        reward_components=dict(proposer_initial_reward_components),
                        elapsed_seconds=time.perf_counter() - t_step_wall,
                    )

                # Solver attempts the task
                t_solver_wall = time.perf_counter()
                solver_prompt_text = create_solver_prompt(self, task_data) # Call imported function
                solver_prompt_inputs = self.adapter.tokenizer(solver_prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=self.config.get("max_prompt_length_solver", 1024))
                solver_prompt_ids_batch = solver_prompt_inputs.input_ids.to(self.device)
                solver_prompt_mask_batch = solver_prompt_inputs.attention_mask.to(self.device)
                
                max_new_tokens_solver = self.config.get("max_new_tokens_solver", 512)
                max_prompt_len_for_solver_gen = model_max_len - max_new_tokens_solver - safety_margin
                if max_prompt_len_for_solver_gen <= 0:
                    max_prompt_len_for_solver_gen = self.config.get("min_max_prompt_len_for_generation", 64)
                    logger.warning(f"Calculated max_prompt_len_for_solver_gen ({max_prompt_len_for_solver_gen}) is low or negative. Using {max_prompt_len_for_solver_gen}.")

                generated_solution_texts = self.adapter.generate(
                prompt=solver_prompt_text, max_new_tokens=max_new_tokens_solver,
                temperature=current_temp, top_p=current_top_p,
                num_return_sequences=max(1, int(self.config.get("solver_num_return_sequences", 1))),
                    max_prompt_length=max_prompt_len_for_solver_gen
                )
                generated_solution_text = generated_solution_texts[0]
                solver_action_inputs = self.adapter.tokenizer(generated_solution_text, return_tensors="pt", padding=True, truncation=True, max_length=max_new_tokens_solver)
                solver_action_ids_batch = solver_action_inputs.input_ids.to(self.device)
                solver_action_mask_batch = solver_action_inputs.attention_mask.to(self.device)

                solver_full_sequence_ids = torch.cat([solver_prompt_ids_batch, solver_action_ids_batch], dim=1)
                solver_full_sequence_mask = torch.cat([solver_prompt_mask_batch, solver_action_mask_batch], dim=1)

                if solver_full_sequence_ids.shape[1] > model_max_len:
                    logger.warning(f"Solver full sequence length {solver_full_sequence_ids.shape[1]} exceeds model max {model_max_len}. Truncating.")
                    solver_full_sequence_ids = solver_full_sequence_ids[:, :model_max_len]
                    solver_full_sequence_mask = solver_full_sequence_mask[:, :model_max_len]
                    current_solver_prompt_len = solver_prompt_ids_batch.shape[1]
                    new_solver_action_len = max(0, model_max_len - current_solver_prompt_len)
                    solver_action_ids_batch_truncated = solver_action_ids_batch[:, :new_solver_action_len]
                    solver_action_mask_batch_truncated = solver_action_mask_batch[:, :new_solver_action_len]
                else:
                    solver_action_ids_batch_truncated = solver_action_ids_batch
                    solver_action_mask_batch_truncated = solver_action_mask_batch
                
                # This is where the solver's attempt result would be generated.
                # For now, let's assume HFRewardManager.calculate_solver_reward
                # also returns a structure compatible with solver_attempts_results for the proposer.
                # And that the DummyExecutor provides a compatible dict.

                # The HFRewardManager.calculate_solver_reward expects `execution_result`
                # from the executor. Let's assume our DummyExecutor.solution_check provides that.
                
                # We need the output of the actual solution attempt by the dummy executor
                # based on `task_data` and `generated_solution_text`.
                
                # --- THIS PART NEEDS REFINEMENT for actual execution result for solver ---
                # For now, let's mock one attempt result for the proposer based on solver's main success
                # In a real scenario, `solver_attempts_results` would be populated by running the
                # generated_solution_text against task_data multiple times if n_samples > 1
                
                # Let's call the dummy executor to get a result for the solver's attempt
                # This result would then inform the proposer's learnability.
                
                # We need to adapt the HFRewardManager or trainer logic.
                # HFRewardManager.calculate_proposer_reward expects a list of solver attempt results.
                # HFRewardManager.calculate_solver_reward gives reward for ONE attempt.

                # For now, let's get the solver reward first.
                # We need an execution_result for the solver.
                # The dummy executor's `solution_check` directly provides this.
                mock_execution_result_for_solver = self.python_executor.solution_check(
                    solution_code=generated_solution_text, # Or rather, the code extracted from it
                    input_str=task_data.get("input"),
                    expected_output_str=task_data.get("output")
                )

                solver_experience_tuple = self.reward_manager.calculate_solver_reward(
                    solver_code_str=_extract_code_from_solution(generated_solution_text), 
                    task=task_data, 
                    execution_result=mock_execution_result_for_solver # Pass the dummy result
                )
                solver_reward = solver_experience_tuple[0]
                solver_reward_components = solver_experience_tuple[1]
                epoch_total_solver_reward += solver_reward
                self.metrics["solver_rewards"].append(solver_reward)
                for k, v in solver_reward_components.items(): self.metrics["avg_solver_reward_components"][k] += v

                if train_step_trace_enabled():
                    mc = train_step_trace_max_chars()
                    log_training_step_detail(
                        log_path=self._step_trace_log_path,
                        epoch=epoch,
                        step=step,
                        role="solver",
                        problem_type=chosen_problem_type,
                        prompt_text=solver_prompt_text,
                        model_raw_output=generated_solution_text,
                        proposer_reward=float(proposer_reward),
                        solver_reward=float(solver_reward),
                        execution_summary=summarize_execution_for_trace(
                            mock_execution_result_for_solver,
                            min(mc, 800),
                        ),
                        reward_components=dict(solver_reward_components),
                        elapsed_seconds=time.perf_counter() - t_solver_wall,
                        extra={"task_keys_preview": list(task_data.keys())[:12]},
                    )

                # Now, potentially update the proposer's reward based on this solver attempt.
                # This is the tricky part with the current HFRewardManager structure.
                # For simplicity, we might re-calculate proposer reward here, or the initial one was final.
                # Let's assume the initial proposer reward was final for now, and learnability was based on empty solver attempts.
                # A more complex loop would store the proposer experience *after* solver attempts.

                with torch.no_grad():
                    _, solver_old_log_prob_seq, solver_old_value = get_model_outputs_for_ppo(
                        self, # Pass trainer instance
                        full_sequence_ids=solver_full_sequence_ids,
                        full_sequence_mask=solver_full_sequence_mask,
                        action_ids=solver_action_ids_batch_truncated,
                        action_mask=solver_action_mask_batch_truncated
                    )
                if solver_old_log_prob_seq is not None and solver_old_value is not None:
                     self._store_experience_in_buffer(solver_prompt_ids_batch, solver_action_ids_batch_truncated, solver_old_log_prob_seq, solver_old_value, solver_reward, solver_full_sequence_ids)

            if len(self.ppo_buffer["rewards"]) >= self.config["ppo_update_threshold"]:
                # Memory cleanup before PPO update
                self._cleanup_memory(force=True)

                if self.adapter.use_separate_value_model:
                    self.adapter.actor_model.train()
                    self.adapter.critic_model.train()
                elif hasattr(self.adapter, 'model') and self.adapter.model is not None:
                    self.adapter.model.train()

                logger.info(f"{Fore.GREEN}Performing PPO update with {len(self.ppo_buffer['rewards'])} experiences.{Style.RESET_ALL}")
                ppo_metrics = perform_ppo_update(self) # Call the imported function

                # Memory cleanup after PPO update
                self._cleanup_memory(force=True)
                if ppo_metrics:
                    if "avg_policy_loss" in ppo_metrics: epoch_policy_losses.append(ppo_metrics["avg_policy_loss"])
                    if "avg_value_loss" in ppo_metrics: epoch_value_losses.append(ppo_metrics["avg_value_loss"])
                    if "avg_entropy" in ppo_metrics: epoch_entropies.append(ppo_metrics["avg_entropy"])
                    if ppo_metrics.get("valid_minibatch_updates", 0) == 0:
                        logger.warning(
                            "PPO update had zero valid minibatches this interval. "
                            f"skipped_minibatches={ppo_metrics.get('skipped_minibatches', 0)}. "
                            "Continuing training and waiting for valid experiences."
                        )

                if self.adapter.use_separate_value_model:
                    self.adapter.actor_model.eval()
                    self.adapter.critic_model.eval()
                elif hasattr(self.adapter, 'model') and self.adapter.model is not None:
                    self.adapter.model.eval()
            
            # Enhanced progress bar with colored emojis and better formatting
            exp_count = self.metrics['total_experiences']
            prop_reward = f"{np.mean(self.metrics['proposer_rewards'][-20:]) if self.metrics['proposer_rewards'] else 0:.2f}"
            solv_reward = f"{np.mean(self.metrics['solver_rewards'][-20:]) if self.metrics['solver_rewards'] else 0:.2f}"
            buf_status = f"{len(self.ppo_buffer['rewards'])}/{self.config['ppo_update_threshold']}"
            diff_level = str(self.current_difficulty)

            pbar_generation.set_postfix_str(
                f"{Fore.GREEN} {exp_count} {Fore.YELLOW} {prop_reward} {Fore.BLUE} {solv_reward} {Fore.MAGENTA} {buf_status} {Fore.CYAN} {diff_level}"
            )

        # Final PPO update for any remaining experiences
        if len(self.ppo_buffer["rewards"]) > 0:
            if self.adapter.use_separate_value_model:
                self.adapter.actor_model.train()
                self.adapter.critic_model.train()
            elif hasattr(self.adapter, 'model') and self.adapter.model is not None:
                self.adapter.model.train()
            logger.info(f"{Fore.GREEN}Performing final PPO update for epoch {epoch} with {len(self.ppo_buffer['rewards'])} experiences.{Style.RESET_ALL}")
            ppo_metrics = perform_ppo_update(self) # Call the imported function
            if ppo_metrics:
                if "avg_policy_loss" in ppo_metrics: epoch_policy_losses.append(ppo_metrics["avg_policy_loss"])
                if "avg_value_loss" in ppo_metrics: epoch_value_losses.append(ppo_metrics["avg_value_loss"])
                if "avg_entropy" in ppo_metrics: epoch_entropies.append(ppo_metrics["avg_entropy"])
                if ppo_metrics.get("valid_minibatch_updates", 0) == 0:
                    logger.warning(
                        f"Final epoch PPO update had zero valid minibatches (skipped_minibatches={ppo_metrics.get('skipped_minibatches', 0)}). "
                        "Continuing training unless stop criteria are met."
                    )
        
        # Update curriculum difficulty and problem type weights at the end of the epoch
        update_curriculum_difficulty(self, epoch) # Call imported function
        update_problem_type_weights(self)      # Call imported function

        # Aggregate and log epoch metrics
        self.metrics["avg_policy_loss"] = np.mean(epoch_policy_losses) if epoch_policy_losses else 0
        self.metrics["avg_value_loss"] = np.mean(epoch_value_losses) if epoch_value_losses else 0
        self.metrics["avg_entropy"] = np.mean(epoch_entropies) if epoch_entropies else 0
        self.metrics["avg_proposer_reward"] = np.mean(self.metrics["proposer_rewards"]) if self.metrics["proposer_rewards"] else 0
        self.metrics["avg_solver_reward"] = np.mean(self.metrics["solver_rewards"]) if self.metrics["solver_rewards"] else 0
        
        # Normalize sum of component rewards by total experiences to get average per experience
        num_exp_this_epoch = self.metrics["total_experiences"] - self.metrics.get("_last_epoch_total_experiences", 0)
        if num_exp_this_epoch > 0:
            for pt in ["proposer", "solver"]:
                for k in list(self.metrics[f"avg_{pt}_reward_components"].keys()): # Iterate over a copy of keys
                    self.metrics[f"avg_{pt}_reward_components"][k] /= num_exp_this_epoch
        self.metrics["_last_epoch_total_experiences"] = self.metrics["total_experiences"]
        
        # Log epoch summary with colors
        logger.info(f"{Fore.GREEN}Epoch {epoch} Complete - "
                   f"Proposer Reward: {np.mean(self.metrics['proposer_rewards'][-num_generation_steps:]) if len(self.metrics['proposer_rewards']) >= num_generation_steps else 0:.3f}, "
                   f"Solver Reward: {np.mean(self.metrics['solver_rewards'][-num_generation_steps:]) if len(self.metrics['solver_rewards']) >= num_generation_steps else 0:.3f}, "
                   f"Policy Loss: {np.mean(epoch_policy_losses) if epoch_policy_losses else 0:.3f}{Style.RESET_ALL}")
        
        # Update advanced features with epoch metrics
        epoch_metrics = {
            'policy_loss': np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0,
            'value_loss': np.mean(epoch_value_losses) if epoch_value_losses else 0.0,
            'entropy': np.mean(epoch_entropies) if epoch_entropies else 0.0,
            'total_loss': (np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0) + 
                         (np.mean(epoch_value_losses) if epoch_value_losses else 0.0),
            'mean_reward_proposer': np.mean(self.metrics['proposer_rewards'][-num_generation_steps:]) if len(self.metrics['proposer_rewards']) >= num_generation_steps else 0.0,
            'mean_reward_solver': np.mean(self.metrics['solver_rewards'][-num_generation_steps:]) if len(self.metrics['solver_rewards']) >= num_generation_steps else 0.0,
            'task_success_rate': self.metrics.get('solver_success_rate', 0.0),
            'r_learnability': self.metrics['avg_proposer_reward_components'].get('r_learnability'),
            'r_correctness': self.metrics['avg_solver_reward_components'].get('r_correctness'),
            'gradient_norm': self.metrics.get('last_gradient_norm', 0.0),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'tokens_generated': self.metrics.get('tokens_generated_this_epoch', 0),
            'context_length_used': self.context_trainer.current_context_length,
        }
        
        # Update progressive context based on performance
        self.context_trainer.update_epoch(epoch, epoch_metrics)
        
        # Update curriculum based on performance
        self.curriculum_manager.update_epoch(epoch, epoch_metrics)
        
        # Record metrics
        self.metrics_tracker.end_epoch(epoch, epoch_metrics)
        self._append_run_epoch_summary_jsonl(epoch, epoch_metrics, num_generation_steps)

        # Check if we should stop training
        if self.metrics_tracker.should_stop_training():
            logger.info(f"{Fore.YELLOW}Training convergence detected by metrics tracker{Style.RESET_ALL}")
            self.early_stop = True

        # Memory cleanup at end of epoch
        self._cleanup_memory(force=True)

        return self.metrics # Return all metrics for this epoch

    def _store_experience_in_buffer(self, prompt_ids_batch, action_ids_batch, old_log_prob_seq, old_value, reward, full_sequence_ids_batch):
        """Helper to store one experience item into the PPO buffer."""
        if old_log_prob_seq is None or old_value is None:
            logger.warning("Cannot store experience: old_log_prob_seq or old_value is None.")
            return

        # Store sequence IDs as detached CPU Tensors. Squeeze assumes batch size of 1 for these individual adds.
        self.ppo_buffer["prompts_ids"].append(prompt_ids_batch.squeeze(0).detach().cpu()) 
        self.ppo_buffer["prompts_mask"].append(torch.ones_like(prompt_ids_batch.squeeze(0)).detach().cpu()) # Assuming mask is all ones for prompt
        self.ppo_buffer["generated_ids"].append(action_ids_batch.squeeze(0).detach().cpu())
        self.ppo_buffer["input_ids_for_model"].append(full_sequence_ids_batch.squeeze(0).detach().cpu())
        
        # Store scalar values as 0-dimensional CPU Tensors
        self.ppo_buffer["log_probs"].append(torch.tensor(old_log_prob_seq.item(), device='cpu')) 
        self.ppo_buffer["values"].append(torch.tensor(old_value.item(), device='cpu'))    
        self.ppo_buffer["rewards"].append(torch.tensor(reward, device='cpu'))                
        self.ppo_buffer["dones"].append(torch.tensor(True, device='cpu'))                 
        
        self.metrics["total_experiences"] += 1

    # _calculate_gae is now in hf_ppo_utils.py

    def train(self, num_epochs: int):
        logger.info(f"{Fore.CYAN}Starting HuggingFace RL Training...{Style.RESET_ALL}")
        
        # Initialize early stop flag
        self.early_stop = False
        
        # Check for existing checkpoints to resume
        start_epoch = 0
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "hf_checkpoints"))
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*"))
            if checkpoints:
                # Sort by epoch number extracted from folder name
                try:
                    latest_checkpoint = max(checkpoints, key=lambda p: int(str(p.name).split('_')[-1]))
                    if latest_checkpoint.exists():
                        logger.info(f"{Fore.YELLOW}Found checkpoint: {latest_checkpoint}. Attempting to resume...{Style.RESET_ALL}")
                        start_epoch = load_checkpoint(self, str(latest_checkpoint))
                        logger.info(f"{Fore.GREEN}Resuming training from epoch {start_epoch}{Style.RESET_ALL}")
                except Exception as e:
                    logger.warning(f"Failed to resume from checkpoint: {e}. Starting from scratch.")

        # Print configuration summary
        logger.info(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}Training Configuration:{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Model: {self.adapter.model_name}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Total Epochs: {num_epochs}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Start Epoch: {start_epoch}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Batch Size: {self.config['batch_size']}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Learning Rate: {self.config['learning_rate']}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}PPO Update Threshold: {self.config['ppo_update_threshold']}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Checkpoint Interval: {self.config['checkpoint_interval']} epochs{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}\n")
        
        if self.config.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.info(f"{Fore.YELLOW}torch.autograd.set_detect_anomaly(True) enabled for the training session.{Style.RESET_ALL}")

        try:
            for epoch in range(start_epoch, num_epochs):
                self.train_epoch(epoch)
                
                # Check for early stopping
                if self.early_stop:
                    logger.info(f"{Fore.YELLOW}Early stopping triggered at epoch {epoch}{Style.RESET_ALL}")
                    break
                
                # Combined checkpoint score with Q117 easy-spike guard (Q119 best-checkpoint bar)
                recent_success = [
                    m.task_success_rate
                    for m in self.metrics_tracker.epoch_metrics[-20:]
                ]
                checkpoint_score = score_checkpoint_from_metrics(
                    self.metrics,
                    recent_task_success_rates=recent_success,
                )
                
                # Enhanced checkpoint saving with pruning
                should_save = (
                    (epoch + 1) % self.config["checkpoint_interval"] == 0 or
                    epoch == num_epochs - 1 or
                    checkpoint_score > max(self.checkpoint_scores.values(), default=float('-inf'))
                )
                
                if should_save:
                    logger.info(f"{Fore.YELLOW}Saving checkpoint at epoch {epoch}...{Style.RESET_ALL}")
                    self._save_checkpoint_with_pruning(epoch, checkpoint_score)
                
                if (epoch + 1) % self.config["evaluation_interval"] == 0:
                    # Could add evaluation logic here
                    pass
            
            # Save final checkpoint
            final_checkpoint_epoch = max(0, num_epochs - 1)
            logger.info(f"{Fore.YELLOW}Saving final checkpoint...{Style.RESET_ALL}")
            self._save_checkpoint_with_pruning(final_checkpoint_epoch, float('inf'))  # Ensure final is kept
            
            # Save metrics summary
            self.metrics_tracker.save_metrics()
            training_summary = self.metrics_tracker.get_training_summary()
            training_progress = training_summary.get("training_progress")
            
            # Log training summary with advanced metrics
            logger.info(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Training Complete!{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Total Experiences: {self.metrics['total_experiences']}{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Valid Tasks Generated: {self.metrics['total_valid_tasks']}{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Final Difficulty Level: {self.curriculum_manager.current_difficulty.name}{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}Final Context Length: {self.context_trainer.current_context_length}{Style.RESET_ALL}")
            if training_progress:
                logger.info(
                    f"{Fore.GREEN}Best Reward: {training_progress.get('best_reward', 0.0):.4f}"
                    f" at epoch {training_progress.get('best_epoch', -1)}{Style.RESET_ALL}"
                )
                logger.info(f"{Fore.GREEN}Convergence: {training_progress.get('convergence_detected', False)}{Style.RESET_ALL}")
                logger.info(
                    f"{Fore.GREEN}Total Tracked Epochs: {training_progress.get('total_epochs', 0)}, "
                    f"Total Hours: {training_progress.get('total_time_hours', 0.0):.4f}{Style.RESET_ALL}"
                )
            else:
                logger.info(f"{Fore.YELLOW}No epoch metrics were recorded. No training progression statistics available yet.{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

            # Print comprehensive training summary
            print_training_summary(self, self.start_time, num_epochs)
        finally:
            self._release_training_resources()
            self._cleanup_memory(force=True)

    def _save_checkpoint_with_pruning(self, epoch: int, score: float):
        """Save checkpoint and prune old ones to manage disk space"""
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        
        # Save the checkpoint
        save_checkpoint(
            self, checkpoint_name,
            epoch, self.optimizer, self.scaler,
            self.metrics, self.experience_buffer if self.config.get("save_buffers_in_checkpoint", False) else None,
            self.ppo_buffer if self.config.get("save_buffers_in_checkpoint", False) else None
        )
        
        # Track this checkpoint's score
        self.checkpoint_scores[checkpoint_name] = score
        
        # Prune if we have too many checkpoints
        if len(self.checkpoint_scores) > self.max_checkpoints:
            self._prune_checkpoints()
    
    def _prune_checkpoints(self):
        """Remove lowest scoring checkpoints to save disk space"""
        import shutil
        
        # Sort checkpoints by score (ascending)
        sorted_checkpoints = sorted(self.checkpoint_scores.items(), key=lambda x: x[1])
        
        # Determine which checkpoints to remove
        num_to_remove = len(sorted_checkpoints) - self.max_checkpoints
        checkpoints_to_remove = [name for name, _ in sorted_checkpoints[:num_to_remove]]
        
        logger.info(f"{Fore.YELLOW}Pruning {num_to_remove} old checkpoints to save disk space{Style.RESET_ALL}")
        
        for checkpoint_name in checkpoints_to_remove:
            checkpoint_path = os.path.join(self.config["checkpoint_dir"], checkpoint_name)
            
            # Remove all files associated with this checkpoint
            patterns = [
                f"{checkpoint_path}.pt",
                f"{checkpoint_path}_actor",  # Directory
                f"{checkpoint_path}_critic.pt",
                f"{checkpoint_path}_optimizer.pt",
            ]
            
            for pattern in patterns:
                try:
                    if os.path.isdir(pattern):
                        shutil.rmtree(pattern)
                        logger.info(f"{Fore.YELLOW}Removed directory: {pattern}{Style.RESET_ALL}")
                    elif os.path.exists(pattern):
                        os.remove(pattern)
                        logger.info(f"{Fore.YELLOW}Removed file: {pattern}{Style.RESET_ALL}")
                except Exception as e:
                    logger.warning(f"{Fore.RED}Failed to remove {pattern}: {e}{Style.RESET_ALL}")
            
            # Remove from tracking
            del self.checkpoint_scores[checkpoint_name]
        
        logger.info(f"{Fore.GREEN}Checkpoint pruning complete. Keeping best {self.max_checkpoints} checkpoints.{Style.RESET_ALL}")

    def get_task_for_generation(self, problem_type: str) -> Dict[str, Any]:
        """Get task parameters for generation based on current curriculum"""
        return {
            'difficulty_level': self.current_difficulty,
            'problem_type': problem_type
        }
    


def test_curriculum_manager():
    """Test that CurriculumLearningManager has current_level property"""
    try:
        from hf_curriculum_learning import CurriculumLearningManager, create_curriculum_config
        config = create_curriculum_config()
        manager = CurriculumLearningManager(config)
        print(f" CurriculumLearningManager.current_level = '{manager.current_level}'")
        print(f" CurriculumLearningManager.current_difficulty_numeric() = {manager.current_difficulty_numeric()}")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing CurriculumLearningManager: {e}")
        return False

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Train AZR HF PPO Trainer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs to run (outer RL epochs)")
    parser.add_argument(
        "--checkpoint-dir",
        default="hf_checkpoints/hf_trainer_qwen3_5b",
        help="Directory to save/load checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for this run",
    )
    parser.add_argument(
        "--seed-tasks-per-type",
        type=int,
        default=0,
        help="Number of seed tasks to generate per problem type before PPO training",
    )
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base weights in 4-bit when bitsandbytes is available (default: off).",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.85,
        help="CUDA memory fraction limit for this process (0-1)",
    )
    parser.add_argument(
        "--cuda-alloc-config",
        type=str,
        default="max_split_size_mb:128,garbage_collection_threshold:0.8",
        help="PYTORCH_CUDA_ALLOC_CONF value for CUDA memory allocator tuning",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["fp16", "bf16", "fp32", "auto"],
        default="auto",
        help="Model loading dtype for actor/critic (auto prefers bf16 on CUDA when supported, else fp16).",
    )
    parser.add_argument(
        "--use-separate-value-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use separate actor and critic (value) models (AZR / PPO-style; use --no-use-separate-value-model for unified).",
    )
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
        help="CPU cap percentage (0-100) for this training process",
    )
    parser.add_argument(
        "--trainer-config",
        type=str,
        default="",
        help="Path to JSON file with hf trainer configuration overrides.",
    )
    parser.add_argument(
        "--trainer-config-json",
        type=str,
        default="",
        help="JSON string containing hf trainer configuration overrides.",
    )
    parser.add_argument(
        "--set-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional config override in KEY=VALUE format (VALUE parsed as JSON when possible).",
    )
    args = parser.parse_args()
    trainer_config_overrides: Dict[str, object] = {}

    def _safe_parse_literal(value: str):
        try:
            return json.loads(value)
        except Exception:
            return value

    if args.trainer_config:
        with open(args.trainer_config, "r", encoding="utf-8") as config_file:
            overrides_from_file = json.load(config_file)
            if not isinstance(overrides_from_file, dict):
                raise ValueError("--trainer-config must contain a JSON object.")
            trainer_config_overrides.update(overrides_from_file)

    if args.trainer_config_json:
        overrides_from_cli = json.loads(args.trainer_config_json)
        if not isinstance(overrides_from_cli, dict):
            raise ValueError("--trainer-config-json must contain a JSON object.")
        trainer_config_overrides.update(overrides_from_cli)

    for item in args.set_config:
        if "=" not in item:
            raise ValueError(f"Invalid --set-config value '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        trainer_config_overrides[key.strip()] = _safe_parse_literal(value.strip())

    cpu_threads = _apply_cpu_cap(args.cpu_cap)
    configure_training_logging(use_rich=args.rich)
    logger.info(f"CPU cap set to {args.cpu_cap:.1f}% (max threads={cpu_threads})")
    try:
        selected_torch_dtype = _parse_torch_dtype(args.model_dtype)
    except Exception as e:
        logger.error(f"Invalid --model-dtype value '{args.model_dtype}': {e}")
        raise

    # Print beautiful status header
    print_status_header()
    print(f"{Fore.WHITE} Output mode: {'Rich' if args.rich and _RICH_AVAILABLE else 'Plain'}{Style.RESET_ALL}")

    # Test curriculum manager fix
    print("\n Testing Curriculum Manager Fix:")
    if test_curriculum_manager():
        print(" Curriculum manager test passed!")
    else:
        print(" Curriculum manager test failed!")

    # Demo the real-time status display
    print(f"\n{Fore.MAGENTA} DEMONSTRATION: Enhanced Terminal Display Features")
    print(f"{Fore.WHITE}┌{'─'*76}┐")
    print(f"{Fore.CYAN}│  Features Added:{Fore.WHITE}")
    print(f"{Fore.CYAN}│   • Color-coded progress bars with text tags")
    print(f"{Fore.CYAN}│   • Real-time status indicators")
    print(f"{Fore.CYAN}│   • Boxed layouts for better readability")
    print(f"{Fore.CYAN}│   • Reduced logging clutter (WARNING+ only)")
    print(f"{Fore.CYAN}│   • Enhanced error and status formatting")
    print(f"{Fore.WHITE}└{'─'*76}┘\n")

    logger.info("Starting HF Trainer Example")
    # Use local Qwen3-0.6B as the fallback model when available.
    model_to_use = "Qwen3-0.6B"
    
    local_model_path = Path("models") / "Qwen3-0.6B"  # Use local folder name
    # Ensure to check for a file that indicates a full download, like config.json or pytorch_model.bin
    hf_model_name = str(local_model_path) if local_model_path.exists() and (local_model_path / "config.json").exists() else model_to_use

    # Define a dedicated cache directory within the models folder
    dedicated_hf_cache_dir = Path("models") / ".hf_cache"
    os.makedirs(dedicated_hf_cache_dir, exist_ok=True)

    paper_style = _azr_optional_env_bool("AZR_PAPER_STYLE_DEFAULTS") is True
    gen_env = _azr_optional_env_positive_int("AZR_HF_GENERATION_STEPS_PER_EPOCH")
    batch_env = _azr_optional_env_positive_int("AZR_HF_BATCH_SIZE")
    ppo_env = _azr_optional_env_positive_int("AZR_HF_PPO_UPDATE_THRESHOLD")
    lr_env = _azr_optional_env_float("AZR_HF_LEARNING_RATE")
    clr_env = _azr_optional_env_float("AZR_HF_CRITIC_LEARNING_RATE")

    if gen_env is not None:
        gen_steps_main = gen_env
    elif paper_style:
        gen_steps_main = 10
    else:
        gen_steps_main = 1

    if batch_env is not None:
        batch_main = batch_env
    elif paper_style:
        batch_main = 16
    else:
        batch_main = 1

    if ppo_env is not None:
        ppo_thr_main = ppo_env
    elif paper_style:
        ppo_thr_main = 64
    else:
        ppo_thr_main = 1

    seed_tasks = args.seed_tasks_per_type
    if paper_style:
        if seed_tasks == 0 and not os.environ.get("AZR_SEED_TASKS_PER_TYPE", "").strip():
            seed_tasks = 6
            logger.info(
                "AZR_PAPER_STYLE_DEFAULTS=1: seed_tasks_per_type=6 (no AZR_SEED_TASKS_PER_TYPE in env; CLI was 0)."
            )

    lr_critic_main: Dict[str, float] = {}
    if lr_env is not None:
        lr_critic_main["learning_rate"] = float(lr_env)
        lr_critic_main["critic_learning_rate"] = float(clr_env if clr_env is not None else lr_env)
    elif paper_style:
        lr_critic_main["learning_rate"] = 5e-7
        lr_critic_main["critic_learning_rate"] = 5e-7
    if clr_env is not None and "critic_learning_rate" not in lr_critic_main:
        lr_critic_main["critic_learning_rate"] = float(clr_env)

    if gen_env is not None or batch_env is not None or ppo_env is not None:
        logger.info(
            "__main__ throughput from env: generation_steps_per_epoch=%s, batch_size=%s, ppo_update_threshold=%s.",
            gen_steps_main,
            batch_main,
            ppo_thr_main,
        )
    elif paper_style:
        logger.info(
            "AZR_PAPER_STYLE_DEFAULTS=1: __main__ throughput generation_steps_per_epoch=%s, batch_size=%s, ppo_update_threshold=%s.",
            gen_steps_main,
            batch_main,
            ppo_thr_main,
        )

    # Trainer runtime configuration:
    # - rl_epochs: total outer training epochs
    # - generation_steps_per_epoch: how many proposer/solver task-generation steps per epoch before PPO updates; higher = more experiences per epoch
    # - ppo_update_threshold: number of experiences to accumulate before triggering a PPO update
    # - batch_size: PPO minibatch size used inside each PPO update
    # - max_new_tokens: maximum tokens to generate per call in this example entrypoint (can be overridden per role inside trainer)
    # - checkpoint_dir: where checkpoints from this run will be stored
    # Default __main__ path uses minimal 1/1/1 unless AZR_PAPER_STYLE_DEFAULTS=1 (paper-like throughput).
    config = {
        "rl_epochs": args.epochs,
        "generation_steps_per_epoch": gen_steps_main,
        "ppo_update_threshold": ppo_thr_main,
        "batch_size": batch_main,
        "max_new_tokens": 256, 
        "checkpoint_dir": args.checkpoint_dir, 
        "gradient_accumulation_steps": 4, 
        "gradient_checkpointing": True, 
        # GradScaler is for AMP (fp32 weights + autocast); loading weights in fp16/bf16 yields fp16
        # parameter grads that torch.amp.GradScaler cannot unscale ("Attempting to unscale FP16 gradients").
        "mixed_precision": str(args.model_dtype).lower().strip() in ("fp32", "float32", "float"),
        "empty_cache_frequency": 1, # Empty EVERY step
        "load_in_4bit": args.use_4bit,
        "model_dtype": args.model_dtype,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "cuda_alloc_config": args.cuda_alloc_config,
        "use_separate_value_model": args.use_separate_value_model,
        "seed_tasks_per_type": seed_tasks,
    }
    config.update(lr_critic_main)
    if lr_critic_main:
        logger.info("__main__ actor/critic learning rates from env or paper preset: %s", lr_critic_main)
    config.update(trainer_config_overrides)

    try:
        use_4bit = bool(config.get("load_in_4bit", False))
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
                logger.info(f"{Fore.GREEN}bitsandbytes detected. 4-bit loading enabled.{Style.RESET_ALL}")
            except ImportError:
                logger.warning(f"{Fore.RED}bitsandbytes NOT found. 4-bit loading disabled for this run.{Style.RESET_ALL}")
                use_4bit = False

        hf_adapter_instance = HuggingFaceAdapter(
            model_name=hf_model_name, 
            auth_token=None,  # Or your HF token if needed
            use_separate_value_model=args.use_separate_value_model,
            hf_cache_dir=str(dedicated_hf_cache_dir), # Use the dedicated cache directory
            load_in_4bit=use_4bit, # Pass 4-bit config
            torch_dtype_for_actor_critic=selected_torch_dtype # Pass selected precision
        )
        experience_buffer_instance = ExperienceBuffer(capacity=100)
        
        # Initialize sophisticated DatasetManager
        dataset_manager = DatasetManager(config)
        
        # Use real CodeExecutor with timeout
        python_executor_instance = CodeExecutor(timeout_seconds=5)
        
        hf_reward_manager_instance = HFRewardManager(python_executor=python_executor_instance)

        trainer = HuggingFaceRLTrainer(
            hf_adapter=hf_adapter_instance,
            experience_buffer=experience_buffer_instance,
            hf_reward_manager=hf_reward_manager_instance,
            python_executor=python_executor_instance,
            config=config
        )
        
        # Inject DatasetManager into trainer (or pass in init)
        trainer.dataset_manager = dataset_manager

        # Print comprehensive status after trainer initialization
        print_training_config(config)
        print_system_status(hf_adapter_instance, trainer.device, config)
        print_advanced_features_status()
        
        # --- SEEDING PHASE ---
        logger.info(f"\n{Fore.MAGENTA} STARTING SEEDING PHASE...{Style.RESET_ALL}")
        dataset_manager.generate_seeds(hf_adapter_instance, num_seeds=config.get("seed_tasks_per_type", 0))
        
        logger.info(
            f"Starting training run with model {hf_model_name!r} for {int(args.epochs)} epoch(s)..."
        )
        trainer.train(args.epochs)
        logger.info("Training run finished.")
    except Exception as e:
        logger.error(f"Error in HuggingFaceRLTrainer example usage: {e}", exc_info=True) 
        raise
