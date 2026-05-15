"""Evaluation script for AZR model on standard benchmarks.

Evaluates on:
- Code: HumanEval, MBPP
- Math: GSM8K, MATH
- ProgramBench (optional): aggregate scores from an existing upstream eval run directory.
  Pass ``--benchmarks programbench`` and ``--programbench-run-dir <dir>`` where ``dir``
  contains nested ``*/*.eval.json`` after running ``programbench eval`` on a supported host.
"""

import os
import json
import time
import torch
import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_benchmark_data import apply_benchmark_offline_env, load_azr_benchmark_split
import logging
from tqdm import tqdm
import re
import math
import numbers

# Import our modules
from azr_hf_adapter import HuggingFaceAdapter
from code_executor import CodeExecutor

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    RichHandler = None
    Panel = None
    Table = None
    _RICH_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
_RICH_CONSOLE = Console(highlight=False, force_terminal=True) if _RICH_AVAILABLE else None
TASK_PROGRESS_EVERY = 25


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

    try:
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(max_threads)
    except Exception:
        pass

    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(max_threads)))
    except Exception:
        pass

    if _RICH_AVAILABLE and _RICH_CONSOLE is not None:
        _RICH_CONSOLE.print(f"[blue]CPU cap set to {cap:.1f}% (max threads={max_threads})[/blue]")
    else:
        logger.info(f"CPU cap set to {cap:.1f}% (max threads={max_threads})")

    return max_threads


def _safe_rich_value(value, fmt: str) -> str:
    """Format numbers safely for rich/plain output."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        try:
            return fmt.format(value)
        except Exception:
            return str(value)
    return str(value)


def _get_accuracy_style(accuracy):
    if accuracy is None:
        return "white"
    if accuracy >= 0.7:
        return "green"
    if accuracy >= 0.4:
        return "yellow"
    return "red"


def _format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, float(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _optional_positive_int_env(name: str) -> Optional[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    v = int(raw)
    return v if v > 0 else None


def _default_benchmark_batch_size(fast: bool) -> int:
    raw = (os.environ.get("AZR_BENCHMARK_BATCH_SIZE") or "").strip()
    if raw:
        return max(1, int(raw))
    return 4 if fast else 1


def _can_benchmark_microbatch(gen_kwargs: dict) -> bool:
    return int(gen_kwargs.get("num_return_sequences", 1) or 1) == 1 and int(gen_kwargs.get("num_beams", 1) or 1) <= 1


def _benchmark_gen_microbatch_size(batch_size: int, gen_kwargs: dict) -> int:
    if batch_size <= 1:
        return 1
    if not _can_benchmark_microbatch(gen_kwargs):
        return 1
    return batch_size


def resolve_max_new_tokens_by_benchmark(fast: bool) -> Optional[Dict[str, int]]:
    """Tighter caps when AZR_BENCHMARK_FAST=1; None means use evaluator defaults."""
    if not fast:
        return None
    return {"humaneval": 256, "mbpp": 128, "gsm8k": 256, "math": 512}


def apply_benchmark_speed_from_env(args) -> None:
    """Apply AZR_BENCHMARK_FAST, AZR_BENCHMARK_MAX_TASKS_PER_DATASET, AZR_BENCHMARK_BATCH_SIZE to parsed CLI args."""
    fast = _truthy_env("AZR_BENCHMARK_FAST")
    args.benchmark_fast = fast
    cap = _optional_positive_int_env("AZR_BENCHMARK_MAX_TASKS_PER_DATASET")
    if cap is not None and args.limit > cap:
        logger.info(
            "AZR_BENCHMARK_MAX_TASKS_PER_DATASET=%s caps --limit from %s to %s",
            cap,
            args.limit,
            cap,
        )
        args.limit = cap
    if fast and args.samples_per_task > 1:
        logger.info(
            "AZR_BENCHMARK_FAST=1: clamping --samples-per-task from %s to 1 for wall-clock speed",
            args.samples_per_task,
        )
        args.samples_per_task = 1
    try:
        args.benchmark_batch_size = _default_benchmark_batch_size(fast)
    except ValueError as e:
        raise ValueError(
            f"Invalid AZR_BENCHMARK_BATCH_SIZE: {os.environ.get('AZR_BENCHMARK_BATCH_SIZE')!r}"
        ) from e
    if fast:
        logger.info(
            "AZR_BENCHMARK_FAST=1: benchmark_batch_size=%s (default 4 when AZR_BENCHMARK_BATCH_SIZE unset; set AZR_BENCHMARK_BATCH_SIZE=1 to disable micro-batching)",
            args.benchmark_batch_size,
        )
    if args.benchmark_batch_size > 1 and not fast:
        logger.info(
            "AZR_BENCHMARK_BATCH_SIZE=%s: micro-batching when num_return_sequences=1",
            args.benchmark_batch_size,
        )


def _log_task_heartbeat(
    benchmark_name: str,
    completed: int,
    total: Optional[int],
    start_ts: float,
    correct: Optional[int] = None,
) -> None:
    elapsed = time.perf_counter() - start_ts
    throughput = completed / elapsed if elapsed > 0 else 0.0
    percent = ""
    if total and total > 0:
        percent = f" ({(completed / total * 100):.1f}% complete)"
    correct_text = ""
    if correct is not None:
        correct_text = f" | correct={correct}"
    logger.info(
        f"[{benchmark_name}] task {completed}"
        f"{f'/ {total}' if total else ''}{percent}{correct_text} | "
        f"elapsed={_format_elapsed(elapsed)} | throughput={throughput:.2f} tasks/s"
    )


def configure_logging(results_dir: str, use_rich: bool = True) -> None:
    """Configure logging handlers for a specific output directory."""
    Path(results_dir).mkdir(exist_ok=True)

    # Build handlers first, then configure logging in one shot to avoid duplicate handlers.
    handlers = [logging.FileHandler(Path(results_dir) / 'evaluation.log')]

    if use_rich and _RICH_AVAILABLE:
        stream_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    handlers.append(stream_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


def _is_numeric_score(value) -> bool:
    """True for real scalars suitable for accuracy / aggregate math (excludes bool)."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return True
    # NumPy scalar / other .item() scalars loaded from JSON-adjacent pipelines
    if hasattr(value, "item"):
        try:
            inner = value.item()
        except Exception:
            return False
        return isinstance(inner, numbers.Real) and not isinstance(inner, bool)
    return False


class BenchmarkEvaluator:
    """Evaluator for standard benchmarks"""
    
    def __init__(
        self,
        model_name: str,
        results_dir: str = "evaluation_results",
        use_separate_value_model: bool = True,
        generation_temperature: float = 0.2,
        top_p: float = 0.95,
        samples_per_task: int = 8,
        load_in_4bit: bool = False,
        max_new_tokens_by_benchmark: Optional[Dict[str, int]] = None,
    ):
        # Resolve to local copy under models/<basename> if available
        base_name = Path(model_name).name if (os.path.sep in model_name or '/' in model_name) else model_name.split('/')[-1]
        local_candidate = Path("models") / base_name
        if local_candidate.exists() and (local_candidate / "config.json").exists():
            resolved_model = str(local_candidate)
            logger.info(f"Using local model at {resolved_model} instead of remote '{model_name}'")
        else:
            resolved_model = model_name
        self.model_name = resolved_model

        # Prepare results and cache dirs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.hf_cache_dir = Path("models") / ".hf_cache"
        self.hf_cache_dir.mkdir(exist_ok=True)
        self.generation_temperature = generation_temperature
        self.generation_top_p = top_p
        self.samples_per_task = max(1, int(samples_per_task))
        self.generation_kwargs = build_code_gen_kwargs(
            self.samples_per_task,
            self.generation_temperature,
            self.generation_top_p,
        )
        self.max_new_tokens: Dict[str, int] = {
            "humaneval": 512,
            "mbpp": 256,
            "gsm8k": 512,
            "math": 1024,
        }
        if max_new_tokens_by_benchmark:
            self.max_new_tokens.update(max_new_tokens_by_benchmark)
        
        # Initialize model adapter with local cache
        self.adapter = HuggingFaceAdapter(
            self.model_name,
            hf_cache_dir=str(self.hf_cache_dir),
            use_separate_value_model=use_separate_value_model,
            load_in_4bit=load_in_4bit,
        )
        
        # Initialize code executor for code benchmarks
        self.executor = CodeExecutor(timeout_seconds=10)
        
        self.results = {}
    
    def evaluate_humaneval(self) -> Dict:
        """Evaluate on HumanEval benchmark"""
        logger.info("Starting HumanEval evaluation...")
        
        try:
            # Load HumanEval dataset
            logger.info("HumanEval: loading dataset (openai_humaneval, split=test)...")
            dataset = load_azr_benchmark_split("openai_humaneval", "test", None, logger=logger)
            logger.info(f"HumanEval: dataset loaded with {len(dataset)} tasks")
            
            results = []
            correct = 0
            total = len(dataset)
            start_ts = time.perf_counter()
            
            for idx, example in enumerate(tqdm(dataset, desc="HumanEval"), start=1):
                task_id = example['task_id']
                prompt = example['prompt']
                test = example['test']
                entry_point = example['entry_point']
                
                # Generate solution
                generated = self.adapter.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens["humaneval"],
                    **self.generation_kwargs
                )[0]
                
                # Extract code from generation
                if "```python" in generated:
                    code = generated.split("```python")[1].split("```")[0]
                else:
                    code = generated
                
                # Combine with test code
                full_code = prompt + code + "\n\n" + test
                
                # Execute and check
                result = self.executor.execute(
                    code=full_code,
                    test_input="",  # Tests are self-contained
                    timeout=5
                )
                
                passed = result['success'] and not result.get('error')
                if passed:
                    correct += 1

                if idx % TASK_PROGRESS_EVERY == 0 or idx == total:
                    _log_task_heartbeat("HumanEval", idx, total, start_ts, correct=correct)
                
                results.append({
                    'task_id': task_id,
                    'passed': passed,
                    'generated': code,
                    'error': result.get('error', '')
                })

            elapsed = time.perf_counter() - start_ts
            logger.info(
                "HumanEval complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                total,
                correct,
                (correct / total if total else 0.0),
                _format_elapsed(elapsed),
            )
            accuracy = correct / total if total > 0 else 0
            
            return {
                'benchmark': 'HumanEval',
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating HumanEval: {e}")
            return {'benchmark': 'HumanEval', 'error': str(e)}
    
    def evaluate_mbpp(self) -> Dict:
        """Evaluate on MBPP benchmark"""
        logger.info("Starting MBPP evaluation...")
        
        try:
            # Load MBPP dataset
            logger.info("MBPP: loading dataset (mbpp, sanitized, split=test)...")
            dataset = load_azr_benchmark_split("mbpp", "test", "sanitized", logger=logger)
            logger.info(f"MBPP: dataset loaded with {len(dataset)} tasks")
            
            results = []
            correct = 0
            total = len(dataset)
            start_ts = time.perf_counter()
            
            for idx, example in enumerate(tqdm(dataset, desc="MBPP"), start=1):
                task_id = example.get('task_id', idx)
                # Some variants use 'text', others may use 'prompt' or 'description'
                description = example.get('text') or example.get('prompt') or example.get('description') or ""
                # Tests can be under 'test_list', or occasionally single 'test'
                tests = example.get('test_list') or example.get('tests') or []
                if not tests and 'test' in example:
                    tests = [example['test']]
                
                # Create prompt
                prompt = f"Write a Python function to solve this problem:\n{description}\n\n"
                
                # Generate solution
                generated = self.adapter.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens["mbpp"],
                    **self.generation_kwargs
                )[0]
                
                # Extract code
                if "```python" in generated:
                    code = generated.split("```python")[1].split("```")[0]
                else:
                    code = generated
                
                # Test each case
                all_passed = True
                for test in tests:
                    full_code = code + "\n\n" + test
                    result = self.executor.execute(
                        code=full_code,
                        test_input="",
                        timeout=5
                    )
                    if not result.get('success', False):
                        all_passed = False
                        break
                
                if all_passed:
                    correct += 1

                if idx % TASK_PROGRESS_EVERY == 0 or idx == total:
                    _log_task_heartbeat("MBPP", idx, total, start_ts, correct=correct)
                
                results.append({
                    'task_id': task_id,
                    'passed': all_passed,
                    'generated': code
                })

            elapsed = time.perf_counter() - start_ts
            logger.info(
                "MBPP complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                total,
                correct,
                (correct / total if total else 0.0),
                _format_elapsed(elapsed),
            )
            accuracy = correct / total if total > 0 else 0
            
            return {
                'benchmark': 'MBPP',
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating MBPP: {e}")
            return {'benchmark': 'MBPP', 'error': str(e)}
    
    def evaluate_gsm8k(self) -> Dict:
        """Evaluate on GSM8K benchmark"""
        logger.info("Starting GSM8K evaluation...")
        
        try:
            # Load GSM8K dataset
            logger.info("GSM8K: loading dataset (gsm8k, main, split=test)...")
            dataset = load_azr_benchmark_split("gsm8k", "test", "main", logger=logger)
            logger.info(f"GSM8K: raw dataset size = {len(dataset)}")
            
            results = []
            correct = 0
            total = min(len(dataset), 500)  # Evaluate on subset for speed
            subset = dataset.select(range(total))
            start_ts = time.perf_counter()
            
            for idx, example in enumerate(tqdm(subset, desc="GSM8K"), start=1):
                question = example['question']
                answer = example['answer']
                
                # Extract numerical answer
                true_answer = answer.split("####")[-1].strip()
                
                # Create prompt with chain-of-thought
                prompt = f"""Solve this math problem step by step.

Question: {question}

Let's solve this step by step:
"""
                
                # Generate solution
                generated = self.adapter.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens["gsm8k"],
                    **self.generation_kwargs
                )[0]
                
                # Extract answer (look for patterns like "= X" or "answer is X")
                import re
                answer_patterns = [
                    r'answer is (\-?\d+(?:\.\d+)?)',
                    r'= (\-?\d+(?:\.\d+)?)\s*$',
                    r'equals (\-?\d+(?:\.\d+)?)',
                    r'result is (\-?\d+(?:\.\d+)?)'
                ]
                
                predicted_answer = None
                for pattern in answer_patterns:
                    matches = re.findall(pattern, generated, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        predicted_answer = matches[-1]  # Take last match
                        break
                
                # Check if answer is correct
                is_correct = False
                if predicted_answer:
                    try:
                        pred_num = float(predicted_answer)
                        true_num = float(true_answer)
                        is_correct = abs(pred_num - true_num) < 1e-5
                    except (ValueError, TypeError):
                        pass
                
                if is_correct:
                    correct += 1

                if idx % TASK_PROGRESS_EVERY == 0 or idx == total:
                    _log_task_heartbeat("GSM8K", idx, total, start_ts, correct=correct)
                
                results.append({
                    'idx': idx,
                    'correct': is_correct,
                    'predicted': predicted_answer,
                    'true': true_answer,
                    'generated': generated
                })

            elapsed = time.perf_counter() - start_ts
            logger.info(
                "GSM8K complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                total,
                correct,
                (correct / total if total else 0.0),
                _format_elapsed(elapsed),
            )
            accuracy = correct / total if total > 0 else 0
            
            return {
                'benchmark': 'GSM8K',
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating GSM8K: {e}")
            return {'benchmark': 'GSM8K', 'error': str(e)}
    
    def evaluate_math(self) -> Dict:
        """Evaluate on MATH benchmark"""
        logger.info("Starting MATH evaluation...")
        
        try:
            # Load MATH dataset
            logger.info("MATH: loading dataset (hendrycks/competition_math, split=test)...")
            dataset = load_azr_benchmark_split("hendrycks/competition_math", "test", None, logger=logger)
            logger.info(f"MATH: raw dataset size = {len(dataset)}")
            
            results = []
            correct = 0
            total = min(len(dataset), 200)  # Evaluate on subset for speed
            subset = dataset.select(range(total))
            start_ts = time.perf_counter()
            
            for idx, example in enumerate(tqdm(subset, desc="MATH"), start=1):
                problem = example['problem']
                solution = example['solution']
                level = example['level']
                problem_type = example['type']
                
                # Create prompt
                prompt = f"""Solve this math problem step by step.

Problem: {problem}

Solution:
"""
                
                # Generate solution
                generated = self.adapter.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens["math"],
                    **self.generation_kwargs
                )[0]
                
                # Extract answer (MATH uses boxed format)
                import re
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                matches = re.findall(boxed_pattern, generated)
                
                predicted_answer = matches[-1] if matches else None
                
                # Extract true answer
                true_matches = re.findall(boxed_pattern, solution)
                true_answer = true_matches[-1] if true_matches else None
                
                # Simple string comparison (more sophisticated comparison needed for real eval)
                is_correct = predicted_answer == true_answer if predicted_answer and true_answer else False
                
                if is_correct:
                    correct += 1

                if idx % TASK_PROGRESS_EVERY == 0 or idx == total:
                    _log_task_heartbeat("MATH", idx, total, start_ts, correct=correct)
                
                results.append({
                    'idx': idx,
                    'level': level,
                    'type': problem_type,
                    'correct': is_correct,
                    'predicted': predicted_answer,
                    'true': true_answer
                })

            elapsed = time.perf_counter() - start_ts
            logger.info(
                "MATH complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                total,
                correct,
                (correct / total if total else 0.0),
                _format_elapsed(elapsed),
            )
            accuracy = correct / total if total > 0 else 0
            
            # Calculate accuracy by level
            level_stats = {}
            for r in results:
                level = r['level']
                if level not in level_stats:
                    level_stats[level] = {'total': 0, 'correct': 0}
                level_stats[level]['total'] += 1
                if r['correct']:
                    level_stats[level]['correct'] += 1
            
            for level in level_stats:
                level_stats[level]['accuracy'] = level_stats[level]['correct'] / level_stats[level]['total']
            
            return {
                'benchmark': 'MATH',
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'level_stats': level_stats,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating MATH: {e}")
            return {'benchmark': 'MATH', 'error': str(e)}
    
    def run_all_evaluations(self) -> Dict:
        """Run all benchmark evaluations"""
        logger.info(f"Starting evaluation of {self.model_name} on all benchmarks...")
        
        all_results = {}
        
        # Code benchmarks
        if True:  # Can toggle specific benchmarks
            all_results['humaneval'] = self.evaluate_humaneval()
            all_results['mbpp'] = self.evaluate_mbpp()
        
        # Math benchmarks
        if True:
            all_results['gsm8k'] = self.evaluate_gsm8k()
            all_results['math'] = self.evaluate_math()
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: Dict) -> None:
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = self.results_dir / f"eval_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Remove detailed results for summary
            summary = {}
            for benchmark, data in results.items():
                if isinstance(data, dict) and 'results' in data:
                    summary[benchmark] = {k: v for k, v in data.items() if k != 'results'}
                else:
                    summary[benchmark] = data
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {json_path}")
        
        # Save detailed report
        report_path = self.results_dir / f"eval_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(f"# Evaluation Report for {self.model_name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Benchmark | Total | Correct | Accuracy |\n")
            f.write("|-----------|--------|---------|----------|\n")
            
            for benchmark, data in results.items():
                if isinstance(data, dict) and 'accuracy' in data:
                    f.write(f"| {benchmark} | {data.get('total', 'N/A')} | "
                           f"{data.get('correct', 'N/A')} | "
                           f"{data.get('accuracy', 0):.2%} |\n")
            
            # Add level-wise results for MATH
            if 'math' in results and 'level_stats' in results['math']:
                f.write("\n### MATH Level-wise Results\n\n")
                f.write("| Level | Total | Correct | Accuracy |\n")
                f.write("|-------|--------|---------|----------|\n")
                
                for level, stats in results['math']['level_stats'].items():
                    f.write(f"| {level} | {stats['total']} | "
                           f"{stats['correct']} | {stats['accuracy']:.2%} |\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary"""
        # Calculate averages
        code_benchmarks = ['humaneval', 'mbpp']
        math_benchmarks = ['gsm8k', 'math']

        code_scores = []
        math_scores = []

        if _RICH_AVAILABLE and _RICH_CONSOLE is not None:
            table = Table(title=f"Evaluation Summary - {self.model_name}", show_lines=True)
            table.add_column("Benchmark")
            table.add_column("Status")
            table.add_column("Accuracy")
            table.add_column("Correct")
            table.add_column("Total")
            for benchmark, data in results.items():
                bn = str(benchmark).lower()
                acc = data.get("accuracy") if isinstance(data, dict) else None
                if isinstance(data, dict) and _is_numeric_score(acc):
                    accuracy = float(acc)
                    total = data.get('total', 'N/A')
                    correct = data.get('correct', 'N/A')
                    style = _get_accuracy_style(accuracy)
                    table.add_row(
                        str(benchmark).upper(),
                        "[green]ok[/green]",
                        f"[{style}]{accuracy:.2%}[/{style}]",
                        str(correct),
                        str(total),
                    )

                    if bn in code_benchmarks:
                        code_scores.append(accuracy)
                    elif bn in math_benchmarks:
                        math_scores.append(accuracy)
                elif isinstance(data, dict):
                    table.add_row(
                        str(benchmark).upper(),
                        "[red]error[/red]",
                        "N/A",
                        "N/A",
                        "N/A",
                    )

            summary_parts = []
            if code_scores:
                summary_parts.append(f"Code Average: {_safe_rich_value(np.mean(code_scores), '{:.2%}')}")
            if math_scores:
                summary_parts.append(f"Math Average: {_safe_rich_value(np.mean(math_scores), '{:.2%}')}")
            if code_scores and math_scores:
                summary_parts.append(f"Overall Average: {_safe_rich_value(np.mean(code_scores + math_scores), '{:.2%}')}")

            _RICH_CONSOLE.print(Panel.fit(
                "\n".join(summary_parts) if summary_parts else "No aggregate metrics available.",
                title="[bold cyan]AZR Benchmark Summary[/bold cyan]",
                border_style="bright_blue",
            ))
            _RICH_CONSOLE.print(table)
        else:
            print("\n" + "="*60)
            print(f"EVALUATION SUMMARY - {self.model_name}")
            print("="*60)
            for benchmark, data in results.items():
                bn = str(benchmark).lower()
                acc = data.get("accuracy") if isinstance(data, dict) else None
                if isinstance(data, dict) and _is_numeric_score(acc):
                    accuracy = float(acc)
                    print(f"{benchmark.upper()}: {accuracy:.2%} "
                          f"({data.get('correct', 0)}/{data.get('total', 0)})")

                    if bn in code_benchmarks:
                        code_scores.append(accuracy)
                    elif bn in math_benchmarks:
                        math_scores.append(accuracy)

            print("-"*60)

            if code_scores:
                print(f"Code Average: {np.mean(code_scores):.2%}")
            if math_scores:
                print(f"Math Average: {np.mean(math_scores):.2%}")
            if code_scores and math_scores:
                print(f"Overall Average: {np.mean(code_scores + math_scores):.2%}")

            print("="*60)


def set_global_seed(seed: int) -> None:
    """Set all relevant global RNG seeds for deterministic generation/evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_python_code(text: str) -> str:
    """Extract the first Python code block if fenced; otherwise return text as-is."""
    if "```python" in text:
        try:
            return text.split("```python", 1)[1].split("```", 1)[0]
        except Exception:
            return text
    if "```" in text:
        try:
            return text.split("```", 1)[1].split("```", 1)[0]
        except Exception:
            return text
    return text

def infer_function_name_from_tests(tests: list[str]) -> Optional[str]:
    """Attempt to infer the required function name from MBPP tests."""
    candidates = []
    for t in tests or []:
        # Common pattern: assert fn_name(
        m = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
        if m:
            candidates.append(m.group(1))
        # Patterns like: print(fn_name( ... ))
        if not m:
            m2 = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
            if m2:
                candidates.append(m2.group(1))
    return candidates[0] if candidates else None

def evaluate_programbench_run_dir(run_dir: str) -> Dict[str, Any]:
    """Aggregate scores from an existing ProgramBench agent run (nested ``*/*.eval.json``).

    Full instance evaluation (Docker, Linux x86_64) is done by the upstream ``programbench`` tool;
    this helper mirrors ``programbench info`` so AZR can report mean instance score alongside
    HumanEval/MBPP/GSM8K. See https://github.com/AshutoshBuilds/ProgramBench
    """
    run_path = Path(run_dir).expanduser()
    if not str(run_dir).strip():
        return {
            "benchmark": "ProgramBench",
            "error": (
                "ProgramBench was requested but no run directory was set. Pass "
                "--programbench-run-dir for evaluate_benchmarks, or set baseline/improved run dirs "
                "from run_pre_post_benchmarks. Omit ``programbench`` from --benchmarks if you are "
                "not using ProgramBench yet."
            ),
        }
    if not run_path.is_dir():
        return {"benchmark": "ProgramBench", "error": f"ProgramBench run dir not found: {run_path}"}

    try:
        from programbench.eval.eval import EvaluationResult
        from programbench.eval.eval_batch import BatchEvalSummary, InstanceEvalSummary
        from programbench.utils.load_data import get_active_branches, get_ignored_tests, load_all_instances
    except ImportError:
        return {
            "benchmark": "ProgramBench",
            "error": (
                "programbench is not installed. pip install programbench "
                "(https://github.com/AshutoshBuilds/ProgramBench)"
            ),
        }

    eval_paths = sorted(run_path.glob("*/*.eval.json"))
    if not eval_paths:
        return {
            "benchmark": "ProgramBench",
            "error": (
                f"No instance eval files (pattern: <instance>/*.eval.json) under {run_path}. "
                "After `programbench eval <run_dir>` on a supported host, point --programbench-run-dir at that run directory."
            ),
        }

    instances = {i["instance_id"]: i for i in load_all_instances(include_tests=True)}
    summaries = []
    for p in eval_paths:
        iid = p.parent.name
        result = EvaluationResult.model_validate_json(p.read_text(encoding="utf-8"))
        inst = instances.get(iid)
        if inst is not None:
            active = get_active_branches(inst)
            ignored_tests = get_ignored_tests(inst)
            ignored_branches = {b for b in result.test_branches if b not in set(active)}
            result = result.for_branches(active).without_ignored(ignored_tests)
            if ignored_branches:
                result.warnings = [
                    w for w in result.warnings if not any(f"branch {b}" in w for b in ignored_branches)
                ]
        summaries.append(InstanceEvalSummary.from_eval_result(iid, result))

    batch = BatchEvalSummary(summaries=summaries)
    n = len(summaries)
    solved = sum(1 for s in summaries if s.score >= 1.0 - 1e-12)
    short_results = [{"instance_id": s.instance_id, "score": s.score} for s in summaries[:200]]
    return {
        "benchmark": "ProgramBench",
        "total": n,
        "correct": solved,
        "accuracy": float(batch.average_pass_rate),
        "note": (
            "accuracy = mean per-instance score; correct = count with score==1 "
            "(aligned with programbench info)."
        ),
        "results": short_results,
    }


def build_code_gen_kwargs(samples_per_task: int, temperature: float, top_p: float) -> dict:
    """Return generation kwargs for code tasks supporting greedy, beam, or sampling modes.
    - If samples_per_task <= 1: greedy decoding (no temperature/top_p — avoids Transformers unused-kwarg warnings)
    - If samples_per_task > 1 and temperature > 0: sampling with num_return_sequences=samples_per_task
    - If samples_per_task > 1 and temperature <= 0: beam search with num_beams=num_return_sequences=samples_per_task
    """
    if samples_per_task <= 1:
        # Omit temperature/top_p when not sampling — avoids Transformers "may be ignored" warnings.
        return {
            'do_sample': False,
            'num_return_sequences': 1,
        }
    if temperature and temperature > 0:
        return {
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': samples_per_task,
        }
    # Beam search path
    return {
        'do_sample': False,
        'num_beams': samples_per_task,
        'num_return_sequences': samples_per_task,
    }


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AZR model on benchmarks")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B",
                        help="Model name or path")
    parser.add_argument("--benchmarks", type=str, nargs='+', 
                        default=['humaneval', 'mbpp', 'gsm8k'],
                        help="Benchmarks to evaluate on")
    parser.add_argument("--results-dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max samples per benchmark (HumanEval/MBPP/GSM8K/MATH capped path); may be capped lower by AZR_BENCHMARK_MAX_TASKS_PER_DATASET")
    parser.add_argument("--samples-per-task", type=int, default=8,
                        help="Number of generations per task (used for approximate pass@k)")
    parser.add_argument("--passk", type=int, default=1,
                        help="Accept task as passed if any of k samples succeed (approx pass@k)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Generation top_p")
    parser.add_argument("--k-reference", type=int, default=6, help="Number of few-shot examples for proposer prompts")
    parser.add_argument(
        "--use-separate-value-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load actor + critic ValueModel (paper-style PPO). Use --no-use-separate-value-model for unified actor-critic.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rich styled terminal output",
    )
    parser.add_argument(
        "--cpu-cap",
        type=float,
        default=20.0,
        help="CPU cap percentage (0-100) for this process",
    )
    parser.add_argument(
        "--programbench-run-dir",
        type=str,
        default="",
        help=(
            "Directory with ProgramBench per-instance *.eval.json (after `programbench eval`). "
            "See https://github.com/AshutoshBuilds/ProgramBench"
        ),
    )
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Load base weights in 4-bit when bitsandbytes is available. "
            "If omitted, uses AZR_USE_4BIT from the environment when set; otherwise matches hf_trainer (off)."
        ),
    )

    args = parser.parse_args()
    apply_benchmark_offline_env()
    if args.use_4bit is None:
        raw = (os.environ.get("AZR_USE_4BIT") or "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            args.use_4bit = True
        elif raw in ("0", "false", "no", "off", ""):
            args.use_4bit = False
        else:
            raise ValueError(
                f"Invalid AZR_USE_4BIT environment value: {os.environ.get('AZR_USE_4BIT')!r}. "
                "Use true/false, 1/0, yes/no, or on/off."
            )
    args.benchmarks = list(dict.fromkeys(
        str(item).strip().lower() for item in args.benchmarks
    ))
    # Accept `humaneval,mbpp` as a single argv token (common from shells / env) so eval branches run.
    _bench_flat: list[str] = []
    for item in args.benchmarks:
        for part in str(item).split(","):
            p = part.strip().lower()
            if p:
                _bench_flat.append(p)
    args.benchmarks = list(dict.fromkeys(_bench_flat))
    _apply_cpu_cap(args.cpu_cap)
 
    configure_logging(args.results_dir, use_rich=args.rich)

    if args.seed is not None:
        set_global_seed(args.seed)

    apply_benchmark_speed_from_env(args)

    tok_profile = resolve_max_new_tokens_by_benchmark(args.benchmark_fast)
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        model_name=args.model,
        results_dir=args.results_dir,
        use_separate_value_model=args.use_separate_value_model,
        generation_temperature=args.temperature,
        top_p=args.top_p,
        samples_per_task=args.samples_per_task,
        load_in_4bit=args.use_4bit,
        max_new_tokens_by_benchmark=tok_profile,
    )
    
    # Run evaluations
    if 'all' in args.benchmarks:
        results = evaluator.run_all_evaluations()
    else:
        results = {}
        if 'humaneval' in args.benchmarks:
            # HumanEval has ~164 tasks; cap by args.limit
            def eval_he_capped():
                try:
                    logger.info("HumanEval (capped): loading dataset (openai_humaneval, split=test)...")
                    dataset = load_azr_benchmark_split("openai_humaneval", "test", None, logger=logger)
                    subset = dataset.select(range(min(len(dataset), args.limit)))
                    logger.info(f"HumanEval (capped): dataset loaded with {len(subset)} tasks")
                    # Inline eval mirroring evaluate_humaneval but using subset
                    logger.info("Evaluating on HumanEval (capped)...")
                    correct = 0
                    results_local = []
                    start_ts = time.perf_counter()
                    subset_list = list(subset)
                    gen_kwargs = evaluator.generation_kwargs
                    mt = evaluator.max_new_tokens["humaneval"]
                    bs = _benchmark_gen_microbatch_size(args.benchmark_batch_size, gen_kwargs)
                    if bs > 1:
                        logger.info("HumanEval (capped): micro-batch size=%s", bs)
                    completed = 0
                    with tqdm(total=len(subset_list), desc="HumanEval") as pbar:
                        for chunk_start in range(0, len(subset_list), bs):
                            chunk = subset_list[chunk_start : chunk_start + bs]
                            prompts: List[str] = []
                            metas: List[Tuple[Any, ...]] = []
                            for example in chunk:
                                task_id = example["task_id"]
                                prompt = example["prompt"]
                                test = example["test"]
                                entry_point = example["entry_point"]
                                prompt_with_constraints = (
                                    f"Implement the Python function `{entry_point}` exactly as specified.\n"
                                    f"- Do not include any tests, comments, prints, or imports.\n"
                                    f"- Only output the function definition in a Python fenced block.\n\n"
                                    f"Problem:\n{prompt}\n\n"
                                    f"Output format:\n```python\n# your function here\n```\n"
                                )
                                prompts.append(prompt_with_constraints)
                                metas.append((task_id, prompt, test, entry_point))
                            per_prompt_seqs: List[List[str]] = []
                            if bs <= 1 or not _can_benchmark_microbatch(gen_kwargs):
                                for p in prompts:
                                    per_prompt_seqs.append(
                                        evaluator.adapter.generate(p, max_new_tokens=mt, **gen_kwargs)
                                    )
                            else:
                                flat = evaluator.adapter.generate_batch(
                                    prompts, max_new_tokens=mt, **gen_kwargs
                                )
                                per_prompt_seqs = [[s] for s in flat]
                            for (task_id, prompt, test, entry_point), seqs in zip(metas, per_prompt_seqs):
                                passed_any = False
                                first_code = None
                                for gen in seqs:
                                    code = extract_python_code(gen)
                                    if first_code is None:
                                        first_code = code
                                    full_code = prompt + code + "\n\n" + test
                                    result = evaluator.executor.execute(
                                        code=full_code, test_input="", timeout=5
                                    )
                                    if result.get("success", False) and not result.get("error"):
                                        passed_any = True
                                        break
                                if passed_any:
                                    correct += 1
                                completed += 1
                                if completed % TASK_PROGRESS_EVERY == 0 or completed == len(subset_list):
                                    _log_task_heartbeat(
                                        "HumanEval (capped)",
                                        completed,
                                        len(subset_list),
                                        start_ts,
                                        correct=correct,
                                    )
                                results_local.append(
                                    {
                                        "task_id": task_id,
                                        "passed": passed_any,
                                        "generated": first_code,
                                        "samples": len(seqs),
                                    }
                                )
                            pbar.update(len(chunk))
                    elapsed = time.perf_counter() - start_ts
                    logger.info(
                        "HumanEval (capped) complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                        len(subset),
                        correct,
                        (correct / len(subset) if len(subset) else 0.0),
                        _format_elapsed(elapsed),
                    )
                    total = len(subset); accuracy = correct / total if total > 0 else 0
                    return {'benchmark': 'HumanEval', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
                except Exception as e:
                    logger.error(f"Error evaluating HumanEval (capped): {e}")
                    return {'benchmark': 'HumanEval', 'error': str(e)}
            results['humaneval'] = eval_he_capped()
        if 'mbpp' in args.benchmarks:
            # MBPP sanitized split, cap by args.limit
            try:
                logger.info("MBPP (capped): loading dataset (mbpp, sanitized, split=test)...")
                dataset = load_azr_benchmark_split("mbpp", "test", "sanitized", logger=logger)
                subset = dataset.select(range(min(len(dataset), args.limit)))
                logger.info(f"MBPP (capped): dataset loaded with {len(subset)} tasks")
                logger.info("Evaluating on MBPP (capped)...")
                correct = 0
                results_local = []
                start_ts = time.perf_counter()
                subset_list = list(subset)
                gen_kwargs = evaluator.generation_kwargs
                mt = evaluator.max_new_tokens["mbpp"]
                bs = _benchmark_gen_microbatch_size(args.benchmark_batch_size, gen_kwargs)
                if bs > 1:
                    logger.info("MBPP (capped): micro-batch size=%s", bs)
                completed = 0
                with tqdm(total=len(subset_list), desc="MBPP") as pbar:
                    for chunk_start in range(0, len(subset_list), bs):
                        chunk = subset_list[chunk_start : chunk_start + bs]
                        prompts: List[str] = []
                        metas: List[Tuple[Any, ...]] = []
                        for example in chunk:
                            task_id = example.get("task_id", None)
                            description = (
                                example.get("text")
                                or example.get("prompt")
                                or example.get("description")
                                or ""
                            )
                            tests = example.get("test_list") or example.get("tests") or []
                            if not tests and "test" in example:
                                tests = [example["test"]]
                            fn_name = infer_function_name_from_tests(tests) or "solution"
                            prompt_with_constraints = (
                                f"Implement the function `{fn_name}` only.\n"
                                f"- Do not include tests, comments, prints, or imports.\n"
                                f"- Only output the function definition in a Python fenced block.\n\n"
                                f"Problem:\n{description}\n\n"
                                f"Output format:\n```python\n# your function here\n```\n"
                            )
                            prompts.append(prompt_with_constraints)
                            metas.append((task_id, tests))
                        per_prompt_seqs: List[List[str]] = []
                        if bs <= 1 or not _can_benchmark_microbatch(gen_kwargs):
                            for p in prompts:
                                per_prompt_seqs.append(
                                    evaluator.adapter.generate(p, max_new_tokens=mt, **gen_kwargs)
                                )
                        else:
                            flat = evaluator.adapter.generate_batch(
                                prompts, max_new_tokens=mt, **gen_kwargs
                            )
                            per_prompt_seqs = [[s] for s in flat]
                        for (task_id, tests), seqs in zip(metas, per_prompt_seqs):
                            passed_any = False
                            first_code = None
                            for gen in seqs:
                                code = extract_python_code(gen)
                                if first_code is None:
                                    first_code = code
                                all_passed = True
                                for test in tests:
                                    full_code = code + "\n\n" + test
                                    result = evaluator.executor.execute(
                                        code=full_code, test_input="", timeout=5
                                    )
                                    if not result.get("success", False):
                                        all_passed = False
                                        break
                                if all_passed:
                                    passed_any = True
                                    break
                            if passed_any:
                                correct += 1
                            completed += 1
                            if completed % TASK_PROGRESS_EVERY == 0 or completed == len(subset_list):
                                _log_task_heartbeat(
                                    "MBPP (capped)",
                                    completed,
                                    len(subset_list),
                                    start_ts,
                                    correct=correct,
                                )
                            results_local.append(
                                {
                                    "task_id": task_id,
                                    "passed": passed_any,
                                    "generated": first_code,
                                    "samples": len(seqs),
                                }
                            )
                        pbar.update(len(chunk))
                elapsed = time.perf_counter() - start_ts
                logger.info(
                    "MBPP (capped) complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                    len(subset),
                    correct,
                    (correct / len(subset) if len(subset) else 0.0),
                    _format_elapsed(elapsed),
                )
                total = len(subset); accuracy = correct / total if total > 0 else 0
                results['mbpp'] = {'benchmark': 'MBPP', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating MBPP (capped): {e}")
                results['mbpp'] = {'benchmark': 'MBPP', 'error': str(e)}
        if 'gsm8k' in args.benchmarks:
            # Use existing method but override its internal cap with args.limit
            try:
                logger.info("GSM8K (capped): loading dataset (gsm8k, main, split=test)...")
                dataset = load_azr_benchmark_split("gsm8k", "test", "main", logger=logger)
                total = min(len(dataset), args.limit)
                correct = 0; results_local = []
                logger.info(f"GSM8K (capped): dataset loaded with {total} candidate tasks")
                import re

                start_ts = time.perf_counter()
                subset_list = list(dataset.select(range(total)))
                gen_kwargs = evaluator.generation_kwargs
                mt = evaluator.max_new_tokens["gsm8k"]
                bs = _benchmark_gen_microbatch_size(args.benchmark_batch_size, gen_kwargs)
                if bs > 1:
                    logger.info("GSM8K (capped): micro-batch size=%s", bs)
                answer_patterns = [
                    r"answer is (\-?\d+(?:\.\d+)?)",
                    r"= (\-?\d+(?:\.\d+)?)\s*$",
                    r"equals (\-?\d+(?:\.\d+)?)",
                    r"result is (\-?\d+(?:\.\d+)?)",
                ]
                completed = 0
                with tqdm(total=len(subset_list), desc="GSM8K") as pbar:
                    for chunk_start in range(0, len(subset_list), bs):
                        chunk = subset_list[chunk_start : chunk_start + bs]
                        prompts: List[str] = []
                        metas: List[Tuple[str, int]] = []
                        for offset, example in enumerate(chunk):
                            question = example["question"]
                            answer = example["answer"]
                            true_answer = answer.split("####")[-1].strip()
                            prompt = f"""Solve this math problem step by step.

Question: {question}

Let's solve this step by step:
"""
                            prompts.append(prompt)
                            idx = chunk_start + offset + 1
                            metas.append((true_answer, idx))
                        per_prompt_seqs: List[List[str]] = []
                        if bs <= 1 or not _can_benchmark_microbatch(gen_kwargs):
                            for p in prompts:
                                per_prompt_seqs.append(
                                    evaluator.adapter.generate(p, max_new_tokens=mt, **gen_kwargs)
                                )
                        else:
                            flat = evaluator.adapter.generate_batch(
                                prompts, max_new_tokens=mt, **gen_kwargs
                            )
                            per_prompt_seqs = [[s] for s in flat]
                        for (true_answer, idx), seqs in zip(metas, per_prompt_seqs):
                            predicted_answer = None
                            passed_any = False
                            for gen in seqs:
                                for pattern in answer_patterns:
                                    matches = re.findall(pattern, gen, re.IGNORECASE | re.MULTILINE)
                                    if matches:
                                        predicted_answer = matches[-1]
                                        break
                                if predicted_answer:
                                    try:
                                        if abs(float(predicted_answer) - float(true_answer)) < 1e-5:
                                            passed_any = True
                                            break
                                    except (ValueError, TypeError):
                                        pass
                            if passed_any:
                                correct += 1
                            completed += 1
                            if completed % TASK_PROGRESS_EVERY == 0 or completed == total:
                                _log_task_heartbeat(
                                    "GSM8K (capped)", completed, total, start_ts, correct=correct
                                )
                            results_local.append(
                                {
                                    "idx": idx,
                                    "correct": passed_any,
                                    "predicted": predicted_answer,
                                    "true": true_answer,
                                    "samples": len(seqs),
                                }
                            )
                        pbar.update(len(chunk))
                elapsed = time.perf_counter() - start_ts
                logger.info(
                    "GSM8K (capped) complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                    total,
                    correct,
                    (correct / total if total else 0.0),
                    _format_elapsed(elapsed),
                )
                accuracy = correct / total if total > 0 else 0
                results['gsm8k'] = {'benchmark': 'GSM8K', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating GSM8K (capped): {e}")
                results['gsm8k'] = {'benchmark': 'GSM8K', 'error': str(e)}
        if 'math' in args.benchmarks:
            try:
                logger.info("MATH (capped): loading dataset (hendrycks/competition_math, split=test)...")
                dataset = load_azr_benchmark_split("hendrycks/competition_math", "test", None, logger=logger)
                total = min(len(dataset), args.limit)
                logger.info(f"MATH (capped): dataset loaded with {total} candidate tasks")
                correct = 0; results_local = []
                import re

                boxed_pattern = r"\\boxed\{([^}]+)\}"
                start_ts = time.perf_counter()
                subset_list = list(dataset.select(range(total)))
                gen_kwargs = evaluator.generation_kwargs
                mt = evaluator.max_new_tokens["math"]
                bs = _benchmark_gen_microbatch_size(args.benchmark_batch_size, gen_kwargs)
                if bs > 1:
                    logger.info("MATH (capped): micro-batch size=%s", bs)
                completed = 0
                with tqdm(total=len(subset_list), desc="MATH") as pbar:
                    for chunk_start in range(0, len(subset_list), bs):
                        chunk = subset_list[chunk_start : chunk_start + bs]
                        prompts: List[str] = []
                        metas: List[Tuple[Any, ...]] = []
                        for offset, example in enumerate(chunk):
                            problem = example["problem"]
                            solution = example["solution"]
                            level = example["level"]
                            problem_type = example["type"]
                            prompt = f"""Solve this math problem step by step.

Problem: {problem}

Solution:
"""
                            prompts.append(prompt)
                            idx = chunk_start + offset + 1
                            metas.append((solution, level, problem_type, idx))
                        per_prompt_texts: List[str] = []
                        if bs <= 1 or not _can_benchmark_microbatch(gen_kwargs):
                            for p in prompts:
                                per_prompt_texts.append(
                                    evaluator.adapter.generate(p, max_new_tokens=mt, **gen_kwargs)[0]
                                )
                        else:
                            per_prompt_texts = evaluator.adapter.generate_batch(
                                prompts, max_new_tokens=mt, **gen_kwargs
                            )
                        for (solution, level, problem_type, idx), generated in zip(
                            metas, per_prompt_texts
                        ):
                            matches = re.findall(boxed_pattern, generated)
                            predicted_answer = matches[-1] if matches else None
                            true_matches = re.findall(boxed_pattern, solution)
                            true_answer = true_matches[-1] if true_matches else None
                            is_correct = (
                                predicted_answer == true_answer
                                if predicted_answer and true_answer
                                else False
                            )
                            if is_correct:
                                correct += 1
                            completed += 1
                            if completed % TASK_PROGRESS_EVERY == 0 or completed == total:
                                _log_task_heartbeat(
                                    "MATH (capped)", completed, total, start_ts, correct=correct
                                )
                            results_local.append(
                                {
                                    "idx": idx,
                                    "level": level,
                                    "type": problem_type,
                                    "correct": is_correct,
                                    "predicted": predicted_answer,
                                    "true": true_answer,
                                }
                            )
                        pbar.update(len(chunk))
                elapsed = time.perf_counter() - start_ts
                logger.info(
                    "MATH (capped) complete: total=%s correct=%s accuracy=%.4f elapsed=%s",
                    total,
                    correct,
                    (correct / total if total else 0.0),
                    _format_elapsed(elapsed),
                )
                accuracy = correct / total if total > 0 else 0
                results['math'] = {'benchmark': 'MATH', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating MATH (capped): {e}")
                results['math'] = {'benchmark': 'MATH', 'error': str(e)}
        if 'programbench' in args.benchmarks:
            logger.info("ProgramBench: aggregating existing eval artifacts (no model generation)...")
            results['programbench'] = evaluate_programbench_run_dir(args.programbench_run_dir)
        
        evaluator.save_results(results)
        evaluator.print_summary(results)


if __name__ == "__main__":
    main()
