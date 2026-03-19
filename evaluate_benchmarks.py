"""
Evaluation script for AZR model on standard benchmarks.
Evaluates on:
- Code: HumanEval, MBPP, CruxEval, LiveCodeBench
- Math: GSM8K, MATH
"""

import os
import json
import time
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any
from datetime import datetime
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm
import re
import math

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


class BenchmarkEvaluator:
    """Evaluator for standard benchmarks"""
    
    def __init__(
        self,
        model_name: str,
        results_dir: str = "evaluation_results",
        use_separate_value_model: bool = False,
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
        
        # Initialize model adapter with local cache
        self.adapter = HuggingFaceAdapter(
            self.model_name,
            hf_cache_dir=str(self.hf_cache_dir),
            use_separate_value_model=use_separate_value_model,
        )
        
        # Initialize code executor for code benchmarks
        self.executor = CodeExecutor(timeout_seconds=10)
        
        self.results = {}
    
    def evaluate_humaneval(self) -> Dict:
        """Evaluate on HumanEval benchmark"""
        logger.info("Evaluating on HumanEval...")
        
        try:
            # Load HumanEval dataset
            dataset = load_dataset("openai_humaneval", split="test")
            
            results = []
            correct = 0
            total = len(dataset)
            
            for idx, example in enumerate(tqdm(dataset, desc="HumanEval")):
                task_id = example['task_id']
                prompt = example['prompt']
                test = example['test']
                entry_point = example['entry_point']
                
                # Generate solution
                generated = self.adapter.generate(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
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
                
                results.append({
                    'task_id': task_id,
                    'passed': passed,
                    'generated': code,
                    'error': result.get('error', '')
                })
            
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
        logger.info("Evaluating on MBPP...")
        
        try:
            # Load MBPP dataset
            dataset = load_dataset("mbpp", "sanitized", split="test")
            
            results = []
            correct = 0
            total = len(dataset)
            
            for idx, example in enumerate(tqdm(dataset, desc="MBPP")):
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
                    max_new_tokens=256,
                    temperature=0.2,
                    top_p=0.95
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
                
                results.append({
                    'task_id': task_id,
                    'passed': all_passed,
                    'generated': code
                })
            
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
        logger.info("Evaluating on GSM8K...")
        
        try:
            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split="test")
            
            results = []
            correct = 0
            total = min(len(dataset), 500)  # Evaluate on subset for speed
            
            for idx, example in enumerate(tqdm(dataset.select(range(total)), desc="GSM8K")):
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
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
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
                    except:
                        pass
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'idx': idx,
                    'correct': is_correct,
                    'predicted': predicted_answer,
                    'true': true_answer,
                    'generated': generated
                })
            
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
        logger.info("Evaluating on MATH...")
        
        try:
            # Load MATH dataset
            dataset = load_dataset("hendrycks/competition_math", split="test")
            
            results = []
            correct = 0
            total = min(len(dataset), 200)  # Evaluate on subset for speed
            
            for idx, example in enumerate(tqdm(dataset.select(range(total)), desc="MATH")):
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
                    max_new_tokens=1024,
                    temperature=0.2,
                    top_p=0.95
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
                
                results.append({
                    'idx': idx,
                    'level': level,
                    'type': problem_type,
                    'correct': is_correct,
                    'predicted': predicted_answer,
                    'true': true_answer
                })
            
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
                if isinstance(data, dict) and 'accuracy' in data:
                    accuracy = data['accuracy']
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

                    if benchmark in code_benchmarks:
                        code_scores.append(accuracy)
                    elif benchmark in math_benchmarks:
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
                if isinstance(data, dict) and 'accuracy' in data:
                    print(f"{benchmark.upper()}: {data['accuracy']:.2%} "
                          f"({data.get('correct', 0)}/{data.get('total', 0)})")

                    if benchmark in code_benchmarks:
                        code_scores.append(data['accuracy'])
                    elif benchmark in math_benchmarks:
                        math_scores.append(data['accuracy'])

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

def infer_function_name_from_tests(tests: list[str]) -> str | None:
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

def build_code_gen_kwargs(samples_per_task: int, temperature: float, top_p: float) -> dict:
    """Return generation kwargs for code tasks supporting greedy, beam, or sampling modes.
    - If samples_per_task <= 1: greedy decoding
    - If samples_per_task > 1 and temperature > 0: sampling with num_return_sequences=samples_per_task
    - If samples_per_task > 1 and temperature <= 0: beam search with num_beams=num_return_sequences=samples_per_task
    """
    if samples_per_task <= 1:
        return {
            'do_sample': False,
            'temperature': 0.0,
            'top_p': 1.0,
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
        'temperature': 0.0,
        'top_p': 1.0,
        'num_beams': samples_per_task,
        'num_return_sequences': samples_per_task,
    }


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AZR model on benchmarks")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Model name or path")
    parser.add_argument("--benchmarks", type=str, nargs='+', 
                        default=['humaneval', 'mbpp', 'gsm8k'],
                        help="Benchmarks to evaluate on")
    parser.add_argument("--results-dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max samples per benchmark for quick evaluation (applies to supported datasets)")
    parser.add_argument("--samples-per-task", type=int, default=1,
                        help="Number of generations per task (used for approximate pass@k)")
    parser.add_argument("--passk", type=int, default=1,
                        help="Accept task as passed if any of k samples succeed (approx pass@k)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Generation top_p")
    parser.add_argument(
        "--use-separate-value-model",
        action="store_true",
        default=False,
        help="Enable separate critic value model path (falls back to single-model when False)"
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
 
    args = parser.parse_args()
    args.benchmarks = list(dict.fromkeys(
        str(item).strip().lower() for item in args.benchmarks
    ))
    _apply_cpu_cap(args.cpu_cap)
 
    configure_logging(args.results_dir, use_rich=args.rich)

    if args.seed is not None:
        set_global_seed(args.seed)
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        model_name=args.model,
        results_dir=args.results_dir,
        use_separate_value_model=args.use_separate_value_model,
    )
    
    # Run evaluations
    if 'all' in args.benchmarks:
        results = evaluator.run_all_evaluations()
    else:
        results = {}
        if 'humaneval' in args.benchmarks:
            # HumanEval has ~164 tasks; cap by args.limit
            original_fn = evaluator.evaluate_humaneval
            def eval_he_capped():
                try:
                    dataset = load_dataset("openai_humaneval", split="test")
                    subset = dataset.select(range(min(len(dataset), args.limit)))
                    # Inline eval mirroring evaluate_humaneval but using subset
                    logger.info("Evaluating on HumanEval (capped)...")
                    correct = 0; results_local = []
                    for example in tqdm(subset, desc="HumanEval"):
                        task_id = example['task_id']; prompt = example['prompt']; test = example['test']; entry_point = example['entry_point']
                        # Enforce function name and code-only output
                        prompt_with_constraints = (
                            f"Implement the Python function `{entry_point}` exactly as specified.\n"
                            f"- Do not include any tests, comments, prints, or imports.\n"
                            f"- Only output the function definition in a Python fenced block.\n\n"
                            f"Problem:\n{prompt}\n\n"
                            f"Output format:\n```python\n# your function here\n```\n"
                        )
                        he_kwargs = build_code_gen_kwargs(args.samples_per_task, args.temperature, args.top_p)
                        gens = evaluator.adapter.generate(
                            prompt_with_constraints,
                            max_new_tokens=512,
                            **he_kwargs
                        )
                        passed_any = False
                        first_code = None
                        for gen in gens:
                            code = extract_python_code(gen)
                            if first_code is None:
                                first_code = code
                            full_code = prompt + code + "\n\n" + test
                            result = evaluator.executor.execute(code=full_code, test_input="", timeout=5)
                            if result.get('success', False) and not result.get('error'):
                                passed_any = True
                                break
                        if passed_any: correct += 1
                        results_local.append({'task_id': task_id, 'passed': passed_any, 'generated': first_code, 'samples': len(gens)})
                    total = len(subset); accuracy = correct / total if total > 0 else 0
                    return {'benchmark': 'HumanEval', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
                except Exception as e:
                    logger.error(f"Error evaluating HumanEval (capped): {e}")
                    return {'benchmark': 'HumanEval', 'error': str(e)}
            results['humaneval'] = eval_he_capped()
        if 'mbpp' in args.benchmarks:
            # MBPP sanitized split, cap by args.limit
            try:
                dataset = load_dataset("mbpp", "sanitized", split="test")
                subset = dataset.select(range(min(len(dataset), args.limit)))
                logger.info("Evaluating on MBPP (capped)...")
                correct = 0; results_local = []
                for example in tqdm(subset, desc="MBPP"):
                    task_id = example.get('task_id', None)
                    description = example.get('text') or example.get('prompt') or example.get('description') or ""
                    tests = example.get('test_list') or example.get('tests') or []
                    if not tests and 'test' in example:
                        tests = [example['test']]
                    fn_name = infer_function_name_from_tests(tests) or "solution"
                    prompt = f"Write a Python function named `{fn_name}` to solve this problem:\n{description}\n\n"
                    prompt_with_constraints = (
                        f"Implement the function `{fn_name}` only.\n"
                        f"- Do not include tests, comments, prints, or imports.\n"
                        f"- Only output the function definition in a Python fenced block.\n\n"
                        f"Problem:\n{description}\n\n"
                        f"Output format:\n```python\n# your function here\n```\n"
                    )
                    mbpp_kwargs = build_code_gen_kwargs(args.samples_per_task, args.temperature, args.top_p)
                    gens = evaluator.adapter.generate(
                        prompt_with_constraints,
                        max_new_tokens=256,
                        **mbpp_kwargs
                    )
                    passed_any = False
                    first_code = None
                    for gen in gens:
                        code = extract_python_code(gen)
                        if first_code is None:
                            first_code = code
                        all_passed = True
                        for test in tests:
                            full_code = code + "\n\n" + test
                            result = evaluator.executor.execute(code=full_code, test_input="", timeout=5)
                            if not result.get('success', False):
                                all_passed = False
                                break
                        if all_passed:
                            passed_any = True
                            break
                    if passed_any: correct += 1
                    results_local.append({'task_id': task_id, 'passed': passed_any, 'generated': first_code, 'samples': len(gens)})
                total = len(subset); accuracy = correct / total if total > 0 else 0
                results['mbpp'] = {'benchmark': 'MBPP', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating MBPP (capped): {e}")
                results['mbpp'] = {'benchmark': 'MBPP', 'error': str(e)}
        if 'gsm8k' in args.benchmarks:
            # Use existing method but override its internal cap with args.limit
            try:
                dataset = load_dataset("gsm8k", "main", split="test")
                total = min(len(dataset), args.limit)
                correct = 0; results_local = []
                import re
                for idx, example in enumerate(tqdm(dataset.select(range(total)), desc="GSM8K")):
                    question = example['question']; answer = example['answer']; true_answer = answer.split("####")[-1].strip()
                    prompt = f"""Solve this math problem step by step.
 
  Question: {question}
 
  Let's solve this step by step:
  """
                    gens = evaluator.adapter.generate(
                        prompt,
                        max_new_tokens=512,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_return_sequences=args.samples_per_task,
                        do_sample=True
                    )
                    answer_patterns = [r'answer is (\-?\d+(?:\.\d+)?)', r'= (\-?\d+(?:\.\d+)?)\s*$', r'equals (\-?\d+(?:\.\d+)?)', r'result is (\-?\d+(?:\.\d+)?)']
                    predicted_answer = None
                    passed_any = False
                    for gen in gens:
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
                            except:
                                pass
                    if passed_any: correct += 1
                    results_local.append({'idx': idx, 'correct': passed_any, 'predicted': predicted_answer, 'true': true_answer, 'samples': len(gens)})
                accuracy = correct / total if total > 0 else 0
                results['gsm8k'] = {'benchmark': 'GSM8K', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating GSM8K (capped): {e}")
                results['gsm8k'] = {'benchmark': 'GSM8K', 'error': str(e)}
        if 'math' in args.benchmarks:
            try:
                dataset = load_dataset("hendrycks/competition_math", split="test")
                total = min(len(dataset), args.limit)
                correct = 0; results_local = []
                import re
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                for idx, example in enumerate(tqdm(dataset.select(range(total)), desc="MATH")):
                    problem = example['problem']; solution = example['solution']; level = example['level']; problem_type = example['type']
                    prompt = f"""Solve this math problem step by step.

Problem: {problem}

Solution:
"""
                    generated = self.adapter.generate(
                        prompt,
                        max_new_tokens=1024,
                        temperature=0.2,
                        top_p=0.95
                    )[0]
                    matches = re.findall(boxed_pattern, generated)
                    predicted_answer = matches[-1] if matches else None
                    true_matches = re.findall(boxed_pattern, solution)
                    true_answer = true_matches[-1] if true_matches else None
                    is_correct = predicted_answer == true_answer if predicted_answer and true_answer else False
                    if is_correct: correct += 1
                    results_local.append({'idx': idx, 'level': level, 'type': problem_type, 'correct': is_correct, 'predicted': predicted_answer, 'true': true_answer})
                accuracy = correct / total if total > 0 else 0
                results['math'] = {'benchmark': 'MATH', 'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results_local}
            except Exception as e:
                logger.error(f"Error evaluating MATH (capped): {e}")
                results['math'] = {'benchmark': 'MATH', 'error': str(e)}
        
        evaluator.save_results(results)
        evaluator.print_summary(results)


if __name__ == "__main__":
    main()
