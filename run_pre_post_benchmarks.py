"""Pre/post benchmark runner for AZR model comparison.

Runs deterministic benchmark evaluations for a baseline model and an improved
model/checkpoint using the same protocol, then writes a single machine-readable
and human-readable comparison report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import numbers
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List
import math

REPO_ROOT = Path(__file__).resolve().parent

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Panel = None
    Table = None
    _RICH_AVAILABLE = False


_RICH_CONSOLE = Console(highlight=False, force_terminal=True) if _RICH_AVAILABLE else None


def _status(message: str, style: str = "") -> None:
    if _RICH_AVAILABLE and _RICH_CONSOLE is not None:
        if style:
            _RICH_CONSOLE.print(f"[{style}]{message}[/{style}]")
        else:
            _RICH_CONSOLE.print(f"[blue]{message}[/blue]")
    else:
        print(message)


def _run_command_with_live_logs(command: List[str], log_path: Path) -> int:
    """Run a command and stream stdout to both terminal and a log file."""
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
            cwd=str(REPO_ROOT),
        )

        assert process.stdout is not None
        for line in process.stdout:
            _status(line.rstrip())
            log_file.write(line)
        process.stdout.close()
        return_code = process.wait()
        return return_code


def _apply_cpu_cap(cpu_cap_percent: float, *, context: str = "process") -> int:
    """Apply a CPU usage cap by limiting thread pool sizes."""
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
        import torch

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
        _RICH_CONSOLE.print(f"[blue]CPU cap ({context}): {cap:.1f}%[/blue] -> max threads {max_threads}")
    else:
        print(f"CPU cap ({context}): {cap:.1f}% -> max threads {max_threads}")

    return max_threads


def _safe_path(value: str) -> str:
    return str(Path(value))


def _normalize_benchmark_list(names: List[str]) -> List[str]:
    """Split comma-separated tokens so `humaneval,mbpp` does not become one unknown benchmark name."""
    out: List[str] = []
    for raw in names:
        for part in str(raw).split(","):
            p = part.strip().lower()
            if p:
                out.append(p)
    return list(dict.fromkeys(out))


def _capture_env_snapshot(keys: List[str]) -> Dict[str, str]:
    return {key: os.environ.get(key, "") for key in keys if key in os.environ}


def run_eval_once(
    model: str,
    label: str,
    benchmark_list: List[str],
    limit: int,
    samples_per_task: int,
    passk: int,
    temperature: float,
    top_p: float,
    seed: int,
    use_separate_value_model: bool,
    use_rich: bool,
    results_root: Path,
    python_executable: str,
    cpu_cap_percent: float,
    programbench_run_dir: str = "",
    k_reference: int = 6,
    use_4bit: bool = False,
) -> Dict:
    """Run evaluate_benchmarks.py for one model and return the latest summary."""

    run_dir = results_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "comparison_eval.log"

    command = [
        python_executable,
        str(REPO_ROOT / "evaluate_benchmarks.py"),
        "--model",
        _safe_path(model),
        "--results-dir",
        _safe_path(str(run_dir)),
        "--limit",
        str(limit),
        "--samples-per-task",
        str(samples_per_task),
        "--passk",
        str(passk),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--k-reference",
        str(k_reference),
        "--seed",
        str(seed),
        "--benchmarks",
        *benchmark_list,
    ]

    bench_lower = [str(b).strip().lower() for b in benchmark_list]
    if "programbench" in bench_lower and str(programbench_run_dir or "").strip():
        command.extend(["--programbench-run-dir", _safe_path(programbench_run_dir)])

    if use_separate_value_model:
        command.append("--use-separate-value-model")
    else:
        command.append("--no-use-separate-value-model")
    if use_4bit:
        command.append("--use-4bit")
    else:
        command.append("--no-use-4bit")
    if use_rich:
        command.extend(["--rich"])
    else:
        command.extend(["--no-rich"])
    command.extend(["--cpu-cap", str(cpu_cap_percent)])

    _status(f"Running benchmark eval '{label}'")
    _status(f"Model: {model}")
    _status(f"Benchmarks: {', '.join(benchmark_list)}")
    _status(f"Logging to: {log_path}")
    return_code = _run_command_with_live_logs(command, log_path)
    if return_code != 0:
        _status(f"Benchmark eval '{label}' exited with code {return_code}", style="yellow")
    else:
        _status(f"Benchmark eval '{label}' completed")

    # Always persist command metadata for reproducibility
    meta_path = run_dir / "run_metadata.json"
    run_metadata = {
        "model": model,
        "label": label,
        "benchmarks": benchmark_list,
        "limit": limit,
        "samples_per_task": samples_per_task,
        "passk": passk,
        "temperature": temperature,
        "top_p": top_p,
        "k_reference": k_reference,
        "seed": seed,
        "actor_critic_mode": "separate" if use_separate_value_model else "unified",
        "use_separate_value_model": use_separate_value_model,
        "use_4bit": use_4bit,
        "paper_protocol": {
            "k_reference": k_reference,
            "samples_per_task": samples_per_task,
            "temperature": temperature,
            "top_p": top_p,
            "generation_num_return_sequences": samples_per_task,
            "fallback_supported": True,
        },
        "env_snapshot": _capture_env_snapshot(
            [
                "AZR_BENCHMARK_SAMPLES_PER_TASK",
                "AZR_BENCHMARK_TEMPERATURE",
                "AZR_BENCHMARK_TOP_P",
                "AZR_BENCHMARK_FAST",
                "AZR_BENCHMARK_MAX_TASKS_PER_DATASET",
                "AZR_BENCHMARK_BATCH_SIZE",
                "AZR_GENU_LOG_WARN_CAP",
                "AZR_USE_SEPARATE_VALUE_MODEL",
                "AZR_USE_4BIT",
                "AZR_TRAIN_BATCH_SIZE",
                "AZR_PPO_UPDATE_THRESHOLD",
            ]
        ),
        "programbench_run_dir": programbench_run_dir or None,
        "return_code": return_code,
        "command": command,
        "log_file": str(log_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    eval_files = sorted(run_dir.glob("eval_results_*.json"))
    if not eval_files:
        raise RuntimeError(
            f"No eval_results_*.json file produced for '{label}'. See {log_path}"
        )

    latest = eval_files[-1]
    results = json.loads(latest.read_text(encoding="utf-8"))

    return {
        "label": label,
        "model": model,
        "results_path": str(latest),
        "log_path": str(log_path),
        "return_code": return_code,
        "summary": results,
        "metadata": run_metadata,
    }


def compute_comparison(baseline: Dict, improved: Dict, benchmarks: List[str]) -> Dict:
    """Compute benchmark-by-benchmark deltas (improved - baseline)."""

    rows = []
    for benchmark in benchmarks:
        base_item = baseline["summary"].get(benchmark, {})
        imp_item = improved["summary"].get(benchmark, {})

        base_accuracy = base_item.get("accuracy")
        imp_accuracy = imp_item.get("accuracy")

        delta = None
        if (
            isinstance(base_accuracy, numbers.Real)
            and not isinstance(base_accuracy, bool)
            and isinstance(imp_accuracy, numbers.Real)
            and not isinstance(imp_accuracy, bool)
        ):
            delta = float(imp_accuracy) - float(base_accuracy)

        rows.append({
            "benchmark": benchmark,
            "baseline_accuracy": base_accuracy,
            "baseline_total": base_item.get("total"),
            "baseline_correct": base_item.get("correct"),
            "improved_accuracy": imp_accuracy,
            "improved_total": imp_item.get("total"),
            "improved_correct": imp_item.get("correct"),
            "delta_accuracy": delta,
        })

    return {
        "benchmarks": rows,
        "overall": {
            "baseline_return_code": baseline["return_code"],
            "improved_return_code": improved["return_code"],
            "baseline_model": baseline["model"],
            "improved_model": improved["model"],
            "baseline_results_path": baseline["results_path"],
            "improved_results_path": improved["results_path"],
        },
        "timedelta_utc": datetime.now(timezone.utc).isoformat(),
        "seed": baseline["metadata"]["seed"],
        "benchmarks_requested": benchmarks,
    }


def write_markdown_report(comparison: Dict, output_path: Path) -> None:
    lines = [
        "# AZR Benchmark Comparison",
        "",
        f"Generated: {comparison['timedelta_utc']}",
        "",
        "## Run Metadata",
        f"- Baseline model: `{comparison['overall']['baseline_model']}`",
        f"- Improved model: `{comparison['overall']['improved_model']}`",
        f"- Baseline return code: `{comparison['overall']['baseline_return_code']}`",
        f"- Improved return code: `{comparison['overall']['improved_return_code']}`",
        "",
        "| Benchmark | Baseline Accuracy | Improved Accuracy | Delta |",
        "| --- | --- | --- | --- |",
    ]

    for row in comparison["benchmarks"]:
        base_acc = "N/A" if row["baseline_accuracy"] is None else f"{row['baseline_accuracy']:.4f}"
        imp_acc = "N/A" if row["improved_accuracy"] is None else f"{row['improved_accuracy']:.4f}"
        if row["delta_accuracy"] is None:
            delta = "N/A"
        else:
            delta = f"{row['delta_accuracy']:+.4f}"
        lines.append(
            f"| {row['benchmark']} | {base_acc} | {imp_acc} | {delta} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_run_report(comparison: Dict, baseline: Dict, improved: Dict, use_rich: bool) -> None:
    if use_rich and _RICH_AVAILABLE and _RICH_CONSOLE is not None:
        if _RICH_AVAILABLE and Panel is not None and Table is not None:
            meta = (
                f"[cyan]Baseline[/cyan]: {comparison['overall']['baseline_model']}\n"
                f"[cyan]Improved[/cyan]: {comparison['overall']['improved_model']}\n"
                f"[cyan]Seed[/cyan]: {comparison['seed']}\n"
                f"[cyan]Return codes[/cyan]: "
                f"{comparison['overall']['baseline_return_code']} / {comparison['overall']['improved_return_code']}"
            )
            _RICH_CONSOLE.print(Panel.fit(meta, title="[bold]AZR Benchmark Comparison[/bold]", border_style="green"))

            table = Table(show_lines=True)
            table.add_column("Benchmark")
            table.add_column("Baseline")
            table.add_column("Improved")
            table.add_column("Δ")
            for row in comparison["benchmarks"]:
                def fmt(val):
                    return "N/A" if val is None else f"{val:.4f}"

                delta = fmt(row["delta_accuracy"])
                if row["delta_accuracy"] is not None and row["delta_accuracy"] > 0:
                    delta = f"[green]{delta}[/green]"
                elif row["delta_accuracy"] is not None and row["delta_accuracy"] < 0:
                    delta = f"[red]{delta}[/red]"
                table.add_row(row["benchmark"], fmt(row["baseline_accuracy"]), fmt(row["improved_accuracy"]), delta)
            _RICH_CONSOLE.print(table)
            _RICH_CONSOLE.print(Panel.fit(
                "\n".join(
                    [
                        f"Baseline results: [blue]{baseline['results_path']}[/blue]",
                        f"Improved results: [blue]{improved['results_path']}[/blue]",
                    ]
                ),
                title="[bold]Artifacts[/bold]",
                border_style="blue",
            ))
    else:
        print(f"[done] Comparison JSON: {comparison['comparison_json']}")
        print(f"[done] Comparison Report: {comparison['comparison_md']}")
        print(f"baseline results: {baseline['results_path']}")
        print(f"improved results: {improved['results_path']}")


def _build_output_paths(results_root: Path, run_id: str) -> Dict[str, Path]:
    return {
        "json": results_root / f"comparison_results_{run_id}.json",
        "md": results_root / f"comparison_report_{run_id}.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic pre/post benchmark comparison for AZR models."
    )
    parser.add_argument("--baseline-model", required=True, help="Baseline model path/name")
    parser.add_argument("--improved-model", required=True, help="Improved model path/name")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp", "gsm8k"],
        help="Benchmarks to evaluate",
    )
    parser.add_argument("--limit", type=int, default=3, help="Sample cap per benchmark")
    parser.add_argument("--samples-per-task", type=int, default=8, help="Samples per task")
    parser.add_argument("--passk", type=int, default=1, help="Pass@k setting")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--k-reference", type=int, default=6, help="Number of few-shot examples for proposer prompts")
    parser.add_argument("--seed", type=int, default=20260319, help="Deterministic RNG seed")
    parser.add_argument(
        "--use-separate-value-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use separate actor + critic for benchmark HuggingFaceAdapter (paper-style). Use --no-use-separate-value-model for unified.",
    )
    parser.add_argument(
        "--results-root",
        default="evaluation_results/comparison",
        help="Directory where comparison runs and report are written",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child benchmark calls",
    )
    parser.add_argument(
        "--cpu-cap",
        type=float,
        default=20.0,
        help="CPU usage cap percentage for evaluator runs",
    )
    parser.add_argument(
        "--rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rich styled terminal output",
    )
    parser.add_argument(
        "--baseline-programbench-run-dir",
        default="",
        help="ProgramBench run directory for baseline (nested <instance>/*.eval.json).",
    )
    parser.add_argument(
        "--improved-programbench-run-dir",
        default="",
        help="ProgramBench run directory for improved model (same layout as baseline).",
    )
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Forward 4-bit loading to evaluate_benchmarks / HuggingFaceAdapter. "
            "If omitted, uses AZR_USE_4BIT when set in the environment; otherwise off (matches hf_trainer default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root) / run_id
    results_root.mkdir(parents=True, exist_ok=True)
    _apply_cpu_cap(args.cpu_cap, context="run_pre_post_benchmarks")

    # Keep benchmark selection resilient to user-provided case variations and comma-glued tokens.
    benchmark_list = _normalize_benchmark_list([b.strip().lower() for b in args.benchmarks])

    baseline = run_eval_once(
        model=args.baseline_model,
        label="baseline",
        benchmark_list=benchmark_list,
        limit=args.limit,
        samples_per_task=args.samples_per_task,
        passk=args.passk,
        temperature=args.temperature,
        top_p=args.top_p,
        k_reference=args.k_reference,
        seed=args.seed,
        use_separate_value_model=args.use_separate_value_model,
        use_rich=args.rich,
        results_root=results_root,
        python_executable=args.python,
        cpu_cap_percent=args.cpu_cap,
        programbench_run_dir=args.baseline_programbench_run_dir,
        use_4bit=args.use_4bit,
    )

    improved = run_eval_once(
        model=args.improved_model,
        label="improved",
        benchmark_list=benchmark_list,
        limit=args.limit,
        samples_per_task=args.samples_per_task,
        passk=args.passk,
        temperature=args.temperature,
        top_p=args.top_p,
        k_reference=args.k_reference,
        seed=args.seed,
        use_separate_value_model=args.use_separate_value_model,
        use_rich=args.rich,
        results_root=results_root,
        python_executable=args.python,
        cpu_cap_percent=args.cpu_cap,
        programbench_run_dir=args.improved_programbench_run_dir,
        use_4bit=args.use_4bit,
    )

    comparison = compute_comparison(
        baseline=baseline,
        improved=improved,
        benchmarks=benchmark_list,
    )

    output_paths = _build_output_paths(results_root, run_id)
    json_path = output_paths["json"]
    md_path = output_paths["md"]

    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_markdown_report(comparison, md_path)

    comparison["comparison_json"] = str(json_path)
    comparison["comparison_md"] = str(md_path)

    _print_run_report(
        comparison=comparison,
        baseline=baseline,
        improved=improved,
        use_rich=args.rich,
    )


if __name__ == "__main__":
    main()
