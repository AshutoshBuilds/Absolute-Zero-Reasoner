#!/usr/bin/env python3
"""Single-run research harness for AZR training + benchmark comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from autoevo.contracts import ExperimentGoal, ExperimentManifest, RunPhase
from autoevo.evidence import ComparisonArtifact, ScorePolicy, evaluate_comparison


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(raw: str, *, name: str) -> Dict[str, Any]:
    payload = json.loads(raw) if raw else {}
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return payload


def _safe_weights(raw: Mapping[str, Any], *, name: str) -> Dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    weights: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} value for '{key}' must be a number.") from exc
    return weights


_TRAINING_INSTABILITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"probability tensor contains .* (inf|nan)", re.IGNORECASE),
    re.compile(r"cannot contain", re.IGNORECASE),
    re.compile(r"CRITICAL: NaN or Inf detected", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
)
_TRAINING_NON_FATAL_CONTINUATION: re.Pattern[str] = re.compile(r"Skipping backward pass for this minibatch", re.IGNORECASE)
_TRAINING_RECOVERABLE_LOSS_HEADER: re.Pattern[str] = re.compile(r"CRITICAL:\s*NaN or Inf", re.IGNORECASE)
_TRAINING_RECOVERABLE_LOSS_CONTINUATION: re.Pattern[str] = re.compile(
    r"total PPO loss|inside autocast|Skipping backward pass for this minibatch",
    re.IGNORECASE,
)


def _read_json(path: Path, *, field_name: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} at {path} must be a JSON object.")
    return payload


def _latest_file(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _latest_benchmark_artifacts(
    benchmark_root: Path,
    exclude_run_names: Optional[set[str]] = None,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    run_dirs = [
        d
        for d in benchmark_root.iterdir()
        if d.is_dir() and (exclude_run_names is None or d.name not in exclude_run_names)
    ]
    if not run_dirs:
        return None, None, None, None
    run_dir = sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]
    if run_dir is None:
        return None, None, None, None

    comparison_files = sorted(run_dir.glob("comparison_results_*.json"), key=lambda p: p.stat().st_mtime)
    if comparison_files:
        return run_dir, _latest_file(comparison_files), None, None

    baseline_eval = sorted((run_dir / "baseline").glob("eval_results_*.json"), key=lambda p: p.stat().st_mtime)
    improved_eval = sorted((run_dir / "improved").glob("eval_results_*.json"), key=lambda p: p.stat().st_mtime)
    if baseline_eval and improved_eval:
        return run_dir, None, _latest_file(baseline_eval), _latest_file(improved_eval)
    return run_dir, None, None, None


def _build_fallback_comparison(
    baseline_results: Path,
    improved_results: Path,
    baseline_model: str,
    improved_model: str,
    benchmarks: list[str],
    run_id: Optional[str] = None,
    benchmark_root: Optional[Path] = None,
) -> Dict[str, Any]:
    baseline_summary = _read_json(baseline_results, field_name="baseline evaluation JSON")
    improved_summary = _read_json(improved_results, field_name="improved evaluation JSON")

    rows = []
    for benchmark in benchmarks:
        base_item = baseline_summary.get(benchmark, {})
        imp_item = improved_summary.get(benchmark, {})
        base_accuracy = base_item.get("accuracy")
        imp_accuracy = imp_item.get("accuracy")
        delta = None
        if isinstance(base_accuracy, (int, float)) and isinstance(imp_accuracy, (int, float)):
            delta = imp_accuracy - base_accuracy

        rows.append(
            {
                "benchmark": benchmark,
                "baseline_accuracy": base_accuracy,
                "baseline_total": base_item.get("total"),
                "baseline_correct": base_item.get("correct"),
                "improved_accuracy": imp_accuracy,
                "improved_total": imp_item.get("total"),
                "improved_correct": imp_item.get("correct"),
                "delta_accuracy": delta,
            }
        )

    comparison: Dict[str, Any] = {
        "benchmarks": rows,
        "overall": {
            "baseline_return_code": 0,
            "improved_return_code": 0,
            "baseline_model": baseline_model,
            "improved_model": improved_model,
            "baseline_results_path": str(baseline_results),
            "improved_results_path": str(improved_results),
        },
        "timedelta_utc": datetime.now(timezone.utc).isoformat(),
        "seed": None,
        "benchmarks_requested": benchmarks,
        "notes": "Built from eval_results_*.json fallback due comparison_results_*.json schema differences.",
    }

    if run_id and benchmark_root:
        fallback_path = benchmark_root / f"comparison_results_fallback_{run_id}.json"
        fallback_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparison["comparison_json"] = str(fallback_path)
    return comparison


def _scan_training_instability(*log_paths: str) -> list[str]:
    """Scan training logs for NaN/Inf signatures before accepting a run."""
    matches: list[str] = []
    expect_recoverable_continuation = False

    for path in log_paths:
        if not path:
            continue
        log_file = Path(path)
        if not log_file.exists():
            continue

        log_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in log_lines:
            if expect_recoverable_continuation:
                if _TRAINING_NON_FATAL_CONTINUATION.search(line):
                    expect_recoverable_continuation = False
                    continue
                if _TRAINING_RECOVERABLE_LOSS_CONTINUATION.search(line):
                    # keep skipping this multi-line warning block
                    continue
                expect_recoverable_continuation = False

            if _TRAINING_RECOVERABLE_LOSS_HEADER.search(line):
                expect_recoverable_continuation = True
                continue
            if any(pattern.search(line) for pattern in _TRAINING_INSTABILITY_PATTERNS):
                matches.append(line.strip())
    return matches


def _load_json(path: Optional[str], *, name: str) -> Dict[str, Any]:
    if not path:
        return {}
    return _safe_json(Path(path).read_text(encoding="utf-8"), name=name)


def _resolve_executable(path_like: str) -> str:
    candidate = Path(path_like)
    if candidate.exists():
        return str(candidate)
    fallback = ROOT_DIR / candidate
    return str(fallback) if fallback.exists() else path_like


def _validate_runtime(python_executable: str) -> tuple[bool, str]:
    check_command = [
        python_executable,
        "-c",
        "import torch, transformers  # noqa: F401\nprint('runtime-ok')",
    ]
    try:
        proc = subprocess.run(
            check_command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ.copy(), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        return False, (proc.stderr or proc.stdout or "").strip() or "Runtime check failed."
    except Exception as exc:
        return False, str(exc)


def _run_with_logs(command: list[str], cwd: Path, log_prefix: Path, timeout_seconds: Optional[float]) -> tuple[int, str, str, float]:
    out_path = log_prefix.with_suffix(".stdout.log")
    err_path = log_prefix.with_suffix(".stderr.log")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    start = time.time()
    code = -1
    try:
        with out_path.open("w", encoding="utf-8") as out_file, err_path.open("w", encoding="utf-8") as err_file:
            proc = subprocess.run(command, cwd=str(cwd), check=False, env=env, stdout=out_file, stderr=err_file, timeout=timeout_seconds)
            code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out_path.write_text(f"\nTIMEOUT {timeout_seconds}s: {exc}\n", encoding="utf-8")
        err_path.write_text(f"\nTIMEOUT {timeout_seconds}s: {exc}\n", encoding="utf-8")
    elapsed = time.time() - start
    return code, str(out_path), str(err_path), elapsed


def _latest_ckpt(checkpoint_dir: Path) -> str:
    candidates = sorted(checkpoint_dir.glob("checkpoint_epoch_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
    if not candidates:
        raise RuntimeError(f"No checkpoint_epoch_* found in {checkpoint_dir}")
    return str(candidates[-1])


def _build_training_command(args: argparse.Namespace, trainer_config: Dict[str, Any]) -> list[str]:
    python_executable = _resolve_executable(args.python)
    command = [
        python_executable,
        str(Path("hf_trainer.py")),
        "--epochs",
        str(args.epochs),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--seed",
        str(args.seed),
        "--seed-tasks-per-type",
        str(args.seed_tasks_per_type),
        "--cpu-cap",
        str(args.cpu_cap),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--cuda-alloc-config",
        args.cuda_alloc_config,
        "--model-dtype",
        args.model_dtype,
    ]
    if args.use_4bit:
        command.append("--use-4bit")
    command.append("--use-separate-value-model" if args.use_separate_value_model else "--no-use-separate-value-model")
    command.extend(["--trainer-config-json", json.dumps(trainer_config, sort_keys=True)])
    command.append("--rich" if not args.no_rich else "--no-rich")
    return command


def _benchmark(args: argparse.Namespace, baseline_model: str, improved_model: str, root: Path) -> Dict[str, Any]:
    python_executable = _resolve_executable(args.python)
    benchmark_script = str(ROOT_DIR / "run_pre_post_benchmarks.py")
    benchmark_root = root / "benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        benchmark_script,
        "--baseline-model",
        baseline_model,
        "--improved-model",
        improved_model,
        "--results-root",
        str(benchmark_root),
        "--benchmarks",
        *args.benchmarks,
        "--limit",
        str(args.benchmark_limit),
        "--samples-per-task",
        str(args.samples_per_task),
        "--passk",
        str(args.pass_k),
        "--temperature",
        str(args.benchmark_temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.benchmark_seed),
        "--cpu-cap",
        str(args.benchmark_cpu_cap),
        "--python",
        python_executable,
    ]
    benchmark_use_separate = args.benchmark_overrides.get(
        "use_separate_value_model", args.use_separate_value_model
    )
    if isinstance(benchmark_use_separate, str):
        benchmark_use_separate = benchmark_use_separate.strip().lower() in {"1", "true", "t", "yes", "y"}
    elif isinstance(benchmark_use_separate, (int, float)):
        benchmark_use_separate = bool(benchmark_use_separate)
    elif not isinstance(benchmark_use_separate, bool):
        benchmark_use_separate = bool(args.use_separate_value_model)
    if benchmark_use_separate:
        command.append("--use-separate-value-model")
    command.extend(["--no-rich"] if args.no_rich else ["--rich"])

    out_path = root / "benchmark.stdout.log"
    err_path = root / "benchmark.stderr.log"
    timeout_seconds = args.benchmark_timeout_minutes * 60 if args.benchmark_timeout_minutes else None
    existing_runs = {entry.name for entry in benchmark_root.iterdir() if entry.is_dir()}
    return_code = -1
    try:
        with out_path.open("w", encoding="utf-8") as out_file, err_path.open("w", encoding="utf-8") as err_file:
            proc = subprocess.run(
                command,
                check=False,
                cwd=str(ROOT_DIR),
                env={**os.environ.copy(), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                stdout=out_file,
                stderr=err_file,
                timeout=timeout_seconds,
            )
            return_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out_path.write_text(f"\nBENCHMARK TIMEOUT after {timeout_seconds}s: {exc}\n", encoding="utf-8")
        err_path.write_text(f"\nBENCHMARK TIMEOUT after {timeout_seconds}s: {exc}\n", encoding="utf-8")
        return_code = 124
    except Exception as exc:  # defensive: ensure benchmark failure is visible
        err_path.write_text(f"Benchmark launch failed: {exc}\n", encoding="utf-8")
        return_code = 1

    run_dir, comparison_path, baseline_eval_path, improved_eval_path = _latest_benchmark_artifacts(
        benchmark_root,
        exclude_run_names=existing_runs,
    )
    if run_dir is None:
        raise RuntimeError(f"No benchmark output directory produced for {baseline_model} -> {improved_model}")

    if comparison_path:
        comparison = _read_json(comparison_path, field_name="comparison JSON")
        comparison_json = comparison_path
    else:
        if baseline_eval_path is None or improved_eval_path is None:
            raise RuntimeError(
                f"No benchmark comparison artifacts found for {baseline_model}->{improved_model}."
            )
        comparison = _build_fallback_comparison(
            baseline_results=baseline_eval_path,
            improved_results=improved_eval_path,
            baseline_model=baseline_model,
            improved_model=improved_model,
            benchmarks=list(args.benchmarks),
            run_id=run_dir.name,
            benchmark_root=run_dir,
        )
        comparison_json = Path(comparison["comparison_json"])

    if comparison is None or comparison_json is None:
        raise RuntimeError(
            "No benchmark comparison artifact produced for "
            f"{baseline_model}->{improved_model}. Return code: {return_code}"
        )
    comparison["benchmark_run"] = {
        "command": command,
        "return_code": return_code,
        "stdout": str(out_path),
        "stderr": str(err_path),
    }
    return {
        "return_code": return_code,
        "comparison_json": str(comparison_json),
        "comparison": comparison,
        "stdout": str(out_path),
        "stderr": str(err_path),
    }


def _write_comparison_report(run_root: Path, manifest: Dict[str, Any]) -> None:
    decision = manifest.get("decision", {})
    rows = []
    for row in decision.get("benchmarks", []):
        rows.append(
            "- {benchmark}: baseline={baseline:.4f} improved={improved:.4f} delta={delta:+.4f}".format(
                benchmark=row.get("benchmark", "unknown"),
                baseline=float(row.get("baseline_accuracy", 0.0)),
                improved=float(row.get("improved_accuracy", 0.0)),
                delta=float(row.get("delta_accuracy", 0.0)),
            )
        )
    if not rows:
        rows.append("- No benchmark rows were produced (run rejected early or timed out).")

    policy = decision.get("policy", {})
    content = "\n".join(
        [
            "# AZR Self-Evolution Run Comparison",
            "",
            f"- Experiment ID: {manifest.get('experiment_id')}",
            f"- Status: {manifest.get('status')}",
            f"- Baseline model: {manifest.get('baseline_model')}",
            f"- Candidate model: {manifest.get('candidate_model')}",
            f"- Accepted: {decision.get('accept')}",
            f"- Score delta: {decision.get('score_delta')}",
            f"- Reject reason: {decision.get('reject_reason') or 'None'}",
            "",
            "## Benchmark rows",
            *rows,
            "",
            "## Policy",
            f"- weights: {policy.get('weights', {})}",
            f"- min_delta: {policy.get('min_delta')}",
            f"- max_regression: {policy.get('max_regression')}",
            "",
            f"- Train command: {manifest.get('train_manifest', {}).get('command')}",
            f"- Train return code: {manifest.get('train_manifest', {}).get('return_code')}",
            f"- Benchmark return code: {manifest.get('benchmark_manifest', {}).get('return_code')}",
            f"- Comparison JSON: {manifest.get('benchmark_manifest', {}).get('comparison_json')}",
            f"- Train checkpoint dir: {manifest.get('train_manifest', {}).get('artifacts', {}).get('checkpoint_dir')}",
        ]
    )
    (run_root / "comparison_report.md").write_text(content + "\n", encoding="utf-8")


def run_single_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    goal = ExperimentGoal(
        primary_metric=args.primary_metric,
        target_delta=args.target_delta,
        max_cpu_percent=args.cpu_cap,
        max_gpu_fraction=args.gpu_memory_fraction,
        reject_regression=args.reject_regression,
        benchmark_limit=args.benchmark_limit,
        samples_per_task=args.samples_per_task,
        pass_k=args.pass_k,
        temperature=args.benchmark_temperature,
        top_p=args.top_p,
    )
    policy = ScorePolicy(
        weights=_safe_weights(args.score_weights, name="--score-weights"),
        min_delta=goal.target_delta,
        max_regression=goal.reject_regression,
    )

    trainer_config = {}
    trainer_config.update(_load_json(args.trainer_config_file, name="--trainer-config"))
    trainer_config.update(_safe_json(args.trainer_config_json, name="--trainer-config-json"))
    trainer_config.update(args.trainer_config_override)

    runtime_ok, runtime_report = _validate_runtime(args.python)
    if not runtime_ok:
        elapsed = 0.0
        benchmark_manifest: Dict[str, Any] = {}
        baseline_model = args.baseline_model
        candidate_model = baseline_model
        status = "failed"
        decision = {
            "accept": False,
            "score_delta": None,
            "reject_reason": runtime_report,
            "benchmarks": [],
        }
        train_manifest = ExperimentManifest(
            experiment_id=args.experiment_id,
            phase=RunPhase.FAILED,
            command=args.python,
            started_at_utc=_utc_now(),
            ended_at_utc=_utc_now(),
            elapsed_seconds=elapsed,
            return_code=1,
            artifacts={"checkpoint_dir": args.checkpoint_dir},
            error=runtime_report,
        )
        payload = {
            "experiment_id": args.experiment_id,
            "run_root": str(run_root),
            "status": "failed",
            "start_utc": train_manifest.started_at_utc,
            "baseline_model": args.baseline_model,
            "candidate_model": args.baseline_model,
            "decision": decision,
            "trainer_config": trainer_config,
            "goal": goal.__dict__,
            "train_manifest": train_manifest.__dict__,
            "benchmark_manifest": benchmark_manifest,
            "runtime_report": runtime_report,
        }
        manifest_path = run_root / "manifest.json"
        config_path = run_root / "run_config.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
        return payload

    training_command = _build_training_command(args, trainer_config)
    start = _utc_now()
    train_rc, out_log, err_log, elapsed = _run_with_logs(training_command, Path("."), run_root / "train", args.train_timeout_minutes * 60 if args.train_timeout_minutes else None)
    end = _utc_now()

    baseline_model = args.baseline_model
    candidate_model = baseline_model
    decision: Dict[str, Any] = {
        "accept": False,
        "score_delta": None,
        "reject_reason": "training_or_benchmark_failed",
        "benchmarks": [],
    }
    benchmark_manifest: Dict[str, Any] = {}
    status = "failed"

    train_manifest = ExperimentManifest(
        experiment_id=args.experiment_id,
        phase=RunPhase.TRAIN if train_rc == 0 else RunPhase.FAILED,
        command=" ".join(training_command),
        started_at_utc=start,
        ended_at_utc=end,
        elapsed_seconds=elapsed,
        return_code=train_rc,
        stdout_path=out_log,
        stderr_path=err_log,
        artifacts={"checkpoint_dir": args.checkpoint_dir},
        error=None if train_rc == 0 else f"Training failed with return code {train_rc}",
    )

    if train_rc == 0:
        try:
            training_instability = _scan_training_instability(out_log, err_log)
            if training_instability:
                raise RuntimeError(
                    "Training instability detected. "
                    f"Evidence: {training_instability[:3]}"
                )
            candidate_model = _latest_ckpt(Path(args.checkpoint_dir))
            bench = _benchmark(args, baseline_model, candidate_model, run_root)
            benchmark_manifest = bench
            decision = evaluate_comparison(bench["comparison"], policy)
            if isinstance(bench.get("return_code"), int) and bench.get("return_code") != 0:
                decision["reject_reason"] = f"benchmark_return_code_{bench.get('return_code')}"
            decision["improved_model"] = candidate_model
            decision["baseline_model"] = baseline_model
            decision["artifacts"] = ComparisonArtifact(
                comparison=bench["comparison"], baseline=None, improved=None
            ).__dict__
            train_manifest.phase = RunPhase.BENCHMARK
            if decision.get("accept"):
                status = "accepted"
            else:
                status = "complete"
        except Exception as exc:
            status = "failed"
            decision = {
                "accept": False,
                "score_delta": None,
                "reject_reason": str(exc),
                "benchmarks": [],
            }

    payload = {
        "experiment_id": args.experiment_id,
        "run_root": str(run_root),
        "status": status,
        "start_utc": start,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "decision": decision,
        "trainer_config": trainer_config,
        "goal": goal.__dict__,
        "train_manifest": train_manifest.__dict__,
        "benchmark_manifest": benchmark_manifest,
    }

    _write_comparison_report(run_root, payload)

    manifest_path = run_root / "manifest.json"
    config_path = run_root / "run_config.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single AZR self-improvement experiment.")
    parser.add_argument("--run-root", default="autoevo/run_rounds")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--checkpoint-dir", default="hf_checkpoints/hf_trainer_qwen3_5b")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-tasks-per-type", type=int, default=0)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-separate-value-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-dtype", default="fp16")
    parser.add_argument("--cpu-cap", type=float, default=20.0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--cuda-alloc-config", default="max_split_size_mb:128,garbage_collection_threshold:0.8")
    parser.add_argument("--train-timeout-minutes", type=float, default=720.0)
    parser.add_argument("--benchmark-timeout-minutes", type=float, default=480.0)
    parser.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp", "gsm8k"])
    parser.add_argument("--benchmark-limit", type=int, default=100)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--benchmark-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--benchmark-seed", type=int, default=20260319)
    parser.add_argument("--benchmark-cpu-cap", type=float, default=20.0)
    parser.add_argument("--trainer-config-file", default="")
    parser.add_argument("--trainer-config-json", default="{}")
    parser.add_argument("--trainer-config-override", type=json.loads, default="{}")
    parser.add_argument("--benchmark-overrides", type=json.loads, default="{}")
    parser.add_argument("--primary-metric", default="humaneval")
    parser.add_argument("--target-delta", type=float, default=0.0)
    parser.add_argument("--reject-regression", type=float, default=-0.01)
    parser.add_argument(
        "--score-weights",
        type=json.loads,
        default='{"humaneval":0.5,"mbpp":0.25,"gsm8k":0.25}',
    )
    parser.add_argument("--no-rich", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.experiment_id:
        args.experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(json.dumps(run_single_experiment(args), indent=2))
