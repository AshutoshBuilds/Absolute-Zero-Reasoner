"""Autonomous multi-round self-evolution pilot loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from autoevo.contracts import ExperimentGoal
from autoevo.contracts import ExperimentSpec, parse_experiment_file


DEFAULT_HYPOTHESES: List[ExperimentSpec] = [
    ExperimentSpec(
        hypothesis_id="gen_steps_down_1",
        title="Reduce generation steps for speed",
        description="Decrease per-epoch generation for a slightly more stable run profile.",
        trainer_config={"generation_steps_per_epoch": 8},
        benchmark_overrides={},
    ),
    ExperimentSpec(
        hypothesis_id="gen_steps_down_2",
        title="Reduce PPO batch threshold",
        description="Lower PPO threshold with matching context-aware minibatches.",
        trainer_config={"ppo_update_threshold": 16},
        benchmark_overrides={},
    ),
    ExperimentSpec(
        hypothesis_id="lr_reweight",
        title="Lower critic learning rate",
        description="Small critic step size to avoid aggressive value collapse.",
        trainer_config={"critic_learning_rate": 5e-9},
        benchmark_overrides={},
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_run_command(args: argparse.Namespace, hypothesis: ExperimentSpec, baseline_model: str, run_name: str, root: Path) -> List[str]:
    payload = dict(hypothesis.trainer_config)
    payload.update(hypothesis.benchmark_overrides or {})
    override_json = json.dumps(payload, sort_keys=True)
    run_checkpoint_dir = root / run_name / "checkpoint"
    command = [
        sys.executable,
        str(Path("scripts/run_research_harness.py")),
        "--run-root",
        str(root),
        "--run-name",
        run_name,
        "--experiment-id",
        hypothesis.hypothesis_id,
        "--baseline-model",
        baseline_model,
        "--checkpoint-dir",
        str(run_checkpoint_dir),
        "--python",
        args.python,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--no-use-4bit" if not args.use_4bit else "--use-4bit",
        "--use-separate-value-model" if args.use_separate_value_model else "--no-use-separate-value-model",
        "--model-dtype",
        args.model_dtype,
        "--cpu-cap",
        str(args.cpu_cap),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--cuda-alloc-config",
        args.cuda_alloc_config,
        "--train-timeout-minutes",
        str(args.train_timeout_minutes),
        "--benchmark-timeout-minutes",
        str(args.benchmark_timeout_minutes),
        "--benchmarks",
        *args.benchmarks,
        "--benchmark-limit",
        str(args.benchmark_limit),
        "--samples-per-task",
        str(args.samples_per_task),
        "--pass-k",
        str(args.pass_k),
        "--benchmark-temperature",
        str(args.benchmark_temperature),
        "--top-p",
        str(args.top_p),
        "--benchmark-seed",
        str(args.benchmark_seed),
        "--benchmark-cpu-cap",
        str(args.benchmark_cpu_cap),
        "--primary-metric",
        args.primary_metric,
        "--target-delta",
        str(args.target_delta),
        "--reject-regression",
        str(args.reject_regression),
        "--score-weights",
        json.dumps(args.score_weights, sort_keys=True),
        "--trainer-config-json",
        override_json,
        "--trainer-config-override",
        "{}",
        "--benchmark-overrides",
        json.dumps(hypothesis.benchmark_overrides or {}, sort_keys=True),
        "--seed-tasks-per-type",
        str(args.seed_tasks_per_type),
    ]
    if args.no_rich:
        command.extend(["--no-rich"])
    return command


def _load_hypotheses(args: argparse.Namespace) -> List[ExperimentSpec]:
    if args.hypothesis_file:
        return parse_experiment_file(args.hypothesis_file)
    return DEFAULT_HYPOTHESES


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_round_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data = {"rounds": []}
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    rounds = data.get("rounds", [])
    rounds.append(payload)
    data["rounds"] = rounds
    data["updated_utc"] = _utc_now()
    data["latest_round"] = len(rounds)
    
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _accept_by_goal(comparison_payload: Dict[str, Any], goal: ExperimentGoal) -> bool:
    if not comparison_payload:
        return False
    rows = comparison_payload.get("benchmarks", [])
    if not rows:
        return False
    for row in rows:
        if row.get("benchmark") != goal.primary_metric:
            continue
        delta = row.get("delta_accuracy")
        if delta is None:
            return False
        return float(delta) >= goal.target_delta
    return False


def run_rounds(args: argparse.Namespace) -> Dict[str, Any]:
    goal = ExperimentGoal(
        primary_metric=args.primary_metric,
        target_delta=args.target_delta,
        reject_regression=args.reject_regression,
        max_cpu_percent=args.cpu_cap,
        max_gpu_fraction=args.gpu_memory_fraction,
        benchmark_limit=args.benchmark_limit,
        samples_per_task=args.samples_per_task,
        pass_k=args.pass_k,
        temperature=args.benchmark_temperature,
        top_p=args.top_p,
    )
    state_path = Path(args.state_file)
    current_state = {
        "created_utc": _utc_now(),
        "base_model": args.baseline_model,
        "best_model": args.baseline_model,
        "rounds": [],
        "status": "running",
    }
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        current_state["created_utc"] = state.get("created_utc", current_state["created_utc"])
        current_state["base_model"] = state.get("best_model", args.baseline_model)

    best_model = current_state["base_model"]
    hypotheses = _load_hypotheses(args)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    accepted_rounds = len([round_entry for round_entry in current_state.get("rounds", []) if round_entry.get("status") == "accepted"])
    for index in range(args.rounds):
        if index >= len(hypotheses):
            break
        if accepted_rounds >= args.max_accepted_rounds:
            break

        hypothesis = hypotheses[index]
        run_name = f"round_{index+1:02d}_{hypothesis.hypothesis_id}"
        command = _build_run_command(args, hypothesis, best_model, run_name, run_root)
        manifest_path = run_root / run_name / "manifest.json"

        proc = subprocess.run(
            command,
            check=False,
            cwd=str(Path(".").resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if proc.returncode != 0 or not manifest_path.exists():
            round_payload = {
                "round": index + 1,
                "hypothesis_id": hypothesis.hypothesis_id,
                "status": "failed",
                "error": proc.stdout[:4096] if proc.stdout else f"run_research_harness failed (code={proc.returncode}).",
            }
        else:
            run_payload = _read_manifest(manifest_path)
            decision = run_payload.get("decision", {})
            comparison = decision.get("benchmarks", [])
            accepted = bool(decision.get("accept")) and _accept_by_goal(
                {"benchmarks": comparison},
                goal=goal,
            )
            round_payload = {
                "round": index + 1,
                "hypothesis_id": hypothesis.hypothesis_id,
                "status": "accepted" if accepted else "rejected",
                "manifest_path": str(manifest_path),
                "baseline_model": run_payload.get("baseline_model"),
                "candidate_model": run_payload.get("candidate_model"),
                "decision": decision,
                "benchmarks": comparison,
                "goal_primary": goal.primary_metric,
                "goal": goal.__dict__,
            }
            if accepted:
                best_model = run_payload.get("candidate_model", best_model)
                accepted_rounds += 1
        current_state["rounds"].append(round_payload)
        _append_round_log(state_path, round_payload)

    current_state["best_model"] = best_model
    current_state["status"] = "complete"
    current_state["updated_utc"] = _utc_now()
    _append_round_log(state_path, current_state)
    return current_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conservative autonomous self-evolution pilot.")
    parser.add_argument("--run-root", default="autoevo/run_rounds", help="Directory to write harness outputs.")
    parser.add_argument("--state-file", default="autoevo/run_rounds/ledger.json", help="State ledger path.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for training/eval.")
    parser.add_argument("--baseline-model", default="models/gemma-4-E4B")
    parser.add_argument("--checkpoint-dir", default="hf_checkpoints/hf_trainer_qwen3_5b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-accepted-rounds", type=int, default=1)
    parser.add_argument("--hypothesis-file", default="", help="Optional JSON file with hypothesis specs.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-tasks-per-type", type=int, default=0)
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable 4-bit model loading."
    )
    parser.add_argument(
        "--use-separate-value-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable separate value model."
    )
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
    parser.add_argument("--primary-metric", default="humaneval")
    parser.add_argument("--target-delta", type=float, default=0.0)
    parser.add_argument("--reject-regression", type=float, default=-0.01)
    parser.add_argument("--no-rich", action="store_true", default=False)
    parser.add_argument(
        "--score-weights",
        type=json.loads,
        default='{"humaneval": 0.5, "mbpp": 0.25, "gsm8k": 0.25}',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_rounds(args)
    print(json.dumps(result, indent=2))
