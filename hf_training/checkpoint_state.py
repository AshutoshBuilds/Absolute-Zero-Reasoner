"""
Checkpoint scoring for HF RL training.

Q117 guard: reject easy-spike checkpoints where combined reward looks strong only
because tasks are trivially solvable (dead learnability + near-perfect correctness).
Q119 intent: best checkpoint should reflect sustained learnability, not one-off spikes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

# --- Q117 / Q119 thresholds (conservative; documented in Q&A.md) ---
# r_learnability = 1 - solver_success on proposed tasks; ~0.0625 => ~93.75% trivial success.
Q117_LEARNABILITY_DEAD_MAX = 0.075
# Solver correctness component near 1.0 on easy spike epochs (e.g. epoch 367 ~0.999).
Q117_CORRECTNESS_SPIKE_MIN = 0.95
# Optional corroboration when task success is present on the same epoch.
Q117_TASK_SUCCESS_SPIKE_MIN = 0.90
# Last-N success collapse: peak in window was high but recent tail dropped sharply.
Q117_SUCCESS_COLLAPSE_PEAK_MIN = 0.85
Q117_SUCCESS_COLLAPSE_DROP_MIN = 0.30
Q117_SUCCESS_COLLAPSE_WINDOW = 20
Q117_SUCCESS_COLLAPSE_TAIL = 5


def is_q117_easy_spike(
    r_learnability: Optional[float] = None,
    r_correctness: Optional[float] = None,
    task_success_rate: Optional[float] = None,
    recent_task_success_rates: Optional[Sequence[float]] = None,
) -> bool:
    """
    Return True when checkpoint metrics match a Q117-class easy spike signature.

    Primary gate (used when learnability + correctness are available):
      r_learnability <= Q117_LEARNABILITY_DEAD_MAX AND r_correctness >= Q117_CORRECTNESS_SPIKE_MIN
      (optional: task_success_rate >= Q117_TASK_SUCCESS_SPIKE_MIN when provided)

    Secondary gate (only when recent_task_success_rates is available):
      peak success in the last window was high but the recent tail collapsed by
      Q117_SUCCESS_COLLAPSE_DROP_MIN, together with dead learnability when known.
    """
    learnability_dead = (
        r_learnability is not None
        and math.isfinite(r_learnability)
        and r_learnability <= Q117_LEARNABILITY_DEAD_MAX
    )
    correctness_spike = (
        r_correctness is not None
        and math.isfinite(r_correctness)
        and r_correctness >= Q117_CORRECTNESS_SPIKE_MIN
    )

    if learnability_dead and correctness_spike:
        if task_success_rate is None or not math.isfinite(task_success_rate):
            return True
        if task_success_rate >= Q117_TASK_SUCCESS_SPIKE_MIN:
            return True

    if recent_task_success_rates is not None:
        window = [
            float(x)
            for x in recent_task_success_rates[-Q117_SUCCESS_COLLAPSE_WINDOW:]
            if x is not None and math.isfinite(float(x))
        ]
        if len(window) >= Q117_SUCCESS_COLLAPSE_TAIL + 2:
            peak = max(window)
            tail = window[-Q117_SUCCESS_COLLAPSE_TAIL:]
            tail_mean = sum(tail) / len(tail)
            collapsed = (
                peak >= Q117_SUCCESS_COLLAPSE_PEAK_MIN
                and (peak - tail_mean) >= Q117_SUCCESS_COLLAPSE_DROP_MIN
            )
            if collapsed and (learnability_dead or r_learnability is None):
                return True

    return False


def combined_checkpoint_score(
    avg_proposer_reward: float,
    avg_solver_reward: float,
    r_learnability: Optional[float] = None,
    r_correctness: Optional[float] = None,
    task_success_rate: Optional[float] = None,
    recent_task_success_rates: Optional[Sequence[float]] = None,
) -> float:
    """
    Combined score for best-checkpoint selection and prune ranking.

    Base: (avg_proposer_reward + avg_solver_reward) / 2.
    Q117 spikes score -inf so they cannot win best-checkpoint or prune-anchor retention.
    """
    if not math.isfinite(avg_proposer_reward) or not math.isfinite(avg_solver_reward):
        return float("-inf")

    base = (avg_proposer_reward + avg_solver_reward) / 2.0

    if is_q117_easy_spike(
        r_learnability=r_learnability,
        r_correctness=r_correctness,
        task_success_rate=task_success_rate,
        recent_task_success_rates=recent_task_success_rates,
    ):
        return float("-inf")

    return base


def _mean_tail(values: Sequence[float], window: int) -> float:
    if not values:
        return 0.0
    tail = list(values)[-window:]
    return sum(tail) / len(tail)


def score_checkpoint_from_metrics(
    metrics: Mapping[str, Any],
    recent_task_success_rates: Optional[Sequence[float]] = None,
    reward_window: int = 20,
) -> float:
    """
    Score a checkpoint from trainer metrics dict (save/prune callers).

    Uses recent proposer/solver reward tails for the combined average when lists exist;
    falls back to aggregated avg_* fields otherwise.
    """
    proposer_rewards = metrics.get("proposer_rewards") or []
    solver_rewards = metrics.get("solver_rewards") or []

    if proposer_rewards or solver_rewards:
        avg_proposer = _mean_tail(proposer_rewards, reward_window)
        avg_solver = _mean_tail(solver_rewards, reward_window)
    else:
        avg_proposer = float(metrics.get("avg_proposer_reward", 0.0))
        avg_solver = float(metrics.get("avg_solver_reward", 0.0))

    proposer_components: Dict[str, Any] = dict(metrics.get("avg_proposer_reward_components") or {})
    solver_components: Dict[str, Any] = dict(metrics.get("avg_solver_reward_components") or {})

    r_learnability = proposer_components.get("r_learnability")
    r_correctness = solver_components.get("r_correctness")
    task_success = metrics.get("solver_success_rate", metrics.get("task_success_rate"))

    return combined_checkpoint_score(
        avg_proposer_reward=float(avg_proposer),
        avg_solver_reward=float(avg_solver),
        r_learnability=float(r_learnability) if r_learnability is not None else None,
        r_correctness=float(r_correctness) if r_correctness is not None else None,
        task_success_rate=float(task_success) if task_success is not None else None,
        recent_task_success_rates=recent_task_success_rates,
    )
