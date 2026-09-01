"""
Recovered trainer scoring shim.

Operators may exec a restored trainer from _recover/; duplicate checkpoint scoring
must use the same Q117 guard as hf_training.checkpoint_state so spikes cannot win
best-checkpoint selection.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from hf_training.checkpoint_state import (
    combined_checkpoint_score,
    score_checkpoint_from_metrics,
)


def score_epoch_checkpoint(
    avg_proposer_reward: float,
    avg_solver_reward: float,
    r_learnability: Optional[float] = None,
    r_correctness: Optional[float] = None,
    task_success_rate: Optional[float] = None,
    recent_task_success_rates: Optional[Sequence[float]] = None,
) -> float:
    """Restored-trainer entry point (~lines 2897-2911 in hf_trainer_restored)."""
    return combined_checkpoint_score(
        avg_proposer_reward=avg_proposer_reward,
        avg_solver_reward=avg_solver_reward,
        r_learnability=r_learnability,
        r_correctness=r_correctness,
        task_success_rate=task_success_rate,
        recent_task_success_rates=recent_task_success_rates,
    )


def score_epoch_checkpoint_from_metrics(
    metrics: Mapping[str, Any],
    recent_task_success_rates: Optional[Sequence[float]] = None,
) -> float:
    return score_checkpoint_from_metrics(
        metrics,
        recent_task_success_rates=recent_task_success_rates,
    )
