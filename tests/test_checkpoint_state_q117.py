"""CPU unit tests for Q117 best-checkpoint scoring guard (Aug 2026 run fixtures)."""

import math

import pytest

from hf_training.checkpoint_state import combined_checkpoint_score, is_q117_easy_spike
from _recover.hf_trainer_restored import score_epoch_checkpoint


# Verified component numbers from training_run_logs/2026-08-16/local_hf_train_233022
EPOCH_367 = {
    "avg_proposer_reward": 0.550,
    "avg_solver_reward": 0.548,
    "r_learnability": 0.0625,
    "r_correctness": 0.999,
    "task_success_rate": 0.9375,
    "combined_raw": 0.549,
}

EPOCH_727 = {
    "avg_proposer_reward": 0.200,
    "avg_solver_reward": 0.198,
    "r_learnability": 0.42,
    "r_correctness": 0.38,
    "task_success_rate": 0.20,
    "combined_raw": 0.199,
}

EPOCH_731 = {
    "avg_proposer_reward": 0.306,
    "avg_solver_reward": 0.304,
    "r_learnability": 0.35,
    "r_correctness": 0.45,
    "task_success_rate": 0.4375,
    "combined_raw": 0.305,
}


def _score(fixture: dict) -> float:
    return combined_checkpoint_score(
        avg_proposer_reward=fixture["avg_proposer_reward"],
        avg_solver_reward=fixture["avg_solver_reward"],
        r_learnability=fixture["r_learnability"],
        r_correctness=fixture["r_correctness"],
        task_success_rate=fixture["task_success_rate"],
    )


def test_epoch_367_is_q117_easy_spike():
    assert is_q117_easy_spike(
        r_learnability=EPOCH_367["r_learnability"],
        r_correctness=EPOCH_367["r_correctness"],
        task_success_rate=EPOCH_367["task_success_rate"],
    )


def test_epoch_367_scores_negative_infinity():
    score = _score(EPOCH_367)
    assert score == float("-inf")


def test_epoch_727_and_731_beat_epoch_367_guarded_score():
    score_367 = _score(EPOCH_367)
    score_727 = _score(EPOCH_727)
    score_731 = _score(EPOCH_731)

    assert score_367 == float("-inf")
    assert score_727 == EPOCH_727["combined_raw"]
    assert score_731 == EPOCH_731["combined_raw"]
    assert score_727 > score_367
    assert score_731 > score_367
    assert score_731 > score_727


def test_restored_trainer_shim_matches_checkpoint_state():
    restored = score_epoch_checkpoint(
        avg_proposer_reward=EPOCH_367["avg_proposer_reward"],
        avg_solver_reward=EPOCH_367["avg_solver_reward"],
        r_learnability=EPOCH_367["r_learnability"],
        r_correctness=EPOCH_367["r_correctness"],
        task_success_rate=EPOCH_367["task_success_rate"],
    )
    assert restored == float("-inf")
    assert restored == _score(EPOCH_367)


def test_ranking_report_367_vs_727_vs_731():
    """Document ranking for PR/report: 731 > 727 > 367 (367 rejected)."""
    scores = {
        367: _score(EPOCH_367),
        727: _score(EPOCH_727),
        731: _score(EPOCH_731),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    assert ranked[0][0] == 731
    assert ranked[1][0] == 727
    assert ranked[2][0] == 367
    assert math.isfinite(scores[727])
    assert math.isfinite(scores[731])
