"""Tests for AZR_HF_* and paper-preset interaction in hf_trainer."""

import os

import pytest

import hf_trainer as ht


@pytest.fixture(autouse=True)
def clear_hf_env(monkeypatch):
    for key in (
        "AZR_PAPER_STYLE_DEFAULTS",
        "AZR_HF_LEARNING_RATE",
        "AZR_HF_CRITIC_LEARNING_RATE",
        "AZR_HF_GENERATION_STEPS_PER_EPOCH",
        "AZR_HF_BATCH_SIZE",
        "AZR_HF_PPO_UPDATE_THRESHOLD",
        "AZR_SEED_TASKS_PER_TYPE",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def test_paper_preset_skips_lr_when_azr_hf_lr_set(monkeypatch):
    monkeypatch.setenv("AZR_PAPER_STYLE_DEFAULTS", "1")
    monkeypatch.setenv("AZR_HF_LEARNING_RATE", "1e-6")
    cfg: dict = {"generation_steps_per_epoch": 10, "batch_size": 16, "ppo_update_threshold": 64}
    ht._maybe_apply_azr_paper_style_defaults(cfg, set())
    assert "learning_rate" not in cfg
    ht._apply_azr_hf_env_trainer_hyperparams(cfg)
    assert cfg["learning_rate"] == pytest.approx(1e-6)
    assert cfg["critic_learning_rate"] == pytest.approx(1e-6)


def test_apply_hf_env_overrides_paper_preset_batch(monkeypatch):
    monkeypatch.setenv("AZR_PAPER_STYLE_DEFAULTS", "1")
    cfg = {
        "learning_rate": 5e-7,
        "critic_learning_rate": 5e-7,
        "generation_steps_per_epoch": 10,
        "batch_size": 16,
        "ppo_update_threshold": 64,
    }
    monkeypatch.setenv("AZR_HF_BATCH_SIZE", "8")
    ht._apply_azr_hf_env_trainer_hyperparams(cfg)
    assert cfg["batch_size"] == 8


def test_seed_env_allow_zero(monkeypatch):
    monkeypatch.setenv("AZR_SEED_TASKS_PER_TYPE", "0")
    cfg = {"seed_tasks_per_type": 6}
    ht._apply_azr_hf_env_trainer_hyperparams(cfg)
    assert cfg["seed_tasks_per_type"] == 0
