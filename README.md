# Absolute Zero Model (AZR) Workspace

This workspace has two training tracks:

1. **Local/Dev Track (single-machine, desktop-friendly)**
   - Uses `hf_trainer.py` with HuggingFace adapters.
   - Single-process style, CPU-capped and tuned for stability.
   - Suitable for iterations and diagnostics on RTX 3090 Ti.

2. **Official Protocol Track (distributed, VPS/GPU scale)**
   - Uses the original repo stack from a separate checkout of `AshutoshBuilds/Absolute-Zero-Reasoner`.
   - Ray + vLLM + veRL scripts in that external stack.
   - Suitable for H200/H100/A100 scale runs and closer parity with upstream protocol.
   
Note: this workspace no longer includes a nested `Absolute-Zero-Reasoner/` checkout. For official protocol execution, clone upstream separately and point `AZR_OFFICIAL_REPO_PATH` to it:

```bash
git clone https://github.com/AshutoshBuilds/Absolute-Zero-Reasoner.git /path/to/official-azr
export AZR_OFFICIAL_REPO_PATH=/path/to/official-azr
```

## Why two modes

- Your local GPU is great for experiments, but large self-play PPO runs are better on distributed GPUs.
- The two stacks are not identical today:
  - Local stack is `hf_trainer.py` and local utilities.
  - Official stack is `python -m absolute_zero_reasoner.main_azr_ppo` with Ray/vLLM.

## Local training (Windows, recommended for iteration)

Use the launcher:

```powershell
./scripts/run_local_hf_training.ps1 `
  -Epochs 100 `
  -CheckpointDir hf_checkpoints\hf_trainer_qwen3_5b `
  -Seed 42 `
  -CpuCap 20
```

Useful overrides:

- `-NoRich` to disable pretty terminal output.
- `-SeedTasksPerType N` to create more seed tasks per type.
- `-RunBenchmark` to run the local comparison automatically after training.
- `-BaselineModelForBenchmark` to define baseline in auto-benchmark (default `models/Qwen3.5-0.8B`).
- `-RunLogRoot` to change telemetry/run log folder.
- `-ResourceSampleSeconds` to control CPU/GPU/RAM sampling interval.
- Additional args can be appended after `--` and are passed through to `hf_trainer.py`.

Example with telemetry + auto-benchmark:

```powershell
./scripts/run_local_hf_training.ps1 `
  -Epochs 100 `
  -CheckpointDir hf_checkpoints\hf_trainer_qwen3_5b `
  -Seed 42 `
  -CpuCap 20 `
  -RunBenchmark `
  -ResourceSampleSeconds 15 `
  -BaselineModelForBenchmark models\Qwen3.5-0.8B
```

Run artifacts are stored under:
- `training_run_logs/local_hf_train_<timestamp>/`
- benchmark outputs under `evaluation_results/comparison/run_local_<timestamp>/`.

## Official protocol training (VPS, Ray/vLLM)

Use the official shell launcher:

```bash
bash scripts/run_remote_official_azr.sh 7b
```

Modes available:

- `7b`
- `coder7b`
- `14b`
- `coder14b`
- `llama`

This forwards directly to:
- `<official-repo-checkout>/scripts/selfplay/7b.sh`
- `<official-repo-checkout>/scripts/selfplay/coder7b.sh`
- and so on.

## Benchmark comparison (local/evaluation)

Use:

```bash
.\azr_venv\Scripts\python.exe run_pre_post_benchmarks.py --baseline-model models/Qwen3.5-0.8B --improved-model hf_trainer_after_20260319_scoped/checkpoint_epoch_0 --benchmarks humaneval mbpp gsm8k --limit 100 --samples-per-task 1 --passk 1 --temperature 0.2 --top-p 0.95 --seed 20260319 --use-separate-value-model --cpu-cap 20 --rich
```

## Repository structure snapshot

- `hf_trainer.py`, `evaluate_benchmarks.py`, `optimize_hyperparameters.py`: local workflow entrypoints.
- `Absolute-Zero-Reasoner` (nested checkout) is intentionally excluded; official protocol work is kept in a separate checked-out path.
- `Documents/original_protocol_comparison_notes.md`: detailed local-vs-original protocol notes.
- `training_metrics/`, `evaluation_results/`, `hf_checkpoints/` (current default checkpoint location for local HF runs).