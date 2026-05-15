# Absolute Zero Model (AZR) Workspace

## Branch and workflow policy

Work on the local feature branch `feature/azr-local-hf-work` (tracking `origin/feature/azr-local-hf-work`).
Keep `main` aligned with the upstream path and do not re-add or use a nested `Absolute-Zero-Reasoner/` checkout for this repo’s workflows.

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

Use `.env` as the primary config source, then run:

```powershell
.\scripts\run_local_hf_training.ps1
```

Useful knobs in `.env`:

- `AZR_PYTHON_EXE`
- `AZR_PYTHON_EXE_FOR_BENCHMARK`
- `AZR_EPOCHS`
- `AZR_CHECKPOINT_DIR`
- `AZR_SEED`
- `AZR_SEED_TASKS_PER_TYPE`
- `AZR_CPU_CAP`
- `AZR_NO_RICH`
- `AZR_RUN_BENCHMARK`
- `AZR_BASELINE_MODEL_FOR_BENCHMARK`
- `AZR_BENCHMARK_LIMIT`
- `AZR_BENCHMARK_LIST` (comma-separated)
- `AZR_BENCHMARK_SEED`
- `AZR_BENCHMARK_SAMPLES_PER_TASK`
- `AZR_BENCHMARK_PASSK`
- `AZR_BENCHMARK_TEMPERATURE`
- `AZR_BENCHMARK_TOP_P`
- `AZR_BENCHMARK_CPU_CAP`
- `AZR_BENCHMARK_DATA_ROOT` (default `benchmark_data` under repo root; local `save_to_disk` snapshots)
- `AZR_BENCHMARK_ALLOW_ONLINE` (default off: no Hub fetch when a snapshot is missing; set `1` to download or runtime-cache to disk)
- `AZR_BENCHMARK_ALLOW_ONLINE_LOAD` (legacy alias for `AZR_BENCHMARK_ALLOW_ONLINE` when the latter is unset)
- `AZR_BENCHMARK_PREFETCH_ONLINE` (for `scripts/prefetch_benchmark_datasets.py`: `0` skips download)
- `AZR_BENCHMARK_HUB_REVISION` (optional Git revision for `load_dataset` during prefetch / Hub fallback)
- `AZR_BENCHMARK_OFFLINE` (`1` sets `HF_DATASETS_OFFLINE` / `HF_HUB_OFFLINE` when `evaluate_benchmarks.py` starts)
- `AZR_NO_BENCHMARK_RICH`
- `AZR_RUN_LOG_ROOT`
- `AZR_GPU_MEMORY_FRACTION`
- `AZR_CUDA_ALLOC_CONFIG`
- `AZR_USE_SEPARATE_VALUE_MODEL` (default `true`: separate actor + critic `ValueModel`; `false` for unified actor-critic)
- `AZR_MODEL_DTYPE` (`auto` | `fp16` | `bf16` | `fp32`; `auto` prefers bf16 on CUDA when supported)
- `AZR_GEN_LOGITS_FP32` (unset on CUDA: fp32-friendly generation logits in `hf_generation_utils`; `0` disables)
- `AZR_RESOURCE_SAMPLE_SECONDS`
- `AZR_USE_4BIT` (default `false` for paper-style full weights; `true` for bitsandbytes 4-bit dev runs)
- `AZR_CUDA_VISIBLE_DEVICES`

Booleans in `.env` can use `true/false`, `1/0`, `yes/no`, or `on/off` (case-insensitive).

CLI args still override env values when passed explicitly:

```powershell
.\scripts\run_local_hf_training.ps1 -Epochs 120 -RunBenchmark -NoRich
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

Prefetch standard benchmark splits once (avoids many Hub requests during post-train eval):

```powershell
.\azr_venv\Scripts\python.exe scripts\prefetch_benchmark_datasets.py
```

Optional: `set HF_HUB_ENABLE_HF_TRANSFER=1` after `pip install hf_transfer` for faster downloads. Then run eval with `AZR_BENCHMARK_ALLOW_ONLINE=0` (default) and optionally `AZR_BENCHMARK_OFFLINE=1` so only local snapshot I/O runs.

Use:

```bash
.\azr_venv\Scripts\python.exe run_pre_post_benchmarks.py --baseline-model models/Qwen3-0.6B --improved-model hf_trainer_after_20260319_scoped/checkpoint_epoch_0 --benchmarks humaneval mbpp gsm8k --limit 100 --samples-per-task 1 --passk 1 --temperature 0.2 --top-p 0.95 --seed 20260319 --use-separate-value-model --cpu-cap 20 --rich
```

## Self-evolution pilot (conservative)

Run a controlled 3-round candidate loop:

```bash
python orchestrate_self_evolution.py --rounds 3 --max-accepted-rounds 1 --epochs 20 --baseline-model models/Qwen3-0.6B --checkpoint-dir hf_checkpoints\\Qwen3-0.6B --use-4bit --model-dtype fp16
```

Run one isolated candidate experiment:

```bash
python scripts/run_research_harness.py --run-root autoevo/run_rounds --run-name pilot_01 --baseline-model models/Qwen3-0.6B --checkpoint-dir hf_checkpoints\\Qwen3-0.6B --epochs 20 --trainer-config-json '{"learning_rate": 1e-8, "generation_steps_per_epoch": 8}'
```

Artifacts from the pilot are tracked under `autoevo/run_rounds/` and each run writes:

- `manifest.json` (config, manifests, benchmark delta decision payload)
- `run_config.json` (resolved arguments)
- `train.stdout.log` / `train.stderr.log`
- `benchmark.stdout.log` / `benchmark.stderr.log` (when enabled)

## Repository structure snapshot

- `hf_trainer.py`, `evaluate_benchmarks.py`, `optimize_hyperparameters.py`: local workflow entrypoints.
- Official protocol work is intentionally kept in a separate checked-out `Absolute-Zero-Reasoner` path, not in this local workspace.
- `Documents/original_protocol_comparison_notes.md`: detailed local-vs-original protocol notes.
- `training_metrics/`, `evaluation_results/`, `hf_checkpoints/` (current default checkpoint location for local HF runs).

