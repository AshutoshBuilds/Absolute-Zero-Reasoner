## Hybrid AZR Training Execution

Goal: Keep local iteration fast and stable on 3090 Ti while enabling official-style Ray/vLLM training on VPS GPUs (H200/H100/A100).

### Architecture split

## Local mode (Windows desktop)

- Use: `hf_trainer.py`
- Why: small, single-process path with strict CPU cap and explicit resource controls.
- Primary target: iteration, debugging, smoke checks, and small experiments.

## VPS mode (H200/H100/A100)

- Use: official scripts from a separate checkout of `AshutoshBuilds/Absolute-Zero-Reasoner`.
- Why: distributed Ray + vLLM + veRL stack from the GitHub project.
- Primary target: full protocol parity and large-scale runs.

### Implementation

- Add a Windows launcher for local HF path:
  - `scripts/run_local_hf_training.ps1`
- Add a Linux launcher for official remote mode:
  - `scripts/run_remote_official_azr.sh`
- Update README with mode matrix and exact command examples.
- Record every environment/protocol change in `changelog.md` and `Q&A.md`.

### Runbook (short)

#### Local (Windows)

```powershell
./scripts/run_local_hf_training.ps1 -Epochs 200 -CheckpointDir hf_checkpoints\hf_trainer_qwen3_5b -CpuCap 20
./scripts/run_local_hf_training.ps1 -Epochs 200 -RunBenchmark -ResourceSampleSeconds 15 -BaselineModelForBenchmark models/Qwen3.5-0.8B
```

#### Remote (VPS)

```bash
bash scripts/run_remote_official_azr.sh 7b
```

Use an explicit checkout path when running remote mode:

```bash
export AZR_OFFICIAL_REPO_PATH=/path/to/AshutoshBuilds-Absolute-Zero-Reasoner
bash scripts/run_remote_official_azr.sh 7b
```

### Optional local run policy (new)

- Use timestamped run folders in `training_run_logs/` for every launch.
- Store both process logs and benchmark logs there:
  - `hf_trainer_stdout.log`
  - `hf_trainer_stderr.log`
  - `resource_log.tsv`
  - `benchmark_stdout.log` / `benchmark_stderr.log` (when `-RunBenchmark` is set)
- Keep CPU cap low on desktop and tune:
  - `-CpuCap 20` for safe iteration.
  - `-ResourceSampleSeconds 15` for frequent telemetry.
- Auto-benchmark flow (when enabled):
  - Run local training
  - Select newest local checkpoint in `--checkpoint-dir`
  - Launch `run_pre_post_benchmarks.py` with local defaults.

### Verification checklist

- Confirm mode in use from startup message.
- Confirm output folder exists:
  - local: `hf_checkpoints\hf_trainer_*` (or explicit `CheckpointDir` under `hf_checkpoints\`)
  - official: configured checkpoint / artifact path in selected `.sh` script.
- Confirm runtime stack:
  - local: no Ray/vLLM import required in main path.
  - official: run against explicit official checkout path and confirm Ray/vLLM training process starts and logs worker initialization.
- Compare final metrics with `run_pre_post_benchmarks.py` protocols as configured.

