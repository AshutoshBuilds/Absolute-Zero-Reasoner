# AZR Protocol Comparison Notes (Original Repo vs Local Branch)

## Scope checked
- Source of truth for original protocol: `Absolute-Zero-Reasoner/README.md` and `Absolute-Zero-Reasoner/absolute_zero_reasoner/main_azr_ppo.py` plus training shell scripts under `Absolute-Zero-Reasoner/scripts/*`.
- Source of truth for local protocol: `run_pre_post_benchmarks.py`, `evaluate_benchmarks.py`, `hf_trainer.py`, and related entrypoints in this workspace.

## Original GitHub protocol (as observed)
- Launch flow in original repo relies on `python -m absolute_zero_reasoner.main_azr_ppo` with Hydra config (`main_azr_ppo.py`).
- Training flow is distributed: `ray` initialized in main, worker roles `ActorRollout`, `Critic`, optional `RefPolicy`.
- Rollouts are wired to `vllm` via `actor_rollout_ref.rollout.name=vllm` and tensor/model parallel settings.
- Default training scripts include model choices like `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-Coder-7B`, and many PPO/actor-critic parameters are set in shell scripts.
- README “Evaluation Code” section is explicitly marked `TODO`, so benchmark execution is not fully documented there.

## Local branch protocol currently implemented
- `run_pre_post_benchmarks.py` orchestrates two benchmark runs (baseline + improved) and then computes deltas.
- Benchmark CLI defaults are normalized to `--benchmarks humaneval mbpp gsm8k` (math excluded by default) with a soft CPU cap (`--cpu-cap 20.0`).
- Local evaluator uses HuggingFace model loading through `HuggingFaceAdapter` and local cache resolution under `models/<model-name>` when available.
- Evaluation is performed by `evaluate_benchmarks.py` custom loops over HF datasets, capped subsets by `--limit`, and custom pass logic (`samples-per-task`, pass@k approximations).

## Expected-vs-observed delta snapshot
### Expected protocol intent from previous user runs
- Keep protocol consistent with earlier local baseline/improved runs:
  - `--benchmarks humaneval mbpp gsm8k`
  - deterministic seed `20260319`
  - `--samples-per-task 1 --passk 1`
  - `--temperature 0.2 --top-p 0.95`
  - `--use-separate-value-model`
  - CPU-limited run to keep desktop responsive

### Observed results (latest full run, limit=100)
- Baseline: `models\Qwen3.5-0.8B`
- Improved: `hf_trainer_after_20260319_scoped\checkpoint_epoch_0`
- `run_pre_post_benchmarks.py` command used:
  - `--benchmarks humaneval mbpp gsm8k --limit 100 --samples-per-task 1 --passk 1 --temperature 0.2 --top-p 0.95 --seed 20260319 --use-separate-value-model --cpu-cap 20 --rich`
- Baseline summary:
  - HumanEval `55/100` (`0.5500`)
  - MBPP `0/100` (`0.0000`)
  - GSM8K `1/100` (`0.0100`)
- Improved summary:
  - HumanEval `58/100` (`0.5800`)
  - MBPP `0/100` (`0.0000`)
  - GSM8K `1/100` (`0.0100`)
- Delta (improved − baseline):
  - HumanEval `+0.0300`
  - MBPP `+0.0000`
  - GSM8K `+0.0000`

### Comparison to previous scoped local run (limit=3)
- `run_post_fix_scoped` showed a noisy result in that shorter sample:
  - HumanEval `1.0000 -> 0.0000` (delta `-1.0000`)
  - MBPP and GSM8K both `0.0000`
- That earlier run is now superseded by this full 100-sample protocol for better signal.

## Practical conclusion
- Original repo protocol is primarily about distributed veRL + vLLM orchestration and extensive CLI config.
- Local branch now has a stable, reproducible, CPU-aware benchmark protocol with richer CLI output and deterministic reruns, but uses a simplified single-process execution and different evaluation implementation surface.
- Current local comparison is internally consistent; the key "protocol difference" versus original is orchestration/stack, not benchmark metric math (since both compare HumanEval/MBPP/GSM8K style outcomes, with local math benchmark currently optional/out-of-scope).
