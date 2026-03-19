---
name: qwen3.5-entrypoint-model-integration
overview: Migrate the active HuggingFace AZR entrypoints to a single Qwen3.5 default target, verify repository availability, and add a deterministic model-download path into `models/`, while capturing follow-up hardening tasks identified from prior review.
todos:
  - id: todo-01-verify-qwen-id
    content: Run a metadata probe for `Qwen/Qwen3.5-0.5B` using `huggingface_hub` and capture whether the repo is public, private/gated, or missing.
    status: completed
  - id: todo-02-update-entry-defaults
    content: Keep `hf_trainer.py` defaults in sync by setting `Qwen/Qwen3.5-0.5B` and local checkpoint preference (`models/Qwen3.5-0.5B`) if present.
    status: completed
  - id: todo-03-update-eval-defaults
    content: Set `evaluate_benchmarks.py` CLI `--model` default to `Qwen/Qwen3.5-0.5B` and retain local path detection in `BenchmarkEvaluator`.
    status: completed
  - id: todo-04-update-hpo-defaults
    content: Set `optimize_hyperparameters.py` default `model_name`/`--model` arguments to `Qwen/Qwen3.5-0.5B`, and route through local path resolution before adapter creation.
    status: completed
  - id: todo-05-update-test-defaults
    content: Set `test_azr_system.py` default `model_name` and main test invocation to `Qwen/Qwen3.5-0.5B`, plus local path fallback if needed.
    status: completed
  - id: todo-06-update-model-download-script
    content: Change `download_model.py` to target `Qwen/Qwen3.5-0.5B` and keep Windows-safe directory handling (`local_dir` and no symlink reliance).
    status: completed
  - id: todo-07-attempt-model-download
    content: Execute `python download_model.py` once after update to validate repository access and confirm cached files under `models/Qwen3.5-0.5B`.
    status: completed
  - id: todo-08-handle-unavailable-fallback
    content: If step 1 reveals the repo is unavailable, choose the next approved Qwen model (and record the alternative) before continuing training/evaluation runs.
    status: completed
  - id: todo-09-document-change
    content: Append a changelog entry in IST format describing the model switch, download attempt, and availability result.
    status: completed
  - id: todo-10-fix-adapter-foundation
    content: "In a second pass, address remaining open issues from prior review: `azr_hf_adapter.py` exception swallowing, single-model PPO path robustness, quantization symmetry in critic loading, and PPO log-prob slicing edge cases."
    status: completed
  - id: todo-11-add-checkpoints-qa
    content: Update `Q&A.md` with clarifying question/answer whenever model availability or fallback decision is confirmed with the user.
    status: completed
isProject: false
---

- Confirm model availability before switching defaults by probing Hugging Face metadata and handling unavailable/private cases explicitly.
- Update every requested entry script to use a shared default model ID and local-resolve path pattern (`models/<model-name>` with `config.json` fallback).
- Adjust download helper to fetch that model into the repository `models` folder without symlink dependence on Windows.
- Record all user-visible changes in the required logs with IST timestamps.
- Add actionable follow-up todos for unresolved but high-impact stability issues discovered during prior adapter and PPO-path review.