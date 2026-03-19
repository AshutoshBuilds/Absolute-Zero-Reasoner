# Absolute Zero Reasoner (AZR) Implementation Plan

**Overall Goal:** Understand, set up, run, and potentially extend the [original Absolute-Zero-Reasoner (AZR) repository](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) to replicate and explore the self-play reasoning system described in the paper "Absolute Zero: Reinforced Self-play Reasoning with Zero Data," **leveraging local Ollama models for LLM inference.**

**Our Current Workspace Contains:**
-   An external/reference `Absolute-Zero-Reasoner` checkout for protocol comparison (run separately when using official stack workflows).
-   This plan file (`AZR_Implementation_Plan.md`).
-   Our custom `code_executor.py` (kept for reference; the original repo's executor will likely be used for code task execution).
-   An `archived_tictactoe_project/` directory with our previous Tic-Tac-Toe RL agent.
-   An `our_initial_azr_prototype/` directory with the mock AZR components we built initially.
-   Cursor rules in `.cursor/rules/` summarizing our findings about the original repository.

---

**Revised Implementation Phases (Focusing on Original Repository with Ollama Integration)**

**Phase 0: Environment Setup, Ollama Integration Strategy & Continued Exploration**

*   **P0.1: Model Selection & Access (Decision: Use Local Ollama Models):**
    *   **Primary Approach:** Utilize locally served models via Ollama (e.g., `llama3.1:8b`, `huihui_ai/qwen2.5-coder-abliterate:7b` as available). This simplifies GPU management for LLM inference on Windows and leverages existing model availability.
    *   **Implication:** The model interaction parts of AZR (primarily `actor_rollout_wg.generate_sequences` used by `CodeIORayPPOTrainer`) will need to be adapted to call the Ollama API instead of relying on `vLLM` loading models directly from Hugging Face paths.
    *   The `actor_rollout_ref.model.path` and `critic.model.path` in Hydra configs will need to be re-purposed or new configs added to specify Ollama model names/endpoints.
    *   The Critic model might still need to be a standard Hugging Face model loaded via `verl` if fine-tuning it with RL is a direct goal and Ollama doesn't support the required training interface. This needs investigation. (For now, focus on Proposer/Solver LLM via Ollama).
*   **P0.2: Set Up Original Repository's Environment (Status: Mostly Complete):**
    *   Conda environment `azr` (Python 3.10) is set up.
    *   CUDA toolkit (12.4 via Conda) and CUDA-enabled PyTorch (`2.5.1+cu121`) are installed.
    *   Most `requirements.txt` dependencies are installed, with workarounds for Windows-incompatible packages (`triton`, `uvloop`, etc., have been commented out; `flash-attn` installed via wheel).
    *   **Remaining Concern:** Potential version conflicts (`verl` vs. `vllm` vs. `transformers`) flagged by pip. Using Ollama might bypass some `vLLM` direct dependencies, potentially mitigating these.
*   **P0.3: Python Code Executor (Decision: Use Original Repo's Executor):**
    *   The original repo’s `absolute_zero_reasoner/utils/code_utils/python_executor.py` (from the official checkout) will be used for executing the *generated code tasks*. It has features like `pebble.ProcessPool` for timeouts and `checks.contains_banned_imports`.
    *   Our `code_executor.py` is archived for reference.
*   **P0.4: Initial Codebase Walkthrough (Status: Good Progress):**
    *   Key components and execution flow are generally understood. Rules have been created.
*   **P0.5: Ollama Integration Research & Prototyping (New Immediate Step):**
    *   Research Python client libraries for Ollama or direct HTTP API interaction (e.g., using `requests`).
    *   Prototype a Python function/class that can take a prompt, send it to a specified local Ollama model, and retrieve the generated text.
    *   Understand how to manage conversation history/context with Ollama if needed for multi-turn interactions within the `<think>`/`<answer>` generation. Ensure prompt templates from AZR (`<think>`, `<answer>` tags) can be used with Ollama.

---

**Phase 1: Adapting AZR for Ollama & Understanding Core Logic**

*   **Step 1.1: Configuration for Ollama:**
    *   Modify/add Hydra configurations (`configs/azr_ppo_trainer.yaml` and script overrides) to specify Ollama model names (e.g., `llama3.1:8b`) instead of Hugging Face paths for the actor/proposer/solver roles.
    *   Determine if critic model also needs to be Ollama-based or can remain as a Hugging Face model loaded by `verl`.
*   **Step 1.2: Adapt Model Generation/Rollout (`CodeIORayPPOTrainer` & `verl` workers):**
    *   Identify where `actor_rollout_wg.generate_sequences` is called (within `_compute_batch` of `CodeIORayPPOTrainer`).
    *   Modify the relevant `verl` worker class (e.g., `ActorRolloutRefWorker`) or create a new one. This worker's `generate_sequences` method should be changed from using `vLLM` to calling the Ollama API via our prototype from P0.5.
    *   This change aims to decouple the LLM generation step from direct `vLLM` CUDA kernel dependencies that are problematic on Windows.
*   **Step 1.3: Task/Data Representation & Management (`DatasetManager`):**
    *   Largely use existing `DatasetManager`. Task structures (`program_str`, `input_args`, etc.) remain relevant.
*   **Step 1.4: Proposer & Solver Logic (Model Interaction via Ollama):**
    *   Focus on crafting the correct prompts for Ollama to generate:
        *   Task Proposals (code, inputs, messages) with appropriate `<think>`/`<answer>` structures if the model is expected to produce those tags itself.
        *   Task Solutions (predicted outputs, inferred inputs, synthesized programs).
*   **Step 1.5: Task Validation & Ground Truth (`AZREnvironment` / `PythonExecutor`):**
    *   This part remains largely the same. The `AZREnvironment` (using the original repo's `PythonExecutor`) will validate tasks *proposed* by the Ollama-backed Proposer and verify solutions *generated* by the Ollama-backed Solver.

---

**Phase 2: Reward System & RL Algorithm (Leveraging Original Repo)**

*   **Step 2.1: Reward Calculation (`CodeIORewardManager`):**
    *   The existing `CodeIORewardManager` should still be largely usable. Its inputs (model generations, ground truth from executor) will now partly originate from Ollama (for generations) and partly from the `PythonExecutor` (for ground truth of code execution).
    *   Carefully verify that the output format from Ollama is compatible with what `CodeIORewardManager` expects for parsing and reward calculation (e.g., extraction of answers, thoughts).
*   **Step 2.2: RL Algorithm (TRR++ / PPO via `verl`):**
    *   The core RL update mechanism provided by `verl` and `CodeIORayPPOTrainer` should remain applicable. The main change is *how* actions (text generations) are produced and how their log-probabilities are obtained if needed by the PPO algorithm (Ollama API might not provide logprobs directly, which could be a challenge for PPO; REINFORCE might be simpler if only rewards are available).
    *   **Crucial Point**: If Ollama API does not provide log probabilities for generated tokens, we might need to simplify the RL algorithm (e.g., to a simpler REINFORCE if PPO's requirements can't be met) or find ways to estimate them if the chosen Ollama model can be run locally via `transformers` for a second pass to get logprobs (less efficient). For now, assume we can proceed and address this if it becomes a blocker for PPO. The paper uses TRR++, which is a policy gradient method. Simpler policy gradient methods can work without explicit logprobs of actions if rewards are properly attributed. The `adv_estimator: reinforce_plus_plus` in their config suggests a REINFORCE-based approach, which is more amenable to external black-box generators. 

---

**Phase 3: Running Experiments & Evaluation (with Ollama)**

*   **Step 3.1: Initial Run & Debugging:**
    *   Attempt to run a small-scale experiment using modified scripts (`scripts/selfplay/*.sh` adapted to call the Python main script with Ollama-specific configs).
    *   Focus on the Propose -> Validate (with PythonExecutor) -> Add to Buffer -> Sample from Buffer -> Solve (with Ollama) -> Evaluate Solution (with PythonExecutor) -> Calculate Reward flow.
*   **Step 3.2 - 3.4 (Seeding, Monitoring, Evaluation Protocol):** Largely as before, but adapted for the Ollama-based setup.

---

**Phase 4: Potential Extensions & Our Contributions**

*   Remains similar, with an added focus on the performance and limitations of using Ollama vs. directly hosted/fine-tuned models like in the original paper.
*   If the Python executor in the original repo still needs an upgrade (as per their roadmap), our `code_executor.py` could provide a basis for that.

---
This revised plan prioritizes leveraging your local Ollama models and the existing sophisticated codebase of the Absolute-Zero-Reasoner repository, while adapting the LLM interaction points. 