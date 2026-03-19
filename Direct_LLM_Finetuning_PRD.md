# PRD: Direct LLM Fine-tuning with PPO for AZR-like System

**Author:** AI Assistant (Gemini) & User
**Version:** 1.0
**Date:** 2024-08-23

## 1. Overview

This document outlines the requirements for implementing a direct LLM fine-tuning capability within our Reinforcement Learning (RL) framework, specifically tailored for a PPO-based agent. The primary goal is to enable iterative fine-tuning of a local Language Model (LLM) that acts as a Proposer and/or Solver for coding and reasoning tasks. This approach aligns more closely with the methodology of the original Absolute-Zero-Reasoner (AZR) project, allowing the model's weights to be updated directly based on rewards obtained during the RL loop.

## 2. Goals

*   Enable iterative, online fine-tuning of an LLM using PPO.
*   Allow the LLM to learn and improve its task generation (Proposer role) and solution generation (Solver role) capabilities.
*   Integrate seamlessly with existing components like the Experience Buffer, Reward Manager, and Python Executor.
*   Prioritize Parameter-Efficient Fine-Tuning (PEFT) techniques (e.g., LoRA, QLoRA) to manage resource constraints.
*   Provide a more effective and reactive learning mechanism compared to external/offline fine-tuning approaches.

## 3. Core Components & Requirements

### 3.1. Model Selection & Loading
*   **R3.1.1:** The system must support loading LLMs from Hugging Face Hub (`transformers` library).
*   **R3.1.2:** Initially target models suitable for code/reasoning tasks and manageable for fine-tuning (e.g., smaller variants of Gemma, Llama, Phi, Mistral).
*   **R3.1.3:** Implement support for Parameter-Efficient Fine-Tuning (PEFT) via the Hugging Face `peft` library (e.g., LoRA, QLoRA). This includes adding adapter layers and training only these adapters.
*   **R3.1.4:** The system must manage model placement on available devices (GPU preferred, CPU fallback) using `torch.device` and potentially the `accelerate` library.

### 3.2. Tokenizer
*   **R3.2.1:** The system must load and use the tokenizer corresponding to the selected LLM (`AutoTokenizer`).
*   **R3.2.2:** The tokenizer will be responsible for encoding prompts and decoding model generations, handling special tokens (padding, EOS, BOS) correctly.

### 3.3. RL Trainer (e.g., `HuggingFaceRLTrainer`)
*   **R3.3.1:** The trainer must hold a direct reference to the trainable PyTorch model (or its PEFT adapters) and the tokenizer.
*   **R3.3.2 (Optimizer):**
    *   **R3.3.2.1:** Implement a PyTorch optimizer (e.g., `torch.optim.AdamW`) to update the trainable model parameters.
*   **R3.3.3 (Generation Logic):**
    *   **R3.3.3.1:** The trainer must use the local, trainable model's `generate()` method for task and solution generation.
    *   **R3.3.3.2:** Generation parameters (temperature, top_p, max_new_tokens, etc.) must be configurable.
*   **R3.3.4 (PPO Update Logic):**
    *   **R3.3.4.1 (Actor-Critic):** Implement an actor-critic architecture.
        *   *Option A (Preferred):* Unified model with a value head added to the LLM.
        *   *Option B:* Separate critic model.
    *   **R3.3.4.2 (Forward Pass):** During PPO updates, re-run the model on collected experiences to get:
        *   Log probabilities of the generated actions (tokens) under the current policy.
        *   Value estimates from the critic/value head.
    *   **R3.3.4.3 (Advantage Calculation):** Compute advantages, preferably using Generalized Advantage Estimation (GAE).
    *   **R3.3.4.4 (Loss Calculation):** Calculate PPO-specific losses:
        *   Policy Loss (Actor) using the clipped surrogate objective.
        *   Value Loss (Critic) using Mean Squared Error.
        *   (Optional) Entropy bonus for exploration.
    *   **R3.3.4.5 (Training Step):** Perform backpropagation, gradient clipping, and optimizer steps.

### 3.4. Experience Buffer
*   **R3.4.1:** Adapt the existing `ExperienceBuffer` to store PPO-specific information, including log probabilities of actions taken and value estimates.

### 3.5. Reward Manager & Python Executor
*   **R3.5.1:** These components will continue to be used as-is for task validation, solution execution, and reward calculation. No major changes are anticipated for them in direct relation to enabling direct fine-tuning.

## 4. Implementation Phases

1.  **Phase 1: Environment Setup & Basic Model Integration**
    *   Install `transformers`, `torch`, `peft`, `accelerate`, `bitsandbytes`.
    *   Modify the trainer to load a Hugging Face model (with PEFT) and tokenizer.
    *   Implement generation using `model.generate()`.
    *   Test this pipeline with PPO updates initially disabled or virtual.
2.  **Phase 2: Implement Value Function/Critic**
    *   Choose and implement the actor-critic architecture (unified model with value head is preferred for simplicity if feasible).
3.  **Phase 3: Full PPO Update Implementation**
    *   Integrate calculations for log probabilities, value estimates, advantages, and PPO losses.
    *   Implement the training step including backpropagation and optimizer updates on the local model's trainable parameters.
4.  **Phase 4: Iterative Testing & Refinement**
    *   Begin with small models and datasets.
    *   Debug loss calculations and gradient updates thoroughly.
    *   Monitor key training metrics (rewards, losses, KL divergence).
    *   Tune hyperparameters for stability and performance.

## 5. Key Considerations

*   **Computational Resources:** Acknowledge the VRAM and processing demands of fine-tuning.
*   **Training Stability:** Implement measures to ensure stable training (e.g., gradient clipping, careful hyperparameter tuning).
*   **Batching:** Ensure correct data batching for generation and training.
*   **Reference Policy (PPO):** Consider implementing a reference policy mechanism for KL divergence calculation if needed for stability, though this can be a V2 feature.

## 6. Ollama Code Strategy

*   **Prioritize Direct Fine-tuning:** All new development will focus on the Hugging Face-based direct fine-tuning approach.
*   **Phased Removal:**
    1.  Achieve a robust and working implementation of the direct fine-tuning system.
    2.  Once stable and demonstrably effective, re-evaluate the need for the Ollama-specific interaction code.
    3.  If the Ollama pathway does not offer significant distinct value for the primary goal of iterative fine-tuning, refactor it out to simplify the main codebase. (The existing Ollama code can be preserved in version control history or a separate branch for archival/reference).

## 7. Future Considerations (Post V1)

*   Support for a wider range of models and PEFT techniques.
*   Advanced curriculum learning strategies integrated with the fine-tuning process.
*   More sophisticated methods for managing training stability (e.g., adaptive KL penalties).
*   Distributed training support for larger models if necessary. 