import re
import ast
import logging
import difflib
import math
from typing import Dict, Any, List, Set, Tuple, Optional, Union

# Import common utilities
from azr_common_utils import (
    _clean_input_string,
    _evaluate_input,
    _extract_code_from_solution, # May not be used directly here, but good to have if needed
    _compare_outputs,
    _calculate_ast_complexity,
    contains_banned_imports
)

logger = logging.getLogger(__name__)

class HFRewardManager:
    """Reward manager for HuggingFace-based AZR"""
    
    def __init__(self, difficulty: float = 1.0, python_executor: Any = None):
        """Initialize the reward manager
        
        Args:
            difficulty: The target difficulty level (1-5)
            python_executor: An instance of a Python code executor.
        """
        self.difficulty = difficulty
        self.python_executor = python_executor # Store the executor
        self._last_comparison_similarity = 0.0 # Retained from Ollama version, might be useful
        self.previous_tasks_summary: Set[str] = set() # For novelty calculation

        if self.python_executor is None:
            logger.warning("HFRewardManager initialized without a Python executor. Code execution capabilities will be limited.")
        
    def update_difficulty(self, difficulty: float):
        """Update the target difficulty level
        
        Args:
            difficulty: New difficulty level (1-5)
        """
        self.difficulty = difficulty

    # --- Proposer Reward Calculation ---
    def calculate_proposer_reward(
        self, 
        generated_task: Dict[str, Any],
        proposer_raw_output: str,
        solver_attempts_results: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the reward for the proposer based on the generated task and solver's performance.
        
        Args:
            generated_task: The task dictionary ({"code", "input", "output", ...}) from parsing.
            proposer_raw_output: The raw string output from the proposer LLM.
            solver_attempts_results: A list of dictionaries, each from eval_solution_function 
                                     for a solver's attempt on this task.
        Returns:
            A tuple (total_reward, reward_components_dict).
        """
        reward_components = {
            "r_parsable": 0.0,
            "r_valid_spec": 0.0,    # Task spec (code, input, output) is valid and runnable
            "r_safety": 0.0,        # Code safety (no banned imports)
            "r_complexity": 0.0,    # Code complexity (AST based)
            "r_difficulty": 0.0,  # Alignment with target difficulty
            "r_learnability": 0.0,  # Based on solver's success on this proposed task
            "r_novelty": 0.0,       # Uniqueness of the task
            "r_code_quality_proposer": 0.0 # Quality of the *example solution* provided by proposer
        }

        # 1. Parsability Reward (Implicitly handled by _parse_generated_tasks in trainer)
        # If generated_task is None or marked with parsing_failed, it means parsing failed.
        # The trainer should ideally pass a flag or handle this.
        # For now, assume if we get a task dict, it was somewhat parsable.
        # A more direct way: check if proposer_raw_output led to a valid generated_task.
        if not generated_task or generated_task.get("parsing_failed", False):
            reward_components["r_parsable"] = -1.0 # Penalize unparsable output severely
            total_reward = self._aggregate_rewards(reward_components)
            return total_reward, reward_components
        else:
            reward_components["r_parsable"] = 1.0

        task_code = generated_task.get("code", "")
        task_input_str = generated_task.get("input", "")
        task_expected_output_str = generated_task.get("output", "")

        # 2. Safety Reward
        is_banned, reason = contains_banned_imports(task_code)
        if not is_banned:
            reward_components["r_safety"] = 1.0
        else:
            reward_components["r_safety"] = -1.0 # Penalize unsafe code
            logger.debug(f"Proposer task safety penalty: {reason}")
            total_reward = self._aggregate_rewards(reward_components)
            return total_reward, reward_components # Stop if unsafe

        # 3. Task Specification Validity (r_valid_spec)
        #    - Can the example solution be run with the example input to get the example output?
        if self.python_executor and task_code and task_input_str:
            try:
                # We need to use the executor's solution_check or equivalent method.
                # Assuming python_executor has a method like `check_solution(code, input_str, expected_output_str)`
                # that returns a dict like {'valid': bool, 'output': str, 'error': str or None}
                
                # If using the solution_check method we added to an executor earlier:
                if hasattr(self.python_executor, 'solution_check'):
                    validation_result = self.python_executor.solution_check(
                        solution_code=task_code, 
                        input_str=task_input_str, 
                        expected_output_str=task_expected_output_str
                    )
                    if validation_result.get("valid", False):
                        reward_components["r_valid_spec"] = 1.0
                    else:
                        reward_components["r_valid_spec"] = -0.5 # Penalize if example solution doesn't work
                        logger.debug(f"Proposer task spec invalid: {validation_result.get('output')}")
                else:
                    # Fallback if no such specific method - this is harder to score directly here
                    # We might need to execute it and compare. For now, assume neutral if no check method.
                    reward_components["r_valid_spec"] = 0.2 # Small bonus for providing all parts
                    logger.warning("Python executor does not have 'solution_check'. Proposer r_valid_spec might be inaccurate.")

            except Exception as e:
                logger.warning(f"Error validating proposer's example solution: {e}")
                reward_components["r_valid_spec"] = -0.75 # Penalize if execution itself errors
        else:
            reward_components["r_valid_spec"] = -1.0 # Missing critical components

        # 4. Complexity Reward (for the proposer's example solution code)
        complexity_score = _calculate_ast_complexity(task_code)
        # Normalize complexity: target complexity around 5-15 for reward range.
        # This is a simple way, could be a Gaussian centered at target.
        if 3 <= complexity_score <= 20: # Good range
            reward_components["r_complexity"] = 0.2 + 0.8 * ((complexity_score - 3) / 17)
        elif complexity_score < 3:
            reward_components["r_complexity"] = 0.1 * (complexity_score / 3) # Penalize too simple
        else: # complexity_score > 20
            reward_components["r_complexity"] = max(0, 0.2 - 0.5 * ((complexity_score - 20) / 20)) # Penalize too complex

        # 5. Difficulty Alignment Reward
        # This uses the current `self.difficulty` (target) vs. `complexity_score` (actual)
        # Scale self.difficulty (1-5) to roughly match complexity_score typical range (e.g., 1-25)
        target_complexity_min = (self.difficulty -1) * 4 + 2 # e.g. diff 1 -> 2, diff 3 -> 10, diff 5 -> 18
        target_complexity_max = self.difficulty * 5 + 3       # e.g. diff 1 -> 8, diff 3 -> 18, diff 5 -> 28
        if target_complexity_min <= complexity_score <= target_complexity_max:
            reward_components["r_difficulty"] = 1.0
        elif complexity_score < target_complexity_min:
            reward_components["r_difficulty"] = 0.5 * (complexity_score / target_complexity_min if target_complexity_min > 0 else 0)
        else: # complexity_score > target_complexity_max
            reward_components["r_difficulty"] = 0.5 * (target_complexity_max / complexity_score if complexity_score > 0 else 0)
        
        # 6. Learnability Reward (based on solver attempts)
        # AZR Logic: Reward tasks that are VALID but HARD for the current solver.
        # Target: Proposer should generate tasks where solver accuracy is low (or target ~0.5).
        # Here we implement a simple "harder is better" reward, bounded by validity.
        
        if not reward_components["r_valid_spec"] > 0:
             # If task spec is invalid, learnability is irrelevant/penalized
             reward_components["r_learnability"] = -1.0
        elif solver_attempts_results:
            successful_solves = sum(1 for res in solver_attempts_results if res.get("success_bool", False))
            total_attempts = len(solver_attempts_results)
            success_rate = successful_solves / total_attempts if total_attempts > 0 else 0.0
            
            # Reward Difficulty: 1.0 - success_rate
            # If solver solves it 100% (easy), reward is 0.0.
            # If solver solves it 0% (hard), reward is 1.0.
            # This assumes the task IS valid (checked above).
            reward_components["r_learnability"] = 1.0 - success_rate
            
            logger.debug(f"Proposer Learnability: Success Rate {success_rate:.2f} -> Reward {reward_components['r_learnability']:.2f}")
        else:
            # No solver attempts (e.g., if task was filtered out before solver stage)
            reward_components["r_learnability"] = -0.5 # Penalize if not solvable or not attempted

        # 7. Novelty Reward
        task_summary = self._summarize_task(task_code, task_input_str, task_expected_output_str)
        if task_summary not in self.previous_tasks_summary:
            reward_components["r_novelty"] = 1.0
            self.previous_tasks_summary.add(task_summary)
        else:
            reward_components["r_novelty"] = 0.1 # Small reward for re-generating, but prefer novelty
        # Consider more sophisticated novelty (e.g., embedding similarity to previous tasks)

        # 8. Code Quality of Proposer's Example Solution
        reward_components["r_code_quality_proposer"] = self._evaluate_code_quality_metrics(task_code)

        total_reward = self._aggregate_rewards(reward_components)
        return total_reward, reward_components

    # --- Solver Reward Calculation ---
    def calculate_solver_reward(
        self, 
        solver_code_str: str, 
        task: Dict[str, Any],
        execution_result: Dict[str, Any] # Result from python_executor.check_solution or equivalent
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the reward for the solver based on its generated solution.

        Args:
            solver_code_str: The code string generated by the solver.
            task: The task dictionary the solver was attempting.
            execution_result: A dictionary from the executor, typically including:
                              {'valid': bool, 'output': str, 'error': str|None, 'similarity': float, 'reason': str}
                              The 'valid' field from executor is the primary correctness signal.
                              'similarity' and 'reason' can come from _compare_outputs.

        Returns:
            A tuple (total_reward, reward_components_dict).
        """
        reward_components = {
            "r_correctness": 0.0,     # Based on executor's direct output comparison
            "r_safety_solver": 0.0,   # Safety of solver's code
            "r_efficiency": 0.0,      # Placeholder for code efficiency (e.g. execution time)
            "r_complexity_solver": 0.0, # AST complexity of solver's code
            "r_code_quality_solver": 0.0 # Adherence to good coding practices (docstrings, type hints etc.)
        }

        # 1. Correctness Reward (Primary)
        # Relies on the 'valid' and 'similarity' fields from execution_result
        # execution_result['valid'] should be the direct True/False from executor's comparison logic
        # execution_result['similarity'] can be a more nuanced score from _compare_outputs
        
        is_correct = execution_result.get("valid", False)
        similarity_score = execution_result.get("similarity", 0.0) # From _compare_outputs
        
        if is_correct:
            reward_components["r_correctness"] = 1.0 + (similarity_score - 0.5) * 0.2 # Max 1.1 for perfect, slight bonus for high similarity
        else:
            # If not fully correct, give partial credit based on similarity
            reward_components["r_correctness"] = -1.0 + similarity_score * 0.8 # Range from -1.0 to -0.2 for incorrect

        # 2. Safety of Solver's Code
        is_banned_solver, reason_solver = contains_banned_imports(solver_code_str)
        if not is_banned_solver:
            reward_components["r_safety_solver"] = 0.2 # Small bonus for safe code
        else:
            reward_components["r_safety_solver"] = -1.0 # Penalize unsafe code
            logger.debug(f"Solver code safety penalty: {reason_solver}")
            # If unsafe, perhaps other rewards should be negated or reduced.
            # For now, let it be an additive penalty.

        # 3. Efficiency Reward (Placeholder)
        # This would require timing execution, which is complex with external executors.
        # For now, neutral. Could be based on (1 / (1 + execution_time)) or similar.
        reward_components["r_efficiency"] = 0.0 

        # 4. Complexity of Solver's Code
        solver_complexity = _calculate_ast_complexity(solver_code_str)
        # Penalize overly complex solutions compared to task's original solution (if available)
        # or a general notion of simplicity.
        # For now, a gentle penalty for very high complexity.
        if solver_complexity > 30:
            reward_components["r_complexity_solver"] = -0.5 * ((solver_complexity - 30) / 30)
        elif solver_complexity < 2:
             reward_components["r_complexity_solver"] = -0.2 # Slightly penalize trivial (empty) solutions
        else:
            reward_components["r_complexity_solver"] = 0.1 # Small bonus for reasonable complexity
        
        # 5. Code Quality of Solver's Code
        reward_components["r_code_quality_solver"] = self._evaluate_code_quality_metrics(solver_code_str)

        total_reward = self._aggregate_rewards(reward_components)
        return total_reward, reward_components

    def _aggregate_rewards(self, components: Dict[str, float]) -> float:
        """Aggregates reward components into a single scalar value."""
        # Simple weighted sum for now. Weights can be tuned.
        weights = {
            "r_parsable": 1.5, "r_valid_spec": 1.5, "r_safety": 2.0,
            "r_complexity": 0.5, "r_difficulty": 0.7,
            "r_learnability": 1.5, "r_novelty": 0.8, 
            "r_code_quality_proposer": 0.5,
            # Solver rewards
            "r_correctness": 3.0, "r_safety_solver": 1.5,
            "r_efficiency": 0.3, "r_complexity_solver": 0.3,
            "r_code_quality_solver": 0.7 
        }
        
        # Apply weights to existing components
        weighted_sum = sum(components.get(comp, 0.0) * weights.get(comp, 1.0) for comp in components)
        
        # Normalize by sum of weights for components that were present
        num_components_considered = sum(weights.get(comp, 1.0) for comp in components if comp in components)
        
        if num_components_considered == 0: return 0.0
        
        # Scale to a typical range e.g. -1 to 1 or 0 to 1, depending on preference.
        # The current reward components are roughly in [-1, 1] or [0, 1].
        # A simple average of weighted components:
        return weighted_sum / num_components_considered


    def _summarize_task(self, code: str, input_str: str, output_str: str) -> str:
        """Creates a simple hashable summary of a task for novelty checking."""
        # Normalize code a bit (remove whitespace, comments) for more robust summary
        code_lines = []
        for line in code.splitlines():
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                code_lines.append(stripped_line)
        normalized_code = "".join(code_lines)
        return f"code:{normalized_code}|input:{input_str}|output:{output_str}"

    def _evaluate_code_quality_metrics(self, code: str) -> float:
        """Evaluates basic code quality metrics like presence of docstrings, type hints."""
        if not code.strip(): return -0.5 # Penalize empty code

        quality_score = 0.0
        num_functions = 0
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    num_functions += 1
                    # Docstring check
                    if ast.get_docstring(node):
                        quality_score += 0.4
                    else:
                        quality_score -= 0.1 # Minor penalty for missing docstring
                    
                    # Type hints for arguments
                    arg_hints = sum(1 for arg_node in node.args.args if arg_node.annotation)
                    if node.args.args: # If there are args
                        quality_score += 0.3 * (arg_hints / len(node.args.args))
                    else: # No args, no penalty/bonus for arg hints
                        quality_score += 0.1 # Small bonus if no args and no hints needed

                    # Return type hint
                    if node.returns:
                        quality_score += 0.3
                    else:
                        # Allow no return hint if function clearly doesn't return (e.g. only print statements)
                        # This is a heuristic: check if no return statements or all are `return None`
                        has_explicit_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))
                        if has_explicit_return: # Has explicit return but no hint
                             quality_score -= 0.1 # Minor penalty

                    # Basic input validation check (e.g., isinstance, assert)
                    has_validation = any(
                        (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'isinstance') or 
                        isinstance(n, ast.Assert)
                        for n in ast.walk(node)
                    )
                    if has_validation:
                        quality_score += 0.2
        except SyntaxError:
            return -1.0 # Penalize syntax errors heavily in quality score
        
        if num_functions == 0: # Not a function definition, or malformed
            if "def " not in code: # Truly no attempt at a function
                return -0.75
            return -0.25 # Malformed function

        # Normalize score: max possible for one function is ~1.2 (0.4+0.3+0.3+0.2)
        # Return score in range roughly -1 to 1.
        # Current quality_score can range, e.g. perfect = 0.4+0.3+0.3+0.2 = 1.2
        # Worst case (no doc, no hints, has args/return) = -0.1 + 0 - 0.1 = -0.2
        # Let's scale it simply: if >0, it's good. If negative, it's bad.
        return max(-1.0, min(1.0, quality_score / (num_functions if num_functions > 0 else 1))) 

    # The following methods (_clean_input_string, _evaluate_input, _compare_outputs)
    # were previously here but are now intended to be imported from azr_common_utils.py
    # Ensure they are removed if they were copied over during initial refactor.

    # Example: solution_check (this is more of an executor utility but can be part of reward manager logic if executor is simple)
    # This demonstrates how _compare_outputs and _evaluate_input from common_utils would be used.
    def solution_check(self, solution_code: str, input_str: str, expected_output_str: str) -> Dict[str, Any]:
        """
        Checks if a given solution_code produces the expected_output for a given input_str.
        This method is primarily for the executor, but a version can live here if the reward manager
        needs to perform its own execution checks or augment the executor's results.
        Uses helper functions for parsing input and comparing outputs.

        Returns a dictionary: 
        {'valid': bool, 'output': Any, 'error': str|None, 'similarity': float, 'reason': str}
        'valid' is True if the actual output matches the expected output based on _compare_outputs.
        'similarity' and 'reason' also come from _compare_outputs.
        """
        result = {
            'valid': False, 
            'output': None, 
            'error': None, 
            'similarity': 0.0, 
            'reason': 'INIT'
        }

        if self.python_executor is None:
            result['error'] = "No Python executor available in HFRewardManager."
            result['reason'] = "NO_EXECUTOR"
            return result
        
        # Use the executor's actual execution method. 
        # This `solution_check` in the reward manager is more about defining HOW to use the executor's output.
        # The executor should ideally return a structured result like this itself.
        # For now, let's assume a generic `execute_code` method on the executor that takes full code to run.
        
        # We need to: 
        # 1. Parse the `input_str` into Python object(s).
        # 2. Construct the code to run (solution_code + function call with parsed input).
        # 3. Execute and capture output/errors.
        # 4. Compare output with `expected_output_str` using `_compare_outputs`.

        # This is a simplified simulation of what an executor would do.
        # The HFRewardManager would typically rely on the output of a dedicated executor.
        # However, if we need to call _compare_outputs, we need the actual output.
        
        # Let's assume the python_executor itself has a `run_solution_with_input` method:
        # def run_solution_with_input(self, solution_code: str, input_str: str) -> Dict[str, Any]:
        #    # ... internally parses input, execs code, calls func, returns {'output': ..., 'error': ...}
        #    pass

        if not hasattr(self.python_executor, 'run_solution_with_input') and not hasattr(self.python_executor, 'solution_check'):
            result['error'] = "Executor lacks `run_solution_with_input` or `solution_check` method."
            result['reason'] = "EXECUTOR_METHOD_MISSING"
            return result

        try:
            if hasattr(self.python_executor, 'solution_check'): # If executor has its own comprehensive check
                exec_res = self.python_executor.solution_check(solution_code, input_str, expected_output_str)
                actual_output = exec_res.get('output', None)
                result['error'] = exec_res.get('error', None)
            elif hasattr(self.python_executor, 'run_solution_with_input'): # If executor just runs code
                exec_res = self.python_executor.run_solution_with_input(solution_code, input_str)
                actual_output = exec_res.get('output', None)
                result['error'] = exec_res.get('error', None)
            else: # Should not happen due to check above
                raise ValueError("Suitable executor method not found after check")

            if result['error']:
                result['output'] = None # Error occurred during execution
                result['valid'] = False
                result['similarity'] = 0.0
                result['reason'] = f"EXECUTION_ERROR: {str(result['error'])[:100]}" # Truncate long errors
                return result
            
            result['output'] = actual_output
            is_match, score, reason_code = _compare_outputs(actual_output, expected_output_str)
            result['valid'] = is_match
            result['similarity'] = score
            result['reason'] = reason_code

        except Exception as e:
            logger.error(f"Error in HFRewardManager.solution_check when using executor: {e}")
            result['error'] = str(e)
            result['valid'] = False
            result['similarity'] = 0.0
            result['reason'] = "REWARD_MANAGER_CHECK_ERROR"
            
        return result 