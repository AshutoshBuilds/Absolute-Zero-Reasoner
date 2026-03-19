import logging
import random
from typing import Dict

# Import utilities from azr_common_utils if they are used by prompt creation
# (e.g., _extract_code_from_solution, though it seems to be used by the caller of _create_solver_prompt)
# For now, assuming they are not directly needed within these two functions themselves.
# If _extract_code_from_solution is needed *inside* _create_solver_prompt, it should be imported here.
from azr_common_utils import _extract_code_from_solution # Added as it IS used by _create_solver_prompt

logger = logging.getLogger(__name__) # Use a logger specific to this module

TOPICS = [
    "math", "string manipulation", "list processing", "dictionary operations", 
    "basic algorithms", "geometry", "data parsing", "validation", "sorting", 
    "searching", "recursion", "pattern matching", "conversion", "statistics",
    "text analysis", "matrix operations", "file path handling", "json processing",
    "date and time", "set operations", "stack/queue logic"
]

CONSTRAINTS = [
    "Use a list comprehension",
    "Use a dictionary",
    "Use recursion",
    "Use the 'math' module",
    "Use string formatting",
    "Handle a specific edge case",
    "Return a boolean value",
    "Return a tuple",
    "Process a list of strings",
    "Calculate a cumulative sum",
    "Filter elements based on a condition",
    "Transform keys in a dictionary"
]

def create_proposer_prompt(trainer_instance, problem_type: str, phase: str = "train") -> str:
    """Create a prompt for the proposer model with curriculum-based difficulty."""
    memory = "" # No memory prompt for now, can be added from trainer_instance if needed
    current_difficulty = trainer_instance.current_difficulty # Access from trainer_instance
    
    # Select a random topic and constraint to encourage diversity
    topic = random.choice(TOPICS)
    constraint = random.choice(CONSTRAINTS)
    
    difficulty_descriptions = {
        1: "extremely simple, with straightforward logic and minimal steps",
        2: "simple, with basic logic and a few steps",
        3: "moderate difficulty, requiring multiple steps and some logical thinking",
        4: "challenging, requiring advanced logical thinking and multiple steps",
        5: "very challenging, requiring complex logic and multiple advanced steps"
    }
    difficulty_desc = difficulty_descriptions.get(current_difficulty, difficulty_descriptions[3])
    
    base_prompt = f"""You are a task generation bot. Your SOLE purpose is to generate a Python coding challenge formatted as a SINGLE, VALID JSON object.
DO NOT output ANY text before or after the JSON object.
DO NOT use markdown like ```json ... ```.
Your entire response MUST be ONLY the JSON object.

The JSON object must contain these exact keys: "code", "input", "output".
- "code": A string with the complete Python function definition.
- "input": A string representing the example input. For multiple arguments, use a tuple string like \"(10, 'hello')\".
- "output": A string representing the expected output. String outputs must be quoted like \"\\\"Result String\\\"\". Numeric/boolean outputs like \"123\", \"True\".

Problem Type: {problem_type}
Topic: {topic}
Constraint: {constraint}
Difficulty Level: {difficulty_desc} (level {current_difficulty}/5)
{memory}

Generate a Python coding challenge problem that:
1. Has a clear Python function to implement (provide the full function).
2. Includes one clear example input and its corresponding expected output.
3. Is solvable using only the Python standard library.
4. Has deterministic behavior.
5. Is related to the topic: '{topic}' and follows constraint: '{constraint}'.
6. IS NOT "add_numbers" or "greet" or a simple calculator. Be creative!

Example of the EXACT output format expected:
{{
  \"code\": \"def count_vowels(s):\\n    return sum(1 for char in s.lower() if char in 'aeiou')\",
  \"input\": \"\\\"Hello World\\\"\",
  \"output\": \"3\"
}}

Another example (string output):
{{
  \"code\": \"def reverse_list(lst):\\n    return lst[::-1]\",
  \"input\": \"[1, 2, 3]\",
  \"output\": \"[3, 2, 1]\"
}}

Now, generate the JSON object for the new task:"""
    return base_prompt

def create_solver_prompt(trainer_instance, task: Dict, phase: str = "train") -> str:
    """Create a prompt for the solver model."""
    problem_desc = task.get("problem_description", "")
    code_snippet = _extract_code_from_solution(task.get("code", ""))
    input_example = task.get("input", "")
    output_example = task.get("output", "")

    if not problem_desc:
        problem_desc = f"Implement the Python function provided below. When called with input `{input_example}`, it should produce output `{output_example}`."
    
    prompt = f"""You need to implement a Python function based on the following task.

Problem Context:
{problem_desc}

Provided Function (from the task definition, this is the function you need to implement or complete if it's a stub):
```python
{code_snippet}
```

Example Input:
`{input_example}`

Expected Output for this input:
`{output_example}`

Please provide ONLY the complete Python function implementation that solves this problem. 
Your function should work correctly for the example input to produce the example output.
Do not add any explanatory text outside the function's docstring or comments. 
Only use the Python standard library.
Your response should be a single Python code block.

Example of expected response format:
```python
def function_name(param1, param2):
    # Your implementation here
    result = param1 + param2
    return result
```
"""
    return prompt 