import logging
import random
from typing import Dict
from typing import Any, Iterable, List, Optional

# Import utilities from azr_common_utils if they are used by prompt creation
# (e.g., _extract_code_from_solution, though it seems to be used by the caller of _create_solver_prompt)
# For now, assuming they are not directly needed within these two functions themselves.
# If _extract_code_from_solution is needed *inside* _create_solver_prompt, it should be imported here.
from azr_common_utils import _extract_code_from_solution # Added as it IS used by _create_solver_prompt

logger = logging.getLogger(__name__) # Use a logger specific to this module
K_REFERENCE_COUNT = 6

DEFAULT_MEMORY_EXAMPLES: List[Dict[str, Any]] = [
    {
        "code": "def count_vowels(s):\n    return sum(1 for char in s.lower() if char in 'aeiou')",
        "input": "\"Hello World\"",
        "output": "3",
    },
    {
        "code": "def reverse_list(lst):\n    return lst[::-1]",
        "input": "[1, 2, 3]",
        "output": "[3, 2, 1]",
    },
    {
        "code": "def is_palindrome(s):\n    clean = str(s).replace(' ', '').lower()\n    return clean == clean[::-1]",
        "input": "\"Racecar\"",
        "output": "True",
    },
    {
        "code": "def sum_dict_values(d):\n    return sum(d.values()) if d else 0",
        "input": "{'a': 1, 'b': 2, 'c': 3}",
        "output": "6",
    },
    {
        "code": "def flatten_matrix(matrix):\n    return [item for row in matrix for item in row]",
        "input": "[[1, 2], [3, 4]]",
        "output": "[1, 2, 3, 4]",
    },
    {
        "code": "def safe_divide(a, b):\n    return a / b if b != 0 else 0",
        "input": "(10, 2)",
        "output": "5",
    },
]


def _coerce_memory_examples(seed_tasks: Optional[Iterable[Dict[str, Any]]], k_reference: int) -> List[Dict[str, Any]]:
    k_reference = max(0, int(k_reference))
    selected_examples: List[Dict[str, Any]] = []

    if k_reference == 0:
        return selected_examples

    if seed_tasks:
        shuffled = list(seed_tasks)
        random.shuffle(shuffled)
        selected_examples.extend(shuffled[:k_reference])

    if len(selected_examples) < k_reference:
        needed = k_reference - len(selected_examples)
        selected_examples.extend(DEFAULT_MEMORY_EXAMPLES[:needed])

    if len(selected_examples) > k_reference:
        selected_examples = selected_examples[:k_reference]

    return selected_examples


def _format_memory_examples(example_tasks: List[Dict[str, Any]]) -> str:
    if not example_tasks:
        return ""

    lines = []
    for idx, item in enumerate(example_tasks, start=1):
        code = item.get("code", "").replace("{", "{{").replace("}", "}}")
        task_input = item.get("input", "")
        output = item.get("output", "")
        lines.append(
            f"\nExample {idx}:\n"
            f"Code: {code}\n"
            f"Input: {task_input}\n"
            f"Output: {output}"
        )
    return "".join(lines)

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

def create_proposer_prompt(
    trainer_instance,
    problem_type: str,
    phase: str = "train",
    seed_tasks: Optional[Iterable[Dict[str, Any]]] = None,
    k_reference: int = K_REFERENCE_COUNT,
) -> str:
    """Create a prompt for the proposer model with curriculum-based difficulty."""
    memory_examples = _coerce_memory_examples(seed_tasks=seed_tasks, k_reference=k_reference)
    memory = _format_memory_examples(memory_examples)
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