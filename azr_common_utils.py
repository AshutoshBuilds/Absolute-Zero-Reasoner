import ast
import json
import re
import math
from typing import Any, Tuple, List, Dict, Union
from hf_parsing_utils import (
    parse_generated_tasks as _parse_generated_tasks,
    strip_leading_trailing_code_fences,
)

# --- Input Parsing and Cleaning Utilities ---

def _clean_input_string(input_str: str) -> str:
    """Cleans an input string by removing common problematic characters and attempting to fix escape sequences."""
    if not isinstance(input_str, str):
        return str(input_str) # Should already be a Python object if not a string

    cleaned_str = input_str.strip()

    # Handle common direct escape issues, e.g., "Here\\'s an example" -> "Here's an example"
    # Or "Path is C:\\\\Users" -> "Path is C:\\Users"
    # This is tricky because over-unescaping can break valid paths or literal backslashes.
    # A common pattern is an extra backslash before a quote or another backslash.
    cleaned_str = cleaned_str.replace("\\\\'", "'").replace('\\\\"', '"')
    # Be careful with general backslash replacement, as it might affect valid escape sequences like \\n
    # Try to fix extra escaping for common sequences like \\n -> \\n (no change), \\\\n -> \\n
    cleaned_str = re.sub(r'\\\\(n|t|r|b|f|\\|\\")', r'\\\1', cleaned_str)


    # Try to handle strings that look like they were repr()ed twice or have layers of quotes
    if cleaned_str.startswith("'") and cleaned_str.endswith("'") and len(cleaned_str) > 1:
        inner_content = cleaned_str[1:-1]
        # If the inner content itself looks like a quoted string, unwrap it
        if (inner_content.startswith("'") and inner_content.endswith("'")) or \
           (inner_content.startswith('"') and inner_content.endswith('"')):
            try:
                # Attempt to evaluate the inner content as if it's a string literal
                # This can sometimes fix issues like "'\"hello\"'" -> "hello"
                evaluated_inner = ast.literal_eval(inner_content)
                if isinstance(evaluated_inner, str):
                    cleaned_str = evaluated_inner
            except (SyntaxError, ValueError):
                # If ast.literal_eval fails, it's not a simple string literal, proceed with current cleaned_str
                pass


    # Attempt to fix JSON-like strings with unquoted keys or single quotes
    if cleaned_str.startswith("{") and cleaned_str.endswith("}"):
        try:
            # Replace single quotes with double quotes for JSON compatibility,
            # but be careful not to mess up single quotes within strings.
            # This regex tries to replace single quotes used for dictionary keys/values.
            json_like_str = re.sub(r"(?<!\\)'", '"', cleaned_str)
            # Add quotes to unquoted keys
            json_like_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_like_str)
            # Attempt to parse with json.loads
            json.loads(json_like_str)
            cleaned_str = json_like_str # If it parses, it's likely a valid transformation
        except json.JSONDecodeError:
            pass # Stick with original if it's not easily fixable JSON

    # Final pass with ast.literal_eval if it looks like a Python literal
    # This is a powerful tool but can fail on complex, non-literal strings.
    try:
        # Heuristic: if it starts and ends with typical literal markers and isn't just a plain word
        if (cleaned_str.startswith(("{", "[", "(", "'", '"')) and \
            cleaned_str.endswith(("}", "]", ")", "'", '"'))) or \
           cleaned_str.lower() in ("true", "false", "none") or \
           cleaned_str.replace('.', '', 1).isdigit() or \
           (cleaned_str.startswith('-') and cleaned_str[1:].replace('.', '', 1).isdigit()):
            
            # Attempt to evaluate as a raw string literal to handle escapes properly
            # e.g., if cleaned_str is "{\\"key\\": \\"value\\"}"
            try:
                # First, try directly
                eval_direct = ast.literal_eval(cleaned_str)
                if isinstance(eval_direct, str): # If it became a string, it might need another pass
                     cleaned_str = eval_direct
                # else: it's already a Python object, which _evaluate_input will handle
            except (SyntaxError, ValueError):
                 # If direct fails, try decoding unicode escapes then evaluate
                try:
                    decoded_str = bytes(cleaned_str, "utf-8").decode("unicode_escape")
                    if decoded_str != cleaned_str: # if decoding did something
                        eval_decoded = ast.literal_eval(decoded_str)
                        if isinstance(eval_decoded, str):
                            cleaned_str = eval_decoded
                except (SyntaxError, ValueError, UnicodeDecodeError):
                    pass # Stick with previous cleaning if this also fails
    except Exception:
        pass # If any error occurs, fall back to the string as is.

    return cleaned_str

def _evaluate_input(input_str: str) -> Any:
    """
    Evaluates a string representation of Python input into actual Python objects.
    Handles various formats: single values, lists, tuples, dictionaries, multiple arguments.
    """
    if not isinstance(input_str, str):
        return input_str # Already a Python object

    cleaned_input_str = _clean_input_string(input_str)

    # Attempt 1: Direct ast.literal_eval (safest for simple literals)
    try:
        return ast.literal_eval(cleaned_input_str)
    except (ValueError, SyntaxError, TypeError):
        pass # Try other methods

    # Attempt 2: Handle semicolon-separated multiple arguments for a function
    # e.g., "'Alice'; 10; [1, 2]" -> ("Alice", 10, [1, 2])
    if ';' in cleaned_input_str:
        parts = cleaned_input_str.split(';')
        if len(parts) > 1:
            try:
                evaluated_parts = [_evaluate_input(part.strip()) for part in parts]
                # If all parts evaluated successfully, return as a tuple (representing *args)
                return tuple(evaluated_parts)
            except Exception:
                 # If splitting by semicolon and evaluating parts fails,
                 # it might be a string containing semicolons, not multiple arguments.
                 pass # Fall through to other parsing attempts

    # Attempt 3: Handle inputs that look like function calls or comma-separated values not in a list/tuple
    # e.g., "func(1, 'a')" or "1, 'a'" (intended as *args for a function)
    # Check for comma outside of brackets/parentheses/braces
    # This is a bit heuristic
    # if re.search(r",(?![^(\[{]*[)\]}])", cleaned_input_str): # More robust comma check needed
    # Simplified: if there's a comma and it's not clearly a single list/tuple/dict literal
    if ',' in cleaned_input_str and not (
        (cleaned_input_str.startswith('[') and cleaned_input_str.endswith(']')) or
        (cleaned_input_str.startswith('(') and cleaned_input_str.endswith(')')) or
        (cleaned_input_str.startswith('{') and cleaned_input_str.endswith('}'))
    ):
        # Try to parse as a tuple of arguments
        try:
            # Wrap with parentheses to make it a valid tuple literal for ast.literal_eval
            return ast.literal_eval(f"({cleaned_input_str})")
        except (ValueError, SyntaxError):
            pass


    # Attempt 4: JSON decoding for complex nested structures or if ast.literal_eval is too strict
    try:
        # Ensure double quotes for JSON standard
        json_compatible_str = cleaned_input_str.replace("'", '"')
        # Attempt to fix unquoted keys for JSON (basic cases)
        json_compatible_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_compatible_str)
        return json.loads(json_compatible_str)
    except json.JSONDecodeError:
        pass

    # Attempt 5: Try to parse as a list if it has list-like separators (e.g. items separated by comma, space, or newline)
    # This is more aggressive and should be a later attempt.
    if re.search(r"[,;\n]|\s{2,}", cleaned_input_str.strip()): # Check for common delimiters
        # Split by common delimiters (comma, semicolon, newline, multiple spaces)
        # and filter out empty strings
        potential_items = [item.strip() for item in re.split(r"[,;\n]|\s{2,}", cleaned_input_str) if item.strip()]
        if len(potential_items) > 1:
            try:
                evaluated_items = [_evaluate_input(item) for item in potential_items]
                # If items were individually parsable, return as a list
                # This could turn "1, 2, 3" into [1, 2, 3] or "'a' 'b'" into ['a', 'b']
                return evaluated_items
            except Exception:
                pass # If sub-parsing fails, this wasn't the right structure

    # Final Fallback: return the cleaned string itself if no parsing method worked.
    # This means it will be treated as a single string argument.
    return cleaned_input_str


# --- Code Extraction Utilities ---

def _extract_code_from_solution(solution_text: str, is_proposer: bool = False) -> str:
    """
    Extracts Python code from a given text, typically the output of an LLM.
    Handles markdown code blocks and attempts to find function definitions.
    """
    if not isinstance(solution_text, str):
        return ""

    solution_text = strip_leading_trailing_code_fences(solution_text)

    # 1. Look for Python markdown code blocks
    match_python_block = re.search(r"```python\n(.*?)\n```", solution_text, re.DOTALL)
    if match_python_block:
        return match_python_block.group(1).strip()

    # 2. Look for generic markdown code blocks if Python-specific not found
    match_generic_block = re.search(r"```(.*?)```", solution_text, re.DOTALL)
    if match_generic_block:
        # Check if the content of the generic block looks like Python
        potential_code = match_generic_block.group(1).strip()
        if 'def ' in potential_code or 'import ' in potential_code or 'print(' in potential_code:
            return potential_code
        # If it's from proposer and doesn't look like a full function, it might be just a task description
        # or input/output examples. In that case, we might want to return empty or the raw text.
        # For now, if it's a generic block, assume it's code.
        return potential_code


    # 3. If no markdown, try to find 'def' statements and extract the function
    # This is more brittle and might grab too much or too little.
    # It will try to find the first 'def' and then gather indented lines.
    func_defs = []
    lines = solution_text.splitlines()
    in_function = False
    current_func_lines = []
    base_indent = ""

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("def "):
            if in_function: # Found a new function, save the previous one
                func_defs.append("\\n".join(current_func_lines))
            current_func_lines = [line]
            in_function = True
            # Determine base indentation of this function definition
            match_indent = re.match(r"^(\\s*)def", line)
            base_indent = match_indent.group(1) if match_indent else ""
        elif in_function:
            # Continue if line is empty, a comment, or has greater indent than base_indent.
            # Or if it has the same indent but isn't a new 'def' (e.g. decorators, or subsequent code)
            # Stop if line has less indent than base_indent or is a new 'def' at same/less indent.
            current_line_indent_match = re.match(r"^(\\s*)", line)
            current_line_indent = current_line_indent_match.group(1) if current_line_indent_match else ""

            if not stripped_line: # Blank line
                current_func_lines.append(line)
            elif stripped_line.startswith("#"): # Comment
                current_func_lines.append(line)
            elif len(current_line_indent) > len(base_indent): # Clearly indented part of the function
                current_func_lines.append(line)
            elif len(current_line_indent) == len(base_indent) and not stripped_line.startswith("def "):
                 # Same indent level, but not a new function definition (e.g. code after function, or a method in a class)
                 # This logic is tricky; for now, assume it's part of the same logical block if not a new def.
                 current_func_lines.append(line)
            else: # Dedented or new function definition
                if current_func_lines: # Save the function we were building
                    func_defs.append("\\n".join(current_func_lines))
                in_function = False
                current_func_lines = []
                base_indent = ""
                # Re-evaluate current line if it's a new def
                if stripped_line.startswith("def "):
                    current_func_lines = [line]
                    in_function = True
                    match_indent = re.match(r"^(\\s*)def", line)
                    base_indent = match_indent.group(1) if match_indent else ""


    if in_function and current_func_lines: # Add the last function
        func_defs.append("\\n".join(current_func_lines))

    if func_defs:
        return "\\n\\n".join(func_defs).strip() # Join all found function definitions

    # 4. Fallback: if it's not from the proposer (i.e., it's a solver's attempt)
    # and we haven't found code yet, return the whole text as a last resort.
    # Proposer outputs are more structured (task, input, output), so less likely to be raw code.
    if not is_proposer and not func_defs:
        return solution_text.strip()

    return "" # Default to empty string if no code is found


# --- Output Comparison Utilities ---
def _compare_outputs(actual_output: Any, expected_output_str: str, problem_type: str = "unknown") -> Tuple[bool, float, str]:
    """
    Compares the actual output from code execution with the expected output string.
    Returns: (is_match, score, reason_code)
    Score is between 0.0 and 1.0.
    Reason codes:
        "EXACT_MATCH", "NORMALIZED_STRING_MATCH", "NUMERIC_MATCH", "BOOL_MATCH",
        "LIST_EXACT_MATCH", "LIST_NORMALIZED_MATCH", "LIST_ORDER_AGNOSTIC_MATCH",
        "LIST_TYPE_MATCH", "PARTIAL_LIST_MATCH",
        "DICT_EXACT_MATCH", "DICT_NORMALIZED_MATCH", "PARTIAL_DICT_MATCH",
        "TYPE_MISMATCH", "VALUE_MISMATCH", "FORMAT_ERROR_EXPECTED_EVAL", "NO_MATCH"
    """
    expected_output: Any
    try:
        # Try to evaluate the expected output string to a Python object for richer comparison
        expected_output = _evaluate_input(expected_output_str)
    except (SyntaxError, ValueError):
        # If expected_output_str cannot be evaluated, treat it as a literal string.
        expected_output = expected_output_str
        # We might also want to flag that evaluation failed if strictness is required.
        # For now, we proceed with string comparison.

    actual_output_str = str(actual_output)

    # 1. Direct exact match (covers non-string types like numbers, booleans if evaluated correctly)
    if actual_output == expected_output:
        return True, 1.0, "EXACT_MATCH"

    # 2. String representation match (if direct object match failed or one is string)
    if actual_output_str == expected_output_str:
        return True, 1.0, "EXACT_STRING_REPR_MATCH"

    # 3. Normalized string comparison (case, whitespace, quotes)
    # Only do this if both are strings or can be treated as strings meaningfully
    norm_actual = actual_output_str.strip().lower()
    norm_expected = expected_output_str.strip().lower()

    if norm_actual.strip("\'\"`") == norm_expected.strip("\'\"`"): # Strip quotes as well
        return True, 0.95, "NORMALIZED_STRING_MATCH" # High score for minor formatting

    # 4. Numeric comparison with tolerance (if both seem numeric)
    try:
        # Check if both can be converted to float
        num_actual = float(actual_output) # actual_output might already be a number
        num_expected = float(expected_output) # expected_output might already be a number
        if math.isclose(num_actual, num_expected, rel_tol=1e-6, abs_tol=1e-9):
            return True, 0.98, "NUMERIC_MATCH"
    except (ValueError, TypeError):
        pass # Not both numeric, or conversion failed

    # 5. Boolean semantic matching (e.g., "true" vs True, "1" vs True)
    # This can be tricky; keep it simple for now.
    actual_bool, expected_bool = None, None
    # Convert actual_output to boolean if possible
    if isinstance(actual_output, bool): actual_bool = actual_output
    elif actual_output_str.lower() in ["true", "1", "yes"]: actual_bool = True
    elif actual_output_str.lower() in ["false", "0", "no"]: actual_bool = False
    # Convert expected_output to boolean if possible
    if isinstance(expected_output, bool): expected_bool = expected_output
    elif expected_output_str.lower() in ["true", "1", "yes"]: expected_bool = True
    elif expected_output_str.lower() in ["false", "0", "no"]: expected_bool = False

    if actual_bool is not None and expected_bool is not None and actual_bool == expected_bool:
        return True, 0.9, "BOOL_MATCH"


    # 6. List comparison (if both are lists)
    if isinstance(actual_output, list) and isinstance(expected_output, list):
        if len(actual_output) == len(expected_output):
            if actual_output == expected_output: # Handles list of complex objects
                return True, 1.0, "LIST_EXACT_MATCH"

            # Element-wise normalized string comparison if elements are simple
            try:
                match_count = 0
                all_simple = True
                for act, exp in zip(actual_output, expected_output):
                    if not (isinstance(act, (str, int, float, bool)) and isinstance(exp, (str, int, float, bool))):
                        all_simple = False
                        break
                    norm_act_elem = str(act).strip().lower().strip("\"'")
                    norm_exp_elem = str(exp).strip().lower().strip("\"'")
                    if norm_act_elem == norm_exp_elem:
                        match_count += 1
                if all_simple and match_count == len(actual_output):
                    return True, 0.9, "LIST_NORMALIZED_MATCH"
                elif all_simple and match_count > 0: # Partial match
                     return False, 0.3 * (match_count / len(actual_output)), "PARTIAL_LIST_MATCH_NORMALIZED_ELEMENTS"

            except TypeError: # Comparison between types failed
                pass

            # Order-agnostic comparison for lists of simple types (numbers, strings, bools)
            try:
                # Ensure all elements are hashable for set comparison, or sortable for sorted list comparison
                # This is a simplification; true order-agnostic requires more complex matching for unhashable types.
                if all(isinstance(x, (int, float, str, bool, tuple)) for x in actual_output) and \
                   all(isinstance(x, (int, float, str, bool, tuple)) for x in expected_output):
                    if sorted([str(x).strip().lower().strip("\"'") for x in actual_output]) == sorted([str(x).strip().lower().strip("\"'") for x in expected_output]):
                        return True, 0.85, "LIST_ORDER_AGNOSTIC_NORMALIZED_MATCH"
            except TypeError: # Not sortable (e.g., list of lists)
                pass
        else: # Length mismatch
            # Check for sublist or superlist for partial credit
            # This is simplistic, could be improved with sequence matching
            try:
                # Convert to comparable strings for sublist check
                str_actual_set = {str(x).strip().lower().strip("\"'") for x in actual_output}
                str_expected_set = {str(x).strip().lower().strip("\"'") for x in expected_output}
                intersection_len = len(str_actual_set.intersection(str_expected_set))
                union_len = len(str_actual_set.union(str_expected_set))
                if union_len > 0:
                    return False, 0.4 * (intersection_len / union_len), "PARTIAL_LIST_INTERSECTION_MATCH"
            except TypeError:
                pass


    # 7. Dictionary comparison (if both are dicts)
    if isinstance(actual_output, dict) and isinstance(expected_output, dict):
        if actual_output == expected_output: # Handles nested dicts
            return True, 1.0, "DICT_EXACT_MATCH"

        # Key-value normalized string comparison (for dicts with simple string keys/values)
        # This is a simplification. A full comparison would involve deep recursive checks.
        try:
            if sorted(actual_output.keys()) == sorted(expected_output.keys()):
                matches = 0
                all_simple_kv = True
                for k in actual_output:
                    act_v, exp_v = actual_output[k], expected_output[k]
                    if not (isinstance(act_v, (str, int, float, bool)) and isinstance(exp_v, (str, int, float, bool))):
                        all_simple_kv = False; break
                    if str(act_v).strip().lower().strip("\"'") == str(exp_v).strip().lower().strip("\"'"):
                        matches +=1
                if all_simple_kv and matches == len(actual_output):
                    return True, 0.9, "DICT_NORMALIZED_VALUE_MATCH"
                elif all_simple_kv and matches > 0:
                    return False, 0.3 * (matches / len(actual_output)), "PARTIAL_DICT_MATCH_NORMALIZED_VALUES"
        except (TypeError, KeyError):
            pass


    # Fallback: type mismatch or value mismatch
    if type(actual_output) != type(expected_output) and expected_output_str != str(actual_output):
        # Consider if expected_output_str *could* have been evaluated. If not, it was always a string.
        try:
            # Check if it was evaluatable. If expected_output is already a string due to eval failure, this won't re-raise.
            # The goal is to know if expected_output_str *could* have represented a non-string type.
            if not isinstance(expected_output, str):
                 # This means expected_output was successfully evaluated to a non-string type
                 # and it mismatched with actual_output type.
                 return False, 0.0, f"TYPE_MISMATCH (Actual: {type(actual_output).__name__}, Expected Type from Eval: {type(expected_output).__name__})"
            else: # expected_output is a string (either originally or from failed eval of expected_output_str)
                if isinstance(ast.literal_eval(expected_output_str), str):
                    # This means expected_output_str evaluates to a string.
                    # If actual_output is not a string, then it is a type mismatch.
                    if not isinstance(actual_output, str):
                        return False, 0.0, f"TYPE_MISMATCH (Actual: {type(actual_output).__name__}, Expected: str)"
                    # Otherwise, both are strings and didn't match earlier, so it's a value mismatch.
                    return False, 0.0, "VALUE_MISMATCH (Strings, after normalization)"
                else:
                    # This means expected_output_str evaluates to a non-string.
                    # Since actual_output type didn't match this evaluated non-string type (checked by `actual_output == expected_output` earlier)
                    # and actual_output also didn't match its string representation, it is a type mismatch.
                    return False, 0.0, f"TYPE_MISMATCH (Actual: {type(actual_output).__name__}, Expected Type from Eval: {type(ast.literal_eval(expected_output_str)).__name__})"

        except (SyntaxError, ValueError):
            # Expected output_str was always a string (could not be evaluated to anything else) and didn't match.
            if not isinstance(actual_output, str):
                 return False, 0.0, f"TYPE_MISMATCH (Actual: {type(actual_output).__name__}, Expected: str)"
            # If both are strings but didn't match earlier, it's a value mismatch.
            return False, 0.0, "VALUE_MISMATCH (Strings, after normalization)"

    return False, 0.0, "NO_MATCH"


# --- Code Analysis Utilities ---

def _calculate_ast_complexity(code_string: str) -> int:
    """Calculates a simple complexity score based on AST node counts."""
    try:
        tree = ast.parse(code_string)
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
                                 ast.Try, ast.ExceptHandler, ast.Call, ast.BinOp, ast.BoolOp,
                                 ast.Compare, ast.comprehension, ast.Lambda)):
                complexity += 1
            if isinstance(node, ast.Attribute): # Accessing attributes can indicate complexity
                complexity += 0.5
            if isinstance(node, ast.Subscript): # Indexing/slicing
                complexity += 0.5

        # Bonus for docstrings and type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if ast.get_docstring(node):
                    complexity += 2 # Bonus for docstring
                if node.returns: # Return type hint
                    complexity +=1
                for arg_node in node.args.args:
                    if arg_node.annotation: # Argument type hint
                        complexity +=1
        return int(complexity)
    except SyntaxError:
        return 0 # Or handle error appropriately, e.g., by returning a large number or None
    except Exception:
        return 0 # Catch any other AST processing errors


def _check_code_safety(code_string: str, banned_keywords: List[str] = None, banned_modules: List[str] = None) -> Tuple[bool, str]:
    """
    Performs basic safety checks on Python code using AST.
    Checks for banned keywords (e.g., 'eval', 'exec' if not intended) and module imports.
    Returns (is_safe, reason_message).
    """
    if banned_keywords is None:
        banned_keywords = ["eval", "exec", "open", "input", "compile"] # Example defaults
    if banned_modules is None:
        banned_modules = ["os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pickle", "ctypes", "multiprocessing", "threading"] # Example defaults

    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            # Check for banned keywords used as function calls or builtins
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_keywords:
                return False, f"Banned keyword used as function: {node.func.id}"
            if isinstance(node, ast.Name) and node.id in banned_keywords:
                 # This might be too broad, e.g. variable named 'open'.
                 # Context matters, but for a simple check, flag it.
                 # Consider checking if it's in a Call context more specifically.
                 pass # Let's rely on the Call check for now.

            # Check for imports of banned modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in banned_modules:
                        return False, f"Banned module imported: {alias.name}"
                    # Check for submodules too, e.g., os.path
                    for banned_mod in banned_modules:
                        if alias.name.startswith(banned_mod + "."):
                             return False, f"Banned submodule imported: {alias.name}"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module in banned_modules:
                    return False, f"Banned module in from-import: {node.module}"
                # Check for submodules in from-import
                if node.module:
                    for banned_mod in banned_modules:
                        if node.module.startswith(banned_mod + "."):
                             return False, f"Banned submodule in from-import: {node.module}"
                # Check imported names from a banned module, e.g., from os import path
                # This is tricky; if 'os' is banned, 'from os import path' should also be.
                # The above check handles 'node.module in banned_modules'.

            # Check for direct file operations (simplistic check by name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                return False, "Direct file open() call detected."
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                # e.g. io.open() or _io.open()
                # We'd need to know what 'node.func.value' (the object) is.
                # For now, this is a basic check.
                 pass # Can be noisy.

        return True, "Code passed basic safety checks."
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Unexpected error during safety check: {e}"

def contains_banned_imports(code: str, banned_list: List[str] = None) -> Tuple[bool, str]:
    """
    Checks if the given Python code string contains any banned imports using AST parsing.
    This is a more focused version, similar to one in the original AZR codebase.
    """
    if banned_list is None:
        banned_list = [
            "antigravity", "collections.abc", "ctypes", "curses", "distutils", "ensurepip",
            "filecmp", "fileinput", "ftplib", "html", "http", "idlelib", "imaplib",
            "importlib", "inspect", "ipaddress", "json.tool", "lib2to3", "logging.config",
            "logging.handlers", "mailbox", "modulefinder", "multiprocessing", "netrc",
            "nntplib", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools", "pip",
            "platform", "poplib", "pty", "pydoc", "pydoc_data", "pyexpat", "queue",
            "runpy", "sched", "shelve", "shlex", "shutil", "signal", "smtpd", "smtplib",
            "sndhdr", "socket", "socketserver", "sqlite3", "ssl", "stat", "subprocess",
            "sunau", "symbol", "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
            "tempfile", "test", "textwrap", "threading", "tkinter", "token", "tokenize",
            "trace", "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
            "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
            "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
            "zipimport", "zlib", "os", "path", "io", "builtins.open", "builtins.eval", "builtins.exec"
        ] # A more comprehensive list

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    for banned_name in banned_list:
                        # Direct match or submodule of a banned item.
                        if n.name == banned_name or n.name.startswith(banned_name + "."):
                            return True, f"Banned import: {n.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module: # `from . import something` node.module is None
                    for banned_name in banned_list:
                        if node.module == banned_name or node.module.startswith(banned_name + "."):
                            return True, f"Banned import from: {node.module}"
            elif isinstance(node, ast.Call): # Check for builtins like eval, exec, open
                 if isinstance(node.func, ast.Name):
                     func_name = node.func.id
                     if func_name in banned_list or f"builtins.{func_name}" in banned_list:
                         return True, f"Banned function call: {func_name}"
                 # Could also check for ast.Attribute for things like 'os.system' but that's more involved
                 # if the goal is just to check top-level imports and direct calls of certain builtins.

        return False, "No banned imports found."
    except SyntaxError:
        return True, "Syntax error in code, cannot check imports." # Treat syntax error as unsafe
    except Exception:
        return True, "Unexpected error during import check." # Treat other errors as unsafe 


# Lightweight compatibility shims for legacy imports expected by older callers.
class TaskBuffer:
    """
    Minimal compatibility stand-in for the legacy TaskBuffer class.
    It preserves the public constructor and basic queue-like operations.
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._items: list = []

    def append(self, item: Any) -> None:
        if len(self._items) >= self.max_size:
            self._items.pop(0)
        self._items.append(item)

    def pop(self, index: int = -1) -> Any:
        return self._items.pop(index)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


def parse_generated_tasks(
    trainer_instance_or_text,
    generated_text: str = None,
    task_idx: int = 0
) -> list:
    """
    Backward compatible parser wrapper.

    Supports both legacy signatures:
      parse_generated_tasks(generated_text)
      parse_generated_tasks(trainer_instance, generated_text, task_idx)
    """
    if generated_text is None:
        return _parse_generated_tasks(None, trainer_instance_or_text, task_idx)
    return _parse_generated_tasks(trainer_instance_or_text, generated_text, task_idx)