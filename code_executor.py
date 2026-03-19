import io
import sys
import time
import traceback
import multiprocessing
import ast # For parsing the code to find imports
import re # Added for finding function name
import argparse
import os
import math
from typing import Dict, Any

# Import AZR common utils for input evaluation and output comparison
try:
    from azr_common_utils import _evaluate_input, _compare_outputs
except ImportError:
    # Fallback if running standalone or utils not found
    def _evaluate_input(x): return x
    def _compare_outputs(a, b): return str(a).strip() == str(b).strip(), 1.0 if str(a).strip() == str(b).strip() else 0.0, "FALLBACK"

# As per Figure 8 in the AZR paper
FORBIDDEN_MODULES = set([
    "os", # Includes os.sys implicitly by checking for 'os'
    "sys", # Directly forbidden
    "shutil",
    "subprocess",
    "multiprocessing", # To prevent the executed code from spawning more processes
    "threading", # Can be used for DoS, though GIL limits true parallelism for CPU-bound tasks
    "socket",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "ctypes",
    "_thread",
    # Potentially dangerous built-ins if not careful, though harder to restrict
    # exec, eval, open, compile, __import__
    # For now, focusing on module imports as per the paper's direct list.
])

try:
    from rich.console import Console
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    _RICH_AVAILABLE = False


def _resolve_output_printer(use_rich: bool) -> tuple[Any, Any]:
    """Return a tuple of (console, print_fn) for consistent CLI output."""
    if use_rich and _RICH_AVAILABLE and Console is not None:
        console = Console(highlight=False, force_terminal=True)
        return console, console.print
    return None, print


def _apply_cpu_cap(cpu_cap_percent: float) -> int:
    """Apply a soft CPU cap using thread pool environment limits."""
    cpu_count = os.cpu_count() or 1
    cap = max(1.0, min(100.0, float(cpu_cap_percent)))
    max_threads = max(1, math.floor(cpu_count * cap / 100.0))
    max_threads = min(cpu_count, max_threads)

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(max_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(max_threads)))
    except Exception:
        pass

    return max_threads


def _execution_worker(code_string: str, input_args: tuple, function_name: str, result_queue: multiprocessing.Queue):
    """
    Worker function to execute code in a separate process.
    Captures stdout, return value, and errors.
    """
    output_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_stream

    return_value = None
    error_message = None
    local_scope = {}

    try:
        exec(code_string, globals(), local_scope)
        target_function = local_scope.get(function_name)
        if callable(target_function):
            return_value = target_function(*input_args)
        else:
            raise ValueError(f"Function '{function_name}' not found or not callable in the provided code.")
    except Exception:
        error_message = traceback.format_exc()
        return_value = None
    finally:
        sys.stdout = original_stdout
        captured_output = output_stream.getvalue()
        output_stream.close()
        result_queue.put({
            'output': captured_output,
            'return_value': return_value,
            'error': error_message,
        })

def _script_execution_worker(code_string: str, result_queue: multiprocessing.Queue):
    """
    Worker that executes arbitrary code (including inline tests) in a separate process.
    It does not attempt to call any function; success is defined as no exception raised.
    """
    output_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_stream

    error_message = None
    try:
        exec(code_string, {})
    except Exception:
        error_message = traceback.format_exc()
    finally:
        sys.stdout = original_stdout
        captured_output = output_stream.getvalue()
        output_stream.close()
        result_queue.put({
            'output': captured_output,
            'return_value': None,
            'error': error_message,
        })

class CodeExecutor:
    """
    A class to securely execute Python code snippets and capture their output,
    errors, and execution time, with timeout and basic import restrictions.
    """

    def __init__(self, timeout_seconds=5):
        """
        Initializes the CodeExecutor.
        Args:
            timeout_seconds (int): Maximum execution time for a code snippet.
        """
        self.timeout_seconds = timeout_seconds

    def _check_forbidden_imports(self, code_string: str) -> list[str]:
        """
        Parses the code_string to find import statements and checks against FORBIDDEN_MODULES.
        Returns a list of forbidden modules found, or an empty list if none are found.
        """
        found_forbidden = []
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                            found_forbidden.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                        found_forbidden.append(node.module.split('.')[0])
        except SyntaxError:
            # If there's a syntax error, the code won't run anyway.
            # The main execution try-except will catch this. For import checking,
            # we can consider it as no forbidden imports *statically found* if unparsable.
            pass 
        return list(set(found_forbidden)) # Return unique forbidden modules

    def execute_code(self, code_string: str, input_args: tuple = (), function_name: str = 'f'):
        """
        Executes a string of Python code in a separate process with a timeout 
        and checks for forbidden imports.

        Args:
            code_string (str): The Python code to execute. It should define a function.
            input_args (tuple): A tuple of arguments to pass to the target function.
            function_name (str): The name of the function to call within the code_string.

        Returns:
            dict: A dictionary containing:
                'output': The stdout of the executed code (or None if timeout/pre-check fail).
                'return_value': The value returned by the target function (or None).
                'error': Any error message if an exception/timeout occurred, None otherwise.
                'execution_time': Time taken for execution in seconds (or for pre-checks).
        """
        start_time = time.time()

        # 1. Check for forbidden imports
        forbidden_imports_found = self._check_forbidden_imports(code_string)
        if forbidden_imports_found:
            execution_time = time.time() - start_time
            return {
                'output': None,
                'return_value': None,
                'error': f"Execution blocked due to forbidden imports: {', '.join(forbidden_imports_found)}",
                'execution_time': execution_time,
            }

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execution_worker,
            args=(code_string, input_args, function_name, result_queue)
        )

        process.start()
        process.join(timeout=self.timeout_seconds)

        execution_time = time.time() - start_time
        
        if process.is_alive():
            process.terminate() # Terminate the process if it's still running (timed out)
            process.join() # Wait for termination to complete
            return {
                'output': None,
                'return_value': None,
                'error': f"Execution timed out after {self.timeout_seconds} seconds.",
                'execution_time': execution_time,
            }
        else:
            if process.exitcode != 0:
                # Process terminated with an error code, but might not have put error in queue
                # This could happen if the error was so severe it killed the process before queue.put
                # or if an uncatchable signal was received. For now, we rely on the queue.
                pass # The error should ideally be in the queue.

            try:
                # Retrieve results from the queue
                result = result_queue.get_nowait() # Use get_nowait as process should be dead
                result['execution_time'] = execution_time
                return result
            except multiprocessing.queues.Empty:
                 # This case should ideally be rare if the process terminated cleanly or with an error it reported.
                 # If the process died without putting to queue and didn't timeout, it's an unexpected crash.
                return {
                    'output': None,
                    'return_value': None,
                    'error': "Execution process terminated unexpectedly without providing results.",
                    'execution_time': execution_time,
                }
    
    # Alias for compatibility with AZR's PythonExecutor
    run_code = execute_code
    
    def execute_script(self, code_string: str, timeout_seconds: int | None = None) -> dict:
        """
        Execute arbitrary code (which may contain inline asserts/tests) in a sandboxed subprocess.
        Returns a dict with keys: output, return_value(None), error, execution_time, success.
        """
        start_time = time.time()

        # Check for forbidden imports before running
        forbidden_imports_found = self._check_forbidden_imports(code_string)
        if forbidden_imports_found:
            execution_time = time.time() - start_time
            return {
                'output': None,
                'return_value': None,
                'error': f"Execution blocked due to forbidden imports: {', '.join(forbidden_imports_found)}",
                'execution_time': execution_time,
                'success': False,
            }

        effective_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_script_execution_worker,
            args=(code_string, result_queue)
        )
        process.start()
        process.join(timeout=effective_timeout)

        execution_time = time.time() - start_time

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                'output': None,
                'return_value': None,
                'error': f"Execution timed out after {effective_timeout} seconds.",
                'execution_time': execution_time,
                'success': False,
            }

        try:
            result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            return {
                'output': None,
                'return_value': None,
                'error': "Execution process terminated unexpectedly without providing results.",
                'execution_time': execution_time,
                'success': False,
            }

       	# Normalize success flag
        result['execution_time'] = execution_time
        result['success'] = result.get('error') in (None, '')
        return result

    def execute(self, code: str, test_input: str = "", timeout: int | None = None) -> dict:
        """
        Compatibility wrapper expected by evaluate_benchmarks.py.
        Ignores test_input; runs code as a script and returns a dict with 'success' key among others.
        """
        return self.execute_script(code_string=code, timeout_seconds=timeout)
    
    def solution_check(self, solution_code: str, input_str: str, expected_output_str: str) -> Dict[str, Any]:
        """
        Checks if the solution code produces the expected output for the given input.
        Uses AZR common utils for robust input parsing and output comparison.
        Returns a dict compatible with HFRewardManager expectations.
        """
        # 1. Parse input string into Python objects
        try:
            input_val = _evaluate_input(input_str)
        except Exception as e:
            return {
                'valid': False,
                'output': None,
                'error': f"Input parsing error: {e}",
                'similarity': 0.0,
                'reason': "INPUT_PARSE_ERROR"
            }

        # 2. Prepare arguments tuple
        input_args = ()
        if isinstance(input_val, tuple):
            input_args = input_val
        else:
            input_args = (input_val,)

        # 3. Detect function name (default to 'solution' or first found function, fallback to 'f')
        # Simple regex to find the first function definition
        match = re.search(r"def\s+(\w+)\s*\(", solution_code)
        function_name = match.group(1) if match else 'f'

        # 4. Execute code
        exec_result = self.execute_code(solution_code, input_args=input_args, function_name=function_name)

        result = {
            'valid': False,
            'output': exec_result.get('return_value'), # Use return value for comparison
            'error': exec_result.get('error'),
            'similarity': 0.0,
            'reason': 'INIT'
        }

        # 5. Handle Execution Errors
        if result['error']:
             # Check if it was a "function not found" error, maybe try 'solution' or 'solve' if 'f' failed
             if "not found or not callable" in result['error'] and function_name == 'f':
                 # Try 'solution' as fallback
                 exec_result_retry = self.execute_code(solution_code, input_args=input_args, function_name='solution')
                 if not exec_result_retry.get('error'):
                     exec_result = exec_result_retry
                     result['output'] = exec_result.get('return_value')
                     result['error'] = None
                 else:
                     result['reason'] = "EXECUTION_ERROR"
                     return result
             else:
                 result['reason'] = "EXECUTION_ERROR"
                 return result

        # 6. Compare Outputs
        actual_output = exec_result.get('return_value')
        
        try:
            is_match, score, reason = _compare_outputs(actual_output, expected_output_str)
            result['valid'] = is_match
            result['similarity'] = score
            result['reason'] = reason
        except Exception as e:
             result['valid'] = False
             result['error'] = f"Comparison error: {e}"
             result['reason'] = "COMPARISON_ERROR"

        return result

    def check_all(self, code, input_str="", output_str="", check_banned=True, check_syntax=True, 
                  check_determinism=False, banned_keywords=None):
        """Check if the code is valid according to various criteria.
        
        Args:
            code (str): The Python code to check
            input_str (str): Input string to test the code with
            output_str (str): Expected output string
            check_banned (bool): Whether to check for banned imports/keywords
            check_syntax (bool): Whether to check for syntax errors
            check_determinism (bool): Whether to check if the code is deterministic
            banned_keywords (list): Additional banned keywords to check for
            
        Returns:
            dict: Result dictionary with 'valid' field and others
        """
        result = {
            'valid': True,
            'error': None,
            'output': None
        }
        
        # Check syntax
        if check_syntax:
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                result['valid'] = False
                result['error'] = f"SyntaxError: {str(e)}"
                return result
                
        # Check for banned imports/keywords
        if check_banned:
            forbidden_imports = self._check_forbidden_imports(code)
            if forbidden_imports:
                result['valid'] = False
                result['error'] = f"Banned imports found: {', '.join(forbidden_imports)}"
                return result
                
            # Check for additional banned keywords
            if banned_keywords:
                for keyword in banned_keywords:
                    if keyword in code:
                        result['valid'] = False
                        result['error'] = f"Banned keyword found: {keyword}"
                        return result
        
        # Execute code to check for runtime errors and get output
        exec_result = self.execute_code(code, input_args=() if not input_str else (input_str,))
        
        if exec_result.get('error'):
            result['valid'] = False
            result['error'] = exec_result['error']
            return result
            
        result['output'] = exec_result.get('return_value')
        
        # Check if output matches expected output if provided
        if output_str and result['output'] is not None:
            # Try to normalize and compare outputs
            try:
                # Handle case where output_str is a string representation of a value
                expected = eval(output_str)
            except:
                expected = output_str
                
            if str(result['output']).strip() != str(expected).strip():
                result['valid'] = False
                result['error'] = f"Output mismatch: expected '{expected}', got '{result['output']}'"
                return result
        
        # Check determinism if requested
        if check_determinism:
            # Run code multiple times to check for determinism
            outputs = []
            for _ in range(3):
                exec_result = self.execute_code(code, input_args=() if not input_str else (input_str,))
                if exec_result.get('error'):
                    result['valid'] = False
                    result['error'] = f"Determinism check failed: {exec_result['error']}"
                    return result
                outputs.append(str(exec_result.get('return_value')))
            
            # Check if all outputs are the same
            if len(set(outputs)) > 1:
                result['valid'] = False
                result['error'] = f"Code is not deterministic: got different outputs on multiple runs"
                return result
        
        return result
    
    def eval_output_prediction(self, code, input_str, output_prediction):
        """Evaluate if the predicted output matches the actual output of running the code.
        
        Args:
            code (str): The code to execute
            input_str (str): The input to the code
            output_prediction (str): The predicted output
            
        Returns:
            dict: Result dictionary with 'value_match' field indicating if the prediction was correct
        """
        result = {
            'value_match': False,
            'error': None
        }
        
        # Run the code with the input
        exec_result = self.execute_code(code, input_args=() if not input_str else (input_str,))
        
        if exec_result.get('error'):
            result['error'] = exec_result['error']
            return result
        
        actual_output = exec_result.get('return_value')
        
        # Try to normalize and compare outputs
        try:
            # Handle case where output_prediction is a string representation of a value
            predicted = eval(output_prediction)
        except:
            predicted = output_prediction
            
        # Compare as strings with whitespace normalization
        actual_str = str(actual_output).strip()
        predicted_str = str(predicted).strip()
        
        result['value_match'] = actual_str == predicted_str
        
        return result
    
    def eval_input_prediction(self, code, input_prediction, output_str):
        """Evaluate if the predicted input produces the expected output when run with the code.
        
        Args:
            code (str): The code to execute
            input_prediction (str): The predicted input
            output_str (str): The expected output
            
        Returns:
            dict: Result dictionary with 'value_match' field indicating if the prediction was correct
        """
        result = {
            'value_match': False,
            'error': None
        }
        
        # Run the code with the predicted input
        exec_result = self.execute_code(code, input_args=() if not input_prediction else (input_prediction,))
        
        if exec_result.get('error'):
            result['error'] = exec_result['error']
            return result
        
        actual_output = exec_result.get('return_value')
        
        # Try to normalize and compare outputs
        try:
            # Handle case where output_str is a string representation of a value
            expected = eval(output_str)
        except:
            expected = output_str
            
        # Compare as strings with whitespace normalization
        actual_str = str(actual_output).strip()
        expected_str = str(expected).strip()
        
        result['value_match'] = actual_str == expected_str
        
        return result

# Example Usage (for testing purposes):
def run_code_executor_demo(use_rich: bool = True, timeout_seconds: float = 2.0) -> None:
    """Run executable examples for the sandboxed code executor."""
    _, printer = _resolve_output_printer(use_rich)
    executor = CodeExecutor(timeout_seconds=timeout_seconds)
    printer("[bold cyan]CodeExecutor Demo[/bold cyan]")

    code1 = """
def f(a, b):
    print(f'Input args: a={a}, b={b}')
    return a + b
"""
    printer("\n--- Example 1: Simple Addition ---")
    result1 = executor.execute_code(code1, input_args=(5, 10))
    printer(f"Stdout: {result1['output'].strip() if result1['output'] else 'N/A'}")
    printer(f"Return Value: {result1['return_value']}")
    printer(f"Error: {result1['error']}")
    printer(f"Execution Time: {result1['execution_time']:.4f}s")

    code2 = """
def f(x):
    return x / 0
"""
    printer("\n--- Example 2: Division by Zero ---")
    result2 = executor.execute_code(code2, input_args=(5,))
    printer(f"Stdout: {result2['output'].strip() if result2['output'] else 'N/A'}")
    printer(f"Return Value: {result2['return_value']}")
    printer(f"Error:\n{result2['error']}")
    printer(f"Execution Time: {result2['execution_time']:.4f}s")

    code3 = """
def f():
    print("Hello from executed code!")
    print("Another line.")
    return 42
"""
    printer("\n--- Example 3: Stdout Capture ---")
    result3 = executor.execute_code(code3, input_args=())
    printer(f"Stdout:\n{result3['output'].strip() if result3['output'] else 'N/A'}")
    printer(f"Return Value: {result3['return_value']}")
    printer(f"Error: {result3['error']}")
    printer(f"Execution Time: {result3['execution_time']:.4f}s")

    code4 = """
import time
def f():
    print("Starting long task...")
    time.sleep(5) # This should exceed the 2-second timeout
    print("Finished long task.")
    return "Done"
"""
    printer("\n--- Example 4: Timeout Test ---")
    result4 = executor.execute_code(code4)
    printer(f"Stdout: {result4['output']}") # Expected to be None or empty
    printer(f"Return Value: {result4['return_value']}") # Expected to be None
    printer(f"Error: {result4['error']}")
    printer(f"Execution Time: {result4['execution_time']:.4f}s")

    code5 = """
import os
def f():
    return os.getcwd()
"""
    printer("\n--- Example 5: Forbidden Import (os) ---")
    result5 = executor.execute_code(code5)
    printer(f"Stdout: {result5['output']}")
    printer(f"Return Value: {result5['return_value']}")
    printer(f"Error: {result5['error']}")
    printer(f"Execution Time: {result5['execution_time']:.4f}s")

    code6 = """
import sys
def f():
    return sys.version
"""
    printer("\n--- Example 6: Forbidden Import (sys) ---")
    result6 = executor.execute_code(code6)
    printer(f"Stdout: {result6['output']}")
    printer(f"Return Value: {result6['return_value']}")
    printer(f"Error: {result6['error']}")
    printer(f"Execution Time: {result6['execution_time']:.4f}s")

    code7 = """
import parent.os_wrapper # Assuming os_wrapper imports os
def f():
    return "nested attempt"
"""
    # This static check will only find 'parent.os_wrapper'. 
    # If 'parent.os_wrapper' is not in FORBIDDEN_MODULES, it would pass this static check.
    # A dynamic check during exec (if possible in a sandbox) would be needed for deeper inspection.
    # Our current check is for top-level names in FORBIDDEN_MODULES.
    printer("\n--- Example 7: Potentially Deeper Forbidden Import (Static Check Limitations) ---")
    printer("Note: This example highlights static analysis limitations for deeply nested forbidden imports.")
    printer("The current check looks for top-level module names in the forbidden list.")
    # To make this fail our current check, FORBIDDEN_MODULES would need 'parent'
    # or we modify the check to be more sophisticated (e.g. checking all parts of 'alias.name')
    # For simplicity, we assume the AZR paper implies checking for direct imports of listed modules.
    result7 = executor.execute_code(code7)
    printer(f"Stdout: {result7['output']}")
    printer(f"Return Value: {result7['return_value']}")
    printer(f"Error: {result7['error']}")
    printer(f"Execution Time: {result7['execution_time']:.4f}s")

    code8 = """
def f(x)
    return x*2 # Syntax error: missing colon
"""
    printer("\n--- Example 8: Syntax Error (after import check) ---")
    result8 = executor.execute_code(code8, input_args=(7,))
    printer(f"Stdout: {result8['output'].strip() if result8['output'] else 'N/A'}")
    printer(f"Return Value: {result8['return_value']}")
    printer(f"Error:\n{result8['error']}")
    printer(f"Execution Time: {result8['execution_time']:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CodeExecutor examples and output execution diagnostics.")
    parser.add_argument(
        "--rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rich terminal rendering",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Timeout for demo executions in seconds.",
    )
    parser.add_argument(
        "--cpu-cap",
        type=float,
        default=20.0,
        help="CPU cap percentage (0-100) for this process",
    )
    args = parser.parse_args()
    max_threads = _apply_cpu_cap(args.cpu_cap)
    print(f"CPU cap set to {args.cpu_cap:.1f}% (max threads={max_threads})")

    # Important for Windows and macOS with 'spawn' start method
    multiprocessing.freeze_support()
    run_code_executor_demo(use_rich=args.rich, timeout_seconds=args.timeout)