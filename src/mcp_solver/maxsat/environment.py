"""
MaxSAT secure execution environment.

This module provides a secure environment for executing MaxSAT code,
with timeout handling and output capturing. It builds on the PySAT
environment module but adapts it for MaxSAT optimization problems.
"""

import collections
import io
import itertools
import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from multiprocessing import Process, Queue
from typing import Any


# Import path management to ensure we get the correct PySAT
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir in sys.path:
    sys.path.remove(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

# Add site-packages to the front of the path
import site


site_packages = site.getsitepackages()
for p in reversed(site_packages):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import PySAT but protect against failure
try:
    import pysat
    from pysat.card import CardEnc, EncType
    from pysat.examples.rc2 import RC2
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Cadical153, Glucose3, Glucose4, Lingeling, Solver

    # Import our local solution module
    from . import solution
    from .solution import export_solution
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)


# Exception for timeouts
class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


@contextmanager
def time_limit(seconds: float):
    """
    Context manager to limit execution time of a code block.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutException: If execution time exceeds the limit
    """

    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")

    # Set signal handler for SIGALRM
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        # Reset signal handler
        signal.setitimer(signal.ITIMER_REAL, 0)


def safe_import(name, *args, **kwargs):
    """A restricted version of __import__ that only allows importing safe modules."""
    ALLOWED_MODULES = {
        # Standard modules
        "math",
        "random",
        "collections",
        "itertools",
        "re",
        "json",
        # PySAT modules
        "pysat",
        "pysat.formula",
        "pysat.solvers",
        "pysat.card",
        "pysat.examples",
        "pysat.examples.rc2",
        "pysat.pb",
        "pysat.engines",
    }
    if name not in ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' is not allowed in the restricted environment"
        )
    return __import__(name, *args, **kwargs)


def _execute_pysat_code_in_process(code: str, result_queue: Queue) -> None:
    """
    Helper function to execute MaxSAT code in a separate process.
    This function will be run in a separate process.

    Args:
        code: MaxSAT code to execute
        result_queue: Queue to send the result back to the parent process
    """
    try:
        # Capture standard output and error
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Initialize result dictionary
        result = {"success": False, "output": "", "error": None, "solution": None}

        # Process imports (same as in the original function)
        processed_code = ""
        in_import = False
        import_lines = []
        regular_lines = []

        for line in code.split("\n"):
            if line.strip().startswith(("import ", "from ")) or in_import:
                # Track multiline imports
                if line.strip().endswith("\\"):
                    in_import = True
                else:
                    in_import = False
                import_lines.append(line)
            else:
                regular_lines.append(line)

        # Add fixed imports at the top - embed all helpers as strings like PySAT does
        processed_code = """
# Standard library imports provided by the environment
import collections
import itertools
import math
import json
import re
import random

# PySAT imports
from pysat.formula import CNF, WCNF
from pysat.solvers import Glucose3, Cadical153
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2

# Cardinality constraint helpers (embedded directly)
def at_most_k(variables, k):
    \"\"\"Returns clauses ensuring at most k variables are true\"\"\"
    clauses = []
    if k < 0:
        # Trivially unsatisfiable
        clauses.append([])  # Empty clause means contradiction
    elif k == 0:
        # All variables must be false
        for var in variables:
            clauses.append([-var])
    elif k >= len(variables):
        # Trivially satisfiable
        pass
    else:
        # For each combination of k+1 variables, at least one must be false
        from itertools import combinations
        for combo in combinations(variables, k + 1):
            clauses.append([-var for var in combo])
    return clauses

def at_least_k(variables, k):
    \"\"\"Returns clauses ensuring at least k variables are true\"\"\"
    clauses = []
    if k <= 0:
        # Trivially satisfiable
        pass
    elif k > len(variables):
        # Trivially unsatisfiable
        clauses.append([])  # Empty clause means contradiction
    else:
        # For each combination of n-k+1 variables, at least one must be true
        from itertools import combinations
        n = len(variables)
        negated_vars = [-var for var in variables]
        for combo in combinations(negated_vars, n - k + 1):
            clauses.append([-var for var in combo])
    return clauses

def exactly_k(variables, k):
    \"\"\"Returns clauses ensuring exactly k variables are true\"\"\"
    at_most = at_most_k(variables, k)
    at_least = at_least_k(variables, k)
    return at_most + at_least

# Basic constraint helpers
def at_most_one(variables):
    \"\"\"Returns clauses ensuring at most one variable is true\"\"\"
    clauses = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def exactly_one(variables):
    \"\"\"Returns clauses ensuring exactly one variable is true\"\"\"
    clauses = []
    # At least one is true
    clauses.append(list(variables))
    # At most one is true
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def implies(a, b):
    \"\"\"Returns a clause representing a -> b (if a then b)\"\"\"
    return [[-a, b]]

def mutually_exclusive(variables):
    \"\"\"Returns clauses ensuring variables are mutually exclusive\"\"\"
    return at_most_one(variables)

def if_then_else(condition, then_var, else_var):
    \"\"\"Returns clauses for if-then-else construct\"\"\"
    return [[-condition, then_var], [condition, else_var]]

""" + "\n".join(regular_lines)

        # Setup restricted globals for execution (same as in original function)
        restricted_globals = {
            "__builtins__": {
                # Allowed builtins
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "divmod": divmod,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "id": id,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "next": next,
                "print": print,
                "range": range,
                "reversed": reversed,
                "round": round,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "True": True,
                "False": False,
                "None": None,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "__import__": safe_import,
                "__build_class__": __build_class__,  # Needed for class definitions
                "__name__": "__main__",  # Needed for some constructs
                "NameError": NameError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "RuntimeError": RuntimeError,
                "NotImplementedError": NotImplementedError,
                "StopIteration": StopIteration,
                "AssertionError": AssertionError,
                "assert": lambda cond, msg="": None
                if cond
                else (_ for _ in ()).throw(AssertionError(msg)),
                "type": type,
                "repr": repr,
                "hash": hash,
                "getattr": getattr,
                "setattr": setattr,
                "hasattr": hasattr,
                "delattr": delattr,
                "vars": vars,
                "dir": dir,
                "globals": lambda: restricted_globals,
                "locals": locals,
                "callable": callable,
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "oct": oct,
                "bin": bin,
                "format": format,
                "pow": pow,
                "slice": slice,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "super": super,
                "object": object,
            },
            # Provide the PySAT/MaxSAT environment
            "CNF": CNF,
            "WCNF": WCNF,
            "Glucose3": Glucose3,
            "Cadical153": Cadical153,
            "CardEnc": CardEnc,
            "EncType": EncType,
            "RC2": RC2,  # MaxSAT solver
            "export_solution": export_solution,
            "collections": collections,
            "itertools": itertools,
            "math": math,
            "random": random,
            "re": re,
            "json": json,
            "time": time,
            # Don't need to add cardinality functions here - they're embedded in code
        }

        # Add common variable types needed for PySAT code
        restricted_globals["dict_keys"] = type({}.keys())
        restricted_globals["dict_values"] = type({}.values())
        restricted_globals["dict_items"] = type({}.items())
        restricted_globals["map_iterator"] = type(map(lambda x: x, []))
        restricted_globals["filter_iterator"] = type(filter(lambda x: x, []))
        restricted_globals["enumerate_iterator"] = type(enumerate([]))
        restricted_globals["zip_iterator"] = type(zip(strict=False))
        restricted_globals["range_iterator"] = type(range(0))

        # Add logging capability so we can trace execution
        restricted_globals["logging"] = logging

        # Wrap code execution in a try-except block to catch syntax errors
        try:
            # First, compile the code to catch syntax errors before execution
            compiled_code = compile(processed_code, "<maxsat_code>", "exec")

            # Execute code and capture output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the compiled code (no need for time_limit here as we're in a separate process)
                exec(compiled_code, restricted_globals)

            # Extract standard output and error
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Check if _LAST_SOLUTION was set by export_solution
            # Import the solution module to get _LAST_SOLUTION
            from . import solution as sol_module

            solution = getattr(sol_module, "_LAST_SOLUTION", None)

            # Return success result with output and solution
            result["success"] = True
            result["output"] = stdout + "\n" + stderr
            result["solution"] = solution

        except SyntaxError as e:
            # Handle syntax errors with detailed information
            error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n  {' ' * (e.offset - 1)}^"

            result["error"] = error_msg
            result["output"] = error_msg
            result["success"] = False

        except Exception as e:
            # Handle any other exception
            error_msg = f"Error: {type(e).__name__}: {e!s}"
            result["error"] = error_msg
            result["output"] = (
                f"{error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            )

        # Send the result back through the queue
        result_queue.put(result)
    except Exception as e:
        # Handle any unexpected exceptions in the process
        result_queue.put(
            {
                "success": False,
                "error": f"Process error: {e!s}",
                "output": traceback.format_exc(),
                "solution": None,
            }
        )


def execute_pysat_code(code: str, timeout: float = 10.0) -> dict[str, Any]:
    """
    Executes MaxSAT Python code in a secure environment with robust timeout handling.

    This implementation uses a separate process to execute the code, which allows for
    more reliable timeout handling by terminating the entire process if needed.

    Args:
        code: The MaxSAT Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        A dictionary containing the execution results:
        {
            'success': bool,
            'output': str,
            'error': Optional[str],
            'solution': Optional[Dict]
        }
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing MaxSAT code with timeout {timeout} seconds")

    # Initialize result dictionary with default timeout error
    result = {
        "success": True,  # Important: Mark as success but with timeout information
        "output": f"Execution timed out after {timeout} seconds",
        "error": None,  # No error, just a timeout
        "solution": None,
        "timeout": True,
        "error_details": {
            "type": "TimeoutInfo",  # Not an error type
            "timeout": timeout,
            "message": f"MaxSAT execution exceeded the {timeout} second timeout and was terminated",
        },
    }

    # Create a queue for inter-process communication
    result_queue = Queue()

    # Create and start a process to execute the code
    # Use daemon=True to ensure the process doesn't block server shutdown
    process = Process(
        target=_execute_pysat_code_in_process, args=(code, result_queue), daemon=True
    )

    start_time = time.time()

    try:
        # Start the process
        process.start()

        # Use a non-blocking approach to wait for results or timeout
        elapsed = 0
        check_interval = 0.1  # Check every 100ms

        # Check for a result or timeout without blocking
        while elapsed < timeout:
            # Check if process completed and placed result in queue
            if not result_queue.empty():
                result = result_queue.get()
                execution_time = time.time() - start_time

                # Add execution time info
                if "error_details" not in result:
                    result["error_details"] = {}
                result["error_details"]["execution_time"] = execution_time

                logger.debug(
                    f"MaxSAT code execution completed in {execution_time:.2f} seconds"
                )
                return result

            # Sleep briefly to avoid busy waiting
            time.sleep(check_interval)
            elapsed = time.time() - start_time

        # If we've reached here, it's a timeout
        logger.warning(f"MaxSAT code execution timed out after {timeout} seconds")

        # Attempt to terminate the process gently
        if process.is_alive():
            try:
                process.terminate()
            except Exception as e:
                logger.error(f"Error during process termination: {e}")

        return result

    except Exception as e:
        logger.error(f"Error in execute_pysat_code: {e!s}", exc_info=True)
        # Return a success=True result even for errors to maintain connection
        return {
            "success": True,  # Mark as successful to prevent disconnection
            "output": f"Error executing MaxSAT code: {e!s}",
            "error": f"Execution error: {e!s}",
            "error_details": {"type": type(e).__name__, "message": str(e)},
            "solution": None,
            "status": "error",  # Indicate that there was an error even though success=True
        }
