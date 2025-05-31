"""
Z3 environment module for secure execution of Z3 code.

This module provides functions for executing Z3 code in a secure environment
with timeout handling and restricted access to system resources.
"""

import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any


# IMPORTANT: Properly import the Z3 library (not our local package)
# First, remove the current directory from the path to avoid importing ourselves
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

# Now try to import Z3
try:
    import z3

    # Store all Z3 symbols we'll need
    Z3_Int = z3.Int
    Z3_Bool = z3.Bool
    Z3_Real = z3.Real
    Z3_String = z3.String if hasattr(z3, "String") else None
    Z3_Solver = z3.Solver
    Z3_Optimize = z3.Optimize
    Z3_sat = z3.sat
    Z3_unsat = z3.unsat
    Z3_unknown = z3.unknown
    Z3_is_expr = z3.is_expr if hasattr(z3, "is_expr") else lambda x: False
    Z3_is_int = z3.is_int if hasattr(z3, "is_int") else lambda x: False
    Z3_is_real = z3.is_real if hasattr(z3, "is_real") else lambda x: False
    Z3_is_bool = z3.is_bool if hasattr(z3, "is_bool") else lambda x: False
    Z3_is_true = z3.is_true if hasattr(z3, "is_true") else lambda x: False
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)

# Global variable to store the solution
_LAST_SOLUTION = None


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


@contextmanager
def time_limit(seconds: float):
    """
    Context manager for limiting execution time of code blocks.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutException: If execution time exceeds the limit
    """

    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")

    # Set the timeout handler
    previous_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        # Reset the alarm and restore previous handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_z3_code(
    code_string: str, timeout: float = 4.0, auto_extract: bool = True
) -> dict[str, Any]:
    """
    Execute Z3 Python code in a secure environment with timeout handling.

    Args:
        code_string: The Z3 Python code to execute
        timeout: Maximum execution time in seconds
        auto_extract: Whether to add solution extraction code automatically

    Returns:
        Dictionary with execution results
    """
    # Import solution module
    from .solution import export_solution

    # Import templates modules
    from .templates import function_templates, subset_templates, z3_templates

    # Reset last solution
    global _LAST_SOLUTION
    _LAST_SOLUTION = None

    # Pre-process the code to remove any imports
    code_lines = code_string.split("\n")
    processed_code = []

    for line in code_lines:
        # Skip import lines but keep Z3 namespace imports and templates imports
        if (
            line.strip().startswith("from z3 import")
            or line.strip().startswith("import z3")
            or line.strip().startswith("from z3_templates import")
            or line.strip().startswith("from mcp_solver.z3.templates import")
            or line.strip().startswith("from function_templates import")
            or line.strip().startswith("from subset_templates import")
        ):
            continue
        else:
            processed_code.append(line)

    processed_code_string = "\n".join(processed_code)

    # Modify code to catch common variable naming issues
    # Add a preamble that makes solver, s, and z3_solver global to avoid issues
    # with export_solution not being able to access them
    processed_code_string = (
        """
# Make sure 'solver', 's', and 'z3_solver' variables are accessible globally
# This helps with the common issue of local solver variables not being visible to export_solution
global solver, s, z3_solver
solver = None
s = None
z3_solver = None

"""
        + processed_code_string
    )

    # Create restricted globals dict with only necessary functions/modules
    restricted_globals = {
        # Allow a limited subset of builtins
        "Exception": Exception,
        "ImportError": ImportError,
        "NameError": NameError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # Explicitly set open to None to force NameError
        "open": None,
        # Add global keywords
        "global": None,
        "solver": None,
        "s": None,
        "z3_solver": None,
    }

    # Add Z3 module itself
    restricted_globals["z3"] = z3

    # Add imported Z3 symbols to simulate 'from z3 import *'
    restricted_globals["Int"] = Z3_Int
    restricted_globals["Bool"] = Z3_Bool
    restricted_globals["Real"] = Z3_Real
    if Z3_String:
        restricted_globals["String"] = Z3_String
    restricted_globals["Solver"] = Z3_Solver
    restricted_globals["Optimize"] = Z3_Optimize
    restricted_globals["sat"] = Z3_sat
    restricted_globals["unsat"] = Z3_unsat
    restricted_globals["unknown"] = Z3_unknown
    restricted_globals["is_expr"] = Z3_is_expr
    restricted_globals["is_int"] = Z3_is_int
    restricted_globals["is_real"] = Z3_is_real
    restricted_globals["is_bool"] = Z3_is_bool
    restricted_globals["is_true"] = Z3_is_true

    # Add more Z3 symbols - iterate through the z3 module
    for name in dir(z3):
        if not name.startswith("_") and name not in restricted_globals:
            restricted_globals[name] = getattr(z3, name)

    # Add our export_solution function
    restricted_globals["export_solution"] = export_solution

    # Add Z3 templates module functions
    restricted_globals["z3_templates"] = z3_templates
    restricted_globals["array_is_sorted"] = z3_templates.array_is_sorted
    restricted_globals["all_distinct"] = z3_templates.all_distinct
    restricted_globals["array_contains"] = z3_templates.array_contains
    restricted_globals["exactly_k"] = z3_templates.exactly_k
    restricted_globals["at_most_k"] = z3_templates.at_most_k
    restricted_globals["at_least_k"] = z3_templates.at_least_k
    restricted_globals["function_is_injective"] = z3_templates.function_is_injective
    restricted_globals["function_is_surjective"] = z3_templates.function_is_surjective

    # Add function templates
    restricted_globals["function_templates"] = function_templates
    restricted_globals["constraint_satisfaction_template"] = (
        function_templates.constraint_satisfaction_template
    )
    restricted_globals["optimization_template"] = (
        function_templates.optimization_template
    )
    restricted_globals["array_template"] = function_templates.array_template
    restricted_globals["quantifier_template"] = function_templates.quantifier_template
    restricted_globals["demo_template"] = function_templates.demo_template

    # Add subset templates
    restricted_globals["subset_templates"] = subset_templates
    restricted_globals["smallest_subset_with_property"] = (
        subset_templates.smallest_subset_with_property
    )

    # Prepare result dictionary
    result = {
        "status": "unknown",
        "error": None,
        "output": [],
        "solution": None,
        "execution_time": 0,
    }

    # Capture print output
    original_stdout = sys.stdout
    from io import StringIO

    captured_output = StringIO()
    sys.stdout = captured_output

    # Execute code with timeout
    start_time = time.time()

    try:
        with time_limit(timeout):
            # Execute the code in the restricted environment
            local_vars = {}
            exec(processed_code_string, restricted_globals, local_vars)

            # Check solution from local_vars
            if "solution" in local_vars:
                result["solution"] = local_vars["solution"]
                result["status"] = "success"
            # Check for solution in _LAST_SOLUTION
            elif _LAST_SOLUTION is not None:
                result["solution"] = _LAST_SOLUTION
                result["status"] = "success"
            else:
                # Important: Also check export_solution module's _LAST_SOLUTION directly
                # since it might be set by code executed in different scope
                from .solution import _LAST_SOLUTION as solution_last_solution

                if solution_last_solution is not None:
                    result["solution"] = solution_last_solution
                    result["status"] = "success"
                else:
                    result["status"] = "no_solution"
                    result["error"] = (
                        "No solution was exported. Make sure to call export_solution()"
                    )
    except TimeoutException as e:
        result["status"] = "timeout"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e!s}"
        result["traceback"] = traceback.format_exc()
    finally:
        # Restore stdout and record execution time
        sys.stdout = original_stdout
        result["execution_time"] = time.time() - start_time
        result["output"] = (
            captured_output.getvalue().strip().split("\n")
            if captured_output.getvalue()
            else []
        )

    return result
