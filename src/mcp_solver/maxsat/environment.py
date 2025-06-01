"""
MaxSAT secure execution environment.

This module provides a secure environment for executing MaxSAT code,
with timeout handling and output capturing. It builds on the PySAT
environment module but adapts it for MaxSAT optimization problems.
"""

import sys
import os
import tempfile
import logging
import math
import random
import collections
import itertools
import re
import json
import traceback
import time
from typing import Any

# Import specific PySAT environment utilities
from mcp_solver.pysat.environment import (
    execute_pysat_code, 
    safe_import, 
    time_limit, 
    TimeoutException
)

# Define the default timeout
DEFAULT_TIMEOUT_SECONDS = 10.0

# Import PySAT but protect against failure
try:
    import pysat
    from pysat.card import CardEnc, EncType
    from pysat.examples.rc2 import RC2
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Cadical153, Glucose3, Glucose4, Lingeling, Solver
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import our MaxSAT solution functions 
from mcp_solver.maxsat.solution import export_maxsat_solution

# Import PySAT templates and helper functions
from mcp_solver.pysat.templates.cardinality_templates import at_most_k, at_least_k, exactly_k
from mcp_solver.pysat.constraints import at_most_one, exactly_one, implies, mutually_exclusive, if_then_else

# Import MaxSAT templates
from mcp_solver.maxsat.templates import (
    # From basic_templates
    create_maxsat_solver,
    solve_maxsat_problem,
    add_hard_constraint,
    add_soft_constraint,
    encode_binary_variable,
    
    # From optimization_templates
    feature_selection_problem,
    weighted_max_cut,
    set_cover_problem,
    knapsack_problem,
    
    # From variable_mapping
    VariableMap,
    create_variable_map,
    
    # From cardinality_constraints
    at_most_k as maxsat_at_most_k,
    at_least_k as maxsat_at_least_k,
    prefer_at_least_k,
    prefer_at_most_k,
    
    # From objective_helpers
    maximize_sum,
    minimize_sum,
    optimize_net_value,
    calculate_objective_value,
    encode_weighted_selection
)

# We need to create our own version of _execute_pysat_code_in_process that includes
# export_maxsat_solution in the restricted globals, just like PySAT does with export_solution

# Import needed modules for our implementation
from multiprocessing import Process, Queue
import io
from contextlib import redirect_stdout, redirect_stderr

def _execute_maxsat_code_in_process(code: str, result_queue: Queue) -> None:
    """
    Helper function to execute MaxSAT code in a separate process.
    This is based on PySAT's _execute_pysat_code_in_process but includes export_maxsat_solution.

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

        # Add fixed imports and helpers at the top
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

# Define basic helper functions for MaxSAT constraints
def at_most_one(variables):
    clauses = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def exactly_one(variables):
    clauses = at_most_one(variables)
    clauses.append(list(variables))
    return clauses

def implies(a, b):
    return [[-a, b]]

def mutually_exclusive(variables):
    return at_most_one(variables)

def if_then_else(condition, then_var, else_var):
    return [[-condition, then_var], [condition, else_var]]

""" + "\n".join(regular_lines)

        # Setup restricted globals for execution
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
            },
            # Provide the PySAT environment - only include stable solvers
            "CNF": CNF,
            "WCNF": WCNF,
            "Glucose3": Glucose3,  # Standard solver 1
            "Cadical153": Cadical153,  # Standard solver 2
            "CardEnc": CardEnc,
            "EncType": EncType,
            "RC2": RC2,  # MaxSAT solver
            "export_maxsat_solution": export_maxsat_solution,  # Key difference from PySAT
            "collections": collections,
            "itertools": itertools,
            "math": math,
            "random": random,
            "re": re,
            "json": json,
            "time": time,  # Add time module to the restricted globals
            # No need to reference the constraint functions directly here,
            # as they're already defined in the processed_code preamble
            # Add MaxSAT specific template functions
            "create_maxsat_solver": create_maxsat_solver,
            "solve_maxsat_problem": solve_maxsat_problem,
            "add_hard_constraint": add_hard_constraint,
            "add_soft_constraint": add_soft_constraint,
            "encode_binary_variable": encode_binary_variable,
            "feature_selection_problem": feature_selection_problem,
            "weighted_max_cut": weighted_max_cut,
            "set_cover_problem": set_cover_problem,
            "knapsack_problem": knapsack_problem,
            "VariableMap": VariableMap,
            "create_variable_map": create_variable_map,
            # Cardinality constraints (MaxSAT versions)
            "at_most_k": maxsat_at_most_k,
            "at_least_k": maxsat_at_least_k,
            "prefer_at_least_k": prefer_at_least_k,
            "prefer_at_most_k": prefer_at_most_k,
            # Objective helpers
            "maximize_sum": maximize_sum,
            "minimize_sum": minimize_sum,
            "optimize_net_value": optimize_net_value,
            "calculate_objective_value": calculate_objective_value,
            "encode_weighted_selection": encode_weighted_selection,
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

            # Check if _LAST_SOLUTION was set by export_maxsat_solution
            solution = None
            if "_LAST_SOLUTION" in restricted_globals:
                solution = restricted_globals["_LAST_SOLUTION"]

            # Return success result with output and solution
            result["success"] = True
            result["output"] = stdout + "\n" + stderr
            result["solution"] = solution

        except Exception as e:
            # Handle any exception
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

# Custom version of execute_pysat_code for MaxSAT
def execute_maxsat_code(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """
    Executes MaxSAT Python code in a secure environment with robust timeout handling.
    
    This is similar to execute_pysat_code but includes export_maxsat_solution in
    the restricted globals dictionary.

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
        target=_execute_maxsat_code_in_process, args=(code, result_queue), daemon=True
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
        logger.error(f"Error in execute_maxsat_code: {e!s}", exc_info=True)
        # Return a success=True result even for errors to maintain connection
        return {
            "success": True,  # Mark as successful to prevent disconnection
            "output": f"Error executing MaxSAT code: {e!s}",
            "error": f"Execution error: {e!s}",
            "error_details": {"type": type(e).__name__, "message": str(e)},
            "solution": None,
            "status": "error",  # Indicate that there was an error even though success=True
        }

# Override the execute_pysat_code function with our MaxSAT-specific version
execute_pysat_code = execute_maxsat_code

# Add a reference that will be available in the environment
def get_export_maxsat_solution():
    """
    Helper function to get a reference to the export_maxsat_solution function.
    This is needed because the function might not be directly importable in the environment.
    """
    return export_maxsat_solution

# Define the list of exported symbols
__all__ = [
    # Environment functions
    "execute_pysat_code",
    "DEFAULT_TIMEOUT_SECONDS",
    "get_export_maxsat_solution",
    
    # Solution functions
    "export_maxsat_solution",
    
    # Basic templates
    "create_maxsat_solver",
    "solve_maxsat_problem",
    "add_hard_constraint",
    "add_soft_constraint",
    "encode_binary_variable",
    
    # Optimization templates
    "feature_selection_problem",
    "weighted_max_cut",
    "set_cover_problem",
    "knapsack_problem",
    
    # Variable mapping
    "VariableMap",
    "create_variable_map"
]
