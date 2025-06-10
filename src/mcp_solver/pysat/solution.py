"""
PySAT solution module for extracting and formatting solutions from PySAT solvers.

This module provides functions for extracting solution data from PySAT solvers
and converting it to a standardized format.
"""

import logging
import os
import sys
from typing import Any


# IMPORTANT: Properly import the PySAT library (not our local package)
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

# Now try to import PySAT
try:
    from pysat.formula import CNF
    from pysat.solvers import Solver
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import our error handling utilities
from .error_handling import PySATError, validate_variables


# Set up logger
logger = logging.getLogger(__name__)

# Track the last solution - make it global and accessible
_LAST_SOLUTION = None

# Reserved keys that shouldn't be processed as custom dictionaries
RESERVED_KEYS = {
    "satisfiable",
    "model",
    "values",
    "status",
    "error_type",
    "error_message",
    "warnings",
    "unsatisfied",
}


class SolutionError(Exception):
    """
    Custom exception for solution processing errors.

    This exception is used when errors occur during solution processing
    that should be captured and returned as a structured error solution.
    """

    pass


def export_solution(
    data: dict[str, Any] | Solver | None = None,
    variables: dict[str, int] | None = None,
    objective: float | None = None,
) -> dict[str, Any]:
    """
    Extract and format solutions from a PySAT solver or solution data.

    This function processes PySAT solution data and creates a standardized
    output format. It supports both direct dictionary input and PySAT Solver
    objects. All values in custom dictionaries are automatically extracted
    and made available in the flat "values" dictionary.

    Args:
        data: PySAT Solver object or dictionary containing solution data
        variables: Dictionary mapping variable names to their variable IDs
        objective: Optional objective value for optimization problems

    Returns:
        Dictionary with structured solution data, including:
        - satisfiable: Boolean indicating satisfiability
        - status: String status ("sat", "unsat", or "error")
        - values: Flattened dictionary of all values from custom dictionaries
        - model: List of true variable IDs (if satisfiable)
        - Other custom dictionaries provided in the input

    If an error occurs, the returned dictionary will include:
        - satisfiable: False
        - error_type: Type of the error
        - error_message: Detailed error message
        - status: "error"
    """
    global _LAST_SOLUTION

    try:
        solution_data = _process_input_data(data, variables, objective)
        solution_data = _extract_values_from_dictionaries(solution_data)

        # Log the solution data
        logger.debug(f"Setting _LAST_SOLUTION: {solution_data}")
        print(f"DEBUG - _LAST_SOLUTION set to: {solution_data}")

        # Store the solution and return it
        _LAST_SOLUTION = solution_data
        return solution_data

    except Exception as e:
        # Create an error solution with structured error information
        error_solution = _create_error_solution(e)

        # Store and return the error solution
        _LAST_SOLUTION = error_solution
        logger.error(f"Error in export_solution: {e!s}", exc_info=True)
        print(f"DEBUG - _LAST_SOLUTION set to error: {error_solution}")

        return error_solution


# Removed export_maxsat_solution function as it's now in the MaxSAT module


def _process_input_data(
    data: dict[str, Any] | Solver | None,
    variables: dict[str, int] | None = None,
    objective: float | None = None,
) -> dict[str, Any]:
    """
    Process input data from various sources into a standardized solution dictionary.

    Args:
        data: PySAT Solver object or dictionary containing solution data
        variables: Dictionary mapping variable names to their variable IDs
        objective: Optional objective value for optimization problems

    Returns:
        Standardized solution dictionary

    Raises:
        SolutionError: If the input data cannot be processed
    """
    # Initialize solution data
    solution_data: dict[str, Any] = {}

    # Case 1: Direct dictionary data
    if isinstance(data, dict):
        solution_data = data.copy()

    # Case 2: PySAT Solver object
    elif data is not None and hasattr(data, "get_model") and callable(data.get_model):
        # Extract model from solver (solver.solve() should have already been called)
        model = data.get_model()

        if model is not None:
            # Solver has a satisfiable solution
            solution_data["satisfiable"] = True
            solution_data["model"] = model

            # Extract variable assignments if variables dictionary is provided
            if variables:
                # Validate variables dictionary
                errors = validate_variables(variables)
                if errors:
                    error_msg = "; ".join(errors)
                    raise SolutionError(f"Invalid variables dictionary: {error_msg}")

                # Map variable names to their truth values based on the model
                solution_data["assignment"] = {
                    name: (var_id in model) if var_id > 0 else ((-var_id) not in model)
                    for name, var_id in variables.items()
                }
        else:
            # No model means unsatisfiable
            solution_data["satisfiable"] = False

    # Case 3: None or unknown type
    elif data is None:
        # Default to empty unsatisfiable solution
        solution_data["satisfiable"] = False
    else:
        raise SolutionError(f"Unsupported data type: {type(data).__name__}")

    # Ensure the satisfiable flag is set
    if "satisfiable" not in solution_data:
        solution_data["satisfiable"] = False

    # Set the status field to match satisfiability
    solution_data["status"] = (
        "sat" if solution_data.get("satisfiable", False) else "unsat"
    )

    # Include objective value if provided
    if objective is not None:
        solution_data["objective"] = objective

    # Ensure values dictionary exists (may be populated later)
    solution_data["values"] = solution_data.get("values", {})

    return solution_data


def _extract_values_from_dictionaries(solution_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract values from custom dictionaries into a flat values dictionary.

    Args:
        solution_data: The solution dictionary to process

    Returns:
        The processed solution dictionary with extracted values
    """
    # Skip extraction for unsatisfiable solutions that don't need values
    if not solution_data.get("satisfiable", False):
        # Ensure values dictionary exists even for unsatisfiable solutions
        solution_data["values"] = solution_data.get("values", {})
        return solution_data

    # Create a new values dictionary
    values: dict[str, Any] = {}

    # First pass: collect all keys to detect potential collisions
    key_counts: dict[str, int] = {}

    for key, value in solution_data.items():
        if key not in RESERVED_KEYS and isinstance(value, dict):
            for subkey in value.keys():
                key_counts[subkey] = key_counts.get(subkey, 0) + 1

    # Second pass: extract values and handle collisions
    for key, value in solution_data.items():
        if key not in RESERVED_KEYS and isinstance(value, dict):
            for subkey, subvalue in value.items():
                # Only extract leaf nodes (not nested dictionaries)
                if not isinstance(subvalue, dict):
                    if key_counts[subkey] > 1:
                        # This is a collision - prefix with parent dictionary name
                        values[f"{key}.{subkey}"] = subvalue
                    else:
                        # No collision - use the key directly
                        values[subkey] = subvalue

    # Update the values dictionary in the solution
    solution_data["values"] = values

    return solution_data


def _create_error_solution(error: Exception) -> dict[str, Any]:
    """
    Create a standardized error solution dictionary from an exception.

    Args:
        error: The exception that occurred

    Returns:
        A solution dictionary with error information
    """
    # If it's already a structured PySAT error, use its context
    if isinstance(error, PySATError):
        error_context = getattr(error, "context", {})
        error_solution = {
            "satisfiable": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "status": "error",
            "values": {},
        }

        # Add context if available
        if error_context:
            error_solution["error_context"] = error_context

    else:
        # For standard exceptions, create a basic error solution
        error_solution = {
            "satisfiable": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "status": "error",
            "values": {},
        }

    return error_solution
