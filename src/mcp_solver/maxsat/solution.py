"""
MaxSAT solution module for extracting and formatting solutions from MaxSAT solvers.

This module provides functions for extracting solution data from MaxSAT solvers
and converting it to a standardized format optimized for optimization problems.
"""

import logging
import os
import sys
from typing import Any, Union


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
    from pysat.examples.rc2 import RC2
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Solver
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import our error handling utilities

# Set up logger
logger = logging.getLogger(__name__)

# Track the last solution - make it global and accessible
_LAST_SOLUTION = None

# Track whether an optimal RC2 solution has been set
_RC2_SOLUTION_SET = False

# Reserved keys that shouldn't be processed as custom dictionaries
RESERVED_KEYS = {
    "satisfiable",
    "model",
    "values",
    "status",
    "objective",
    "cost",
    "error_type",
    "error_message",
    "warnings",
    "unsatisfied",
    "maxsat_result",
    "soft_clauses",
    "hard_clauses",
    "weights",
}


class SolutionError(Exception):
    """
    Custom exception for solution processing errors.

    This exception is used when errors occur during solution processing
    that should be captured and returned as a structured error solution.
    """

    pass


def export_maxsat_solution(
    data: Union[dict[str, Any], "RC2", None] = None,
    variables: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Export solution data from a MaxSAT optimization problem.

    This function makes it easy to return results from a MaxSAT problem,
    handling both directly provided dictionaries and RC2 solver instances.
    It automatically extracts optimization data and formats it in a standardized way.

    Args:
        data: Either an RC2 solver instance or a dictionary with solution data
        variables: Optional mapping from variable names to their numeric IDs

    Returns:
        Dictionary with the MaxSAT solution data and metadata.

    Examples:
        Basic usage with a direct dictionary:
            ```python
            export_maxsat_solution(
                {
                    "satisfiable": True,
                    "selected_features": {"base": True, "premium": True},
                    "total_value": 75,
                }
            )
            ```

        Usage with an RC2 solver:
            ```python
            with RC2(wcnf) as solver:
                model = solver.compute()
                export_maxsat_solution(solver, var_mapping)
            ```
            
    WARNING: Do not call both export_maxsat_solution with an RC2 solver AND
    export_solution in the same program. This will overwrite your optimal
    MaxSAT solution with a potentially different solution.
    """
    global _LAST_SOLUTION, _RC2_SOLUTION_SET
    try:
        solution_data = {}

        # Case 1: RC2 solver provided
        if isinstance(data, RC2):
            solver = data
            model = solver.model

            if model is not None:
                # Solution found
                solution_data["satisfiable"] = True
                solution_data["status"] = "optimal"
                solution_data["model"] = model
                solution_data["cost"] = solver.cost

                # For optimization problems, provide the objective value
                # (negative cost since MaxSAT minimizes costs, but users often think in terms of maximizing value)
                solution_data["objective"] = -solver.cost

                # Map variable names to values if a mapping was provided
                if variables:
                    solution_data["assignment"] = {
                        name: (var_id in model)
                        if var_id > 0
                        else ((-var_id) not in model)
                        for name, var_id in variables.items()
                    }
            else:
                # No model found, problem is unsatisfiable
                solution_data["satisfiable"] = False
                solution_data["status"] = "unsatisfiable"

        # Case 2: Dictionary data provided
        elif isinstance(data, dict):
            solution_data = data.copy()

            # Ensure required fields are present
            if "satisfiable" not in solution_data:
                solution_data["satisfiable"] = True

            if "status" not in solution_data:
                solution_data["status"] = (
                    "optimal"
                    if solution_data.get("satisfiable", True)
                    else "unsatisfiable"
                )

        # Case 3: No data provided
        else:
            solution_data = {
                "satisfiable": False,
                "status": "error",
                "message": "No valid data provided",
            }

        # Ensure values dictionary exists for the extraction process
        if "values" not in solution_data:
            solution_data["values"] = {}

        # Extract values from custom dictionaries
        solution_data = _extract_values_from_dictionaries(solution_data)

        # Create maxsat_result structure with optimization-specific data
        maxsat_result = {
            "satisfiable": solution_data.get("satisfiable", False),
            "status": solution_data.get("status", "unknown"),
            "values": solution_data.get("values", {}),
        }

        # Add cost/objective if available
        if "cost" in solution_data:
            maxsat_result["cost"] = solution_data["cost"]
        if "objective" in solution_data:
            maxsat_result["objective"] = solution_data["objective"]

        # Include model information if available
        if "model" in solution_data:
            maxsat_result["model"] = solution_data["model"]

        # Include any custom dictionaries
        for key, value in solution_data.items():
            if key not in RESERVED_KEYS and isinstance(value, dict):
                maxsat_result[key] = value

        # Log the solution data
        logger.debug(f"Setting _LAST_SOLUTION: {maxsat_result}")
        print(f"DEBUG - _LAST_SOLUTION set to: {maxsat_result}")

        # Store the MaxSAT solution
        _LAST_SOLUTION = maxsat_result
        
        # Mark if this was an RC2 solver solution
        if isinstance(data, RC2):
            _RC2_SOLUTION_SET = True

        return maxsat_result

    except Exception as e:
        # Create a simple error solution
        error_solution = {
            "satisfiable": False,
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "values": {},
        }

        # Store the error solution
        _LAST_SOLUTION = error_solution
        _RC2_SOLUTION_SET = False
        logger.error(f"Error in export_maxsat_solution: {e!s}", exc_info=True)
        print(f"DEBUG - _LAST_SOLUTION set to error: {error_solution}")

        return error_solution


def extract_weights_mapping(wcnf: "WCNF") -> dict[int, int]:
    """
    Extract mapping of soft clause indices to their weights.

    Args:
        wcnf: WCNF formula with soft clauses

    Returns:
        Dictionary mapping soft clause indices to weights
    """
    weights_mapping = {}
    for i, (_, weight) in enumerate(zip(wcnf.soft, wcnf.wght, strict=False)):
        weights_mapping[i] = weight
    return weights_mapping


def get_soft_clause_info(
    wcnf: "WCNF", mapping: dict[str, Any] | None = None
) -> dict[int, dict[str, Any]]:
    """
    Create a mapping of soft clause indices to their information.

    Args:
        wcnf: WCNF formula with soft clauses
        mapping: Optional mapping from variable IDs to names

    Returns:
        Dictionary mapping clause indices to information about the clause
    """
    clause_info = {}
    for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght, strict=False)):
        info = {"weight": weight, "clause": clause, "literals": []}

        # If variable mapping provided, add literal names
        if mapping:
            for lit in clause:
                var_id = abs(lit)
                is_positive = lit > 0
                var_name = None
                for name, vid in mapping.items():
                    if vid == var_id:
                        var_name = name
                        break

                if var_name:
                    info["literals"].append({"name": var_name, "positive": is_positive})

        clause_info[i] = info

    return clause_info


def get_unsatisfied_soft_clauses(wcnf: "WCNF", model: list[int]) -> dict[str, Any]:
    """
    Identify which soft clauses are unsatisfied by the given model.

    Args:
        wcnf: WCNF formula with soft clauses
        model: List of true variable IDs (positive for true, negative for false)

    Returns:
        Dictionary with information about unsatisfied soft clauses and their cost
    """
    unsatisfied = []
    total_cost = 0

    # Create a set of true literals for faster lookup
    true_lits = set(model)

    # Check each soft clause
    for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght, strict=False)):
        # A clause is satisfied if any of its literals is true
        is_satisfied = any(
            lit in true_lits if lit > 0 else -lit not in true_lits for lit in clause
        )

        if not is_satisfied:
            unsatisfied.append({"index": i, "clause": clause, "weight": weight})
            total_cost += weight

    return {
        "unsatisfied_clauses": unsatisfied,
        "total_cost": total_cost,
        "num_unsatisfied": len(unsatisfied),
    }


def get_maxsat_solution_status() -> dict[str, Any]:
    """
    Get the current MaxSAT solution status.
    
    Returns:
        Dictionary with information about the last MaxSAT solution
        or None if no solution is available
    """
    global _LAST_SOLUTION, _RC2_SOLUTION_SET
    
    if _LAST_SOLUTION is None:
        return {
            "available": False,
            "message": "No MaxSAT solution available"
        }
    
    # Return basic information about the solution
    result = {
        "available": True,
        "satisfiable": _LAST_SOLUTION.get("satisfiable", False),
        "status": _LAST_SOLUTION.get("status", "unknown"),
        "optimal": _RC2_SOLUTION_SET,
    }
    
    # Include cost and objective if available
    if "cost" in _LAST_SOLUTION:
        result["cost"] = _LAST_SOLUTION["cost"]
    if "objective" in _LAST_SOLUTION:
        result["objective"] = _LAST_SOLUTION["objective"]
        
    return result


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
