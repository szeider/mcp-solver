"""
PySAT solution module for extracting and formatting solutions from PySAT solvers.

This module provides functions for extracting solution data from PySAT solvers
and converting it to a standardized format.
"""

import sys
import os
from typing import Dict, Any, Optional, Union, List, TypeVar, cast
import logging
import traceback

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
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Solver
    from pysat.examples.rc2 import RC2
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
    "objective",
    "cost",
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
    data: Union[Dict[str, Any], Solver, None] = None,
    variables: Optional[Dict[str, int]] = None,
    objective: Optional[float] = None,
) -> Dict[str, Any]:
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
        logger.error(f"Error in export_solution: {str(e)}", exc_info=True)
        print(f"DEBUG - _LAST_SOLUTION set to error: {error_solution}")

        return error_solution


def export_maxsat_solution(
    data: Union[Dict[str, Any], 'RC2', None] = None,
    variables: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Export solution data from a MaxSAT optimization problem.
    
    This function makes it easy to return results from a MaxSAT problem,
    handling both directly provided dictionaries and RC2 solver instances.
    It also automatically calls export_solution internally, so you don't need
    to make a separate call to export_solution.
    
    Args:
        data: Either an RC2 solver instance or a dictionary with solution data
        variables: Optional mapping from variable names to their numeric IDs
    
    Returns:
        Dictionary with the MaxSAT solution data and metadata.
        
    Examples:
        Basic usage with a direct dictionary:
            ```python
            export_maxsat_solution({
                "satisfiable": True,
                "selected_features": {"base": True, "premium": True},
                "total_value": 75
            })
            ```
            
        Usage with an RC2 solver:
            ```python
            with RC2(wcnf) as solver:
                model = solver.compute()
                export_maxsat_solution(solver, var_mapping)
            ```
    """
    global _LAST_SOLUTION
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
                        name: (var_id in model) if var_id > 0 else ((-var_id) not in model)
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
                solution_data["status"] = "optimal" if solution_data.get("satisfiable", True) else "unsatisfiable"
        
        # Case 3: No data provided
        else:
            solution_data = {
                "satisfiable": False,
                "status": "error",
                "message": "No valid data provided"
            }
        
        # Ensure values dictionary exists for the extraction process
        if "values" not in solution_data:
            solution_data["values"] = {}
        
        # Extract values from custom dictionaries
        solution_data = _extract_values_from_dictionaries(solution_data)
        
        # Store the MaxSAT solution
        maxsat_result = solution_data
        logger.debug(f"MaxSAT solution exported: {maxsat_result}")
        
        # Also call export_solution to ensure compatibility with MCP framework
        # This eliminates the need for agents to call export_solution separately
        export_result = {
            "satisfiable": maxsat_result.get("satisfiable", True),
            "message": "MaxSAT solution exported successfully",
            "maxsat_data": maxsat_result
        }
        
        # Add a special marker to indicate this is a MaxSAT solution
        export_result["_is_maxsat_solution"] = True
        
        # Don't call export_solution directly as it would overwrite _LAST_SOLUTION
        # Instead, process it and store it directly to preserve the MaxSAT data
        _LAST_SOLUTION = export_result
        
        return maxsat_result
        
    except Exception as e:
        # Create a simple error solution
        error_solution = {
            "satisfiable": False,
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "values": {}
        }
        
        # Store the error solution
        maxsat_error = error_solution
        logger.error(f"Error in export_maxsat_solution: {str(e)}", exc_info=True)
        
        # Create a solution in the format expected by the MCP framework
        export_result = {
            "satisfiable": False,
            "message": f"MaxSAT error: {str(e)}",
            "error_type": type(e).__name__,
            "maxsat_data": maxsat_error,
            "_is_maxsat_solution": True  # Mark this as MaxSAT even if it failed
        }
        
        # Store the formatted solution directly instead of calling export_solution
        _LAST_SOLUTION = export_result
        
        return maxsat_error


def extract_weights_mapping(wcnf: 'WCNF') -> Dict[int, int]:
    """
    Extract mapping of soft clause indices to their weights.
    
    Args:
        wcnf: WCNF formula with soft clauses
    
    Returns:
        Dictionary mapping soft clause indices to weights
    """
    weights_mapping = {}
    for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
        weights_mapping[i] = weight
    return weights_mapping


def get_soft_clause_info(wcnf: 'WCNF', mapping: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Create a mapping of soft clause indices to their information.
    
    Args:
        wcnf: WCNF formula with soft clauses
        mapping: Optional mapping from variable IDs to names
    
    Returns:
        Dictionary mapping clause indices to information about the clause
    """
    clause_info = {}
    for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
        info = {
            "weight": weight,
            "clause": clause,
            "literals": []
        }
        
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
                    info["literals"].append({
                        "name": var_name,
                        "positive": is_positive
                    })
        
        clause_info[i] = info
    
    return clause_info


def _process_input_data(
    data: Union[Dict[str, Any], Solver, None],
    variables: Optional[Dict[str, int]] = None,
    objective: Optional[float] = None,
) -> Dict[str, Any]:
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
    solution_data: Dict[str, Any] = {}

    # Case 1: Direct dictionary data
    if isinstance(data, dict):
        solution_data = data.copy()

    # Case 2: PySAT Solver object
    elif (
        data is not None
        and hasattr(data, "get_model")
        and callable(getattr(data, "get_model"))
    ):
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


def _extract_values_from_dictionaries(solution_data: Dict[str, Any]) -> Dict[str, Any]:
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
    values: Dict[str, Any] = {}

    # First pass: collect all keys to detect potential collisions
    key_counts: Dict[str, int] = {}

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


def _create_error_solution(error: Exception) -> Dict[str, Any]:
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
