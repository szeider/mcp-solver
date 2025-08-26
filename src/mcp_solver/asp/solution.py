"""
ASP solution module for extracting and formatting solutions from ASP solvers.

This module provides functions for extracting solution data from ASP solvers
and converting it to a standardized format.
"""

import logging
from typing import Any

from .error_handling import ASPError, format_solution_error


logger = logging.getLogger(__name__)

_LAST_SOLUTION = None

RESERVED_KEYS = {
    "satisfiable",
    "answer_sets",
    "values",
    "status",
    "error_type",
    "error_message",
    "warnings",
}


def export_solution(
    data: Any = None,
    status: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract and format solutions from an ASP solver or answer set data.

    Args:
        data: List of answer sets, or dictionary containing solution data
        status: Optional status string ("sat", "unsat", "error")
        warnings: Optional list of warning messages

    Returns:
        Dictionary with structured solution data, including:
        - satisfiable: Boolean indicating if at least one answer set was found
        - status: String status ("sat", "unsat", or "error")
        - answer_sets: List of answer sets (each a list of atoms)
        - values: Flattened dictionary of all atoms (True if present in any answer set)
        - warnings: List of warning messages (if any)

    If an error occurs, the returned dictionary will include:
        - satisfiable: False
        - error_type: Type of the error
        - error_message: Detailed error message
        - status: "error"
    """
    global _LAST_SOLUTION
    try:
        # If data is an exception, format as error immediately
        if isinstance(data, Exception):
            error_solution = format_solution_error(data)
            _LAST_SOLUTION = error_solution
            return error_solution
        solution_data = _process_input_data(data, status, warnings)
        solution_data = _extract_values_from_answer_sets(solution_data)
        # Ensure success flag for normal results to prevent server from overriding status
        solution_data["success"] = True
        logger.debug(f"Setting _LAST_SOLUTION: {solution_data}")
        _LAST_SOLUTION = solution_data
        return solution_data
    except Exception as e:
        error_solution = format_solution_error(e)
        _LAST_SOLUTION = error_solution
        logger.error(f"Error in export_solution: {e!s}", exc_info=True)
        return error_solution


def _process_input_data(
    data: Any,
    status: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Process input data from various sources into a standardized solution dictionary.
    """
    solution_data: dict[str, Any] = {}
    # Case 1: Direct dictionary data
    if isinstance(data, dict):
        solution_data = data.copy()
    # Case 2: List of answer sets (each a list of atoms)
    elif isinstance(data, list) and (not data or isinstance(data[0], list)):
        solution_data["answer_sets"] = data
        solution_data["satisfiable"] = bool(data)
    # Case 3: None or unknown type
    elif data is None:
        solution_data["answer_sets"] = []
        solution_data["satisfiable"] = False
    else:
        raise ASPError(f"Unsupported data type for ASP solution: {type(data).__name__}")
    # Set the status field
    if status:
        solution_data["status"] = status
    else:
        solution_data["status"] = (
            "sat" if solution_data.get("satisfiable", False) else "unsat"
        )
    # Add warnings if provided
    if warnings:
        solution_data["warnings"] = warnings
    # Ensure values dictionary exists (may be populated later)
    solution_data["values"] = solution_data.get("values", {})
    return solution_data


def _extract_values_from_answer_sets(solution_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract values from answer sets into a flat values dictionary.
    """
    answer_sets = solution_data.get("answer_sets", [])
    values: dict[str, Any] = {}
    # For each atom in any answer set, set True
    for answer_set in answer_sets:
        for atom in answer_set:
            if atom not in values:
                values[atom] = True
    solution_data["values"] = values
    return solution_data
