"""
PySAT error handling module.

This module provides enhanced error handling for PySAT operations, including:
- Function wrappers that capture and translate PySAT exceptions
- Context-aware error messages that help users debug problems
- Structured error reporting for better user experience
"""

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar


# Setup logger
logger = logging.getLogger(__name__)

# Type variable for generic function decorator
T = TypeVar("T")

# Map of common PySAT exceptions to user-friendly messages
EXCEPTION_MESSAGES = {
    "TypeError": {
        "object of type 'NoneType' has no len()": "Formula appears to be empty or not properly initialized",
        "append_formula() missing 1 required positional argument: 'formula'": "No formula was provided to the solver",
        "vars() got an unexpected keyword argument": "Incorrect variable definition. Variables must be positive integers",
        "'int' object is not iterable": "A clause must be a list of integers, not a single integer",
    },
    "ValueError": {
        "variable identifier should be a non-zero integer": "Variable IDs must be non-zero integers. Check your variable mapping",
        "literal should be a non-zero integer": "Clause literals must be non-zero integers. Zero is not allowed in clauses",
        "unexpected literal value": "Invalid literal value in clause. Literals must be integers",
    },
    "RuntimeError": {
        "solver is not initialized": "Solver was not properly initialized before use",
        "solver is in an invalid state": "Solver has been corrupted or used after being deleted",
    },
    "AttributeError": {
        "'NoneType' object has no attribute": "Attempted to use a solver or formula that doesn't exist",
        "object has no attribute 'append_formula'": "The solver object doesn't support append_formula. Make sure you're using a compatible solver",
    },
}


class PySATError(Exception):
    """Custom exception class for enhanced PySAT errors."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize PySAT error with enhanced context.

        Args:
            message: User-friendly error message
            original_error: The original exception that was caught
            context: Additional context about the error (e.g., formula size, variable range)
        """
        self.original_error = original_error
        self.context = context or {}
        self.original_traceback = traceback.format_exc()

        # Build enhanced message with context
        enhanced_message = message
        if context:
            enhanced_message += "\n\nContext:"
            for key, value in context.items():
                enhanced_message += f"\n- {key}: {value}"

        if original_error:
            error_type = type(original_error).__name__
            error_msg = str(original_error)
            enhanced_message += f"\n\nOriginal error ({error_type}): {error_msg}"

        super().__init__(enhanced_message)


def pysat_error_handler(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle PySAT exceptions with enhanced error messages.

    Args:
        func: The function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Try to find a matching error message pattern
            friendly_message = None
            if error_type in EXCEPTION_MESSAGES:
                for pattern, message in EXCEPTION_MESSAGES[error_type].items():
                    if pattern in error_msg:
                        friendly_message = message
                        break

            # If no specific message found, use a generic one
            if not friendly_message:
                friendly_message = f"Error in PySAT operation: {error_msg}"

            # Gather context information
            context = {}
            try:
                # Get formula/solver context if available in args
                for arg in args:
                    if hasattr(arg, "nof_clauses"):
                        context["clauses"] = arg.nof_clauses()
                    if hasattr(arg, "nof_vars"):
                        context["variables"] = arg.nof_vars()
                    if hasattr(arg, "clauses") and isinstance(arg.clauses, list):
                        context["clause_count"] = len(arg.clauses)
                        if arg.clauses:
                            context["sample_clause"] = str(arg.clauses[0])
            except Exception:
                # Don't fail during error handling
                pass

            # Log the error with full traceback for debugging
            logger.error(
                f"PySAT error in {func.__name__}: {error_type}: {error_msg}",
                exc_info=True,
            )

            # Raise our enhanced error
            raise PySATError(friendly_message, original_error=e, context=context) from e

    return wrapper


def validate_variables(variables: dict[str, int]) -> list[str]:
    """
    Validate that all variables have proper types and values.

    Args:
        variables: Dictionary mapping variable names to IDs

    Returns:
        List of error messages (empty if no errors)
    """
    errors = []

    if not variables:
        errors.append("Variable dictionary is empty")
        return errors

    for key, var_id in variables.items():
        if not isinstance(var_id, int):
            errors.append(
                f"Variable '{key}' has non-integer ID: {var_id} "
                f"(type: {type(var_id).__name__})"
            )
        elif var_id <= 0:
            errors.append(f"Variable '{key}' has non-positive ID: {var_id}")

    # Check for duplicate IDs
    id_to_keys: dict[int, list[str]] = {}
    for key, var_id in variables.items():
        if var_id not in id_to_keys:
            id_to_keys[var_id] = []
        id_to_keys[var_id].append(key)

    for var_id, keys in id_to_keys.items():
        if len(keys) > 1:
            errors.append(
                f"Multiple variables ({', '.join(keys)}) share the same ID: {var_id}"
            )

    return errors


def validate_formula(formula: Any) -> list[str]:
    """
    Check formula structure for potential issues.

    Args:
        formula: PySAT CNF formula object

    Returns:
        List of error messages (empty if no errors)
    """
    errors = []

    if not hasattr(formula, "clauses"):
        errors.append("Object does not appear to be a valid PySAT formula")
        return errors

    if not formula.clauses:
        errors.append("Formula has no clauses")
        return errors

    max_var = 0
    for i, clause in enumerate(formula.clauses):
        if not clause:
            errors.append(f"Clause {i + 1} is empty")
            continue

        for j, lit in enumerate(clause):
            if not isinstance(lit, int):
                errors.append(
                    f"Clause {i + 1}, literal {j + 1} is not an integer: "
                    f"{lit} (type: {type(lit).__name__})"
                )
            elif lit == 0:
                errors.append(
                    f"Clause {i + 1}, literal {j + 1} is zero (not allowed in PySAT)"
                )
            else:
                max_var = max(max_var, abs(lit))

    # Check if variables seem consistent
    if hasattr(formula, "nof_vars") and callable(formula.nof_vars):
        reported_vars = formula.nof_vars()
        if reported_vars < max_var:
            errors.append(
                f"Formula reports {reported_vars} variables but clauses "
                f"reference variable {max_var}"
            )

    return errors


def format_solution_error(error: Exception) -> dict[str, Any]:
    """
    Format an error for inclusion in the solution.

    Args:
        error: The exception to format

    Returns:
        Dictionary with error information
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # If it's our enhanced error, use its information
    if isinstance(error, PySATError):
        result = {
            "satisfiable": False,
            "error_type": error_type,
            "error_message": error_msg,
        }

        # Include context if available
        if error.context:
            result["error_context"] = error.context

        return result

    # For other errors, create a more basic structure
    return {
        "satisfiable": False,
        "error_type": error_type,
        "error_message": error_msg,
    }
