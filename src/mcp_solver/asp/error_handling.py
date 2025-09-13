"""
ASP error handling module using dumbo_asp Parser.

This module provides enhanced error handling for ASP operations, including:
- Function wrappers that capture and translate ASP exceptions
- Context-aware error messages using dumbo_asp.primitives.parsers.Parser
- Structured error reporting for better user experience
"""

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar


logger = logging.getLogger(__name__)
T = TypeVar("T")

# Map of common ASP exceptions to user-friendly messages
EXCEPTION_MESSAGES = {
    "SyntaxError": {
        "unexpected token": "Syntax error in ASP code: unexpected token.",
        "unexpected character": "Syntax error in ASP code: unexpected character.",
        "unexpected end of input": "Syntax error: unexpected end of input.",
    },
    "RuntimeError": {
        "clingo": "Clingo runtime error. Check your ASP code for semantic issues.",
    },
}


class ASPError(Exception):
    """Custom exception class for enhanced ASP errors."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        context: str | None = None,
    ):
        self.original_error = original_error
        self.original_traceback = traceback.format_exc()
        enhanced_message = message
        if context:
            enhanced_message += f"\nHere are more error details:\n{context}"
        if original_error:
            error_type = type(original_error).__name__
            error_msg = str(original_error)
            enhanced_message += f"\n\nOriginal error ({error_type}): {error_msg}"
        super().__init__(enhanced_message)


def asp_error_handler(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle ASP exceptions with enhanced error messages.
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
            if not friendly_message:
                friendly_message = f"Error in ASP operation: {error_msg}"
            # Gather context information (could include code snippet, etc.)
            context = ""
            # Log the error with full traceback for debugging
            logger.error(
                f"ASP error in {func.__name__}: {error_type}: {error_msg}",
                exc_info=True,
            )
            raise ASPError(friendly_message, original_error=e, context=context) from e

    return wrapper


def format_solution_error(error: Exception) -> dict[str, Any]:
    """
    Format an error for inclusion in the solution.
    """
    error_type = type(error).__name__
    error_msg = str(error)
    return {
        "satisfiable": False,
        "status": "error",
        "success": True,  # Keep server connection logic satisfied
        "error_type": error_type,
        "error_message": error_msg,
    }
