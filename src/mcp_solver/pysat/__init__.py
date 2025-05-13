"""
PySAT solver package for MCP (Model Context Protocol).

This package provides a PySAT backend for constraint solving via MCP.
"""

__version__ = "2.3.0"

from .model_manager import PySATModelManager
from .solution import export_solution
from .templates.mapping import VariableMap

# Export error handling utilities for easier access
from .error_handling import (
    PySATError,
    pysat_error_handler,
    validate_variables,
    validate_formula,
    format_solution_error,
)

# Include common cardinality constraints from templates
from .templates.cardinality_templates import at_most_k, at_least_k, exactly_k

# Include additional constraints from constraints.py
from .constraints import at_most_one, exactly_one

__all__ = [
    "PySATModelManager",
    "export_solution",
    "VariableMap",
    # Error handling
    "PySATError",
    "pysat_error_handler",
    "validate_variables",
    "validate_formula",
    "format_solution_error",
    # Constraints
    "at_most_k",
    "at_least_k",
    "exactly_k",
    "at_most_one",
    "exactly_one",
]
