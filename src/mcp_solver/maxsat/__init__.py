"""MaxSAT Mode for MCP Solver.

This module provides optimization capabilities using MaxSAT.
"""

from .model_manager import MaxSATModelManager
from .solution import export_solution, export_maxsat_solution


__all__ = [
    "MaxSATModelManager",
    "export_solution",
    "export_maxsat_solution",  # Keep for backward compatibility
]
