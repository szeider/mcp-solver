"""
PySAT integration for MCP Solver.

This package provides integration with the PySAT library for SAT solving
capabilities within the MCP Solver framework.
"""

from .constraints import (
    at_most_k, at_least_k, exactly_k,
    at_most_one, exactly_one,
    implies, mutually_exclusive, if_then_else
)

# MaxSAT functionality removed

__all__ = [
    "at_most_k", "at_least_k", "exactly_k",
    "at_most_one", "exactly_one",
    "implies", "mutually_exclusive", "if_then_else"
    # MaxSAT functionality removed
] 