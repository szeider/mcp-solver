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

# Export global MaxSAT functionality
from .global_maxsat import (
    initialize_maxsat, add_hard_clause, add_soft_clause, 
    solve_maxsat, get_current_wcnf,
    add_at_most_k_soft, add_at_least_k_soft, add_exactly_k_soft,
    clear_maxsat
)

__all__ = [
    "at_most_k", "at_least_k", "exactly_k",
    "at_most_one", "exactly_one",
    "implies", "mutually_exclusive", "if_then_else",
    # Global MaxSAT functionality
    "initialize_maxsat", "add_hard_clause", "add_soft_clause", 
    "solve_maxsat", "get_current_wcnf",
    "add_at_most_k_soft", "add_at_least_k_soft", "add_exactly_k_soft",
    "clear_maxsat"
] 