"""
Basic templates for PySAT.

This module provides template functions for common PySAT patterns related to
basic SAT solving.
"""

import sys
from typing import Any


# Import PySAT but protect against failure
try:
    from pysat.formula import CNF
    from pysat.solvers import Cadical153, Glucose3, Glucose4, Lingeling, Solver
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)


def basic_sat_solver(
    clauses: list[list[int]], solver_type=Cadical153
) -> dict[str, Any]:
    """
    Basic SAT solving template.

    Args:
        clauses: List of clauses (each clause is a list of integers)
        solver_type: PySAT solver class to use (default: Cadical153)

    Returns:
        Dictionary with results
    """
    # Create CNF formula
    formula = CNF()
    for clause in clauses:
        formula.append(clause)

    # Create solver and add formula
    solver = solver_type()
    solver.append_formula(formula)

    # Solve
    is_sat = solver.solve()
    model = solver.get_model() if is_sat else None

    # Create result
    result = {
        "is_sat": is_sat,
        "model": model,
        "solver": solver,  # Return solver for cleanup
    }

    return result


def dimacs_to_cnf(dimacs_str: str) -> CNF:
    """
    Convert DIMACS format string to CNF.

    Args:
        dimacs_str: String in DIMACS format

    Returns:
        CNF object
    """
    formula = CNF()
    lines = dimacs_str.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Skip comments and problem line
        if line.startswith("c") or line.startswith("p"):
            continue

        # Parse clause
        if line and not line.startswith("%"):
            clause = [int(lit) for lit in line.split() if lit != "0"]
            if clause:  # Non-empty clause
                formula.append(clause)

    return formula


def sat_to_binary_variables(
    variables: dict[str, int], model: list[int]
) -> dict[str, bool]:
    """
    Convert SAT model to binary variables.

    Args:
        variables: Dictionary mapping variable names to their IDs
        model: SAT model (list of integers)

    Returns:
        Dictionary mapping variable names to boolean values
    """
    result = {}

    for var_name, var_id in variables.items():
        # Handle positive and negative variable IDs
        var_id_abs = abs(var_id)
        var_value = var_id_abs in model
        if var_id < 0:  # If variable ID is negative, negate the value
            var_value = not var_value
        result[var_name] = var_value

    return result
