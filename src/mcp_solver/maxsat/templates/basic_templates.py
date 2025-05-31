"""
Basic templates for MaxSAT.

This module provides template functions for common MaxSAT patterns,
making it easier to build and solve MaxSAT problems.
"""

import sys
from typing import Any, Dict, List, Optional, Union

# Import PySAT but protect against failure
try:
    from pysat.formula import CNF, WCNF
    from pysat.examples.rc2 import RC2
    from pysat.solvers import Cadical153, Solver
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import from our solution module
from mcp_solver.maxsat.solution import export_maxsat_solution


def create_maxsat_solver(wcnf: WCNF) -> RC2:
    """
    Create an RC2 MaxSAT solver for the given WCNF formula.
    
    Args:
        wcnf: WCNF formula with hard and soft constraints
        
    Returns:
        RC2 solver instance
    """
    return RC2(wcnf)


def solve_maxsat_problem(
    wcnf: WCNF, 
    var_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Solve a MaxSAT problem and return the result.
    
    This function solves a MaxSAT problem and exports the result
    using the export_maxsat_solution function.
    
    Args:
        wcnf: WCNF formula with hard and soft constraints
        var_mapping: Optional mapping from variable names to IDs
        
    Returns:
        Dictionary with the MaxSAT solution
    """
    with RC2(wcnf) as solver:
        model = solver.compute()
        
        if model is not None:
            # Use the standard export function to handle the RC2 result
            return export_maxsat_solution(solver, var_mapping)
        else:
            # Problem is unsatisfiable
            return export_maxsat_solution({
                "satisfiable": False,
                "status": "unsatisfiable",
                "message": "No solution exists that satisfies all hard constraints"
            })


def add_hard_constraint(wcnf: WCNF, literals: List[int]) -> None:
    """
    Add a hard constraint (clause) to the WCNF formula.
    
    Args:
        wcnf: WCNF formula to modify
        literals: List of literals forming the clause
    """
    wcnf.append(literals)


def add_soft_constraint(wcnf: WCNF, literals: List[int], weight: int) -> None:
    """
    Add a soft constraint (weighted clause) to the WCNF formula.
    
    Args:
        wcnf: WCNF formula to modify
        literals: List of literals forming the clause
        weight: Weight of the soft constraint (positive integer)
    """
    wcnf.append(literals, weight=weight)


def encode_binary_variable(
    wcnf: WCNF, 
    var_id: int, 
    soft_weight: Optional[int] = None, 
    preferred_value: bool = True
) -> None:
    """
    Encode a binary variable with an optional soft constraint.
    
    This function adds a soft constraint to prefer the variable
    to be either true or false.
    
    Args:
        wcnf: WCNF formula to modify
        var_id: Variable ID
        soft_weight: Weight of the soft constraint (None for no preference)
        preferred_value: Preferred value for the variable (True or False)
    """
    if soft_weight is not None and soft_weight > 0:
        if preferred_value:
            wcnf.append([var_id], weight=soft_weight)
        else:
            wcnf.append([-var_id], weight=soft_weight)


def encode_mutual_exclusion(wcnf: WCNF, var_ids: List[int], hard: bool = True) -> None:
    """
    Encode a mutual exclusion constraint.
    
    This adds constraints ensuring that at most one of the given variables is true.
    
    Args:
        wcnf: WCNF formula to modify
        var_ids: List of variable IDs
        hard: Whether this is a hard constraint (True) or soft constraint (False)
    """
    # For each pair of variables, add a constraint that they can't both be true
    for i in range(len(var_ids)):
        for j in range(i+1, len(var_ids)):
            if hard:
                wcnf.append([-var_ids[i], -var_ids[j]])
            else:
                # Use weight 1 for soft constraint
                wcnf.append([-var_ids[i], -var_ids[j]], weight=1)


def encode_dependency(
    wcnf: WCNF, 
    var_id: int, 
    depends_on: Union[int, List[int]], 
    hard: bool = True
) -> None:
    """
    Encode a dependency constraint.
    
    This adds constraints ensuring that if var_id is true, then depends_on must also be true.
    If depends_on is a list, it ensures that if var_id is true, then at least one of 
    the depends_on variables must be true.
    
    Args:
        wcnf: WCNF formula to modify
        var_id: Variable ID that depends on other variable(s)
        depends_on: Variable ID or list of variable IDs that var_id depends on
        hard: Whether this is a hard constraint (True) or soft constraint (False)
    """
    if isinstance(depends_on, int):
        # Single dependency: var_id → depends_on
        # Equivalent to: ¬var_id ∨ depends_on
        if hard:
            wcnf.append([-var_id, depends_on])
        else:
            # Use weight 1 for soft constraint
            wcnf.append([-var_id, depends_on], weight=1)
    else:
        # Multiple dependencies: var_id → (depends_on[0] ∨ depends_on[1] ∨ ...)
        # Equivalent to: ¬var_id ∨ depends_on[0] ∨ depends_on[1] ∨ ...
        if hard:
            wcnf.append([-var_id] + list(depends_on))
        else:
            # Use weight 1 for soft constraint
            wcnf.append([-var_id] + list(depends_on), weight=1)