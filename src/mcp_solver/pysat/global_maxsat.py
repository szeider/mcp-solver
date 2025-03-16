"""
Global MaxSAT functionality for PySAT integration.

This module provides a global state approach to MaxSAT problem solving
with clearer APIs and better support for weighted constraints.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import itertools

# Import PySAT
try:
    from pysat.formula import CNF, WCNF
    from pysat.examples.rc2 import RC2
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import templates for internal use
from .templates.maxsat_templates import weighted_maxsat_template, maxsat_solver
from .templates.cardinality_templates import at_most_k, at_least_k

# Global WCNF object for direct manipulation
_CURRENT_WCNF = None

def initialize_maxsat():
    """
    Initialize a new MaxSAT problem.
    
    Returns:
        WCNF object ready for constraints
    """
    global _CURRENT_WCNF
    _CURRENT_WCNF = WCNF()
    return _CURRENT_WCNF

def add_hard_clause(literals: List[int]):
    """
    Add a hard clause to the current MaxSAT problem.
    
    Args:
        literals: List of literals in the clause
        
    Returns:
        True if successful, False otherwise
    """
    global _CURRENT_WCNF
    if _CURRENT_WCNF is None:
        _CURRENT_WCNF = initialize_maxsat()
    
    _CURRENT_WCNF.append(literals)
    return True

def add_soft_clause(literals: List[int], weight: int = 1):
    """
    Add a soft clause with the specified weight to the current MaxSAT problem.
    
    Args:
        literals: List of literals in the clause
        weight: Weight of the clause (default: 1)
        
    Returns:
        True if successful, False otherwise
    """
    global _CURRENT_WCNF
    if _CURRENT_WCNF is None:
        _CURRENT_WCNF = initialize_maxsat()
    
    _CURRENT_WCNF.append(literals, weight=weight)
    return True

def get_current_wcnf():
    """
    Get the current WCNF object.
    
    Returns:
        Current WCNF object or None if not initialized
    """
    global _CURRENT_WCNF
    return _CURRENT_WCNF

def solve_maxsat(timeout: Optional[float] = None) -> Tuple[Optional[List[int]], int]:
    """
    Solve the current MaxSAT problem.
    
    Args:
        timeout: Maximum time (in seconds) to spend solving
        
    Returns:
        Tuple of (model, cost) - model may be None if problem is unsatisfiable
    """
    global _CURRENT_WCNF
    if _CURRENT_WCNF is None:
        raise ValueError("No MaxSAT problem initialized. Call initialize_maxsat() first.")
    
    try:
        with RC2(_CURRENT_WCNF) as rc2:
            # Handle timeout in a more compatible way across PySAT versions
            if timeout is not None and hasattr(rc2, "set_timeout"):
                rc2.set_timeout(timeout)
            
            # Compute the solution
            model = rc2.compute()
            cost = rc2.cost if hasattr(rc2, "cost") else 0
            
            return model, cost
    except Exception as e:
        print(f"Error solving MaxSAT problem: {e}")
        return None, 0

# Higher level functions for common MaxSAT patterns

def add_at_most_k_soft(variables: List[int], k: int, weight: int = 1):
    """
    Add a soft constraint that at most k variables can be true.
    
    Args:
        variables: List of variables
        k: Maximum number of variables that can be true
        weight: Weight of the constraint
    """
    # We add multiple soft clauses, one for each combination of k+1 variables
    for combo in itertools.combinations(variables, k+1):
        add_soft_clause([-v for v in combo], weight=weight)

def add_at_least_k_soft(variables: List[int], k: int, weight: int = 1):
    """
    Add a soft constraint that at least k variables must be true.
    
    Args:
        variables: List of variables
        k: Minimum number of variables that must be true
        weight: Weight of the constraint
    """
    # We use De Morgan's law: at least k of variables = at most (n-k) of negated variables
    neg_vars = [-v for v in variables]
    add_at_most_k_soft(neg_vars, len(variables) - k, weight=weight)

def add_exactly_k_soft(variables: List[int], k: int, weight: int = 1):
    """
    Add a soft constraint that exactly k variables must be true.
    
    Args:
        variables: List of variables
        k: Number of variables that must be true
        weight: Weight of the constraint
    """
    # We add both at most k and at least k with the same weight
    add_at_most_k_soft(variables, k, weight=weight)
    add_at_least_k_soft(variables, k, weight=weight)

def clear_maxsat():
    """
    Clear the current MaxSAT problem.
    
    Returns:
        True if successful
    """
    global _CURRENT_WCNF
    _CURRENT_WCNF = None
    return True

def university_course_scheduling_example():
    """
    Example of using the global MaxSAT functionality for university course scheduling.
    
    Returns:
        Dictionary with solving results
    """
    # Initialize a new MaxSAT problem
    initialize_maxsat()
    
    # We'll use 4 courses and 3 time slots
    # Variables: course c in time slot t = 3*(c-1) + t
    
    # Hard constraints
    # Each course must be scheduled exactly once
    
    # At least one time slot per course
    for c in range(1, 5):
        base = 3 * (c - 1)
        add_hard_clause([base + 1, base + 2, base + 3])
    
    # At most one time slot per course
    for c in range(1, 5):
        base = 3 * (c - 1)
        for t1 in range(1, 3):
            for t2 in range(t1 + 1, 4):
                add_hard_clause([-(base + t1), -(base + t2)])
    
    # Soft constraints (preferences)
    # Professor A prefers not to teach course 1 in time slot 2
    add_soft_clause([-2], weight=2)
    
    # Professor B prefers teaching course 2 in time slot 3
    add_soft_clause([6], weight=3)
    
    # Solve the problem
    model, cost = solve_maxsat(timeout=5.0)
    
    # Create variable mapping for solution display
    variables = {}
    for c in range(1, 5):
        for t in range(1, 4):
            var_id = 3 * (c - 1) + t
            variables[f"course_{c}_time_{t}"] = var_id
    
    # Extract solution
    solution = {var_name: var_id in model for var_name, var_id in variables.items()} if model else {}
    
    return {
        "satisfiable": model is not None,
        "model": model,
        "cost": cost,
        "solution": solution
    } 