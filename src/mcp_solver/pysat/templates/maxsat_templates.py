"""
MaxSAT templates for PySAT.

This module provides template functions for common PySAT patterns related to
MaxSAT problems.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

# Import PySAT but protect against failure
try:
    from pysat.formula import CNF, WCNF
    from pysat.card import CardEnc, EncType
    from pysat.examples.rc2 import RC2
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

def weighted_maxsat_template(hard_clauses: List[List[int]], soft_clauses: List[Tuple[List[int], int]]) -> Dict[str, Any]:
    """
    Template for weighted MaxSAT problems.
    
    Args:
        hard_clauses: List of hard clauses (must be satisfied)
        soft_clauses: List of (clause, weight) tuples
        
    Returns:
        Dictionary with WCNF formula and other information
    """
    # Create WCNF formula
    wcnf = WCNF()
    
    # Add hard constraints
    for clause in hard_clauses:
        wcnf.append(clause)
    
    # Add soft constraints with weights
    for clause, weight in soft_clauses:
        wcnf.append(clause, weight=weight)
    
    return {"wcnf": wcnf}

def cardinality_maxsat_template(variables: List[int], weights: List[int], at_most_k: int,
                             incompatible_pairs: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """
    Template for MaxSAT with cardinality constraints.
    
    This template solves the weighted maximum k-subset problem with optional
    incompatibility constraints.
    
    Args:
        variables: List of variable IDs
        weights: List of weights (same length as variables)
        at_most_k: Maximum number of variables to select
        incompatible_pairs: List of incompatible variable pairs (optional)
        
    Returns:
        Dictionary with WCNF formula and other information
    """
    # Create WCNF formula
    wcnf = WCNF()
    
    # Add cardinality constraint as hard constraint
    card_constraint = CardEnc.atmost(variables, at_most_k, encoding=EncType.seqcounter)
    for clause in card_constraint.clauses:
        wcnf.append(clause)
    
    # Add incompatible pairs as hard constraints
    if incompatible_pairs:
        for var1, var2 in incompatible_pairs:
            # Add constraint: NOT(var1 AND var2)
            wcnf.append([-var1, -var2])
    
    # Add soft constraints for maximizing weight
    for var, weight in zip(variables, weights):
        # We want to maximize the weight, so we add a soft constraint
        # to make each variable true, with the weight as the penalty
        wcnf.append([var], weight=weight)
    
    return {"wcnf": wcnf, "variables": variables, "weights": weights}

def maxsat_solver(wcnf: WCNF) -> Dict[str, Any]:
    """
    Solve a MaxSAT problem.
    
    Args:
        wcnf: WCNF formula
        
    Returns:
        Dictionary with results
    """
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    
    return {
        "model": model,
        "cost": cost,
        "rc2": rc2
    }

def minimum_correction_subset(constraints: List[List[int]]) -> Dict[str, Any]:
    """
    Find minimum subset of constraints to remove to make the system satisfiable.
    
    Args:
        constraints: List of constraints (clauses)
        
    Returns:
        Dictionary with results
    """
    # Create WCNF formula
    wcnf = WCNF()
    
    # Add each constraint as a soft clause with weight 1
    for constraint in constraints:
        wcnf.append(constraint, weight=1)
    
    # Solve using RC2
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    
    # Identify which constraints to remove
    if model is not None:
        removed_constraints = []
        for i, constraint in enumerate(constraints):
            # Check if any of the constraint literals are satisfied
            satisfied = any(lit in model for lit in constraint)
            if not satisfied:
                removed_constraints.append(i)
        
        return {
            "is_sat": True,
            "model": model,
            "cost": cost,
            "removed_constraints": removed_constraints,
            "num_removed": len(removed_constraints)
        }
    else:
        return {
            "is_sat": False,
            "model": None,
            "cost": None,
            "removed_constraints": [],
            "num_removed": 0
        }

def partial_maxsat_template(hard_clauses: List[List[int]], soft_clauses: List[List[int]]) -> Dict[str, Any]:
    """
    Template for partial MaxSAT problems (unit weights).
    
    Args:
        hard_clauses: List of hard clauses (must be satisfied)
        soft_clauses: List of soft clauses (each with weight 1)
        
    Returns:
        Dictionary with WCNF formula and other information
    """
    # Create WCNF formula
    wcnf = WCNF()
    
    # Add hard constraints
    for clause in hard_clauses:
        wcnf.append(clause)
    
    # Add soft constraints with unit weights
    for clause in soft_clauses:
        wcnf.append(clause, weight=1)
    
    return {"wcnf": wcnf} 