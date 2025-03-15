"""
PySAT constraint helper functions.

This module provides helper functions for creating common cardinality constraints
with PySAT in a way that's reliable and always works, regardless of the constraint type.
"""

import itertools
from typing import List, Union, Iterable


def at_most_k(variables: List[int], k: int) -> List[List[int]]:
    """
    Create clauses for at most k of variables being true.
    Works for any k value, with optimization for k=1 case.
    
    Args:
        variables: List of variable IDs
        k: Upper bound (any non-negative integer)
    
    Returns:
        List of clauses representing the constraint
    """
    if k >= len(variables):
        return []  # No constraint needed
    
    if k == 0:
        # All variables must be false
        return [[-v] for v in variables]
    
    if k == 1:
        # Optimization for the common at-most-one case
        # This is the pairwise encoding which is most efficient for k=1
        clauses = []
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                clauses.append([-variables[i], -variables[j]])
        return clauses
    
    # General case for k > 1
    clauses = []
    for combo in itertools.combinations(variables, k+1):
        clauses.append([-v for v in combo])
    
    return clauses


def at_least_k(variables: List[int], k: int) -> List[List[int]]:
    """
    Create clauses for at least k of variables being true.
    Works for any k value, with optimization for k=1 case.
    
    Args:
        variables: List of variable IDs
        k: Lower bound (any non-negative integer)
    
    Returns:
        List of clauses representing the constraint
    """
    if k <= 0:
        return []  # No constraint needed
    
    if k == len(variables):
        # All variables must be true
        return [[v] for v in variables]
    
    if k == 1:
        # At least one variable must be true
        # This is a single clause with all variables
        return [variables.copy()]
    
    # Use De Morgan's law for general case
    neg_vars = [-v for v in variables]
    return at_most_k(neg_vars, len(variables) - k)


def exactly_k(variables: List[int], k: int) -> List[List[int]]:
    """
    Create clauses for exactly k of variables being true.
    Works for any k value, with optimization for k=1 case.
    
    Args:
        variables: List of variable IDs
        k: Target value (any non-negative integer)
    
    Returns:
        List of clauses representing the constraint
    """
    if k > len(variables) or k < 0:
        return [[]]  # Unsatisfiable
    
    if k == 0:
        # All variables must be false
        return [[-v] for v in variables]
    
    if k == 1:
        # Exactly one variable is true
        # At least one is true
        at_least = [variables.copy()]
        
        # At most one is true (pairwise encoding)
        at_most = []
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                at_most.append([-variables[i], -variables[j]])
        
        return at_least + at_most
    
    # General case
    return at_most_k(variables, k) + at_least_k(variables, k)


def at_most_one(variables: List[int]) -> List[List[int]]:
    """
    Optimized function for the common at-most-one constraint.
    Uses pairwise encoding for better performance.
    
    Args:
        variables: List of variable IDs
    
    Returns:
        List of clauses representing the constraint
    """
    return at_most_k(variables, 1)


def exactly_one(variables: List[int]) -> List[List[int]]:
    """
    Optimized function for the common exactly-one constraint.
    Uses efficient encoding with pairwise constraints.
    
    Args:
        variables: List of variable IDs
    
    Returns:
        List of clauses representing the constraint
    """
    return exactly_k(variables, 1)


def implies(a: int, b: int) -> List[List[int]]:
    """
    Create a clause for the implication a -> b (if a then b).
    Equivalent to (!a OR b).
    
    Args:
        a: The antecedent variable ID
        b: The consequent variable ID
        
    Returns:
        A list containing one clause that represents the implication
    """
    return [[-a, b]]


def mutually_exclusive(variables: List[int]) -> List[List[int]]:
    """
    Create clauses ensuring that at most one of the variables is true.
    This is equivalent to at_most_one but renamed for clarity in models.
    
    Args:
        variables: List of variable IDs
    
    Returns:
        List of clauses representing mutual exclusion
    """
    return at_most_one(variables)


def if_then_else(condition: int, then_var: int, else_var: int) -> List[List[int]]:
    """
    Create clauses for an if-then-else construct.
    
    Args:
        condition: The condition variable ID
        then_var: The 'then' variable ID
        else_var: The 'else' variable ID
        
    Returns:
        List of clauses representing if-then-else
    """
    # If condition, then then_var
    # If not condition, then else_var
    return [
        [-condition, then_var],   # condition -> then_var
        [condition, else_var]     # !condition -> else_var
    ] 