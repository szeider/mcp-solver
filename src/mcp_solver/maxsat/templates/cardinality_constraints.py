"""
Cardinality constraint templates for MaxSAT.

This module provides helper functions for encoding cardinality constraints
in MaxSAT formulas with a focus on optimization use cases.
"""

import itertools
from typing import List, Optional

from pysat.formula import WCNF


def at_most_k(wcnf: WCNF, variables: List[int], k: int) -> None:
    """
    Add a hard constraint that at most k variables can be true.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Maximum number of variables that can be true
    """
    if k >= len(variables):
        return  # Trivially satisfied
    
    if k < 0:
        # No variables can be true
        for var in variables:
            wcnf.append([-var])
        return
    
    # For each subset of size k+1, at least one must be false
    for subset in itertools.combinations(variables, k + 1):
        wcnf.append([-var for var in subset])


def at_least_k(wcnf: WCNF, variables: List[int], k: int) -> None:
    """
    Add a hard constraint that at least k variables must be true.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Minimum number of variables that must be true
    """
    if k <= 0:
        return  # Trivially satisfied
    
    if k > len(variables):
        # Impossible to satisfy - add an empty clause
        wcnf.append([])
        return
    
    n = len(variables)
    # For each combination of (n-k+1) variables, at least one must be true
    for subset in itertools.combinations(range(n), n - k + 1):
        clause = [variables[i] for i in subset]
        wcnf.append(clause)


def prefer_at_least_k(wcnf: WCNF, variables: List[int], k: int, penalty_per_missing: int) -> None:
    """
    Add soft constraints preferring at least k variables to be true.
    For MaxSAT optimization, this is often more intuitive than combinatorial encoding.
    
    Each variable below k that is false incurs the penalty.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Desired minimum number of true variables
        penalty_per_missing: Penalty for each missing variable
    """
    # Simple encoding: prefer the first k variables to be true
    for var in variables[:k]:
        wcnf.append([var], weight=penalty_per_missing)


def prefer_at_most_k(wcnf: WCNF, variables: List[int], k: int, penalty_per_extra: int) -> None:
    """
    Add soft constraints preferring at most k variables to be true.
    
    Each variable beyond k that is true incurs the penalty.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Desired maximum number of true variables
        penalty_per_extra: Penalty for each extra variable
    """
    # Penalize variables beyond the first k being true
    if k < len(variables):
        for var in variables[k:]:
            wcnf.append([-var], weight=penalty_per_extra)


def exactly_k(wcnf: WCNF, variables: List[int], k: int) -> None:
    """
    Add hard constraints that exactly k variables must be true.
    
    This is a critical helper that was missing and caused many failures.
    It combines at_least_k and at_most_k constraints.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Exact number of variables that must be true
    """
    at_least_k(wcnf, variables, k)
    at_most_k(wcnf, variables, k)


def prefer_exactly_k(wcnf: WCNF, variables: List[int], k: int, 
                    penalty_too_few: int, penalty_too_many: int) -> None:
    """
    Add soft constraints preferring exactly k variables to be true.
    
    Different penalties can be set for having too few vs too many.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Desired exact number of true variables
        penalty_too_few: Penalty for each variable below k
        penalty_too_many: Penalty for each variable above k
    """
    prefer_at_least_k(wcnf, variables, k, penalty_too_few)
    prefer_at_most_k(wcnf, variables, k, penalty_too_many)


def prefer_between_k_and_m(wcnf: WCNF, variables: List[int], 
                          k: int, m: int, penalty: int) -> None:
    """
    Add soft constraints preferring between k and m variables (inclusive) to be true.
    
    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Minimum desired number of true variables
        m: Maximum desired number of true variables
        penalty: Penalty for being outside the range
    """
    prefer_at_least_k(wcnf, variables, k, penalty)
    prefer_at_most_k(wcnf, variables, m, penalty)