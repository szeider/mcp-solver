"""
Cardinality constraint templates for MaxSAT.

This module provides basic cardinality constraint helpers for MaxSAT formulas.
These are the essential functions needed for most optimization problems.
"""

import itertools

from pysat.formula import WCNF


def at_most_k(wcnf: WCNF, variables: list[int], k: int) -> None:
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


def at_least_k(wcnf: WCNF, variables: list[int], k: int) -> None:
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


def exactly_k(wcnf: WCNF, variables: list[int], k: int) -> None:
    """
    Add hard constraints that exactly k variables must be true.

    This combines at_least_k and at_most_k constraints.

    Args:
        wcnf: WCNF formula to modify
        variables: List of variable IDs
        k: Exact number of variables that must be true
    """
    at_least_k(wcnf, variables, k)
    at_most_k(wcnf, variables, k)
