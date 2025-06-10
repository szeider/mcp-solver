"""
PySAT constraint helper functions.

This module provides helper functions for creating common cardinality constraints
with PySAT in a way that's reliable and always works, regardless of the constraint type.
It also includes helper functions for working with MaxSAT soft constraints.
"""

import itertools

# Import the robust implementations from templates
from .templates.cardinality_templates import (
    at_least_k as _at_least_k_robust,
    at_most_k as _at_most_k_robust,
)


def at_most_k(variables: list[int], k: int) -> list[list[int]]:
    """
    Create clauses for at most k of variables being true.
    Works for any k value, with optimization for k=1 case.

    Args:
        variables: List of variable IDs
        k: Upper bound (any non-negative integer)

    Returns:
        List of clauses representing the constraint
    """
    # Delegate to the robust implementation in templates
    return _at_most_k_robust(variables, k)


def at_least_k(variables: list[int], k: int) -> list[list[int]]:
    """
    Create clauses for at least k of variables being true.
    Uses De Morgan's law for efficiency.

    Args:
        variables: List of variable IDs
        k: Lower bound (any non-negative integer)

    Returns:
        List of clauses representing the constraint
    """
    # Delegate to the robust implementation in templates
    return _at_least_k_robust(variables, k)


def exactly_k(variables: list[int], k: int) -> list[list[int]]:
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
            for j in range(i + 1, len(variables)):
                at_most.append([-variables[i], -variables[j]])

        return at_least + at_most

    # General case
    return at_most_k(variables, k) + at_least_k(variables, k)


def at_most_one(variables: list[int]) -> list[list[int]]:
    """
    Optimized function for the common at-most-one constraint.
    Uses pairwise encoding for better performance.

    Args:
        variables: List of variable IDs

    Returns:
        List of clauses representing the constraint
    """
    return at_most_k(variables, 1)


def exactly_one(variables: list[int]) -> list[list[int]]:
    """
    Optimized function for the common exactly-one constraint.
    Uses efficient encoding with pairwise constraints.

    Args:
        variables: List of variable IDs

    Returns:
        List of clauses representing the constraint
    """
    return exactly_k(variables, 1)


def implies(a: int, b: int) -> list[list[int]]:
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


def mutually_exclusive(variables: list[int]) -> list[list[int]]:
    """
    Create clauses ensuring that at most one of the variables is true.
    This is equivalent to at_most_one but renamed for clarity in models.

    Args:
        variables: List of variable IDs

    Returns:
        List of clauses representing mutual exclusion
    """
    return at_most_one(variables)


def if_then_else(condition: int, then_var: int, else_var: int) -> list[list[int]]:
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
        [-condition, then_var],  # condition -> then_var
        [condition, else_var],  # !condition -> else_var
    ]


# MaxSAT helper functions


def soft_clause(literals: int | list[int], weight: int = 1) -> tuple[list[int], int]:
    """
    Create a soft clause with a given weight for MaxSAT formulas.

    In MaxSAT, a soft clause is a constraint that we want to satisfy
    but can be violated at a cost (the weight).

    Args:
        literals: One or more literals (variable IDs) to include in the soft clause
        weight: The weight of the clause (cost if violated)

    Returns:
        A tuple of (clause, weight) for adding to a WCNF formula
    """
    if isinstance(literals, int):
        literals = [literals]
    return (literals, weight)


def soft_at_most_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """
    Create weighted soft at-most-k constraints for MaxSAT.

    Args:
        variables: List of variable IDs
        k: Maximum number of variables that should be true
        weight: Weight of the soft constraint

    Returns:
        List of (clause, weight) pairs for adding to a WCNF formula
    """
    if k >= len(variables):
        return []  # Constraint is always satisfied

    soft_clauses = []

    # Generate all combinations of k+1 variables
    for combo in itertools.combinations(variables, k + 1):
        # At least one variable in the combination should be false
        soft_clauses.append(([-(var) for var in combo], weight))

    return soft_clauses


def soft_at_least_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """
    Create weighted soft at-least-k constraints for MaxSAT.

    Args:
        variables: List of variable IDs
        k: Minimum number of variables that should be true
        weight: Weight of the soft constraint

    Returns:
        List of (clause, weight) pairs for adding to a WCNF formula
    """
    if k <= 0:
        return []  # Constraint is always satisfied

    soft_clauses = []

    # Generate all combinations of n-k+1 negative literals
    negated_vars = [-var for var in variables]
    for combo in itertools.combinations(negated_vars, len(variables) - k + 1):
        # At least one variable in the combination should be true
        soft_clauses.append(([-lit for lit in combo], weight))

    return soft_clauses


def soft_exactly_k(
    variables: list[int], k: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """
    Create weighted soft exactly-k constraints for MaxSAT.

    Args:
        variables: List of variable IDs
        k: Exact number of variables that should be true
        weight: Weight of the soft constraint

    Returns:
        List of (clause, weight) pairs for adding to a WCNF formula
    """
    return soft_at_most_k(variables, k, weight) + soft_at_least_k(variables, k, weight)


def soft_implies(a: int, b: int, weight: int = 1) -> list[tuple[list[int], int]]:
    """
    Create a weighted soft implication constraint (a -> b) for MaxSAT.

    Args:
        a: The antecedent variable ID
        b: The consequent variable ID
        weight: Weight of the soft constraint

    Returns:
        List containing one (clause, weight) pair for adding to a WCNF formula
    """
    return [([-a, b], weight)]


def soft_if_then_else(
    condition: int, then_var: int, else_var: int, weight: int = 1
) -> list[tuple[list[int], int]]:
    """
    Create weighted soft if-then-else constraints for MaxSAT.

    Args:
        condition: The condition variable ID
        then_var: The 'then' variable ID
        else_var: The 'else' variable ID
        weight: Weight of the soft constraint

    Returns:
        List of (clause, weight) pairs for adding to a WCNF formula
    """
    return [
        ([-condition, then_var], weight),  # condition -> then_var
        ([condition, else_var], weight),  # !condition -> else_var
    ]


def add_soft_clauses_to_wcnf(
    wcnf: "WCNF", clauses: list[tuple[list[int], int]]
) -> None:
    """
    Add a list of soft clauses to a WCNF formula.

    Args:
        wcnf: The WCNF formula to update
        clauses: List of (clause, weight) pairs to add

    Returns:
        None (modifies the wcnf in-place)
    """
    for clause, weight in clauses:
        wcnf.append(clause, weight=weight)
