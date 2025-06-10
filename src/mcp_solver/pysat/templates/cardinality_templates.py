"""
Cardinality constraint templates for PySAT.

This module provides template functions for common PySAT patterns related to
cardinality constraints.
"""

import sys
from typing import Any


# Import PySAT but protect against failure
try:
    from pysat.card import CardEnc, EncType
    from pysat.formula import CNF
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)


def at_most_k(
    variables: list[int], k: int, encoding: EncType = EncType.seqcounter
) -> list[list[int]]:
    """
    Create clauses ensuring at most k variables can be true.

    This is a robust implementation that handles edge cases better
    and falls back to direct encoding when needed.

    Args:
        variables: List of variable IDs
        k: Maximum number of variables that can be true
        encoding: Encoding type to try (will fall back to direct encoding if needed)

    Returns:
        List of clauses representing the constraint
    """
    # Handle edge cases
    if k >= len(variables):
        return []  # No constraint needed

    # Use direct encoding for small sets (more reliable across versions)
    if len(variables) <= 20:
        clauses = []
        import itertools

        for combo in itertools.combinations(variables, k + 1):
            clauses.append([-v for v in combo])
        return clauses

    # For larger sets, try PySAT's encoding but handle failures
    try:
        return CardEnc.atmost(variables, k, encoding=encoding).clauses
    except Exception:
        # Fall back to direct encoding if PySAT's fails
        # This will be slower but more reliable
        clauses = []
        import itertools

        for combo in itertools.combinations(variables, k + 1):
            clauses.append([-v for v in combo])
        return clauses


def at_least_k(
    variables: list[int], k: int, encoding: EncType = EncType.seqcounter
) -> list[list[int]]:
    """
    Create clauses ensuring at least k variables must be true.

    Uses De Morgan's law with the robust at_most_k implementation.

    Args:
        variables: List of variable IDs
        k: Minimum number of variables that must be true
        encoding: Encoding type to try (will fall back to direct encoding if needed)

    Returns:
        List of clauses representing the constraint
    """
    # We use De Morgan's law: at least k of variables = at most (n-k) of negated variables
    # For small instances, we can directly call at_most_k with negated vars
    if len(variables) <= 100:  # Arbitrary threshold to avoid excessive computation
        neg_vars = [-v for v in variables]
        return at_most_k(neg_vars, len(variables) - k, encoding=encoding)

    # For larger instances, try PySAT's direct implementation
    try:
        return CardEnc.atleast(variables, k, encoding=encoding).clauses
    except Exception:
        # Fall back to at_most_k with De Morgan's law
        neg_vars = [-v for v in variables]
        return at_most_k(neg_vars, len(variables) - k, encoding=encoding)


def exactly_k(
    variables: list[int], k: int, encoding: EncType = EncType.seqcounter
) -> CNF:
    """
    Create CNF formula enforcing exactly k variables are true.

    Args:
        variables: List of variable IDs
        k: Exact number of variables that must be true
        encoding: Encoding type (default: sequential counter)

    Returns:
        CNF formula with the constraint
    """
    return CardEnc.equals(variables, k, encoding=encoding)


def one_hot_encoding(
    variables: list[int], encoding: EncType = EncType.seqcounter
) -> CNF:
    """
    Create CNF formula enforcing exactly one variable is true.

    Args:
        variables: List of variable IDs
        encoding: Encoding type (default: sequential counter)

    Returns:
        CNF formula with the constraint
    """
    return CardEnc.equals(variables, 1, encoding=encoding)


def add_cardinality_constraint(
    formula: CNF,
    constraint_type: str,
    variables: list[int],
    k: int,
    encoding: EncType = EncType.seqcounter,
) -> CNF:
    """
    Add a cardinality constraint to an existing CNF formula.

    Args:
        formula: Existing CNF formula
        constraint_type: Type of constraint ("atmost", "atleast", "exactly")
        variables: List of variable IDs
        k: Parameter value (max/min/exact number)
        encoding: Encoding type (default: sequential counter)

    Returns:
        CNF formula with the added constraint
    """
    # Create constraint formula
    if constraint_type.lower() == "atmost":
        constraint = CardEnc.atmost(variables, k, encoding=encoding)
    elif constraint_type.lower() == "atleast":
        constraint = CardEnc.atleast(variables, k, encoding=encoding)
    elif constraint_type.lower() in ["exactly", "equals"]:
        constraint = CardEnc.equals(variables, k, encoding=encoding)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    # Add constraint clauses to the formula
    for clause in constraint.clauses:
        formula.append(clause)

    return formula


def create_balanced_partitioning(
    variables: list[int], num_parts: int = 2, encoding: EncType = EncType.seqcounter
) -> dict[str, Any]:
    """
    Create a balanced partitioning of variables.

    Args:
        variables: List of variable IDs
        num_parts: Number of partitions (default: 2)
        encoding: Encoding type (default: sequential counter)

    Returns:
        Dictionary with formulas and information about the partitioning
    """
    if num_parts < 2:
        raise ValueError("Number of partitions must be at least 2")

    n = len(variables)

    # Create result dictionary
    result = {"formula": CNF(), "part_variables": [], "constraints": []}

    # Create part indicator variables
    # p_i_j = 1 means variable i is in part j
    top_var = max(abs(var) for var in variables)
    part_vars = []

    for i, var in enumerate(variables):
        part_var_row = []
        for j in range(num_parts):
            part_var = top_var + 1 + i * num_parts + j
            part_var_row.append(part_var)
        part_vars.append(part_var_row)

        # Each variable must be in exactly one part
        one_hot = CardEnc.equals(part_var_row, 1, encoding=encoding)
        result["constraints"].append(("one_hot", var, one_hot))
        for clause in one_hot.clauses:
            result["formula"].append(clause)

    result["part_variables"] = part_vars

    # Calculate target size for each part
    target_size = n // num_parts
    remainder = n % num_parts

    # Create part size constraints
    for j in range(num_parts):
        # Variables indicating if each element is in part j
        part_j_vars = [part_vars[i][j] for i in range(n)]

        # Part size constraint
        part_size = target_size + (1 if j < remainder else 0)
        part_size_constraint = CardEnc.equals(part_j_vars, part_size, encoding=encoding)
        result["constraints"].append(("part_size", j, part_size_constraint))

        # Add constraints to the formula
        for clause in part_size_constraint.clauses:
            result["formula"].append(clause)

    return result
