"""
Z3 template functions for common quantifier patterns and constraints.

This module provides helper functions that generate common constraint patterns,
especially those involving quantifiers, which can be difficult to write correctly.
"""

import sys
import os
from typing import List, Union, Any

# IMPORTANT: Properly import the Z3 library (not our local package)
# First, remove the current directory from the path to avoid importing ourselves
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if current_dir in sys.path:
    sys.path.remove(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if parent_parent_dir in sys.path:
    sys.path.remove(parent_parent_dir)

# Add site-packages to the front of the path
import site

site_packages = site.getsitepackages()
for p in reversed(site_packages):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now try to import Z3
try:
    from z3 import (
        Int,
        Ints,
        And,
        Or,
        Not,
        Implies,
        ForAll,
        Exists,
        PbEq,
        PbLe,
        PbGe,
        BoolRef,
        ArrayRef,
        ExprRef,
    )
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)


# Array and sequence properties
def array_is_sorted(
    arr: ArrayRef, size: Union[int, ExprRef], strict: bool = False
) -> BoolRef:
    """
    Create a constraint ensuring array is sorted in ascending order.

    Args:
        arr: Z3 array representing the sequence
        size: Size of the array (int or Z3 expression)
        strict: If True, use strict inequality (< instead of <=)

    Returns:
        Z3 constraint expression
    """
    i, j = Ints("_i _j")
    if strict:
        return ForAll([i, j], Implies(And(0 <= i, i < j, j < size), arr[i] < arr[j]))
    else:
        return ForAll([i, j], Implies(And(0 <= i, i < j, j < size), arr[i] <= arr[j]))


def all_distinct(arr: ArrayRef, size: Union[int, ExprRef]) -> BoolRef:
    """
    Create a constraint ensuring all elements in array are distinct.

    Args:
        arr: Z3 array representing the sequence
        size: Size of the array (int or Z3 expression)

    Returns:
        Z3 constraint expression
    """
    i, j = Ints("_i _j")
    return ForAll([i, j], Implies(And(0 <= i, i < j, j < size), arr[i] != arr[j]))


def array_contains(arr: ArrayRef, size: Union[int, ExprRef], value: Any) -> BoolRef:
    """
    Create a constraint ensuring array contains a specific value.

    Args:
        arr: Z3 array representing the sequence
        size: Size of the array (int or Z3 expression)
        value: Value that must be present in the array

    Returns:
        Z3 constraint expression
    """
    i = Int("_i")
    return Exists([i], And(0 <= i, i < size, arr[i] == value))


# Cardinality constraints
def exactly_k(bool_vars: List[BoolRef], k: Union[int, ExprRef]) -> BoolRef:
    """
    Create a constraint ensuring exactly k boolean variables are true.

    Args:
        bool_vars: List of boolean variables
        k: Required number of true variables

    Returns:
        Z3 constraint expression
    """
    return PbEq([(v, 1) for v in bool_vars], k)


def at_most_k(bool_vars: List[BoolRef], k: Union[int, ExprRef]) -> BoolRef:
    """
    Create a constraint ensuring at most k boolean variables are true.

    Args:
        bool_vars: List of boolean variables
        k: Maximum number of true variables

    Returns:
        Z3 constraint expression
    """
    return PbLe([(v, 1) for v in bool_vars], k)


def at_least_k(bool_vars: List[BoolRef], k: Union[int, ExprRef]) -> BoolRef:
    """
    Create a constraint ensuring at least k boolean variables are true.

    Args:
        bool_vars: List of boolean variables
        k: Minimum number of true variables

    Returns:
        Z3 constraint expression
    """
    return PbGe([(v, 1) for v in bool_vars], k)


# Functional properties
def function_is_injective(
    func: ArrayRef,
    domain_size: Union[int, ExprRef],
    range_size: Union[int, ExprRef] = None,
) -> BoolRef:
    """
    Create a constraint ensuring a function is injective (one-to-one).

    Args:
        func: Z3 array representing the function mapping
        domain_size: Size of the domain (int or Z3 expression)
        range_size: Size of the range (optional, not used in constraint)

    Returns:
        Z3 constraint expression
    """
    i, j = Ints("_i _j")
    return ForAll(
        [i, j],
        Implies(
            And(0 <= i, i < domain_size, 0 <= j, j < domain_size, i != j),
            func[i] != func[j],
        ),
    )


def function_is_surjective(
    func: ArrayRef, domain_size: Union[int, ExprRef], range_size: Union[int, ExprRef]
) -> BoolRef:
    """
    Create a constraint ensuring a function is surjective (onto).

    Args:
        func: Z3 array representing the function mapping
        domain_size: Size of the domain
        range_size: Size of the range

    Returns:
        Z3 constraint expression
    """
    j = Int("_j")
    i = Int("_i")
    return ForAll(
        [j],
        Implies(
            And(0 <= j, j < range_size),
            Exists([i], And(0 <= i, i < domain_size, func[i] == j)),
        ),
    )
