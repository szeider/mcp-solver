"""
Subset-related template functions for Z3.

This module provides helper functions for finding optimal subsets with specific properties.
"""

import os
import sys
from collections.abc import Callable
from itertools import combinations
from typing import TypeVar


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
    from z3 import And, Bool, Int, Not, Or, Solver, sat, unsat
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)

# Type variable for generic item types
T = TypeVar("T")


def smallest_subset_with_property(
    items: list[T],
    property_check_func: Callable[[list[T]], bool],
    min_size: int = 1,
    max_size: int | None = None,
) -> list[T] | None:
    """
    Find the smallest subset of items that satisfies a given property.

    Args:
        items: List of items to consider
        property_check_func: Function that takes a list of items and returns True if property is satisfied
        min_size: Minimum size of subset to consider (default: 1)
        max_size: Maximum size of subset to consider (default: len(items))

    Returns:
        The smallest subset satisfying the property, or None if no such subset exists

    Example:
        ```python
        # Define a property checker that returns True if tasks cannot all be scheduled
        def is_unschedulable(tasks):
            s = Solver()
            # ... set up scheduling constraints ...
            return s.check() == unsat


        # Find the smallest set of tasks that cannot be scheduled together
        result = smallest_subset_with_property(all_tasks, is_unschedulable, min_size=2)
        ```
    """
    if max_size is None:
        max_size = len(items)

    # Optional optimization: Check candidate subsets first if provided
    if hasattr(property_check_func, "candidate_subsets"):
        for subset in property_check_func.candidate_subsets:
            if min_size <= len(subset) <= max_size and property_check_func(subset):
                # Found a candidate that works, now minimize it
                return _minimize_subset(subset, property_check_func)

    # Start with the smallest possible size and increase
    for size in range(min_size, max_size + 1):
        print(f"Checking subsets of size {size}...")

        # Check all subsets of this size
        for subset in combinations(items, size):
            subset = list(subset)  # Convert to list for consistency

            if property_check_func(subset):
                return subset

    return None


def _minimize_subset(
    subset: list[T], property_check_func: Callable[[list[T]], bool]
) -> list[T]:
    """Helper function to minimize a subset while maintaining the property"""
    minimal = subset.copy()

    # Try removing each element to see if property still holds
    for item in subset:
        smaller = [x for x in minimal if x != item]
        if smaller and property_check_func(smaller):
            # If property still holds with item removed, recursively minimize further
            return _minimize_subset(smaller, property_check_func)

    return minimal
