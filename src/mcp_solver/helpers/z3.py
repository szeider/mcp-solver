"""Z3 helper library: array/function properties, pseudo-Boolean cardinality
constraints, and optimal-subset search.

The array and function helpers return Z3 ``BoolRef`` constraints (typically
quantified) that you add to a ``Solver``/``Optimize``. Arrays are modelled as
Z3 ``Array(Int, ...)`` values with an explicit ``size``; quantified helpers
range over indices ``0 <= i < size``. Reserved index names ``_i``/``_j`` are
used internally, so avoid those names in your own model.

Example:
    >>> import z3
    >>> arr = z3.Array("a", z3.IntSort(), z3.IntSort())
    >>> s = z3.Solver()
    >>> s.add(array_is_sorted(arr, 3, strict=True))
    >>> s.add(arr[0] == 5, arr[1] == 1)
    >>> s.check()
    unsat
"""

from collections.abc import Callable
from itertools import combinations
from typing import Any, TypeVar

from z3 import (
    And,
    ArrayRef,
    BoolRef,
    Exists,
    ExprRef,
    ForAll,
    Implies,
    Int,
    Ints,
    PbEq,
    PbGe,
    PbLe,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Array and sequence properties
# ---------------------------------------------------------------------------


def array_is_sorted(
    arr: ArrayRef, size: int | ExprRef, strict: bool = False
) -> BoolRef:
    """Constraint that ``arr[0:size]`` is sorted in ascending order.

    Args:
        arr: Z3 array (index sort Int).
        size: Number of leading elements considered (int or Z3 expression).
        strict: If True require strictly increasing (``<``), else ``<=``.

    Returns:
        A quantified Z3 ``BoolRef``.
    """
    i, j = Ints("_i _j")
    guard = And(i >= 0, i < j, j < size)
    if strict:
        return ForAll([i, j], Implies(guard, arr[i] < arr[j]))
    return ForAll([i, j], Implies(guard, arr[i] <= arr[j]))


def all_distinct(arr: ArrayRef, size: int | ExprRef) -> BoolRef:
    """Constraint that all elements of ``arr[0:size]`` are pairwise distinct."""
    i, j = Ints("_i _j")
    return ForAll([i, j], Implies(And(i >= 0, i < j, j < size), arr[i] != arr[j]))


def array_contains(arr: ArrayRef, size: int | ExprRef, value: Any) -> BoolRef:
    """Constraint that ``value`` occurs somewhere in ``arr[0:size]``."""
    i = Int("_i")
    return Exists([i], And(i >= 0, i < size, arr[i] == value))


# ---------------------------------------------------------------------------
# Cardinality constraints (pseudo-Boolean)
# ---------------------------------------------------------------------------


def exactly_k(bool_vars: list[BoolRef], k: int | ExprRef) -> BoolRef:
    """Constraint that exactly ``k`` of ``bool_vars`` are true (``PbEq``)."""
    return PbEq([(v, 1) for v in bool_vars], k)


def at_most_k(bool_vars: list[BoolRef], k: int | ExprRef) -> BoolRef:
    """Constraint that at most ``k`` of ``bool_vars`` are true (``PbLe``)."""
    return PbLe([(v, 1) for v in bool_vars], k)


def at_least_k(bool_vars: list[BoolRef], k: int | ExprRef) -> BoolRef:
    """Constraint that at least ``k`` of ``bool_vars`` are true (``PbGe``)."""
    return PbGe([(v, 1) for v in bool_vars], k)


# ---------------------------------------------------------------------------
# Functional properties
# ---------------------------------------------------------------------------


def function_is_injective(
    func: ArrayRef,
    domain_size: int | ExprRef,
    range_size: int | ExprRef | None = None,
) -> BoolRef:
    """Constraint that ``func`` restricted to ``[0, domain_size)`` is injective.

    Args:
        func: Z3 array modelling the function.
        domain_size: Size of the domain considered.
        range_size: Unused; accepted for symmetry with
            :func:`function_is_surjective`.

    Returns:
        A quantified Z3 ``BoolRef``.
    """
    i, j = Ints("_i _j")
    return ForAll(
        [i, j],
        Implies(
            And(i >= 0, i < domain_size, j >= 0, j < domain_size, i != j),
            func[i] != func[j],
        ),
    )


def function_is_surjective(
    func: ArrayRef, domain_size: int | ExprRef, range_size: int | ExprRef
) -> BoolRef:
    """Constraint that ``func`` maps ``[0, domain_size)`` onto ``[0, range_size)``.

    Every target ``j`` in ``[0, range_size)`` has some ``i`` in
    ``[0, domain_size)`` with ``func[i] == j``.
    """
    j = Int("_j")
    i = Int("_i")
    return ForAll(
        [j],
        Implies(
            And(j >= 0, j < range_size),
            Exists([i], And(i >= 0, i < domain_size, func[i] == j)),
        ),
    )


# ---------------------------------------------------------------------------
# Optimal subset search
# ---------------------------------------------------------------------------


def smallest_subset_with_property(
    items: list[T],
    property_check_func: Callable[[list[T]], bool],
    min_size: int = 1,
    max_size: int | None = None,
) -> list[T] | None:
    """Find a smallest subset of ``items`` for which the property holds.

    Enumerates subsets by increasing size and returns the first one for which
    ``property_check_func`` returns True. Because sizes are tried smallest
    first, the returned subset has minimum cardinality. This is a generic
    driver: ``property_check_func`` typically builds and checks a Z3 model, but
    any boolean predicate over a subset works.

    If ``property_check_func`` carries a ``candidate_subsets`` attribute, those
    candidates are tried first and, on a hit, locally minimized via element
    removal before returning.

    Args:
        items: Items to draw subsets from.
        property_check_func: Predicate on a subset (e.g. returns
            ``solver.check() == unsat``).
        min_size: Smallest subset size to consider.
        max_size: Largest subset size to consider (default ``len(items)``).

    Returns:
        A smallest satisfying subset, or None if none exists in range.

    Example:
        >>> nums = [1, 2, 3, 4]
        >>> smallest_subset_with_property(nums, lambda s: sum(s) >= 5, min_size=1)
        [1, 4]
    """
    if max_size is None:
        max_size = len(items)

    if hasattr(property_check_func, "candidate_subsets"):
        for subset in property_check_func.candidate_subsets:
            if min_size <= len(subset) <= max_size and property_check_func(subset):
                return _minimize_subset(subset, property_check_func)

    for size in range(min_size, max_size + 1):
        for subset in combinations(items, size):
            subset = list(subset)
            if property_check_func(subset):
                return subset

    return None


def _minimize_subset(
    subset: list[T], property_check_func: Callable[[list[T]], bool]
) -> list[T]:
    """Greedily drop elements while the property is preserved."""
    minimal = subset.copy()
    for item in subset:
        smaller = [x for x in minimal if x != item]
        if smaller and property_check_func(smaller):
            return _minimize_subset(smaller, property_check_func)
    return minimal
