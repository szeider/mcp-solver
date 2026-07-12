"""Behavioral tests for mcp_solver.helpers.z3."""

import pytest
import z3

from mcp_solver.helpers import z3 as zh


def int_array(name="a"):
    return z3.Array(name, z3.IntSort(), z3.IntSort())


# ---------------------------------------------------------------------------
# Array properties
# ---------------------------------------------------------------------------


def test_array_is_sorted():
    arr = int_array()
    s = z3.Solver()
    s.add(zh.array_is_sorted(arr, 3))
    s.add(arr[0] == 1, arr[1] == 2, arr[2] == 2)
    assert s.check() == z3.sat

    s = z3.Solver()
    s.add(zh.array_is_sorted(arr, 3, strict=True))
    s.add(arr[0] == 1, arr[1] == 2, arr[2] == 2)  # not strictly increasing
    assert s.check() == z3.unsat


def test_all_distinct():
    arr = int_array()
    s = z3.Solver()
    s.add(zh.all_distinct(arr, 3))
    s.add(arr[0] == 1, arr[1] == 2, arr[2] == 3)
    assert s.check() == z3.sat

    s = z3.Solver()
    s.add(zh.all_distinct(arr, 3))
    s.add(arr[0] == 1, arr[1] == 5, arr[2] == 1)  # duplicate
    assert s.check() == z3.unsat


def test_array_contains():
    arr = int_array()
    s = z3.Solver()
    s.add(zh.array_contains(arr, 3, 7))
    s.add(arr[0] == 1, arr[1] == 7, arr[2] == 3)
    assert s.check() == z3.sat

    s = z3.Solver()
    s.add(zh.array_contains(arr, 3, 7))
    s.add(arr[0] == 1, arr[1] == 2, arr[2] == 3)  # 7 absent
    assert s.check() == z3.unsat


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------


def test_exactly_k():
    xs = [z3.Bool(f"x{i}") for i in range(4)]
    s = z3.Solver()
    s.add(zh.exactly_k(xs, 2))
    assert s.check() == z3.sat
    s.add(z3.Not(z3.Or(*[z3.And(xs[i], xs[j]) for i in range(4) for j in range(i)])))
    # Forcing at most one true contradicts exactly-2.
    s2 = z3.Solver()
    s2.add(zh.exactly_k(xs, 2))
    s2.add(zh.at_most_k(xs, 1))
    assert s2.check() == z3.unsat


def test_at_most_k():
    xs = [z3.Bool(f"x{i}") for i in range(4)]
    s = z3.Solver()
    s.add(zh.at_most_k(xs, 2))
    s.add(xs[0], xs[1], xs[2])  # three true violates at-most-2
    assert s.check() == z3.unsat

    s = z3.Solver()
    s.add(zh.at_most_k(xs, 2))
    s.add(xs[0], xs[1])
    assert s.check() == z3.sat


def test_at_least_k():
    xs = [z3.Bool(f"x{i}") for i in range(4)]
    s = z3.Solver()
    s.add(zh.at_least_k(xs, 3))
    s.add(z3.Not(xs[0]), z3.Not(xs[1]))  # only two can be true
    assert s.check() == z3.unsat

    s = z3.Solver()
    s.add(zh.at_least_k(xs, 3))
    s.add(xs[0], xs[1], xs[2])
    assert s.check() == z3.sat


# ---------------------------------------------------------------------------
# Function properties
# ---------------------------------------------------------------------------


def test_function_is_injective():
    func = int_array("f")
    s = z3.Solver()
    s.add(zh.function_is_injective(func, 3))
    s.add(func[0] == 1, func[1] == 1)  # collision
    assert s.check() == z3.unsat

    s = z3.Solver()
    s.add(zh.function_is_injective(func, 3))
    s.add(func[0] == 1, func[1] == 2, func[2] == 3)
    assert s.check() == z3.sat


def test_function_is_surjective():
    func = int_array("f")
    # Domain 3 onto range 3: needs a bijection-like cover.
    s = z3.Solver()
    s.add(zh.function_is_surjective(func, 3, 3))
    s.add(func[0] == 0, func[1] == 1, func[2] == 2)
    assert s.check() == z3.sat

    # Domain 2 cannot cover range 3.
    s = z3.Solver()
    s.add(zh.function_is_surjective(func, 2, 3))
    assert s.check() == z3.unsat


# ---------------------------------------------------------------------------
# Subset search
# ---------------------------------------------------------------------------


def test_smallest_subset_with_property():
    nums = [1, 2, 3, 4]
    result = zh.smallest_subset_with_property(nums, lambda s: sum(s) >= 5)
    assert result is not None
    assert sum(result) >= 5
    assert len(result) == 2  # no single element reaches 5


def test_smallest_subset_none_when_impossible():
    nums = [1, 2, 3]
    result = zh.smallest_subset_with_property(nums, lambda s: sum(s) >= 100)
    assert result is None


def test_smallest_subset_candidate_minimization():
    nums = [1, 2, 3, 4]

    def prop(s):
        return sum(s) >= 4

    prop.candidate_subsets = [[1, 2, 3, 4]]
    result = zh.smallest_subset_with_property(nums, prop)
    # The full candidate is minimized down; [4] alone already satisfies sum>=4.
    assert sum(result) >= 4
    assert len(result) == 1


def test_smallest_subset_uses_z3_property():
    # A property backed by a real Z3 check: which pairs of intervals conflict.
    intervals = [(0, 2), (1, 3), (5, 6)]

    def overlaps(subset):
        s = z3.Solver()
        t = z3.Int("t")
        for lo, hi in subset:
            s.add(t >= lo, t < hi)
        return s.check() == z3.sat  # a common point exists

    result = zh.smallest_subset_with_property(intervals, overlaps, min_size=2)
    assert result == [(0, 2), (1, 3)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
