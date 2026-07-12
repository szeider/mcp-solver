"""Behavioral tests for mcp_solver.helpers.maxsat (WCNF-mutating helpers)."""

import pytest
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from mcp_solver.helpers import maxsat as mh


def rc2_solve(wcnf):
    """Return (model, cost) from RC2, or (None, None) if unsat."""
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        if model is None:
            return None, None
        return model, rc2.cost


def count_true(model, variables):
    return sum(1 for v in variables if v in model)


def test_at_most_k_hard_bound():
    variables = [1, 2, 3, 4]
    wcnf = WCNF()
    mh.at_most_k(wcnf, variables, 2)
    # Soft: prefer every variable true, so the optimum pushes to the bound.
    for v in variables:
        wcnf.append([v], weight=1)
    model, cost = rc2_solve(wcnf)
    assert count_true(model, variables) == 2
    # Two of four soft "prefer true" clauses stay violated.
    assert cost == 2


def test_at_most_k_trivial_when_k_ge_n():
    wcnf = WCNF()
    mh.at_most_k(wcnf, [1, 2, 3], 3)
    assert wcnf.hard == []


def test_at_most_k_negative_forces_all_false():
    variables = [1, 2, 3]
    wcnf = WCNF()
    mh.at_most_k(wcnf, variables, -1)
    for v in variables:
        wcnf.append([v], weight=1)  # push against the hard bound
    model, cost = rc2_solve(wcnf)
    assert count_true(model, variables) == 0
    assert cost == 3


def test_at_least_k_hard_bound():
    variables = [1, 2, 3, 4]
    wcnf = WCNF()
    mh.at_least_k(wcnf, variables, 2)
    # Soft: prefer every variable false.
    for v in variables:
        wcnf.append([-v], weight=1)
    model, cost = rc2_solve(wcnf)
    assert count_true(model, variables) == 2
    assert cost == 2


def test_at_least_k_trivial_when_k_le_zero():
    wcnf = WCNF()
    mh.at_least_k(wcnf, [1, 2, 3], 0)
    assert wcnf.hard == []


def test_at_least_k_impossible_adds_empty_clause():
    wcnf = WCNF()
    mh.at_least_k(wcnf, [1, 2], 3)
    assert [] in wcnf.hard
    model, cost = rc2_solve(wcnf)
    assert model is None  # unsatisfiable


def test_exactly_k_hard_bound():
    variables = [1, 2, 3, 4]
    wcnf = WCNF()
    mh.exactly_k(wcnf, variables, 2)
    for v in variables:
        wcnf.append([v], weight=1)
    model, _ = rc2_solve(wcnf)
    assert count_true(model, variables) == 2


def test_exactly_k_pins_solution():
    # exactly_k=4 over 4 vars forces all true, regardless of soft prefs.
    variables = [1, 2, 3, 4]
    wcnf = WCNF()
    mh.exactly_k(wcnf, variables, 4)
    for v in variables:
        wcnf.append([-v], weight=1)  # prefer false, but hard bound wins
    model, cost = rc2_solve(wcnf)
    assert count_true(model, variables) == 4
    assert cost == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
