"""Behavioral tests for mcp_solver.helpers.pysat."""

from itertools import combinations

import pytest
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Glucose3

from mcp_solver.helpers import pysat as ph


def enumerate_projected(clauses, query_vars):
    """Return the set of satisfying assignments over query_vars.

    Each assignment is a frozenset of the query vars that are true. Auxiliary
    variables introduced by encodings are projected out; distinct projections
    are enumerated by blocking on the projection.
    """
    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)

    models = set()
    while solver.solve():
        model = solver.get_model()
        truth = {v: (v in model) for v in query_vars}
        models.add(frozenset(v for v in query_vars if truth[v]))
        # Block this exact projection.
        block = [(-v if truth[v] else v) for v in query_vars]
        solver.add_clause(block)
    solver.delete()
    return models


def rc2_solve(wcnf):
    """Return (model, cost) from RC2, or (None, None) if unsat."""
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        if model is None:
            return None, None
        return model, rc2.cost


# ---------------------------------------------------------------------------
# VariableMap
# ---------------------------------------------------------------------------


def test_variable_map_roundtrip():
    vm = ph.VariableMap()
    a = vm.get_id("a")
    b = vm.get_id("b")
    c = vm.get_id("c")
    assert (a, b, c) == (1, 2, 3)
    # Stable across repeated lookups.
    assert vm.get_id("a") == 1
    assert vm.get_name(a) == "a"
    assert vm.get_name(-b) == "b"  # sign ignored
    assert vm.get_name(999) == "unknown_999"
    assert vm.get_mapping() == {"a": 1, "b": 2, "c": 3}


def test_variable_map_interpret_model():
    vm = ph.VariableMap()
    x = vm.get_id("x")
    y = vm.get_id("y")
    z = vm.get_id("z")
    # Model includes an unknown aux id (99) which must be dropped.
    model = [x, -y, z, 99]
    assert vm.interpret_model(model) == {"x": True, "y": False, "z": True}


# ---------------------------------------------------------------------------
# exactly_one / exactly_k
# ---------------------------------------------------------------------------


def test_exactly_one_four_vars_enumerates_four_models():
    variables = [1, 2, 3, 4]
    clauses = ph.exactly_one(variables)
    models = enumerate_projected(clauses, variables)
    expected = {frozenset([v]) for v in variables}
    assert models == expected
    assert len(models) == 4


def test_exactly_k_general_case():
    variables = [1, 2, 3, 4]
    clauses = ph.exactly_k(variables, 2)
    models = enumerate_projected(clauses, variables)
    expected = {frozenset(c) for c in combinations(variables, 2)}
    assert models == expected
    assert len(models) == 6


def test_exactly_k_zero_forces_all_false():
    variables = [1, 2, 3]
    clauses = ph.exactly_k(variables, 0)
    models = enumerate_projected(clauses, variables)
    assert models == {frozenset()}


def test_exactly_k_impossible_is_unsat():
    variables = [1, 2, 3]
    assert ph.exactly_k(variables, 5) == [[]]
    assert ph.exactly_k(variables, -1) == [[]]
    models = enumerate_projected(ph.exactly_k(variables, 5), variables)
    assert models == set()


def test_exactly_k_cardenc_path_25_vars_is_sat():
    # Regression: above the pairwise threshold, at_most_k and at_least_k both
    # allocate CardEnc auxiliaries; without top_id threading their IDs collided
    # and exactly_k(25 vars, 10) was spuriously UNSAT (2026-07-13 campaign).
    variables = list(range(1, 26))
    clauses = ph.exactly_k(variables, 10)
    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)
    assert solver.solve()
    model = solver.get_model()
    assert sum(1 for v in variables if v in model) == 10
    # And the count is enforced, not merely allowed: block-and-check a few
    # models, all must have exactly 10 true.
    for _ in range(5):
        block = [(-v if v in model else v) for v in variables]
        solver.add_clause(block)
        if not solver.solve():
            break
        model = solver.get_model()
        assert sum(1 for v in variables if v in model) == 10
    solver.delete()


def test_exactly_k_cardenc_path_respects_top_id():
    # Formula vars 1..30 exist; exactly_k over the first 25 must not allocate
    # auxiliaries that collide with vars 26..30 when top_id says they exist.
    variables = list(range(1, 26))
    clauses = ph.exactly_k(variables, 10, top_id=30)
    aux = {abs(lit) for cl in clauses for lit in cl} - set(variables)
    assert aux, "expected CardEnc auxiliaries on the >20-vars path"
    assert min(aux) > 30


def test_one_hot_matches_exactly_one():
    variables = [1, 2, 3, 4]
    assert enumerate_projected(
        ph.one_hot_encoding(variables), variables
    ) == enumerate_projected(ph.exactly_one(variables), variables)


# ---------------------------------------------------------------------------
# at_most_k / at_least_k
# ---------------------------------------------------------------------------


def test_at_most_k_below_and_above_boundary():
    variables = [1, 2, 3, 4]
    models = enumerate_projected(ph.at_most_k(variables, 2), variables)
    for m in models:
        assert len(m) <= 2
    # Every subset of size <= 2 must be a model.
    for size in range(3):
        for combo in combinations(variables, size):
            assert frozenset(combo) in models
    # Subsets of size 3 must be excluded.
    assert frozenset([1, 2, 3]) not in models

    # k >= n is trivially satisfied (no clauses).
    assert ph.at_most_k(variables, 4) == []
    assert ph.at_most_k(variables, 5) == []


def test_at_most_k_uses_pairwise_at_boundary_20():
    variables = list(range(1, 21))  # exactly 20 -> pairwise, no aux vars
    clauses = ph.at_most_k(variables, 3)
    used = {abs(lit) for clause in clauses for lit in clause}
    assert used <= set(variables)  # no auxiliary variables introduced


def test_at_most_k_cardenc_path_21_vars():
    variables = list(range(1, 22))  # 21 -> CardEnc path
    clauses = ph.at_most_k(variables, 3, top_id=21)
    used = {abs(lit) for clause in clauses for lit in clause}
    # CardEnc introduces auxiliary variables above the 21 originals.
    assert max(used) > 21

    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)
    # Forcing 4 originals true violates at-most-3 -> UNSAT.
    assert not solver.solve(assumptions=[1, 2, 3, 4])
    # Forcing exactly 3 true is fine -> SAT.
    assert solver.solve(assumptions=[1, 2, 3])
    solver.delete()


def test_at_least_k_de_morgan():
    variables = [1, 2, 3, 4]
    models = enumerate_projected(ph.at_least_k(variables, 3), variables)
    expected = {frozenset(c) for size in (3, 4) for c in combinations(variables, size)}
    assert models == expected


def test_at_most_one_and_mutually_exclusive_agree():
    variables = [1, 2, 3]
    a = enumerate_projected(ph.at_most_one(variables), variables)
    b = enumerate_projected(ph.mutually_exclusive(variables), variables)
    assert a == b
    for m in a:
        assert len(m) <= 1


# ---------------------------------------------------------------------------
# implies / if_then_else
# ---------------------------------------------------------------------------


def test_implies():
    models = enumerate_projected(ph.implies(1, 2), [1, 2])
    # a -> b excludes only (a true, b false).
    assert frozenset([1]) not in models
    assert frozenset([1, 2]) in models
    assert frozenset([2]) in models
    assert frozenset() in models


def test_if_then_else():
    # condition=1, then=2, else=3
    models = enumerate_projected(ph.if_then_else(1, 2, 3), [1, 2, 3])
    for m in models:
        if 1 in m:
            assert 2 in m  # condition -> then
        else:
            assert 3 in m  # !condition -> else


# ---------------------------------------------------------------------------
# Soft-clause family (via RC2)
# ---------------------------------------------------------------------------


def test_soft_clause_optimization_choice():
    # Hard: (1 or 2) and (2 or 3). Soft: prefer each var false (weight 1).
    # Optimum sets only var 2 true (covers both), cost 1.
    wcnf = WCNF()
    wcnf.append([1, 2])
    wcnf.append([2, 3])
    ph.add_soft_clauses_to_wcnf(
        wcnf,
        [ph.soft_clause(-1), ph.soft_clause(-2), ph.soft_clause(-3)],
    )
    model, cost = rc2_solve(wcnf)
    assert cost == 1
    true_vars = {v for v in model if v > 0}
    assert true_vars == {2}


def test_soft_at_most_k_cost():
    # Hard forces all three true; soft at-most-1 has 3 violated clauses.
    wcnf = WCNF()
    for v in (1, 2, 3):
        wcnf.append([v])
    ph.add_soft_clauses_to_wcnf(wcnf, ph.soft_at_most_k([1, 2, 3], 1, weight=2))
    _, cost = rc2_solve(wcnf)
    assert cost == 3 * 2


def test_soft_at_least_k_cost():
    # Hard forces all three false; soft at-least-2 has 3 violated clauses.
    wcnf = WCNF()
    for v in (1, 2, 3):
        wcnf.append([-v])
    ph.add_soft_clauses_to_wcnf(wcnf, ph.soft_at_least_k([1, 2, 3], 2, weight=5))
    _, cost = rc2_solve(wcnf)
    assert cost == 3 * 5


def test_soft_exactly_k_is_union():
    variables = [1, 2, 3]
    combined = ph.soft_exactly_k(variables, 1, weight=1)
    manual = ph.soft_at_most_k(variables, 1, 1) + ph.soft_at_least_k(variables, 1, 1)
    assert combined == manual


def test_soft_implies_and_if_then_else_costs():
    # soft_implies(1, 2): violated only when 1 true, 2 false.
    wcnf = WCNF()
    wcnf.append([1])
    wcnf.append([-2])
    ph.add_soft_clauses_to_wcnf(wcnf, ph.soft_implies(1, 2, weight=4))
    _, cost = rc2_solve(wcnf)
    assert cost == 4

    # soft_if_then_else(1, 2, 3): condition true -> the "then" clause governs.
    wcnf = WCNF()
    wcnf.append([1])  # condition true
    wcnf.append([-2])  # then_var false -> violates [-1, 2]
    wcnf.append([-3])  # else clause [1, 3] satisfied by condition
    ph.add_soft_clauses_to_wcnf(wcnf, ph.soft_if_then_else(1, 2, 3, weight=7))
    _, cost = rc2_solve(wcnf)
    assert cost == 7


def test_soft_clause_wraps_bare_int():
    assert ph.soft_clause(5) == ([5], 1)
    assert ph.soft_clause([5, -6], 3) == ([5, -6], 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
