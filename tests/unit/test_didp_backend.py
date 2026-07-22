"""Smoke test for the didp backend: the didppy API the template documents.

Guards against didppy releases drifting from the API taught in
templates/didp.md (model construction, transitions, CABS, solution fields).
"""

import pytest

didppy = pytest.importorskip("didppy")


def test_didppy_knapsack_end_to_end():
    # 0/1 knapsack, maximization — exercises the exact API surface the
    # template teaches. Optimum is 60 (items 1 and 2, weight 50).
    weights = [10, 20, 30, 40]
    profits = [5, 25, 35, 50]
    capacity = 50
    n = len(weights)

    model = didppy.Model(maximize=True, float_cost=False)
    item = model.add_object_type(number=n)
    r = model.add_int_var(target=capacity)
    i = model.add_element_var(object_type=item, target=0)
    w = model.add_int_table(weights)
    p = model.add_int_table(profits)

    model.add_transition(
        didppy.Transition(
            name="pack",
            cost=p[i] + didppy.IntExpr.state_cost(),
            effects=[(r, r - w[i]), (i, i + 1)],
            preconditions=[i < n, r >= w[i]],
        )
    )
    model.add_transition(
        didppy.Transition(
            name="skip",
            cost=didppy.IntExpr.state_cost(),
            effects=[(i, i + 1)],
            preconditions=[i < n],
        )
    )
    model.add_base_case([i == n])

    solution = didppy.CABS(model, time_limit=10, quiet=True).search()

    assert not solution.is_infeasible
    assert solution.is_optimal
    assert solution.cost == 60
    assert [t.name for t in solution.transitions].count("pack") == 2
