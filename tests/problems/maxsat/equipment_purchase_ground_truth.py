#!/usr/bin/env python3
"""Semantic validator for the equipment purchase MaxSAT problem.

Checks the hard constraints and requires the reported total value to equal
the true maximum, computed here by exhaustive enumeration (stdlib only).

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys
from itertools import combinations

BUDGET = 7000
ITEMS = {
    "Analyzer": {"cost": 3500, "value": 9},
    "Bench": {"cost": 2500, "value": 6},
    "Computer": {"cost": 2000, "value": 5},
    "Desk": {"cost": 1500, "value": 4},
}
NAMES = list(ITEMS)


def totals(selected):
    selected = set(selected)
    cost = sum(ITEMS[name]["cost"] for name in selected)
    value = sum(ITEMS[name]["value"] for name in selected)
    if "Bench" in selected and "Desk" in selected:
        value += 2  # synergy bonus
    return cost, value


def feasible(selected):
    selected = set(selected)
    cost, _ = totals(selected)
    if cost > BUDGET:
        return False
    # Analyzer requires Computer.
    return not ("Analyzer" in selected and "Computer" not in selected)


def optimum():
    best = -1
    for size in range(len(NAMES) + 1):
        for subset in combinations(NAMES, size):
            if feasible(subset):
                _, value = totals(subset)
                best = max(best, value)
    return best


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    selected = data.get("equipment_selected")
    if not isinstance(selected, list):
        return False, "Missing 'equipment_selected' list"
    for name in selected:
        if name not in ITEMS:
            return False, f"Unknown equipment {name!r}"
    if len(set(selected)) != len(selected):
        return False, "Duplicate equipment in selection"

    cost, value = totals(selected)
    if cost > BUDGET:
        return False, f"Budget exceeded: total cost {cost} > {BUDGET}"
    if "Analyzer" in selected and "Computer" not in selected:
        return False, "Dependency violated: Analyzer requires Computer"

    stated_cost = data.get("total_cost")
    if stated_cost is not None and stated_cost != cost:
        return False, (
            f"Reported total_cost {stated_cost} does not match actual {cost}"
        )
    stated_value = data.get("total_value")
    if stated_value is not None and stated_value != value:
        return False, (
            f"Reported total_value {stated_value} does not match actual {value}"
        )

    best = optimum()
    if value != best:
        return False, (f"Suboptimal: total value {value} but the maximum is {best}")

    return True, f"Valid optimal selection (value {value}, cost {cost})"


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"valid": False, "message": "No input provided"}))
        sys.exit(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"valid": False, "message": f"Invalid JSON: {exc}"}))
        sys.exit(1)

    ok, message = validate(data)
    print(json.dumps({"valid": ok, "message": message}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
