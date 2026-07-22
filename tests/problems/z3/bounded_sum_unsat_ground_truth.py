#!/usr/bin/env python3
"""Semantic validator for the bounded-sum (unsatisfiable) problem.

The instance is unsatisfiable by design, so the correct verdict is
{"satisfiable": false}. Unsatisfiability is confirmed here by an in-validator
exhaustive oracle (stdlib only): every constraint forces the five variables to
be distinct integers in [1, 10], so the whole search space is the
P(10, 5) = 30240 ordered 5-tuples of distinct values, and enumerating them
shows none satisfies all six constraints. If a candidate instead claims
satisfiability, the supplied assignment is also checked against every
constraint, which always exposes a violated one.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys
from itertools import combinations, permutations

VARS = ("a", "b", "c", "d", "e")


def has_satisfying_assignment():
    """Exhaustively test all 30240 distinct in-range tuples for a solution."""
    for tup in permutations(range(1, 11), len(VARS)):
        if check_assignment(dict(zip(VARS, tup, strict=False))) is None:
            return True
    return False


def check_assignment(values):
    """Return an error string if any constraint is violated, else None."""
    for name in VARS:
        v = values[name]
        if not 1 <= v <= 10:
            return f"{name}={v} not in [1,10]"
    nums = [values[n] for n in VARS]
    if len(set(nums)) != len(nums):
        return "variables are not all distinct"
    if sum(nums) != 36:
        return f"sum is {sum(nums)}, must be 36"
    if not values["a"] * values["b"] < 20:
        return f"a*b = {values['a'] * values['b']} not < 20"
    if not values["c"] * values["d"] * values["e"] >= 200:
        return f"c*d*e = {values['c'] * values['d'] * values['e']} not >= 200"
    for x, y in combinations(VARS, 2):
        if values[x] + values[y] < 11:
            return f"{x}+{y} = {values[x] + values[y]} < 11"
    return None


# Confirm the instance really is unsatisfiable before trusting that verdict.
assert not has_satisfying_assignment(), "instance unexpectedly satisfiable"


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    satisfiable = data.get("satisfiable")
    if satisfiable is False:
        return True, (
            "correctly reported unsatisfiable "
            "(confirmed by exhaustive search over all 30240 distinct tuples)"
        )

    if satisfiable is True:
        assignment = data.get("assignment")
        if not isinstance(assignment, dict) or set(assignment) != set(VARS):
            return False, (
                "claimed satisfiable but instance is UNSAT; no valid "
                "assignment over a,b,c,d,e can exist"
            )
        for name in VARS:
            if not isinstance(assignment[name], int) or isinstance(
                assignment[name], bool
            ):
                return False, f"{name} must be an integer"
        error = check_assignment(assignment)
        if error is None:
            return False, (
                "assignment reported as valid, but the instance is provably "
                "UNSAT (validator bug if this fires)"
            )
        return False, (
            f"claimed satisfiable, but the assignment violates a constraint: "
            f"{error} (the instance is UNSAT)"
        )

    return False, f"satisfiable must be false for this instance, got {satisfiable!r}"


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

    valid, message = validate(data)
    print(json.dumps({"valid": valid, "message": message}))
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
