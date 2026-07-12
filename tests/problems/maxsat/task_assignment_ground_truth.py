#!/usr/bin/env python3
"""Semantic validator for the task assignment MaxSAT problem.

Checks the hard constraints and requires the reported penalty to equal the
true minimum, computed here by exhaustive enumeration (stdlib only).

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys

TASKS = ["Task A", "Task B"]
EMPLOYEES = ["Alice", "Bob"]


def penalty(assignment):
    total = 0
    if assignment["Task A"] != "Alice":
        total += 3  # Alice prefers Task A
    if assignment["Task B"] != "Bob":
        total += 2  # Bob prefers Task B
    if assignment["Task A"] == "Bob":
        total += 1  # Task A should ideally go to Alice
    return total


def feasible(assignment):
    # Alice cannot do both tasks.
    return not (assignment["Task A"] == "Alice" and assignment["Task B"] == "Alice")


def optimum():
    best = None
    for a in EMPLOYEES:
        for b in EMPLOYEES:
            cand = {"Task A": a, "Task B": b}
            if feasible(cand):
                cost = penalty(cand)
                best = cost if best is None else min(best, cost)
    return best


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    assignment = data.get("assignment")
    if not isinstance(assignment, dict) or set(assignment) != set(TASKS):
        return False, f"'assignment' must map exactly {TASKS}"

    for task, emp in assignment.items():
        if emp not in EMPLOYEES:
            return False, f"{task} assigned to unknown employee {emp!r}"

    if not feasible(assignment):
        return False, "Hard constraint violated: Alice cannot do both tasks"

    actual = penalty(assignment)
    stated = data.get("total_penalty")
    if stated is not None and stated != actual:
        return False, (
            f"Reported total_penalty {stated} does not match actual {actual}"
        )

    best = optimum()
    if actual != best:
        return False, (
            f"Suboptimal: penalty {actual} but the minimum penalty is {best}"
        )

    return True, f"Valid optimal assignment (penalty {actual})"


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
