#!/usr/bin/env python3
"""Semantic validator for the shift_assignment problem (hard constraints + optimality).

Two employees (anna, ben) are assigned to three shifts (morning, afternoon,
evening). The hard constraints are:

  - Each of the three shifts is covered by exactly one employee.
  - There are at most two employees.
  - No employee works more than two shifts.

Each employee has a fixed set of preferred shifts (must match the .md):

  - anna prefers morning, afternoon, and evening.
  - ben prefers morning.

The objective is to maximize satisfied_preferences: the number of shifts whose
assigned employee lists that shift as preferred. Because anna can work at most
two shifts, not all four preferences can be met. Brute force over all feasible
assignments shows the optimum is a unique assignment scoring 3 out of 4:

  {morning: ben, afternoon: anna, evening: anna}

This validator checks the hard constraints, verifies the reported
satisfied_preferences value, and requires it to equal the brute-forced optimum.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import itertools
import json
import sys
from collections import Counter

SHIFTS = ["morning", "afternoon", "evening"]
EMPLOYEES = ["anna", "ben"]

# Per-employee preferred shifts (must match shift_assignment.md).
PREFERENCES = {
    "anna": {"morning", "afternoon", "evening"},
    "ben": {"morning"},
}


def score(assignment):
    """Number of shifts whose assigned employee prefers that shift."""
    return sum(1 for s in SHIFTS if s in PREFERENCES[assignment[s]])


def feasible_assignments():
    """All assignments of shifts to employees obeying the hard constraints."""
    result = []
    for combo in itertools.product(EMPLOYEES, repeat=len(SHIFTS)):
        counts = Counter(combo)
        if any(c > 2 for c in counts.values()):
            continue  # no employee works more than two shifts
        if len(set(combo)) > 2:
            continue  # at most two employees (always true here)
        result.append(dict(zip(SHIFTS, combo, strict=False)))
    return result


def optimum():
    """Return (best_score, [optimal assignments])."""
    scored = [(score(a), a) for a in feasible_assignments()]
    best = max(s for s, _ in scored)
    return best, [a for s, a in scored if s == best]


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    assignment = data.get("assignment")
    if not isinstance(assignment, dict):
        return False, "assignment must map shift names to employees"

    normalized = {}
    for key, value in assignment.items():
        shift = key.strip().lower()
        if not isinstance(value, str):
            return False, f"shift {shift} must map to an employee name string"
        normalized[shift] = value.strip().lower()

    if set(normalized) != set(SHIFTS):
        return False, (
            f"assignment must cover exactly {sorted(SHIFTS)}, got {sorted(normalized)}"
        )

    for shift in SHIFTS:
        emp = normalized[shift]
        if emp not in EMPLOYEES:
            return False, (
                f"shift {shift} assigned to unknown employee {emp!r}; "
                f"expected one of {EMPLOYEES}"
            )

    counts = Counter(normalized[s] for s in SHIFTS)
    distinct = set(counts)
    if len(distinct) > 2:
        return False, (
            f"at most 2 employees allowed, got {len(distinct)}: {sorted(distinct)}"
        )
    for emp, count in counts.items():
        if count > 2:
            return False, f"employee {emp} works {count} shifts, at most 2 allowed"

    # Optimality: the achieved score must equal the brute-forced optimum.
    achieved = score(normalized)
    best, optima = optimum()

    reported = data.get("satisfied_preferences")
    if reported is not None:
        if not isinstance(reported, int) or isinstance(reported, bool):
            return False, "satisfied_preferences must be an integer"
        if reported != achieved:
            return False, (
                f"satisfied_preferences={reported} does not match the actual "
                f"number satisfied by the assignment ({achieved})"
            )

    if achieved != best:
        return False, (
            f"assignment satisfies {achieved} preference(s), but the optimum is "
            f"{best} (e.g. {optima[0]}); solution is feasible but suboptimal"
        )

    return True, (
        f"hard constraints satisfied and optimal: {achieved}/{best} "
        f"preferences satisfied by {dict(normalized)}"
    )


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
