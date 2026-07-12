#!/usr/bin/env python3
"""Semantic validator for the software package selection MaxSAT problem.

Checks the hard dependency constraints and requires the reported objective
(value minus the package-count penalty) to equal the true maximum, computed
here by exhaustive enumeration (stdlib only).

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys
from itertools import product

VALUES = {"Core": 0, "UI": 6, "Auth": 8, "API": 7, "Analytics": 5}
NAMES = list(VALUES)
PREFERENCE_WEIGHT = 4
MIN_PACKAGES = 3


def feasible(selected):
    selected = set(selected)
    if "Core" not in selected:
        return False  # Core required
    if "UI" in selected and "Core" not in selected:
        return False
    if "Auth" in selected and "Core" not in selected:
        return False
    # Analytics requires UI and API.
    return not (
        "Analytics" in selected and not ("UI" in selected and "API" in selected)
    )


def score(selected):
    selected = set(selected)
    value = sum(VALUES[name] for name in selected)
    penalty = PREFERENCE_WEIGHT if len(selected) < MIN_PACKAGES else 0
    return value, penalty, value - penalty


def optimum():
    best = None
    for bits in product((False, True), repeat=len(NAMES)):
        subset = {name for name, chosen in zip(NAMES, bits, strict=False) if chosen}
        if feasible(subset):
            obj = score(subset)[2]
            best = obj if best is None else max(best, obj)
    return best


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    selected = data.get("selected_packages")
    if not isinstance(selected, list):
        return False, "Missing 'selected_packages' list"
    for name in selected:
        if name not in VALUES:
            return False, f"Unknown package {name!r}"
    if len(set(selected)) != len(selected):
        return False, "Duplicate package in selection"

    sel = set(selected)
    if "Core" not in sel:
        return False, "Hard constraint violated: Core must be selected"
    if "Analytics" in sel and not ("UI" in sel and "API" in sel):
        return False, "Dependency violated: Analytics requires UI and API"

    value, penalty, objective = score(sel)
    if data.get("total_value") is not None and data["total_value"] != value:
        return False, (
            f"Reported total_value {data['total_value']} does not match actual {value}"
        )
    if (
        data.get("preference_penalty") is not None
        and data["preference_penalty"] != penalty
    ):
        return False, (
            f"Reported preference_penalty {data['preference_penalty']} does "
            f"not match actual {penalty}"
        )
    if data.get("objective_value") is not None and data["objective_value"] != objective:
        return False, (
            f"Reported objective_value {data['objective_value']} does not "
            f"match actual {objective}"
        )

    best = optimum()
    if objective != best:
        return False, (f"Suboptimal: objective {objective} but the maximum is {best}")

    return True, f"Valid optimal selection (objective {objective})"


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
