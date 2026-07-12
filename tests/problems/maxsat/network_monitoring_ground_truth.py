#!/usr/bin/env python3
"""Semantic validator for the network monitoring MaxSAT problem.

Hard constraints: every edge is covered by a monitored endpoint and at
least one Core server is monitored. The soft objective minimizes
    sum(cost of selected) + sum(20 - value over unselected).
The reported selection must achieve the true minimum, computed here by
exhaustive enumeration (stdlib only).

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys
from itertools import product

# name -> (value, cost)
SERVERS = {
    "Core1": (10, 3),
    "Core2": (10, 3),
    "Web1": (6, 2),
    "Web2": (6, 2),
    "DB1": (8, 3),
    "Edge1": (5, 1),
}
NAMES = list(SERVERS)
CORE = {"Core1", "Core2"}
EDGES = [
    ("Core1", "Core2"),
    ("Core1", "Web1"),
    ("Core1", "DB1"),
    ("Core2", "Web2"),
    ("Core2", "DB1"),
    ("Core2", "Edge1"),
    ("Web1", "Web2"),
    ("Web2", "Edge1"),
]


def edges_covered(selected):
    selected = set(selected)
    return all(u in selected or v in selected for u, v in EDGES)


def has_core(selected):
    return bool(set(selected) & CORE)


def penalty(selected):
    selected = set(selected)
    cost = sum(SERVERS[name][1] for name in selected)
    value_gap = sum(20 - SERVERS[name][0] for name in NAMES if name not in selected)
    return cost + value_gap


def optimum():
    best = None
    for bits in product((False, True), repeat=len(NAMES)):
        subset = [name for name, chosen in zip(NAMES, bits, strict=False) if chosen]
        if has_core(subset) and edges_covered(subset):
            cost = penalty(subset)
            best = cost if best is None else min(best, cost)
    return best


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    selected = data.get("selected_servers")
    if not isinstance(selected, list):
        return False, "Missing 'selected_servers' list"
    for name in selected:
        if name not in SERVERS:
            return False, f"Unknown server {name!r}"
    if len(set(selected)) != len(selected):
        return False, "Duplicate server in selection"

    if not has_core(selected):
        return False, "Hard constraint violated: no Core server is monitored"
    if not edges_covered(selected):
        uncovered = [
            f"{u}-{v}"
            for u, v in EDGES
            if u not in set(selected) and v not in set(selected)
        ]
        return False, f"Uncovered network connection(s): {', '.join(uncovered)}"

    cost = sum(SERVERS[name][1] for name in selected)
    value = sum(SERVERS[name][0] for name in selected)
    if data.get("total_cost") is not None and data["total_cost"] != cost:
        return False, (
            f"Reported total_cost {data['total_cost']} does not match actual {cost}"
        )
    if (
        data.get("total_monitoring_value") is not None
        and data["total_monitoring_value"] != value
    ):
        return False, (
            f"Reported total_monitoring_value "
            f"{data['total_monitoring_value']} does not match actual {value}"
        )

    actual_penalty = penalty(selected)
    if data.get("soft_penalty") is not None and data["soft_penalty"] != actual_penalty:
        return False, (
            f"Reported soft_penalty {data['soft_penalty']} does not match "
            f"actual {actual_penalty}"
        )

    best = optimum()
    if actual_penalty != best:
        return False, (
            f"Suboptimal: soft penalty {actual_penalty} but the minimum is {best}"
        )

    return True, f"Valid optimal selection (soft penalty {actual_penalty})"


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
