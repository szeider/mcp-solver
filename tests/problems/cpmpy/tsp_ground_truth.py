#!/usr/bin/env python3
"""Semantic validator for the 9-city Austria TSP.

Checks the reported tour is a permutation of cities 1..9, recomputes its
closed-cycle length, verifies the reported total, and confirms it equals the
optimum. The optimum is found by brute force here (fix city 1, permute the
other 8: 8! = 40320 tours), which is cheap.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys
from itertools import permutations

# Distances in km, indexed 1..9 (row/col 0 unused).
DIST = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 65, 60, 184, 195, 319, 299, 478, 631],
    [0, 65, 0, 125, 119, 130, 254, 234, 413, 566],
    [0, 60, 125, 0, 184, 157, 281, 261, 440, 593],
    [0, 184, 119, 184, 0, 208, 252, 136, 315, 468],
    [0, 195, 130, 157, 208, 0, 136, 280, 459, 629],
    [0, 319, 254, 281, 252, 136, 0, 217, 391, 566],
    [0, 299, 234, 261, 136, 280, 217, 0, 188, 343],
    [0, 478, 413, 440, 315, 459, 391, 188, 0, 157],
    [0, 631, 566, 593, 468, 629, 566, 343, 157, 0],
]
CITIES = list(range(1, 10))


def tour_length(order):
    n = len(order)
    return sum(DIST[order[i]][order[(i + 1) % n]] for i in range(n))


def optimum():
    best = None
    for perm in permutations(range(2, 10)):
        order = (1, *perm)
        length = tour_length(order)
        if best is None or length < best:
            best = length
    return best


OPT = optimum()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    tour = data.get("tour")
    if not isinstance(tour, list) or len(tour) != 9:
        return False, "tour must be a list of 9 city numbers"
    if sorted(tour) != CITIES:
        return False, f"tour must be a permutation of {CITIES}, got {tour}"

    length = tour_length(tour)

    total = data.get("total_distance")
    if not isinstance(total, int) or isinstance(total, bool):
        return False, f"total_distance must be an integer, got {total!r}"
    if total != length:
        return False, (
            f"total_distance {total} does not match recomputed tour length {length}"
        )

    if length != OPT:
        return False, f"tour length {length} is not optimal (optimum is {OPT})"

    return True, f"valid optimal tour of length {length} km"


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
