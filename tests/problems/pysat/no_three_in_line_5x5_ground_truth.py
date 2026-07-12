#!/usr/bin/env python3
"""Semantic validator for the no-three-in-line problem on a 5x5 grid.

A valid answer places exactly 10 distinct grid points (coordinates in
0..4) such that no three of them are collinear.

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys
from itertools import combinations

GRID = 5
REQUIRED_POINTS = 10


def collinear(a, b, c):
    """Return True if points a, b, c lie on a common straight line."""
    return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    points = data.get("points")
    if not isinstance(points, list):
        return False, "Missing 'points' list"

    normalized = []
    for entry in points:
        if not isinstance(entry, list) or len(entry) != 2:
            return False, f"Each point must be an [x, y] pair, got {entry}"
        x, y = entry
        if not (isinstance(x, int) and isinstance(y, int)):
            return False, f"Coordinates must be integers, got {entry}"
        if not (0 <= x < GRID and 0 <= y < GRID):
            return False, f"Point out of the 5x5 grid: {entry}"
        normalized.append((x, y))

    if len(set(normalized)) != len(normalized):
        return False, "Duplicate points in the placement"

    if len(normalized) != REQUIRED_POINTS:
        return False, (
            f"Expected exactly {REQUIRED_POINTS} points, got {len(normalized)}"
        )

    for a, b, c in combinations(normalized, 3):
        if collinear(a, b, c):
            return False, f"Points {a}, {b}, {c} are collinear"

    return True, "Valid placement of 10 points, no three collinear"


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
