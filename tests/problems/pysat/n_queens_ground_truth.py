#!/usr/bin/env python3
"""Semantic validator for the n=8 n-queens problem.

Reads a candidate solution as JSON from stdin, checks it semantically,
prints {"valid": bool, "message": str} and exits 0 on valid, 1 otherwise.
"""

import json
import sys

N = 8


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    queens = data.get("queens")
    if not isinstance(queens, list):
        return False, "Missing 'queens' list"
    if len(queens) != N:
        return False, f"Expected {N} queens, got {len(queens)}"

    cols = []
    rows = []
    for entry in queens:
        if not isinstance(entry, list) or len(entry) != 2:
            return False, f"Each queen must be a [column, row] pair, got {entry}"
        col, row = entry
        if not (isinstance(col, int) and isinstance(row, int)):
            return False, f"Coordinates must be integers, got {entry}"
        if not (0 <= col < N and 0 <= row < N):
            return False, f"Coordinate out of range [0,{N - 1}]: {entry}"
        cols.append(col)
        rows.append(row)

    if len(set(cols)) != N:
        return False, "Two queens share a column"
    if len(set(rows)) != N:
        return False, "Two queens share a row"

    for i in range(N):
        for j in range(i + 1, N):
            if abs(cols[i] - cols[j]) == abs(rows[i] - rows[j]):
                return False, (f"Queens {queens[i]} and {queens[j]} share a diagonal")

    return True, "Valid 8-queens placement"


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
