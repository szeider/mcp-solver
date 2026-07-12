#!/usr/bin/env python3
"""Semantic validator for the 16x16 Sudoku puzzle.

A valid answer is a fully filled 16x16 grid over the symbols 1-9, A-G in
which every row, column, and 4x4 box contains each symbol exactly once,
and which agrees with all the given clues.

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys

SYMBOLS = set("123456789ABCDEFG")
SIZE = 16
BOX = 4

# Given puzzle; '.' marks an empty cell.
PUZZLE = [
    "..47..A.....3..5",
    ".6.....9..B.....",
    "....8...7....D..",
    "G....3.....1....",
    "..2......F....6.",
    "......7....C....",
    ".5......D......8",
    ".....9......4...",
    "...E.......8....",
    "..1....F.....7..",
    "........B.......",
    ".8..........E...",
    ".......D..9.....",
    "....F.........2.",
    "................",
    "...2............",
]


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    grid = data.get("solution")
    if not isinstance(grid, list) or len(grid) != SIZE:
        return False, f"Solution must be a list of {SIZE} rows"

    for r, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != SIZE:
            return False, f"Row {r} must have {SIZE} cells"
        for c, cell in enumerate(row):
            if cell not in SYMBOLS:
                return False, f"Cell ({r},{c}) has invalid symbol {cell!r}"

    # Row constraint.
    for r in range(SIZE):
        if set(grid[r]) != SYMBOLS:
            return False, f"Row {r} does not contain each symbol exactly once"

    # Column constraint.
    for c in range(SIZE):
        column = {grid[r][c] for r in range(SIZE)}
        if column != SYMBOLS:
            return False, f"Column {c} does not contain each symbol exactly once"

    # Box constraint.
    for br in range(0, SIZE, BOX):
        for bc in range(0, SIZE, BOX):
            block = {grid[br + dr][bc + dc] for dr in range(BOX) for dc in range(BOX)}
            if block != SYMBOLS:
                return False, (f"Box at ({br},{bc}) does not contain each symbol once")

    # Clue agreement.
    for r in range(SIZE):
        for c in range(SIZE):
            clue = PUZZLE[r][c]
            if clue != "." and grid[r][c] != clue:
                return False, (
                    f"Cell ({r},{c}) is {grid[r][c]!r} but clue requires {clue!r}"
                )

    return True, "Valid 16x16 Sudoku solution consistent with all clues"


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
