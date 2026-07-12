#!/usr/bin/env python3
"""Semantic validator for the carpet_cutting strip-packing problem.

Checks the reported placements: each piece matches its dimensions (rotation
allowed), all pieces lie within the 12-ft-wide roll and within the claimed
length, no two pieces overlap, and the claimed roll length is the optimum.

This is 2D strip packing and cannot be brute-forced cheaply, so the optimum is
hardcoded. It was computed once with an exact CP-SAT model (CPMpy/OR-Tools
minimizing the used length over a 12-wide roll): the minimum length is 18 ft.
Re-running that model reproduces 18.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

ROLL_WIDTH = 12
# Piece index (1-based) -> (a, b) nominal dimensions.
PIECES = {1: (6, 8), 2: (4, 5), 3: (5, 7), 4: (8, 10)}
OPTIMAL_LENGTH = 18  # exact minimum, computed once via CP-SAT (see docstring)


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    separated = ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
    return not separated


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    placements = data.get("placements")
    if not isinstance(placements, list) or len(placements) != 4:
        return False, "placements must be a list of 4 objects"

    rects = []
    seen_pieces = set()
    for p in placements:
        if not isinstance(p, dict):
            return False, "each placement must be an object"
        for field in ("piece", "x", "y", "width", "height"):
            if field not in p:
                return False, f"placement missing field {field}"
            if not isinstance(p[field], int) or isinstance(p[field], bool):
                return False, f"placement field {field} must be an integer"

        piece = p["piece"]
        if piece not in PIECES:
            return False, f"unknown piece index {piece}"
        if piece in seen_pieces:
            return False, f"piece {piece} placed more than once"
        seen_pieces.add(piece)

        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        a, b = PIECES[piece]
        if {w, h} != {a, b}:
            return False, (
                f"piece {piece} placed as {w}x{h}, must be {a}x{b} (rotation allowed)"
            )
        if x < 0 or y < 0:
            return False, f"piece {piece} has negative coordinate ({x},{y})"
        if x + w > ROLL_WIDTH:
            return False, (
                f"piece {piece} exceeds roll width: x+width = {x + w} > {ROLL_WIDTH}"
            )
        rects.append((x, y, w, h))

    if seen_pieces != set(PIECES):
        return False, f"placements must cover all pieces {sorted(PIECES)}"

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if overlaps(rects[i], rects[j]):
                return False, f"pieces overlap: {rects[i]} and {rects[j]}"

    used_length = max(y + h for (_, y, _, h) in rects)

    reported = data.get("roll_length_used")
    if not isinstance(reported, int) or isinstance(reported, bool):
        return False, f"roll_length_used must be an integer, got {reported!r}"
    if reported != used_length:
        return False, (
            f"roll_length_used {reported} does not match packing extent {used_length}"
        )
    if used_length > 50:
        return False, f"packing length {used_length} exceeds roll length 50"
    if used_length != OPTIMAL_LENGTH:
        return False, (
            f"roll length {used_length} is not optimal (minimum is {OPTIMAL_LENGTH} ft)"
        )

    return True, f"valid packing using the optimal {used_length} ft of roll"


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
