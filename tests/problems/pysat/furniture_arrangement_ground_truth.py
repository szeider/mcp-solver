#!/usr/bin/env python3
"""Semantic validator for the furniture arrangement problem.

Reads a candidate solution as JSON from stdin, checks it semantically,
prints {"valid": bool, "message": str} and exits 0 on valid, 1 otherwise.
"""

import json
import sys

FURNITURE = {"Sofa", "TV", "Bookshelf", "Dining Table", "Desk"}
ROOMS = {"Living Room", "Bedroom", "Study", "Dining Room"}


def validate(data):
    if data.get("satisfiable") is not True:
        return False, "Problem is satisfiable, but candidate reports otherwise"

    arrangement = data.get("arrangement")
    if not isinstance(arrangement, dict):
        return False, "Missing 'arrangement' object"

    if set(arrangement) != FURNITURE:
        return False, (
            f"Arrangement must place exactly {sorted(FURNITURE)}, "
            f"got {sorted(arrangement)}"
        )

    for item, room in arrangement.items():
        if room not in ROOMS:
            return False, f"Invalid room '{room}' for {item}"

    # Capacity constraints.
    counts = {room: 0 for room in ROOMS}
    for room in arrangement.values():
        counts[room] += 1
    if counts["Living Room"] > 3:
        return False, "Living Room holds more than 3 pieces"
    if counts["Bedroom"] > 2:
        return False, "Bedroom holds more than 2 pieces"

    # Co-location and placement constraints.
    if arrangement["TV"] != arrangement["Sofa"]:
        return False, "TV and Sofa are not in the same room"
    if arrangement["Desk"] != "Study":
        return False, "Desk is not in the Study"
    if arrangement["Bookshelf"] != "Study":
        return False, "Bookshelf is not in the Study"
    if arrangement["Dining Table"] != "Dining Room":
        return False, "Dining Table is not in the Dining Room"
    if arrangement["Sofa"] == "Study":
        return False, "Sofa is placed in the Study (it does not fit)"

    return True, "Valid furniture arrangement"


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
