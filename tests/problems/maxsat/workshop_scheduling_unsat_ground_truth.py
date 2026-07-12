#!/usr/bin/env python3
"""Semantic validator for the workshop scheduling problem (UNSAT by design).

The hard constraints require workshops A, B, and C to all occupy the
Morning slot while forbidding any two workshops from sharing a slot. With
only two slots this is unsatisfiable, so the correct answer is
{"satisfiable": false}.

A candidate that claims satisfiable=true is checked against the hard
constraints and rejected, naming the violated constraint.

Reads a candidate solution as JSON from stdin, prints
{"valid": bool, "message": str}, exits 0 on valid, 1 otherwise.
"""

import json
import sys

WORKSHOPS = ["Workshop A", "Workshop B", "Workshop C"]
SLOTS = {"Morning", "Afternoon"}
# Each workshop is forced into the Morning slot by a hard constraint.
FORCED_MORNING = {"Workshop A", "Workshop B", "Workshop C"}


def validate(data):
    satisfiable = data.get("satisfiable")

    if satisfiable is False:
        return True, "Correctly reports the instance as unsatisfiable"

    if satisfiable is not True:
        return False, "Missing or invalid 'satisfiable' field"

    # Candidate claims SAT; the instance is provably UNSAT. Find the flaw.
    schedule = data.get("schedule") or data.get("assignment")
    if not isinstance(schedule, dict) or set(schedule) != set(WORKSHOPS):
        return False, (
            "Claimed satisfiable, but the hard constraints are unsatisfiable: "
            "A, B, and C are all forced into the Morning slot while no two "
            "workshops may share a slot"
        )

    for ws, slot in schedule.items():
        if slot not in SLOTS:
            return False, f"{ws} assigned to unknown slot {slot!r}"

    # Forced-Morning hard constraints.
    for ws in FORCED_MORNING:
        if schedule[ws] != "Morning":
            return False, f"Hard constraint violated: {ws} must be in Morning"

    # No two workshops in the same slot.
    used = {}
    for ws, slot in schedule.items():
        if slot in used:
            return False, (
                f"Hard constraint violated: {used[slot]} and {ws} share the {slot} slot"
            )
        used[slot] = ws

    # Unreachable: the two constraint groups cannot both hold.
    return False, (
        "Claimed a valid schedule, but the forced-Morning and "
        "no-shared-slot constraints are jointly unsatisfiable"
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

    ok, message = validate(data)
    print(json.dumps({"valid": ok, "message": message}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
