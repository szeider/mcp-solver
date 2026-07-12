"""Semantic validator for the university_scheduling (unsatisfiable) problem.

The chained same-room / consecutive-slot constraints (items 5-7) cannot all be
satisfied, so no valid schedule exists. Item 7 requires EN101 to share the room
of HI101, PS101, and AR101 and to be scheduled in the slot immediately after
HI101. This was confirmed with an exact CP-SAT model, which reports UNSAT (the
instance was also UNSAT under the earlier self-referential wording of item 7,
so the verdict is unchanged). The only correct verdict is
{"satisfiable": false}.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    satisfiable = data.get("satisfiable")
    if satisfiable is False:
        return True, "correctly reported that no valid schedule exists"
    if satisfiable is True:
        return False, (
            "claimed satisfiable, but the chained constraints (items 5-7) make "
            "the instance unsatisfiable"
        )
    return False, f"satisfiable must be false, got {satisfiable!r}"


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
