#!/usr/bin/env python3
"""Semantic validator for the party_invitation problem.

Recomputes the unique answer set:
  - Nothing says Alice is not coming, so Alice is invited (default).
  - Bob is invited only if Alice is NOT invited -> Bob is not invited.
  - Carol needs both Alice and Bob invited -> Carol is not invited.
So the invited set is exactly {alice}.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

EXPECTED = {"alice"}


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    invited = data.get("invited")
    if not isinstance(invited, list):
        return False, "invited must be a list of names"
    for name in invited:
        if not isinstance(name, str):
            return False, f"invited entries must be strings, got {name!r}"

    got = {name.strip().lower() for name in invited}
    if got != EXPECTED:
        return False, (
            f"invited set {sorted(got)} does not match expected {sorted(EXPECTED)}"
        )

    return True, "invited set is exactly {alice}, matching the unique answer set"


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
