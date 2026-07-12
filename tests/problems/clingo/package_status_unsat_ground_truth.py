#!/usr/bin/env python3
"""Semantic validator for the package_status (unsatisfiable) problem.

The package is asserted both lost and delivered, while an integrity constraint
forbids being both at once, so the program has no answer set. The only correct
verdict is {"satisfiable": false}.

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
        return True, "correctly reported no answer set (inconsistent program)"
    if satisfiable is True:
        return False, (
            "claimed satisfiable, but asserting both lost and delivered "
            "violates the ':- lost, delivered' constraint; no answer set exists"
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
