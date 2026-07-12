#!/usr/bin/env python3
"""Semantic validator for the sum-of-cubes induction proof.

The identity sum(i^3, i=1..n) = n^2(n+1)^2/4 is a true theorem (verified here
independently for a range of n), so the only correct verdict is
verified = true. Reads JSON on stdin, prints {"valid": bool, "message": str},
exits 0/1. Plain stdlib only.
"""

import json
import sys


def identity_holds():
    """Independently confirm the closed form for many n."""
    for n in range(0, 200):
        computed = sum(i * i * i for i in range(1, n + 1))
        expected = n * n * (n + 1) * (n + 1) // 4
        if computed != expected:
            return False
    return True


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    # The identity is mathematically true; the correct verdict is fixed.
    assert identity_holds(), "closed form should hold for all n"

    verified = data.get("verified")
    if verified is True:
        return True, (
            "correct verdict: sum(i^3, i=1..n) = n^2(n+1)^2/4 is a true identity"
        )
    if verified is False:
        return False, (
            "verified claimed false, but the identity is a true theorem "
            "(verified independently for n=0..199)"
        )
    return False, f"verified must be true, got {verified!r}"


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
