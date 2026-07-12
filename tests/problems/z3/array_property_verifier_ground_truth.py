#!/usr/bin/env python3
"""Semantic validator for the 8-slot sorted-array problem.

Checks that the reported array satisfies all five conditions. Reads JSON on
stdin, prints {"valid": bool, "message": str}, exits 0/1. Plain stdlib only.
"""

import json
import sys


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    if data.get("satisfiable") is not True:
        return False, (
            f"satisfiable must be true (a valid array exists), got "
            f"{data.get('satisfiable')!r}"
        )

    array = data.get("array")
    if not isinstance(array, list):
        return False, "array must be a list of 8 integers"
    if len(array) != 8:
        return False, f"array must have 8 entries, got {len(array)}"
    for value in array:
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"array entries must be integers, got {value!r}"

    # 1. sorted ascending (non-decreasing)
    for i in range(7):
        if array[i] > array[i + 1]:
            return False, f"not sorted ascending at index {i}: {array}"

    # 2. each value in [1, 15]
    for value in array:
        if not 1 <= value <= 15:
            return False, f"value {value} out of range [1,15]"

    # 3. exactly one 7
    if array.count(7) != 1:
        return False, f"the number 7 must appear exactly once, appears {array.count(7)}"

    # 4. sum equals 60
    if sum(array) != 60:
        return False, f"sum is {sum(array)}, must be 60"

    # 5. no two adjacent elements both even
    for i in range(7):
        if array[i] % 2 == 0 and array[i + 1] % 2 == 0:
            return False, (
                f"adjacent even values at indices {i},{i + 1}: "
                f"{array[i]},{array[i + 1]}"
            )

    return True, f"valid array {array} meets all five conditions"


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
