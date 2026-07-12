#!/usr/bin/env python3
"""Semantic validator for the SEND + MORE = MONEY cryptarithmetic puzzle.

Reads a candidate solution as JSON on stdin, checks it semantically (not by
string comparison), and prints {"valid": bool, "message": str}. Exit code 0
if valid, 1 otherwise. Plain stdlib only.
"""

import json
import sys

LETTERS = "SENDMORY"


def word_value(assignment, word):
    value = 0
    for letter in word:
        value = value * 10 + assignment[letter]
    return value


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    satisfiable = data.get("satisfiable")
    if satisfiable is not True:
        # The puzzle is satisfiable, so any other verdict is wrong.
        return False, f"satisfiable must be true, got {satisfiable!r}"

    assignment = data.get("assignment")
    if not isinstance(assignment, dict):
        return False, "assignment must be an object mapping letters to digits"

    if set(assignment) != set(LETTERS):
        return False, (
            f"assignment must cover exactly {sorted(set(LETTERS))}, "
            f"got {sorted(assignment)}"
        )

    digits = {}
    for letter, digit in assignment.items():
        if not isinstance(digit, int) or isinstance(digit, bool):
            return False, f"digit for {letter} must be an integer, got {digit!r}"
        if not 0 <= digit <= 9:
            return False, f"digit for {letter} out of range [0,9]: {digit}"
        digits[letter] = digit

    used = list(digits.values())
    if len(set(used)) != len(used):
        return False, "letters must map to distinct digits"

    if digits["S"] == 0:
        return False, "leading digit S may not be 0"
    if digits["M"] == 0:
        return False, "leading digit M may not be 0"

    send = word_value(digits, "SEND")
    more = word_value(digits, "MORE")
    money = word_value(digits, "MONEY")

    if send + more != money:
        return False, (
            f"equation fails: SEND({send}) + MORE({more}) "
            f"= {send + more} != MONEY({money})"
        )

    # Validate optional reported word values if present.
    for name, expected in (("SEND", send), ("MORE", more), ("MONEY", money)):
        if name in data and data[name] != expected:
            return False, (
                f"reported {name}={data[name]} does not match assignment "
                f"value {expected}"
            )

    return True, (
        f"valid: SEND={send} + MORE={more} = MONEY={money} with distinct "
        "nonzero-leading digits"
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

    valid, message = validate(data)
    print(json.dumps({"valid": valid, "message": message}))
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
