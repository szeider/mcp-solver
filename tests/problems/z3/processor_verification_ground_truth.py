#!/usr/bin/env python3
"""Semantic validator for the processor parity-property verification.

The property "R3 always equals the parity bit of R0" does NOT hold: after the
sequence, R3 is the low bit of (memory[R0] XOR R0), not the bit-count parity
of R0. The correct verdict is therefore property_holds = false, accompanied by
a counterexample. This validator recomputes the instruction sequence in plain
Python from the counterexample's initial state and confirms it genuinely
falsifies the property.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys


def parity(value):
    return bin(value & 0xFF).count("1") % 2


def compute_r3(r0, memory):
    """Run instructions 1-3 and return R3 (the low bit of memory[R0] XOR R0)."""
    address = r0 & 0x7
    r1 = memory[address]
    r2 = r1 ^ r0
    return r2 & 0x1


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    property_holds = data.get("property_holds")
    if property_holds is True:
        return False, (
            "property_holds claimed true, but the property is false: R3 is "
            "the low bit of memory[R0] XOR R0, not the parity of R0"
        )
    if property_holds is not False:
        return False, (
            f"property_holds must be false for this instance, got {property_holds!r}"
        )

    ce = data.get("counterexample")
    if not isinstance(ce, dict):
        return False, "counterexample object required when property_holds is false"

    regs = ce.get("initial_registers")
    if not isinstance(regs, dict) or "R0" not in regs:
        return False, "counterexample.initial_registers must include R0"
    r0 = regs["R0"]
    if not isinstance(r0, int) or isinstance(r0, bool) or not 0 <= r0 <= 255:
        return False, f"R0 must be an integer in [0,255], got {r0!r}"

    memory = ce.get("initial_memory")
    if not isinstance(memory, list) or len(memory) != 8:
        return False, "counterexample.initial_memory must be a list of 8 integers"
    for value in memory:
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"memory entries must be integers, got {value!r}"
        if not 0 <= value <= 255:
            return False, f"memory value {value} out of range [0,255]"

    r3 = compute_r3(r0, memory)
    expected_parity = parity(r0)
    if r3 == expected_parity:
        return False, (
            f"claimed counterexample does not falsify the property: with "
            f"R0={r0}, memory={memory}, R3={r3} equals parity(R0)="
            f"{expected_parity}"
        )

    return True, (
        f"valid counterexample: R0={r0}, memory={memory} yields R3={r3} but "
        f"parity(R0)={expected_parity}; the property is correctly reported as "
        "not holding"
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
