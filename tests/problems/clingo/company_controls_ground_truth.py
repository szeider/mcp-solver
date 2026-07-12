#!/usr/bin/env python3
"""Semantic validator for the company_controls transitive-control problem.

Recomputes the control relation from the ownership facts:

  owns(c1, c2, 60), owns(c1, c3, 20), owns(c2, c3, 40), owns(c3, c4, 51)

X controls Y when the shares of Y that X owns directly, plus the shares of Y
owned by companies X (transitively) controls, exceed 50%. Iterating this
fixpoint yields the control pairs {(c1,c2), (c1,c3), (c1,c4), (c3,c4)}.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

COMPANIES = ["c1", "c2", "c3", "c4"]
OWNS = {
    ("c1", "c2"): 60,
    ("c1", "c3"): 20,
    ("c2", "c3"): 40,
    ("c3", "c4"): 51,
}


def compute_controls():
    """Least fixpoint of the recursive control definition."""
    controls = set()
    changed = True
    while changed:
        changed = False
        for x in COMPANIES:
            controlled = {w for w in COMPANIES if (x, w) in controls}
            for y in COMPANIES:
                if x == y:
                    continue
                share = OWNS.get((x, y), 0)
                share += sum(OWNS.get((w, y), 0) for w in controlled)
                if share > 50 and (x, y) not in controls:
                    controls.add((x, y))
                    changed = True
    return controls


EXPECTED = compute_controls()


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    pairs = data.get("controls")
    if not isinstance(pairs, list):
        return False, "controls must be a list of [X, Y] pairs"

    got = set()
    for pair in pairs:
        if not isinstance(pair, (list | tuple)) or len(pair) != 2:
            return False, f"each control entry must be a 2-element list, got {pair!r}"
        x, y = pair[0], pair[1]
        if x not in COMPANIES or y not in COMPANIES:
            return False, f"unknown company in pair {pair!r}"
        if x == y:
            return False, f"pair must have X != Y, got {pair!r}"
        got.add((x, y))

    if got != EXPECTED:
        missing = sorted(EXPECTED - got)
        extra = sorted(got - EXPECTED)
        return False, (
            f"control set mismatch: missing {missing}, unexpected {extra}; "
            f"expected {sorted(EXPECTED)}"
        )

    return True, f"control relation matches expected {sorted(EXPECTED)}"


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
