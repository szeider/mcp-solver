#!/usr/bin/env python3
"""Semantic validator for the birds_fly default-reasoning problem.

Recomputes the expected answer-set semantics from the knowledge base:

  - Tweety: bird, not penguin, not injured -> flies by default; mobile;
    has feathers.
  - Opus: penguin -> cannot fly (exception); not mobile; has feathers.
  - Woody: woodpecker and injured -> cannot fly (exception); not mobile;
    has feathers.
  - Polly: airplane, stated to fly (fact); mobile; no feathers (not a bird).

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

EXPECTED = {
    "tweety": {
        "can_fly": True,
        "mobile": True,
        "has_feathers": True,
        "fly_reasoning": "default",
    },
    "opus": {
        "can_fly": False,
        "mobile": False,
        "has_feathers": True,
        "fly_reasoning": "exception",
    },
    "woody": {
        "can_fly": False,
        "mobile": False,
        "has_feathers": True,
        "fly_reasoning": "exception",
    },
    "polly": {
        "can_fly": True,
        "mobile": True,
        "has_feathers": False,
        "fly_reasoning": "fact",
    },
}


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    entities = data.get("entities")
    if not isinstance(entities, dict):
        return False, "entities must be an object keyed by entity name"

    got_keys = {k.lower() for k in entities}
    if got_keys != set(EXPECTED):
        return False, (
            f"entities must cover exactly {sorted(EXPECTED)}, got {sorted(got_keys)}"
        )

    normalized = {k.lower(): v for k, v in entities.items()}
    for name, expected in EXPECTED.items():
        ent = normalized[name]
        if not isinstance(ent, dict):
            return False, f"{name} must map to an object"
        for field in ("can_fly", "mobile", "has_feathers"):
            if field not in ent:
                return False, f"{name} missing field {field}"
            if not isinstance(ent[field], bool):
                return False, f"{name}.{field} must be a boolean"
            if ent[field] != expected[field]:
                return False, (
                    f"{name}.{field} = {ent[field]}, expected {expected[field]}"
                )
        reasoning = ent.get("fly_reasoning")
        if not isinstance(reasoning, str):
            return False, f"{name}.fly_reasoning must be a string"
        if reasoning.strip().lower() != expected["fly_reasoning"]:
            return False, (
                f"{name}.fly_reasoning = {reasoning!r}, expected "
                f"{expected['fly_reasoning']!r}"
            )

    return True, "all four entities match the expected answer-set semantics"


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
