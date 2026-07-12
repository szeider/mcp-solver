"""Semantic validator for the zebra / desks logic puzzle.

Checks that the reported assignment is a complete bijection over desks, names,
subjects, and foods, and that every clue in the statement holds. The puzzle
has a unique solution, so this passes only that solution.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

NAMES = {"emily", "anna", "ben", "liam"}
SUBJECTS = {"biology", "chemistry", "physics", "literature"}
FOODS = {"pizza", "pasta", "apples", "ice cream"}


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    rows = data.get("assignment")
    if not isinstance(rows, list) or len(rows) != 4:
        return False, "assignment must be a list of 4 desk objects"

    by_desk = {}
    for row in rows:
        if not isinstance(row, dict):
            return False, "each assignment entry must be an object"
        for field in ("desk", "name", "subject", "food"):
            if field not in row:
                return False, f"assignment entry missing field {field}"
        desk = row["desk"]
        if not isinstance(desk, int) or desk not in {1, 2, 3, 4}:
            return False, f"desk must be an integer 1-4, got {desk!r}"
        if desk in by_desk:
            return False, f"desk {desk} assigned twice"
        by_desk[desk] = {
            "name": str(row["name"]).strip().lower(),
            "subject": str(row["subject"]).strip().lower(),
            "food": str(row["food"]).strip().lower(),
        }

    if set(by_desk) != {1, 2, 3, 4}:
        return False, "desks must be exactly 1,2,3,4"

    names = {by_desk[d]["name"] for d in by_desk}
    subjects = {by_desk[d]["subject"] for d in by_desk}
    foods = {by_desk[d]["food"] for d in by_desk}
    if names != NAMES:
        return (
            False,
            f"names must be a bijection over {sorted(NAMES)}, got {sorted(names)}",
        )
    if subjects != SUBJECTS:
        return False, (
            f"subjects must be a bijection over {sorted(SUBJECTS)}, got "
            f"{sorted(subjects)}"
        )
    if foods != FOODS:
        return (
            False,
            f"foods must be a bijection over {sorted(FOODS)}, got {sorted(foods)}",
        )

    def desk_of(attr, value):
        for d in by_desk:
            if by_desk[d][attr] == value:
                return d
        return None

    # Clue 1: Biology student at Desk 3.
    if by_desk[3]["subject"] != "biology":
        return False, "clue violated: Biology student must be at Desk 3"

    # Clue 2: Pizza student adjacent to Chemistry student.
    pizza_desk = desk_of("food", "pizza")
    chem_desk = desk_of("subject", "chemistry")
    if abs(pizza_desk - chem_desk) != 1:
        return False, (
            "clue violated: Pizza student must sit next to the Chemistry student"
        )

    # Clue 3: Physics student not at Desk 2.
    if by_desk[2]["subject"] == "physics":
        return False, "clue violated: Physics student must not be at Desk 2"

    # Clue 4: Desk 2 likes Pasta.
    if by_desk[2]["food"] != "pasta":
        return False, "clue violated: Desk 2 student must like Pasta"

    # Clue 5: Anna likes apples.
    if by_desk[desk_of("name", "anna")]["food"] != "apples":
        return False, "clue violated: Anna must like apples"

    # Clue 6: Ben is not studying Biology.
    if by_desk[desk_of("name", "ben")]["subject"] == "biology":
        return False, "clue violated: Ben must not study Biology"

    # Clue 7: Emily at Desk 1.
    if by_desk[1]["name"] != "emily":
        return False, "clue violated: Emily must be at Desk 1"

    # Clue 8: Desk 4 likes ice cream.
    if by_desk[4]["food"] != "ice cream":
        return False, "clue violated: Desk 4 student must like ice cream"

    # Clue 9: Physics student is Anna or Liam (even-length name).
    physics_name = by_desk[desk_of("subject", "physics")]["name"]
    if physics_name not in {"anna", "liam"}:
        return False, (
            "clue violated: Physics student's name must have an even number of "
            "letters (Anna or Liam)"
        )

    return True, "assignment satisfies every clue (the unique solution)"


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
