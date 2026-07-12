"""Semantic validator for the university_scheduling optimization problem.

Checks every hard constraint on the reported schedule, recomputes the number
of soft-preference violations (only item 7: HI101 in the last slot), verifies
it matches the reported value, and confirms it equals the optimum. A feasible
schedule with zero violations exists (found once via CP-SAT), so the optimum
is 0.

Reads JSON on stdin, prints {"valid": bool, "message": str}, exits 0/1.
Plain stdlib only.
"""

import json
import sys

COURSES = [
    "CS101",
    "CS201",
    "MA101",
    "PH101",
    "EN101",
    "BI101",
    "HI101",
    "EC101",
    "PS101",
    "AR101",
]
COMPUTER_LAB = 1  # Room 1
SPECIAL_LAB = 4  # Room 4
OPTIMAL_VIOLATIONS = 0


def validate(data):
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"
    if data.get("satisfiable") is not True:
        return False, f"satisfiable must be true, got {data.get('satisfiable')!r}"

    schedule = data.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 10:
        return False, "schedule must be a list of 10 course objects"

    slot = {}
    room = {}
    for entry in schedule:
        if not isinstance(entry, dict):
            return False, "each schedule entry must be an object"
        for field in ("course", "time_slot", "room"):
            if field not in entry:
                return False, f"schedule entry missing field {field}"
        course = entry["course"]
        if course not in COURSES:
            return False, f"unknown course {course!r}"
        if course in slot:
            return False, f"course {course} scheduled more than once"
        t, r = entry["time_slot"], entry["room"]
        if not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= 5:
            return False, f"{course} time_slot must be integer 1-5, got {t!r}"
        if not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= 4:
            return False, f"{course} room must be integer 1-4, got {r!r}"
        slot[course] = t
        room[course] = r

    if set(slot) != set(COURSES):
        return False, "schedule must assign every course exactly once"

    # No two courses share the same room in the same time slot.
    seen = {}
    for course in COURSES:
        key = (slot[course], room[course])
        if key in seen:
            return False, (
                f"room conflict: {course} and {seen[key]} both in room "
                f"{room[course]} at slot {slot[course]}"
            )
        seen[key] = course

    # Hard room / slot requirements.
    if room["CS101"] != COMPUTER_LAB:
        return False, "CS101 must be in the computer lab (Room 1)"
    if room["CS201"] != COMPUTER_LAB:
        return False, "CS201 must be in the computer lab (Room 1)"
    if slot["CS101"] == slot["CS201"]:
        return False, "CS101 and CS201 must not be at the same time"
    if slot["MA101"] != 1:
        return False, "MA101 must be in the first time slot"
    if room["PH101"] != SPECIAL_LAB:
        return False, "PH101 must be in the special-equipment lab (Room 4)"
    if room["BI101"] != SPECIAL_LAB:
        return False, "BI101 must be in the special-equipment lab (Room 4)"
    if room["EC101"] != 2:
        return False, "EC101 must be in Room 2"
    if slot["PS101"] == 1:
        return False, "PS101 must not be in the first time slot"
    if room["PS101"] == room["MA101"]:
        return False, "PS101 must not be in the same room as MA101"

    # Soft preference: HI101 prefers not the last slot (item 7).
    violations = 1 if slot["HI101"] == 5 else 0

    reported = data.get("preferences_violated")
    if not isinstance(reported, int) or isinstance(reported, bool):
        return False, f"preferences_violated must be an integer, got {reported!r}"
    if reported != violations:
        return False, (
            f"preferences_violated {reported} does not match recomputed {violations}"
        )
    if violations != OPTIMAL_VIOLATIONS:
        return False, (
            f"{violations} preference(s) violated, but the optimum is "
            f"{OPTIMAL_VIOLATIONS}"
        )

    return True, "valid schedule satisfying all hard constraints with 0 violations"


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
