"""Semantic validator for the university_scheduling optimization problem.

Checks every hard constraint on the reported schedule, recomputes the number
of soft-preference violations (only item 7: HI101 in the last slot), verifies
it matches the reported value, and confirms it equals the optimum.

The optimum is recomputed here by an in-validator exact oracle (stdlib only): a
backtracking search over (time_slot, room) assignments enforces exactly the
same hard constraints checked below and computes the minimum number of soft
violations. The soft count is non-negative, so as soon as the search exhibits a
feasible schedule with zero violations that is provably the global optimum; the
oracle finds one, so the optimum is 0.

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
SLOTS = [1, 2, 3, 4, 5]
ROOMS = [1, 2, 3, 4]


def hard_ok(slot, room):
    """Full hard-constraint predicate on a complete (slot, room) assignment.

    Mirrors exactly the hard constraints enforced in validate() below.
    """
    used = set()
    for c in COURSES:
        key = (slot[c], room[c])
        if key in used:
            return False  # no two courses share a room in the same slot
        used.add(key)
    if room["CS101"] != COMPUTER_LAB or room["CS201"] != COMPUTER_LAB:
        return False
    if slot["CS101"] == slot["CS201"]:
        return False
    if slot["MA101"] != 1:
        return False
    if room["PH101"] != SPECIAL_LAB or room["BI101"] != SPECIAL_LAB:
        return False
    if room["EC101"] != 2:
        return False
    if slot["PS101"] == 1:
        return False
    return room["PS101"] != room["MA101"]


def soft_violations(slot):
    """Soft-preference cost: 1 iff HI101 lands in the last slot (item 7)."""
    return 1 if slot["HI101"] == 5 else 0


def optimal_violations():
    """Exact minimum soft-violation count via constraint backtracking.

    Per-course (slot, room) domains are pre-reduced by the pinned hard
    constraints; the recursion prunes on the constraints checkable from a
    partial assignment and confirms completeness with hard_ok() at the leaf.
    The objective is non-negative, so the search stops once it reaches 0.
    """

    def domain(course):
        slots, rooms = SLOTS, ROOMS
        if course in ("CS101", "CS201"):
            rooms = [COMPUTER_LAB]
        if course in ("PH101", "BI101"):
            rooms = [SPECIAL_LAB]
        if course == "EC101":
            rooms = [2]
        if course == "MA101":
            slots = [1]
        if course == "PS101":
            slots = [s for s in SLOTS if s != 1]
        return [(s, r) for s in slots for r in rooms]

    doms = {c: domain(c) for c in COURSES}
    order = sorted(COURSES, key=lambda c: len(doms[c]))
    slot, room = {}, {}
    best = [None]

    def partial_ok():
        used = set()
        for c in slot:
            key = (slot[c], room[c])
            if key in used:
                return False
            used.add(key)
        if "CS101" in slot and "CS201" in slot and slot["CS101"] == slot["CS201"]:
            return False
        return not (
            "PS101" in slot and "MA101" in room and room["PS101"] == room["MA101"]
        )

    def dfs(i):
        if best[0] == 0:
            return  # reached the global lower bound of the objective
        if i == len(order):
            if hard_ok(slot, room):
                v = soft_violations(slot)
                if best[0] is None or v < best[0]:
                    best[0] = v
            return
        c = order[i]
        for s, r in doms[c]:
            slot[c], room[c] = s, r
            if partial_ok():
                dfs(i + 1)
            del slot[c]
            del room[c]

    dfs(0)
    return best[0]


OPTIMAL_VIOLATIONS = optimal_violations()  # previously hardcoded: 0, matches oracle


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
