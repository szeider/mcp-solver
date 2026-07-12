A university needs to schedule 10 courses across 5 time slots and 4 rooms. Each course has specific requirements:

1. Course CS101 requires a computer lab and cannot be scheduled at the same time as CS201
2. Course CS201 requires a computer lab
3. Course MA101 must be in the first time slot
4. Course PH101 requires a lab with special equipment
5. Course EN101 has no special requirements
6. Course BI101 requires a lab with special equipment
7. Course HI101 prefers not to be in the last time slot
8. Course EC101 must be scheduled in Room 2
9. Course PS101 cannot be in the first time slot and must not be in the same room as MA101
10. Course AR101 has no special requirements

The rooms have the following attributes:
- Room 1 is a computer lab
- Room 2 is a regular classroom
- Room 3 is a regular classroom
- Room 4 is a lab with special equipment

Schedule all courses to minimize the number of preferences violated while ensuring all hard constraints are satisfied.

## Output Format

Time slots are numbered 1-5 and rooms 1-4. The only soft preference is item 7
(HI101 prefers not to be in the last time slot, slot 5); every other item is a
hard constraint. Two courses may not share the same room in the same time
slot.

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (a feasible schedule exists).
- `preferences_violated` (integer): the number of soft preferences violated by
  the schedule (0 or 1 here).
- `schedule` (list of 10 objects): one per course, each with:
  - `course` (string): the course code (e.g. `CS101`).
  - `time_slot` (integer 1-5)
  - `room` (integer 1-4)

The optimal schedule violates 0 preferences.

Example:

```json
{
  "satisfiable": true,
  "preferences_violated": 0,
  "schedule": [
    {"course": "CS101", "time_slot": 4, "room": 1},
    {"course": "CS201", "time_slot": 1, "room": 1},
    {"course": "MA101", "time_slot": 1, "room": 3},
    {"course": "PH101", "time_slot": 1, "room": 4},
    {"course": "EN101", "time_slot": 5, "room": 3},
    {"course": "BI101", "time_slot": 5, "room": 4},
    {"course": "HI101", "time_slot": 3, "room": 4},
    {"course": "EC101", "time_slot": 2, "room": 2},
    {"course": "PS101", "time_slot": 4, "room": 4},
    {"course": "AR101", "time_slot": 2, "room": 1}
  ]
}
```