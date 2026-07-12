## Problem: Assigning Shifts to Employees

We need to assign two employees, **anna** and **ben**, to work three shifts
(morning, afternoon, evening). Each shift must be covered by exactly one
employee, and no employee may work more than two shifts. The employees have
preferences about which shifts they would like to work:

| Employee | Preferred shifts             |
| -------- | ---------------------------- |
| anna     | morning, afternoon, evening  |
| ben      | morning                      |

The goal is to:

- Ensure each shift is covered by exactly one employee.
- Assign at most two shifts to any employee.
- Maximize the number of satisfied preferences, i.e. the number of shifts whose
  assigned employee lists that shift as a preferred shift.

Because anna can work at most two shifts, at least one of her three preferences
must go unsatisfied, so not all preferences can be met and a genuine trade-off
is required.

Model and solve the previous problem using answer set programming (ASP),
optimizing for the maximum number of satisfied preferences.

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` if a valid assignment exists.
- `assignment` (object): maps each shift name `morning`, `afternoon`,
  `evening` to the name of the employee (`anna` or `ben`) assigned to it.
  Required when `satisfiable` is `true`.
- `satisfied_preferences` (integer): the number of preferences satisfied by the
  returned assignment. This must equal the maximum achievable value.

The hard constraints require every shift covered by exactly one of the two
employees, with no employee working more than two shifts (so one employee works
two shifts and the other works one). Among all assignments satisfying the hard
constraints, the returned one must maximize `satisfied_preferences`.

Example (this is the unique optimum):

```json
{
  "satisfiable": true,
  "assignment": {"morning": "ben", "afternoon": "anna", "evening": "anna"},
  "satisfied_preferences": 3
}
```

Here ben works his preferred morning shift, and anna works the afternoon and
evening (two of her three preferred shifts); only anna's morning preference is
left unsatisfied, giving three satisfied preferences, which is the maximum.
