A university needs to schedule 10 courses across 5 time slots and 4 rooms. Each course has specific requirements:

## Courses and Basic Requirements:

1. CS101: Requires a computer lab
2. CS201: Requires a computer lab
3. MA101: Must be scheduled in the first time slot
4. PH101: Requires a lab with special equipment
5. EN101: No special room requirements
6. BI101: Requires a lab with special equipment
7. HI101: No special room requirements
8. EC101: Must be scheduled in Room 2
9. PS101: Cannot be scheduled in the first time slot
10. AR101: No special room requirements

## Room Types:

- Room 1: Computer lab
- Room 2: Regular classroom
- Room 3: Regular classroom
- Room 4: Lab with special equipment

## Additional Constraints:

1. No two courses can be scheduled in the same room at the same time
2. CS101 and CS201 cannot be scheduled at the same time
3. PH101 and BI101 must be scheduled in the same time slot
4. PS101 cannot be scheduled in Room 1 or Room 2
5. PS101 and AR101 must be in the same room, with AR101 in the time slot immediately after PS101
6. HI101 must be in the same room as PS101 and AR101, and must be scheduled in the time slot immediately after AR101
7. EN101 must be in the same room as HI101, PS101, and AR101, and must be scheduled in the time slot immediately after EN101

A valid schedule assigns each course to exactly one time slot and one room while satisfying all of the above constraints.

Show that no valid schedule exists.