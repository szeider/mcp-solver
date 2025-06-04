# Workshop Scheduling

A company needs to schedule 3 workshops for its employees.

## Workshops
- Workshop A: Leadership Training
- Workshop B: Technical Skills
- Workshop C: Communication Skills

## Time Slots Available
- Morning (9 AM - 12 PM)
- Afternoon (1 PM - 4 PM)

## Hard Constraints
1. Each workshop must be assigned to exactly one time slot
2. No two workshops can occur at the same time
3. Workshop A must occur in the Morning slot (trainer availability)
4. Workshop B must occur in the Morning slot (equipment setup required)
5. Workshop C must occur in the Morning slot (external trainer constraint)

## Soft Constraints
1. Employees prefer Workshop A in the morning (satisfaction: 5)
2. Employees prefer Workshop B in the afternoon (penalty: 4 if in morning)
3. Employees prefer Workshop C in the afternoon (penalty: 3 if in morning)

## Task
Find a schedule that satisfies all constraints and maximizes employee satisfaction.

## Expected Output
- Assignment of each workshop to a time slot
- Total satisfaction score achieved