# Task Assignment

A manager needs to assign two tasks to two employees with certain constraints and preferences.

## Tasks and Employees
- Task A: Data Analysis
- Task B: Report Writing
- Employee 1: Alice
- Employee 2: Bob

## Constraints

### Hard Constraints
1. Each task must be assigned to exactly one employee
2. Alice cannot do both tasks (she's part-time)

### Preferences (minimize penalties)
1. Alice prefers Task A (penalty of 3 if she doesn't get it)
2. Bob prefers Task B (penalty of 2 if he doesn't get it)
3. Task A should ideally go to Alice (penalty of 1 if assigned to Bob)

## Task
Find the optimal assignment that satisfies all constraints and minimizes the total penalty.

## Expected Output
- Assignment of each task to an employee
- Total penalty incurred
- Verification that constraints are satisfied