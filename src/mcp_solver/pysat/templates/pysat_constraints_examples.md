# PySAT Constraint Helper Functions

This guide explains how to use the cardinality constraint helper functions provided with PySAT mode in MCP Solver.

## Available Helper Functions

The following helper functions are available to create common constraints:

| Function            | Description                                             |
|---------------------|---------------------------------------------------------|
| `at_most_k`         | At most k variables are true                            |
| `at_least_k`        | At least k variables are true                           |
| `exactly_k`         | Exactly k variables are true                            |
| `at_most_one`       | At most one variable is true (optimized for k=1)        |
| `exactly_one`       | Exactly one variable is true (optimized for k=1)        |
| `implies`           | If a is true, then b must be true                       |
| `mutually_exclusive`| At most one variable is true (same as at_most_one)      |
| `if_then_else`      | If condition then x else y                              |

## Basic Usage

All constraint helpers return lists of clauses that you can add to your PySAT formula:

```python
from pysat.formula import CNF

formula = CNF()

# Some variables
courses = [1, 2, 3, 4, 5]  # Variable IDs for courses

# Add constraint: Take at most 3 courses
formula.extend(at_most_k(courses, 3))

# Add constraint: Take at least 2 courses
formula.extend(at_least_k(courses, 2))
```

## Example Scenarios

### 1. Scheduling: At Most 2 Per Day

Suppose variables 1-5 represent courses on Monday, and 6-10 represent courses on Tuesday. To ensure at most 2 courses per day:

```python
monday_courses = [1, 2, 3, 4, 5]
tuesday_courses = [6, 7, 8, 9, 10]

formula = CNF()
formula.extend(at_most_k(monday_courses, 2))  # At most 2 courses on Monday
formula.extend(at_most_k(tuesday_courses, 2))  # At most 2 courses on Tuesday
```

### 2. Assignment: Exactly One Assignment

Suppose variables 11-15 represent assigning a task to different people. To ensure the task is assigned to exactly one person:

```python
task_assignments = [11, 12, 13, 14, 15]

formula = CNF()
formula.extend(exactly_one(task_assignments))  # Task assigned to exactly one person
```

### 3. Selection: Exactly K Items

Suppose variables 20-30 represent possible items to select. To select exactly 5 items:

```python
possible_items = list(range(20, 31))  # Variables 20-30

formula = CNF()
formula.extend(exactly_k(possible_items, 5))  # Select exactly 5 items
```

### 4. Dependencies: If-Then Relationships

Suppose variable 40 represents taking a prerequisite course, and 41 represents taking an advanced course. 
To ensure the prerequisite is taken if the advanced course is taken:

```python
prerequisite = 40
advanced_course = 41

formula = CNF()
formula.extend(implies(advanced_course, prerequisite))  # If taking advanced, must take prerequisite
```

### 5. Mutual Exclusion

Suppose variables 50-55 represent different job positions. To ensure a person can hold at most one position:

```python
job_positions = list(range(50, 56))  # Variables 50-55

formula = CNF()
formula.extend(mutually_exclusive(job_positions))  # Can hold at most one position
```

### 6. If-Then-Else Construct

Suppose variable 60 represents a condition, and we want to enforce: if condition is true, then variable 61 must be true, else variable 62 must be true:

```python
condition = 60
then_var = 61
else_var = 62

formula = CNF()
formula.extend(if_then_else(condition, then_var, else_var))
```

## Complete Working Example

Here's a complete example solving a simple scheduling problem:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a formula
formula = CNF()

# Variables:
# 1, 2, 3 = Courses A, B, C on Monday
# 4, 5, 6 = Courses A, B, C on Tuesday
# 7, 8, 9 = Courses A, B, C on Wednesday

# Each course must be scheduled exactly once
formula.extend(exactly_one([1, 4, 7]))  # Course A on exactly one day
formula.extend(exactly_one([2, 5, 8]))  # Course B on exactly one day
formula.extend(exactly_one([3, 6, 9]))  # Course C on exactly one day

# At most 2 courses per day
formula.extend(at_most_k([1, 2, 3], 2))  # At most 2 courses on Monday
formula.extend(at_most_k([4, 5, 6], 2))  # At most 2 courses on Tuesday
formula.extend(at_most_k([7, 8, 9], 2))  # At most 2 courses on Wednesday

# Dependency: If Course A is on Monday, Course B cannot be on Monday
formula.extend(implies(1, -2))

# Solve the formula
with Glucose3(bootstrap_with=formula) as solver:
    result = solver.solve()
    if result:
        model = solver.get_model()
        print("Solution found!")
        
        # Extract the schedule
        days = ["Monday", "Tuesday", "Wednesday"]
        courses = ["A", "B", "C"]
        
        for day_idx, day_vars in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
            day_courses = []
            for course_idx, var in enumerate(day_vars):
                if var in model:  # If variable is true (positive in model)
                    day_courses.append(courses[course_idx])
            
            print(f"{days[day_idx]}: {', '.join(day_courses) if day_courses else 'No courses'}")
    else:
        print("No solution exists!")

# Export the solution
export_solution({
    "satisfiable": result,
    "model": model if result else None,
    "schedule": {day: courses for day, courses in zip(["Monday", "Tuesday", "Wednesday"], 
                                                      [["A"] if 1 in model else [] + ["B"] if 2 in model else [] + ["C"] if 3 in model else [],
                                                       ["A"] if 4 in model else [] + ["B"] if 5 in model else [] + ["C"] if 6 in model else [],
                                                       ["A"] if 7 in model else [] + ["B"] if 8 in model else [] + ["C"] if 9 in model else []])}
})
```

## Why Use These Helper Functions?

1. **Always Work**: Unlike some PySAT native encodings, these helpers work for any valid k value
2. **Optimization**: Automatically use the most efficient encoding for special cases (like k=1)
3. **Simplicity**: Easy-to-understand function names and parameters
4. **No Exceptions**: You won't encounter UnsupportedBound exceptions

## Advanced Use Cases

For more complex constraints or when performance is critical, you can also use the direct PySAT encodings:

```python
from pysat.card import CardEnc, EncType

# Using direct PySAT CardEnc (note: some encodings have limitations on k values)
atmost_constr = CardEnc.atmost(lits=[1, 2, 3, 4], bound=2, encoding=EncType.seqcounter)
formula.extend(atmost_constr.clauses)
```

However, our helper functions are recommended for most use cases as they're simpler and more reliable. 