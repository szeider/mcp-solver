# PySAT Solver (Lite Mode)

This MCP server provides access to PySAT (Python SAT Solver) through a Python interface. PySAT provides interfaces to several SAT solvers and allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Available Tools

| Tool | Description |
|------|-------------|
| `clear_model` | Reset the PySAT model |
| `add_item` | Add Python code to the model |
| `replace_item` | Replace code in the model |
| `delete_item` | Delete code from the model |
| `solve_model` | Solve the current model using PySAT |
| `get_variable_value` | Get the value of a variable from the solution |
| `get_solution` | Get the complete solution |
| `get_solve_time` | Get the time taken to solve the model |

## Using PySAT

PySAT models are written as Python code. Here's a basic workflow:

1. Create a CNF formula
2. Add clauses (logical constraints)
3. Create a SAT solver and add the formula
4. Solve and get the model (solution)
5. Export the solution

### Example: Simple SAT Problem

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Solve the formula
satisfiable = solver.solve()
model = solver.get_model() if satisfiable else None

# Create a mapping of variable names to IDs
variables = {
    "a": 1,
    "b": 2,
    "c": 3
}

# Print the results
if satisfiable:
    print("Satisfiable!")
    result = {}
    for var_name, var_id in variables.items():
        is_true = var_id in model
        is_false = -var_id in model
        result[var_name] = is_true
    
    # Export solution
    export_solution(result)
else:
    print("Unsatisfiable")
    export_solution({"status": "UNSAT"})

# Free solver memory
solver.delete()
```

## Available Solvers

PySAT includes several SAT solvers with different performance characteristics:

- `Glucose3`: Good general-purpose solver
- `Glucose4`: Updated version of Glucose3
- `Lingeling`: Efficient for large problems
- `MiniSat22`: Classic solver with good stability
- `Minicard`: Extension of MiniSat with cardinality constraints
- `MapleCM`: Award-winning competitive solver

## Advanced Features

### Cardinality Constraints Helper Functions

PySAT mode includes several helper functions for easily creating common cardinality constraints:

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

These helper functions are reliable and work for any valid k values. Here's how to use them:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a formula
formula = CNF()

# Variables representing courses on different days
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
solver = Glucose3(bootstrap_with=formula)
result = solver.solve()

if result:
    model = solver.get_model()
    
    # Extract the schedule
    days = ["Monday", "Tuesday", "Wednesday"]
    courses = ["A", "B", "C"]
    schedule = {}
    
    for day_idx, day_vars in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
        day_courses = []
        for course_idx, var in enumerate(day_vars):
            if var in model:  # If variable is true (positive in model)
                day_courses.append(courses[course_idx])
        
        schedule[days[day_idx]] = day_courses
    
    # Export the solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "schedule": schedule
    })
else:
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
```

### Traditional Cardinality Constraints

```python
from pysat.formula import CNF
from pysat.card import CardEnc, EncType

# Enforce that at most 2 variables can be true
vars = [1, 2, 3, 4, 5]
atmost2 = CardEnc.atmost(vars, 2, encoding=EncType.pairwise)

formula = CNF()
formula.extend(atmost2.clauses)

# Continue with solving as before
```

### MaxSAT Solving

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create weighted MaxSAT problem
wcnf = WCNF()

# Hard clauses (must be satisfied)
wcnf.append([1, 2])  # a OR b

# Soft clauses with weights
wcnf.append([1], weight=5)  # a (weight 5)
wcnf.append([2], weight=3)  # b (weight 3)

# Solve with RC2
with RC2(wcnf) as rc2:
    model = rc2.compute()
    cost = rc2.cost
    print(f"Model: {model}, Cost: {cost}")
```

### Important Tips

1. Always call `solver.delete()` to free memory
2. Use appropriate variable IDs (positive integers)
3. In clauses, positive numbers represent variables, negative ones represent negation
4. For complex problems, use `export_solution()` to format and return results
5. Use the right solver for your problem type

## Special Functions

- `export_solution(data)`: Export data as a solution
- Variable IDs must be positive integers
- Clauses are lists of integers (negative for negated variables)

For more advanced usage, refer to the [PySAT documentation](https://pysathq.github.io/).