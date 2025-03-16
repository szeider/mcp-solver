# PySAT Solver (Lite Mode)

This MCP server provides access to PySAT (Python SAT Solver) through a Python interface. PySAT provides interfaces to several SAT solvers and allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Available Tools

| Tool | Description |
|------|-------------|
| `clear_model` | Reset the PySAT model |
| `add_item` | Add Python code to the model |
| `replace_item` | Replace code in the model |
| `delete_item` | Delete code from the model |
| `solve_model` | Solve the current model using PySAT (requires timeout parameter between 1-10 seconds) |
| `get_variable_value` | Get the value of a variable from the solution |
| `get_solution` | Get the complete solution |
| `get_solve_time` | Get the time taken to solve the model |

> **Note:** MaxSAT optimization functionality is not currently supported. Only standard SAT solving capabilities are available.

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
is_satisfiable = solver.solve()

# Create a mapping of variable names to IDs
variables = {
    "a": 1,
    "b": 2,
    "c": 3
}

# Process the results based on the solver output
if is_satisfiable:
    model = solver.get_model()
    result = {
        "satisfiable": True,
        "assignment": {}
    }
    for var_name, var_id in variables.items():
        result["assignment"][var_name] = var_id in model
    
    # Export solution
    export_solution(result)
else:
    # Export unsatisfiable result
    export_solution({
        "satisfiable": False
    })

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

PySAT mode includes several helper functions for easily creating common cardinality constraints. These functions are located in the `templates/cardinality_templates.py` module but are also available directly in the environment:

| Function            | Description                                             | Return Type       |
|---------------------|---------------------------------------------------------|-------------------|
| `at_most_k`         | At most k variables are true                            | List[List[int]]   |
| `at_least_k`        | At least k variables are true                           | List[List[int]]   |
| `exactly_k`         | Exactly k variables are true                            | List[List[int]]   |
| `at_most_one`       | At most one variable is true (optimized for k=1)        | List[List[int]]   |
| `exactly_one`       | Exactly one variable is true (optimized for k=1)        | List[List[int]]   |
| `implies`           | If a is true, then b must be true                       | List[List[int]]   |
| `mutually_exclusive`| At most one variable is true (same as at_most_one)      | List[List[int]]   |
| `if_then_else`      | If condition then x else y                              | List[List[int]]   |

These functions return lists of clauses that need to be added to your formula. The implementation is robust and handles edge cases gracefully, with fallback mechanisms for compatibility with different environments.

Here's how to use these helper functions:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from mcp_solver.pysat.templates.cardinality_templates import at_most_k, at_least_k, exactly_one, implies

# Create a formula
formula = CNF()

# Variables representing courses on different days
# 1, 2, 3 = Courses A, B, C on Monday
# 4, 5, 6 = Courses A, B, C on Tuesday
# 7, 8, 9 = Courses A, B, C on Wednesday

# Each course must be scheduled exactly once
for clause in exactly_one([1, 4, 7]):  # Course A on exactly one day
    formula.append(clause)
for clause in exactly_one([2, 5, 8]):  # Course B on exactly one day
    formula.append(clause)
for clause in exactly_one([3, 6, 9]):  # Course C on exactly one day
    formula.append(clause)

# At most 2 courses per day
for clause in at_most_k([1, 2, 3], 2):  # At most 2 courses on Monday
    formula.append(clause)
for clause in at_most_k([4, 5, 6], 2):  # At most 2 courses on Tuesday
    formula.append(clause)
for clause in at_most_k([7, 8, 9], 2):  # At most 2 courses on Wednesday
    formula.append(clause)

# Dependency: If Course A is on Monday, Course B cannot be on Monday
for clause in implies(1, -2):
    formula.append(clause)

# Solve the formula
solver = Glucose3(bootstrap_with=formula)
is_satisfiable = solver.solve()

if is_satisfiable:
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

For low-level access, you can also use PySAT's built-in cardinality constraint encodings, but our template implementations are generally more robust and handle edge cases better:

```python
from pysat.formula import CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose3

# Enforce that at most 2 variables can be true
vars = [1, 2, 3, 4, 5]
atmost2 = CardEnc.atmost(vars, 2, encoding=EncType.pairwise)

formula = CNF()
formula.extend(atmost2.clauses)
```

For most use cases, you should prefer the templated versions (`at_most_k`, `at_least_k`, etc.) as they provide more robust behavior and better error handling, especially in environments like Claude Desktop.

### Important Tips

1. Always call `solver.delete()` to free memory when using direct PySAT solvers
2. Use appropriate variable IDs (positive integers)
3. In clauses, positive numbers represent variables, negative ones represent negation
4. For complex problems, use `export_solution()` to format and return results
5. Use the right solver for your problem type
6. When using cardinality constraints, leverage the template functions (`at_most_k`, etc.) for cleaner code

### Handling Solver Results

When working with PySAT solver results:

1. Always store the solver's return value in a variable and use that variable consistently in conditional logic
2. Don't hardcode expected results in print statements
3. The solver returns `True` if satisfiable and `False` if unsatisfiable
4. Only process the model/solution when the solver returns `True`

Example of the correct pattern for direct solver use:
```python
# Correct pattern
is_sat = solver.solve()
if is_sat:  # Use the actual return value
    model = solver.get_model()
    # Process solution
    export_solution({
        "satisfiable": True,
        "model": model
    })
else:
    # Handle unsatisfiable case
    export_solution({"satisfiable": False})
```

This ensures your code correctly handles both satisfiable and unsatisfiable results.

## Example Model

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from mcp_solver.pysat import exactly_one
```

## Special Functions

- `export_solution(data)`: Export data as a solution
- Variable IDs must be positive integers
- Clauses are lists of integers (negative for negated variables)

For more advanced usage, refer to the [PySAT documentation](https://pysathq.github.io/).