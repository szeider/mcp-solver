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

### Basic SAT Problem Example

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

# Solve the formula - IMPORTANT: Use direct conditional check
if solver.solve():
    model = solver.get_model()
    
    # Create a mapping of variable names to IDs
    variables = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    
    # Process the results
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

## Important Guidelines

### Handling Solver Results

When working with PySAT solvers:

1. **IMPORTANT:** Always use the direct conditional check pattern with `if solver.solve():` rather than assigning the result to a variable
2. Don't hardcode expected results in print statements
3. The solver returns `True` if satisfiable and `False` if unsatisfiable
4. Only process the model/solution when the solver returns `True`
5. Always call `solver.delete()` to free memory when using direct PySAT solvers

```python
# Correct pattern
if solver.solve():  # Direct conditional check
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

### Solution Structure and Accessibility

When creating and exporting solutions, you can structure your data in several ways:

1. **Custom Solution Dictionaries**: Create custom dictionaries to organize your solution in a meaningful way for your problem domain.

```python
# Example with a custom "coloring" dictionary for a graph coloring problem
export_solution({
    "satisfiable": True,
    "coloring": {"A": "red", "B": "green", "C": "blue"},
    "model": model
})
```

2. **Values Dictionary**: For standardized variable access, you can also include a "values" dictionary:

```python
# Example with explicit values dictionary
export_solution({
    "satisfiable": True,
    "coloring": {"A": "red", "B": "green", "C": "blue"},
    "values": {"A": "red", "B": "green", "C": "blue"},
    "model": model
})
```

3. **Accessing Variables**: Both approaches are supported:
   - Individual variables can be accessed from the "values" dictionary
   - Custom dictionaries (like "coloring") can be accessed as a whole
   - If you don't include a "values" dictionary, one will be created automatically, and values from your custom dictionaries will be copied into it

This flexible approach allows you to structure your solution data in a way that makes sense for your problem domain while ensuring all values remain accessible.

## Available Solvers

PySAT includes several SAT solvers with different performance characteristics:

- `Glucose3`: Good general-purpose solver
- `Glucose4`: Updated version of Glucose3
- `Lingeling`: Efficient for large problems
- `MiniSat22`: Classic solver with good stability
- `Minicard`: Extension of MiniSat with cardinality constraints
- `MapleCM`: Award-winning competitive solver

## Cardinality Constraints

### Helper Functions

PySAT mode provides helper functions for easily creating common cardinality constraints:

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

Here's how to use these helper functions:

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

# Solve the formula
solver = Glucose3(bootstrap_with=formula)

if solver.solve():
    model = solver.get_model()
    
    # Extract the schedule
    days = ["Monday", "Tuesday", "Wednesday"]
    courses = ["A", "B", "C"]
    schedule = {}
    
    for day_idx, day_vars in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
        day_courses = []
        for course_idx, var in enumerate(day_vars):
            if var in model:  # If variable is true
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

For low-level access, you can also use PySAT's built-in cardinality constraint encodings:

```python
from pysat.formula import CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose3

# Enforce that at most 2 variables can be true
vars = [1, 2, 3, 4, 5]
atmost2 = CardEnc.atmost(vars, 2, encoding=EncType.pairwise)

formula = CNF()
formula.extend(atmost2.clauses)

# Create solver and solve
solver = Glucose3(bootstrap_with=formula)
if solver.solve():
    model = solver.get_model()
    # Process solution
    export_solution({
        "satisfiable": True,
        "model": model
    })
else:
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
```

For most use cases, prefer the templated versions (`at_most_k`, `at_least_k`, etc.) as they provide more robust behavior and better error handling.

## Advanced Example: Complete Solution Handling

Here's a comprehensive example that demonstrates effective solution structuring:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a simple scheduling problem
formula = CNF()

# Variables: Tasks 1-3 assigned to slots A-C
variables = {
    "Task1_SlotA": 1, "Task1_SlotB": 2, "Task1_SlotC": 3,
    "Task2_SlotA": 4, "Task2_SlotB": 5, "Task2_SlotC": 6,
    "Task3_SlotA": 7, "Task3_SlotB": 8, "Task3_SlotC": 9
}

# Add constraints (simplified for example)
# Each task must be assigned to exactly one slot
for task in range(1, 4):
    base = (task - 1) * 3 + 1
    # At least one slot per task
    formula.append([base, base+1, base+2])
    # No more than one slot per task
    formula.append([-base, -(base+1)])
    formula.append([-base, -(base+2)])
    formula.append([-(base+1), -(base+2)])

# Only one task per slot
for slot in range(3):  # Slots A, B, C
    # No two tasks in the same slot
    formula.append([-1-slot, -4-slot])
    formula.append([-1-slot, -7-slot])
    formula.append([-4-slot, -7-slot])

# Solve
solver = Glucose3()
solver.append_formula(formula)

if solver.solve():
    model = solver.get_model()
    
    # Extract the assignment
    schedule = {}
    task_to_slot = {}
    
    for var_name, var_id in variables.items():
        if var_id in model:  # If this assignment is true
            parts = var_name.split("_")
            task = parts[0]
            slot = parts[1]
            
            # Store in a structured way
            if slot not in schedule:
                schedule[slot] = []
            schedule[slot].append(task)
            task_to_slot[task] = slot
    
    # Create solution with both custom dictionaries and values dictionary
    result = {
        "satisfiable": True,
        "schedule": schedule,        # Custom dictionary showing tasks by slot
        "assignment": task_to_slot,  # Custom dictionary showing slot by task
        "model": model,              # Original model
        "values": {}                 # Initialize values dictionary
    }
    
    # Add individual mappings to the values dictionary
    for task, slot in task_to_slot.items():
        result["values"][task] = slot
    
    # Export the solution - makes all dictionaries and values accessible
    export_solution(result)
else:
    export_solution({"satisfiable": False})

# Free solver memory
solver.delete()
```

In this example:
- `schedule` is accessible as a dictionary showing which tasks are in each slot
- `assignment` is accessible as a dictionary showing which slot each task is in
- Individual task assignments (Task1, Task2, Task3) are accessible as variables

## Special Functions

- `export_solution(data)`: Export data as a solution
  - Can be called with a PySAT solver: `export_solution(solver)`
  - Can be called with a dictionary: `export_solution({"satisfiable": True, "coloring": {...}})`
  - Can include both custom dictionaries and a values dictionary
  - Will automatically extract values from custom dictionaries for standardized access
- Variable IDs must be positive integers
- Clauses are lists of integers (negative for negated variables)

For more advanced usage, refer to the [PySAT documentation](https://pysathq.github.io/).