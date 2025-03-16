# PySAT Solver (Lite Mode)

This service provides access to PySAT (Python SAT Solver) with a simplified interface. PySAT allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Available Tools

| Tool           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `clear_model`  | Reset the PySAT model                                        |
| `add_item`     | Add Python code to the model                                 |
| `replace_item` | Replace code in the model                                    |
| `delete_item`  | Delete code from the model                                   |
| `solve_model`  | Solve the current model (requires timeout parameter between 1-10 seconds) |
| `get_model`    | Fetch the current content of the PySAT model                 |

> **Note:** MaxSAT optimization functionality is not supported. Only standard SAT solving capabilities are available.

## Using PySAT - Basic Workflow

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
2. The solver returns `True` if satisfiable and `False` if unsatisfiable
3. Only process the model/solution when the solver returns `True`
4. Always call `solver.delete()` to free memory when using direct PySAT solvers

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

### Solution Structure

When creating and exporting solutions, you should structure your data as follows:

```python
# Standard solution format
export_solution({
    "satisfiable": True,  # Required field
    "model": model,       # Raw SAT model (optional)
    "values": {           # Variable assignments (recommended)
        "var1": value1,
        "var2": value2
    },
    # You can include additional domain-specific dictionaries
    "custom_structure": {...}
})
```

## Available Solvers

PySAT includes several SAT solvers:

- `Glucose3`: Good general-purpose solver (recommended default)
- `Glucose4`: Updated version of Glucose3
- `Lingeling`: Efficient for large problems
- `MiniSat22`: Classic solver with good stability
- `Minicard`: Extension of MiniSat with cardinality constraints
- `MapleCM`: Award-winning competitive solver

## Cardinality Constraints

For convenience, PySAT provides helper functions for common constraints:

| Function      | Description                              |
| ------------- | ---------------------------------------- |
| `at_most_k`   | At most k variables are true             |
| `at_least_k`  | At least k variables are true            |
| `exactly_k`   | Exactly k variables are true             |
| `exactly_one` | Exactly one variable is true (optimized) |
| `implies`     | If a is true, then b must be true        |

Example usage:

```python
# Variables representing courses on different days
# Each course must be scheduled exactly once
for clause in exactly_one([1, 4, 7]):  # Course A on exactly one day
    formula.append(clause)

# At most 2 courses per day
for clause in at_most_k([1, 2, 3], 2):  # At most 2 courses on Monday
    formula.append(clause)
```

## Example: Schedule Problem

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

# Add constraints (simplified)
# Each task must be in exactly one slot and only one task per slot
# ... (constraint code)

# Solve
solver = Glucose3()
solver.append_formula(formula)

if solver.solve():
    model = solver.get_model()
    
    # Extract the assignment
    schedule = {}
    task_to_slot = {}
    
    # ... (process results)
    
    # Export the solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "values": task_to_slot
    })
else:
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
```

## Solution Export

- ```
  export_solution(data)
  ```

  : Export data as a solution

  - Requires at minimum: `{"satisfiable": True/False}`
  - Should include variable assignments in a `values` dictionary
  - Variable IDs must be positive integers
  - Clauses are lists of integers (negative for negated variables)

For more information on PySAT, visit the [PySAT documentation](https://pysathq.github.io/).