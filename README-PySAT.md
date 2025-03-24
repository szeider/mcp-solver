# PySAT Solver (Lite Mode)

This service provides access to PySAT (Python SAT Solver) with a simplified interface. PySAT allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Available Tools

| Tool           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `clear_model`  | Reset the PySAT model                                        |
| `add_item`     | Add Python code to the model                                 |
| `replace_item` | Replace code in the model                                    |
| `delete_item`  | Delete code from the model                                   |
| `solve_model`  | Solve the current model (requires timeout parameter between 1-30 seconds) |
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
    
    # Process the results - create a custom dictionary
    assignment = {}
    for var_name, var_id in variables.items():
        assignment[var_name] = var_id in model
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "assignment": assignment
    })
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
        "model": model,
        "custom_dictionary": {...}
    })
else:
    # Handle unsatisfiable case
    export_solution({"satisfiable": False})
```

### Solution Structure

When creating and exporting solutions, structure your data using custom dictionaries that make sense for your problem domain:

```python
# Example with a custom dictionary for a graph coloring problem
export_solution({
    "satisfiable": True,
    "coloring": {"A": "red", "B": "green", "C": "blue"},
    "model": model
})
```

All values in your custom dictionaries will be automatically extracted and made available in a flat "values" dictionary. This allows you to:

1. Structure your solution data in a way that makes sense for your problem domain
2. Access individual variables directly through the "values" dictionary

#### Key Collision Handling

If multiple custom dictionaries contain the same key, the system will preserve both values by prefixing the keys with their parent dictionary name:

```python
# Example with potential key collision
export_solution({
    "satisfiable": True,
    "employees": {"manager": "Alice", "clerk": "Bob"},
    "rooms": {"manager": "Office A", "meeting": "Room B"}
})
```

This will result in:

- `values["clerk"] = "Bob"`
- `values["meeting"] = "Room B"`
- `values["employees.manager"] = "Alice"`
- `values["rooms.manager"] = "Office A"`

Keys that appear in only one dictionary won't be prefixed.

## Enhanced Error Handling

The system provides robust error detection and reporting to help diagnose problems in your SAT models:

1. **Descriptive Error Messages**: Errors are intercepted and translated into user-friendly messages with context about what went wrong
2. **Variable Validation**: The system automatically checks variable types and values, detecting issues like non-integer variable IDs or duplicate variables
3. **Formula Validation**: Automatically validates clause structure, catching problems like empty clauses, zero literals, or missing variables
4. **Context-Aware Diagnostics**: Error messages include details about the current state of your model, making it easier to identify issues
5. **Solution Structure Validation**: The system validates solution structures before exporting, preventing common serialization errors

These enhancements help you identify and fix issues more quickly, with clear guidance on what might be causing problems in your models.

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

## Example: Graph Coloring Problem

Here's a complete example of solving a graph coloring problem:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Define a simple graph
nodes = ["A", "B", "C", "D"]
edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
colors = ["Red", "Green", "Blue"]

# Create a CNF formula
formula = CNF()

# Create variables for node-color assignments
variables = {}
var_id = 1
for node in nodes:
    for color in colors:
        variables[f"{node}_{color}"] = var_id
        var_id += 1

# Each node must have at least one color
for node in nodes:
    clause = [variables[f"{node}_{color}"] for color in colors]
    formula.append(clause)

# Each node can have at most one color
for node in nodes:
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            color1 = colors[i]
            color2 = colors[j]
            formula.append([-variables[f"{node}_{color1}"], -variables[f"{node}_{color2}"]])

# Adjacent nodes must have different colors
for node1, node2 in edges:
    for color in colors:
        formula.append([-variables[f"{node1}_{color}"], -variables[f"{node2}_{color}"]])

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Solve the formula
if solver.solve():
    model = solver.get_model()
    
    # Extract the coloring assignment
    coloring = {}
    for var_name, var_id in variables.items():
        if var_id in model:  # If this assignment is true
            node, color = var_name.split('_')
            coloring[node] = color
    
    # Export solution with custom dictionary
    export_solution({
        "satisfiable": True,
        "model": model,
        "coloring": coloring  # Values will be automatically extracted
    })
    
    # The exported solution will contain:
    # {
    #    "satisfiable": True,
    #    "model": [...],
    #    "coloring": {"A": "red", "B": "green", ...},
    #    "values": {"A": "red", "B": "green", ...}
    # }
else:
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
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
    
    for var_name, var_id in variables.items():
        if var_id in model:  # If this assignment is true
            task, slot = var_name.split('_')
            schedule[slot] = task
            task_to_slot[task] = slot
    
    # Export the solution with custom dictionaries
    export_solution({
        "satisfiable": True,
        "model": model,
        "schedule": schedule,
        "assignment": task_to_slot
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
  - Structure data using custom dictionaries that reflect your problem domain
  - Values will be automatically extracted into a flat "values" dictionary
  - Variable IDs must be positive integers
  - Clauses are lists of integers (negative for negated variables)

For more information on PySAT, visit the [PySAT documentation](https://pysathq.github.io/).