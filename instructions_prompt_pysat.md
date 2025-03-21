# PySAT Solver

This service provides access to PySAT (Python SAT Solver) with a simplified interface. PySAT allows for propositional constraint modeling using CNF (Conjunctive Normal Form).

## Quick Start Example

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *  # Import cardinality helpers

# Simple problem: Find values for A, B, C such that:
# (A OR B) AND (NOT A OR C) AND (NOT B OR NOT C)

# Create CNF formula
formula = CNF()

# Define variables: 1=A, 2=B, 3=C (simple manual mapping)
A, B, C = 1, 2, 3

# Add clauses: (A OR B) AND (NOT A OR C) AND (NOT B OR NOT C)
formula.append([A, B])        # A OR B
formula.append([-A, C])       # NOT A OR C
formula.append([-B, -C])      # NOT B OR NOT C

# Create solver and add formula
solver = Glucose3()
solver.append_formula(formula)

# Solve and process results
if solver.solve():
    model = solver.get_model()
    
    # Interpret model (positive numbers are True, negative are False)
    solution = {
        "A": A in model,
        "B": B in model,
        "C": C in model
    }
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })

# Free solver memory
solver.delete()
```

## ⚠️ Common Pitfalls

- **Incomplete Variable Lines**: Always complete your variable assignments (e.g., `node_color_vars = [has_color(node, color) for color in colors]`)
- **Dictionary Updates**: Use `node_colors[node] = color` (not `node_colors = color`) to update dictionaries
- **Export Solution**: Always include `export_solution()` with at minimum `{"satisfiable": True/False}`
- **Memory Management**: Always call `solver.delete()` to free memory
- **Variable Ranges**: PySAT variables must be positive integers (1, 2, 3, ...)

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

## ⭐ Pre-Implemented Helper Functions ⭐

PySAT includes ready-to-use constraint helper functions that simplify encoding common logical patterns. To access these helpers, import them explicitly:

```python
# Import helper functions
from pysat.card import *
```

| Function             | Description                                      | Usage Example                                                |
| -------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| `at_most_k`          | At most k variables are true                     | `for clause in at_most_k([1,2,3], 2): formula.append(clause)` |
| `at_least_k`         | At least k variables are true                    | `for clause in at_least_k([1,2,3], 1): formula.append(clause)` |
| `exactly_k`          | Exactly k variables are true                     | `for clause in exactly_k([1,2,3], 2): formula.append(clause)` |
| `at_most_one`        | At most one variable is true (optimized)         | `for clause in at_most_one([1,2,3]): formula.append(clause)` |
| `exactly_one`        | Exactly one variable is true (optimized)         | `for clause in exactly_one([1,2,3]): formula.append(clause)` |
| `implies`            | If a is true, then b must be true                | `for clause in implies(1, 2): formula.append(clause)`        |
| `mutually_exclusive` | Set of variables where at most one can be true   | `for clause in mutually_exclusive([1,2]): formula.append(clause)` |
| `if_then_else`       | Implement if-then-else logic (condition ? a : b) | `for clause in if_then_else(5, 6, 7): formula.append(clause)` |

**IMPORTANT:** These functions return clauses that must be added to your formula using the pattern shown above.

### Complete Example Using Cardinality Constraints

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *  # Import cardinality helpers

# Create formula
formula = CNF()

# Create variables with meaningful names (manual mapping)
a, b, c, d = 1, 2, 3, 4
var_names = {1: "a", 2: "b", 3: "c", 4: "d"}

# Add cardinality constraints
# Example 1: At most 2 of these variables can be true
for clause in at_most_k([a, b, c, d], 2):
    formula.append(clause)
    
# Example 2: Exactly one of these variables must be true
for clause in exactly_one([a, b, c]):
    formula.append(clause)
    
# Example 3: Combining constraints - If 'a' is true, then at most one of b,c,d can be true
formula.append([-a, b, c, d])  # If a is true, at least one of b,c,d must be true
for clause in at_most_one([b, c, d]):  # At most one of b,c,d
    formula.append(clause)

# Solve as usual
solver = Glucose3()
solver.append_formula(formula)
if solver.solve():
    model = solver.get_model()
    
    # Interpret model manually
    solution = {var_names[abs(v)]: (v > 0) for v in model if abs(v) in var_names}
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })

# Always free memory
solver.delete()
```

### Understanding the Helper Functions

Here's what each helper function does:

- `at_most_k([x1,...,xn], k)` ensures no more than k variables can be true simultaneously
- `at_least_k([x1,...,xn], k)` ensures at least k variables must be true
- `exactly_k([x1,...,xn], k)` ensures exactly k variables are true (combines the above two)
- `at_most_one([x1,...,xn])` is an optimized version of at_most_k with k=1
- `exactly_one([x1,...,xn])` is an optimized version of exactly_k with k=1
- `implies(a, b)` encodes logical implication (if a then b)
- `mutually_exclusive([x1,...,xn])` ensures at most one variable is true (similar to at_most_one)
- `if_then_else(c, t, f)` implements "if c then t else f" logic

These implementations use efficient encodings that scale well to larger problems.

## Simple Variable Mapping for Readability

For better code readability, use a simple dictionary-based mapping system for your variables:

```python
# Create a variable mapping dictionary
var_mapping = {}
var_count = 1

# Create variables with meaningful names
def create_var(name):
    nonlocal var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# Create variables
x1 = create_var("x1")
edge_ab = create_var("edge_a_b")

# Lookup variable names
def get_var_name(var_id):
    for name, vid in var_mapping.items():
        if vid == abs(var_id):
            return name
    return f"unknown_{var_id}"

# Interpret model into a dictionary of {name: boolean_value}
def interpret_model(model):
    return {get_var_name(v): (v > 0) for v in model}
```

### Interpreting solutions

```python
# After solving
if solver.solve():
    model = solver.get_model()
    solution = interpret_model(model)
    
    # Check specific variables
    if solution.get("edge_a_b", False):
        print("Edge between A and B exists")
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "assignment": solution,
        "variable_mapping": var_mapping
    })
```

### Namespacing variables for complex problems

```python
# Using functions for different variable types
def state_var(state, time):
    return create_var(f"state_{state}_at_{time}")

def action_var(action, time):
    return create_var(f"action_{action}_at_{time}")

# For a chess board, create variables like "piece_at_a1", "piece_at_b3"
def piece_at(piece, position):
    return create_var(f"piece_{piece}_at_{position}")
```

### Integrating with constraint helpers

```python
# Ensure exactly one color per node using meaningful names
node_colors = [create_var(f"{node}_{color}") for color in colors]
for clause in exactly_one(node_colors):
    formula.append(clause)

# Encode complex boolean expressions
# Example: "if user selected feature A AND feature B, then feature C must be enabled"
a_id = create_var("feature_A")
b_id = create_var("feature_B") 
c_id = create_var("feature_C")

# A ∧ B → C is equivalent to ¬(A ∧ B) ∨ C, which is ¬A ∨ ¬B ∨ C
formula.append([-a_id, -b_id, c_id])
```

## Standard Code Pattern for PySAT Solutions

Most PySAT solutions follow this pattern:

```python
# 1. Create variables and formula
formula = CNF()
var_mapping = {}  # For managing named variables
var_count = 1

# Function to create and track variables
def create_var(name):
    nonlocal var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# 2. Add constraints using helper functions
from pysat.card import *  # Import helpers
x1 = create_var("x1")
x2 = create_var("x2")
x3 = create_var("x3")

for clause in exactly_k([x1, x2, x3], 2):
    formula.append(clause)  # ← Always use this pattern with helpers!

# 3. Solve and process results
solver = Glucose3()
solver.append_formula(formula)
if solver.solve():
    model = solver.get_model()
    
    # Interpret results into a solution dictionary
    solution = {name: (vid in model) for name, vid in var_mapping.items()}
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    # Handle unsatisfiable case
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })
    
# 4. Always free solver memory
solver.delete()
```

## How to Interpret the Solver's Model

When a solution is found, `solver.get_model()` returns a list of literals representing the solution:

- Positive numbers (e.g., 1, 5, 7) represent variables that are TRUE
- Negative numbers (e.g., -2, -3, -4) represent variables that are FALSE

To check if variable X is true in the solution:

```python
model = solver.get_model()
is_true = X in model  # Will be True if variable X is true in the solution
```

With variable mapping, you can easily interpret the model:

```python
# Check if a variable is true in the model
is_x1_true = var_mapping["x1"] in model

# Create a complete solution dictionary
solution = {name: (vid in model) for name, vid in var_mapping.items()}
```

## Using PySAT - Basic Workflow

1. Create a CNF formula
2. Add clauses (logical constraints)
3. Create a SAT solver and add the formula
4. Solve and get the model (solution)
5. Export the solution

### Basic SAT Problem Example with Variable Mapping

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Create variable mapping
var_mapping = {}
var_count = 1

# Create variables
def create_var(name):
    nonlocal var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# Create variables a, b, c
a = create_var("a")
b = create_var("b")
c = create_var("c")

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([a, b])       # Clause 1: a OR b
formula.append([-a, c])      # Clause 2: NOT a OR c
formula.append([-b, -c])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Solve the formula
if solver.solve():
    model = solver.get_model()
    
    # Interpret the model
    solution = {name: (vid in model) for name, vid in var_mapping.items()}
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "assignment": solution,
        "variable_mapping": var_mapping
    })
else:
    # Export unsatisfiable result
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })

# Free solver memory
solver.delete()
```

## Handling Imports

When working with PySAT, the following imports are automatically available:

```python
# Core PySAT modules
from pysat.formula import CNF, WCNF
from pysat.solvers import Glucose3, Cadical153

# Standard Python modules
import math  
import random
import collections
import itertools
import re
import json
```

**IMPORTANT:** To use the helper functions, you must explicitly import them:

```python
from pysat.card import *  # Import all helper functions
```

### Understanding Helper Functions

These helper functions simplify encoding common constraint patterns:

```python
from pysat.card import *

# Helper functions return clauses that you need to add to your formula:
for clause in exactly_k([1, 2, 3, 4, 5], 2):
    formula.append(clause)
    
# Logic operations also return clauses:
for clause in implies(1, 2):  # If var 1 is true, then var 2 must be true
    formula.append(clause)
```

**IMPORTANT:** Always use the pattern shown above - these functions return clauses that must be added to your formula using a for loop.

**Note:** For security reasons, only a limited set of modules can be imported. These include the standard modules shown above and all core `pysat` modules.

## Handling Solver Results

When working with PySAT solvers:

1. **IMPORTANT:** Always use the direct conditional check pattern with `if solver.solve():`
2. The solver returns `True` if satisfiable and `False` if unsatisfiable
3. Only process the model/solution within the conditional branch when the solver returns `True`
4. Always call `solver.delete()` to free memory when using direct PySAT solvers

```python
# Correct pattern
if solver.solve(): 
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

## Solution Structure

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

### Key Collision Handling

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

## Available Solvers

PySAT includes several SAT solvers, but we recommend the default solver `Glucose3`

## Example: Graph Coloring Problem Using Simple Mapping

Here's a complete example of solving a graph coloring problem using simple variable mapping:

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *  # Import cardinality helpers

# Define a simple graph (adjacency list)
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C', 'E'],
    'C': ['A', 'B', 'D'],
    'D': ['A', 'C', 'E'],
    'E': ['B', 'D']
}
colors = ['Red', 'Green', 'Blue']

# Create variable mapping
var_mapping = {}
var_count = 1

# Function to create variables with meaningful names
def create_var(name):
    nonlocal var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# Create variables for node-color pairs
node_colors = {}
for node in graph:
    for color in colors:
        node_colors[(node, color)] = create_var(f"{node}_{color}")

# Create CNF formula
formula = CNF()

# Each node must have exactly one color
for node in graph:
    node_color_vars = [node_colors[(node, color)] for color in colors]
    # Use built-in helper for exactly_one constraint
    for clause in exactly_one(node_color_vars):
        formula.append(clause)

# Adjacent nodes cannot have the same color
for node in graph:
    for neighbor in graph[node]:
        if neighbor > node:  # Avoid duplicate constraints
            for color in colors:
                formula.append([-node_colors[(node, color)], -node_colors[(neighbor, color)]])

# Create solver and add formula
solver = Glucose3()
solver.append_formula(formula)

# Solve and interpret results
if solver.solve():
    model = solver.get_model()
    
    # Create dict of node colors
    coloring = {}
    for (node, color), var_id in node_colors.items():
        if var_id in model:  # If this node-color assignment is true
            coloring[node] = color
    
    # Export the results
    export_solution({
        "satisfiable": True,
        "coloring": coloring,
        "variable_mapping": var_mapping
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No valid coloring exists"
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
