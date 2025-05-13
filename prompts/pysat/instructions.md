# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to PySAT (Python SAT Solver) with a simplified interface for propositional constraint modeling using CNF (Conjunctive Normal Form).



## Overview

The MCP Solver integrates PySAT solving with the Model Context Protocol, allowing you to create, modify, and solve PySAT encodings. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using a SAT solver.



## Quick Start Example

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *  # Import cardinality helpers

# Simple problem: Find values for A, B, C such that:
# (A OR B) AND (NOT A OR C) AND (NOT B OR NOT C)

formula = CNF()

# Define variables: 1=A, 2=B, 3=C
A, B, C = 1, 2, 3

# Add clauses
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

- **Incomplete Variables**: Always complete variable assignments (e.g., `node_color_vars = [has_color(node, color) for color in colors]`)
- **Dictionary Updates**: Use `node_colors[node] = color` (not `node_colors = color`)
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
| `solve_model`  | Solve the current model (requires timeout parameter between 1-30 seconds) |
| `get_model`    | Fetch the current content of the PySAT model                 |

> **MaxSAT Support:** MaxSAT (Maximum Satisfiability) optimization is supported through the RC2 solver. Use WCNF formulas with weighted clauses for optimization problems. See the MaxSAT section below for details.

> **Timeout Handling:** When using `solve_model`, always specify a timeout (in seconds) to prevent long-running computations. If your model times out, you'll receive a response with `"status": "timeout"` and `"timeout": true`, but the connection will be maintained so you can modify and retry your model.



## Solving and Verification

- **Solution Verification:**  
  After solving, verify that the returned solution satisfies all specified constraints. If you get UNSAT, then check that all clauses are indeed justified from the problem description.

## ⭐ Pre-Implemented Helper Functions

Import these constraint helpers to simplify encoding common logical patterns:

```python
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

### Example Using Cardinality Constraints

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *

formula = CNF()

# Create variables with meaningful names
a, b, c, d = 1, 2, 3, 4
var_names = {1: "a", 2: "b", 3: "c", 4: "d"}

# Add cardinality constraints
# At most 2 of these variables can be true
for clause in at_most_k([a, b, c, d], 2):
    formula.append(clause)
    
# Exactly one of these variables must be true
for clause in exactly_one([a, b, c]):
    formula.append(clause)
    
# Combining constraints - If 'a' is true, then at most one of b,c,d can be true
formula.append([-a, b, c, d])  # If a is true, at least one of b,c,d must be true
for clause in at_most_one([b, c, d]):
    formula.append(clause)

# Solve
solver = Glucose3()
solver.append_formula(formula)
if solver.solve():
    model = solver.get_model()
    solution = {var_names[abs(v)]: (v > 0) for v in model if abs(v) in var_names}
    
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })

solver.delete()
```

## Variable Mapping for Readability

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

IMPORTANT: Always use the `global` keyword (not `nonlocal`) for counter variables at the module level. The `nonlocal` keyword is only appropriate when referencing variables from an outer function, not for module-level variables.

### Interpreting Solutions

```python
# After solving
if solver.solve():
    model = solver.get_model()
    solution = interpret_model(model)
    
    # Check specific variables
    if solution.get("edge_a_b", False):
        print("Edge between A and B exists")
    
    export_solution({
        "satisfiable": True,
        "assignment": solution,
        "variable_mapping": var_mapping
    })
```

## MaxSAT Optimization

MaxSAT (Maximum Satisfiability) extends SAT by differentiating between hard constraints (must be satisfied) and soft constraints (can be violated at a cost). This enables solving optimization problems by finding assignments that minimize the total cost of violated soft constraints.

### Basic MaxSAT Example

```python
# % Item 1: Import and setup
from pysat.formula import WCNF, CNF
from pysat.examples.rc2 import RC2
from pysat.solvers import Glucose3

# Create a MaxSAT formula
wcnf = WCNF()

# Variable mapping for readability
var_mapping = {
    "x1": 1,
    "x2": 2
}

# % Item 2: Define constraints
# Hard constraints (must be satisfied)
wcnf.append([1, 2])     # x1 OR x2
wcnf.append([-1, -2])   # NOT x1 OR NOT x2

# Soft constraints with weights (try to satisfy if possible)
wcnf.append([1], weight=1)  # Try to make x1 true (weight 1)
wcnf.append([2], weight=2)  # Try to make x2 true (weight 2)

# % Item 3: Create standard solver for MCP validation
# This is required in addition to the RC2 solver
formula = CNF()
formula.append([1])  # Simple satisfiable clause
solver = Glucose3()
solver.append_formula(formula)

# % Item 4: Solve with RC2 MaxSAT solver
with RC2(wcnf) as rc2_solver:
    # Compute optimal solution
    solution_values = rc2_solver.compute()
    
    if solution_values:
        # The simplest and most reliable approach - let the export_maxsat_solution 
        # function handle all the variable mapping and results formatting
        export_maxsat_solution(rc2_solver, var_mapping)
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })

# % Item 5: Call solver.solve() for MCP validation
# IMPORTANT: MCP requires a call to solver.solve()
solver.solve()
solver.delete()  # Always free solver memory
```

### Feature Selection Example

This example shows a practical application of MaxSAT for optimizing feature selection:

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Problem: Select features to maximize value while respecting dependencies
wcnf = WCNF()

# Variable mapping
var_mapping = {}
var_count = 1

def create_var(name):
    global var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# Define features with their values
features = {
    "base": 0,          # Base product (required)
    "premium": 10,      # Premium upgrade  
    "cloud": 15,        # Cloud storage
    "sync": 7,          # Sync capability
    "mobile": 12,       # Mobile app
    "analytics": 20,    # Analytics dashboard
}

# Create variables for each feature
feature_vars = {f: create_var(f) for f in features}

# Hard constraints (dependencies)
wcnf.append([feature_vars["base"]])  # Base product is required
wcnf.append([-feature_vars["premium"], feature_vars["base"]])  # Premium requires base
wcnf.append([-feature_vars["sync"], feature_vars["cloud"]])    # Sync requires cloud
wcnf.append([-feature_vars["analytics"], feature_vars["premium"]])  # Analytics requires premium

# Soft constraints (feature values as weights)
for feature, value in features.items():
    if value > 0:  # Skip base feature with value 0
        wcnf.append([feature_vars[feature]], weight=value)

# Solve with RC2
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Most reliable approach - let export_maxsat_solution handle the mapping
        # This automatically extracts all variables and formats the result
        export_maxsat_solution(solver, var_mapping)
        
        # The result will include:
        # - satisfiable: True
        # - status: "optimal"
        # - model: List of true variable IDs
        # - assignment: Dictionary mapping feature names to boolean values
        # - cost: Sum of weights of unsatisfied soft constraints
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "message": "No valid feature selection possible"
        })
```

### MaxSAT Solution Export

The `export_maxsat_solution` function simplifies returning results from MaxSAT problems:

```python
# With a direct dictionary
export_maxsat_solution({
    "satisfiable": True,
    "assignment": {"x1": False, "x2": True},
    "cost": 1,
    "objective": 15  # Often the negative of cost for maximization problems
})

# Or directly with the RC2 solver
export_maxsat_solution(solver, var_mapping)
```

Key fields in MaxSAT solutions:
- `satisfiable`: Boolean indicating whether a solution was found
- `status`: Solution status ("optimal", "satisfiable", "unsatisfiable", "error")
- `cost`: Sum of weights of unsatisfied clauses (lower is better)
- `objective`: Value being optimized (often the negative of cost)
- `model`: List of true variable IDs
- `assignment`: Dictionary mapping variable names to boolean values (if var_mapping provided)

### IMPORTANT: Standard MaxSAT Usage Pattern

When working with MaxSAT, ALWAYS use this standard pattern:

```python
# % Item 1: Imports and setup
from pysat.formula import WCNF, CNF
from pysat.examples.rc2 import RC2
from pysat.solvers import Glucose3

# Create a MaxSAT formula
wcnf = WCNF()

# % Item 2: Add constraints
# Add hard and soft constraints...

# % Item A3: Create a standard solver for MCP validation
formula = CNF()
formula.append([1])  # Simple satisfiable clause
solver = Glucose3()
solver.append_formula(formula)

# % Item 4: Solve with RC2 for MaxSAT optimization
with RC2(wcnf) as rc2_solver:
    solution_values = rc2_solver.compute()
    
    if solution_values:
        # MaxSAT solution found
        result = export_maxsat_solution(rc2_solver, var_mapping)
    else:
        # No solution found
        result = export_maxsat_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })

# % Item 5: The MCP framework requires a call to solver.solve()
# and a separate export_solution call
solver.solve()
export_solution({
    "satisfiable": True,
    "message": "MaxSAT solution exported successfully",
    "maxsat_result": result
})
solver.delete()  # Free solver memory
```

Key points about this standardized pattern:
- ALL steps are required - do not skip any of them
- Must use BOTH export_maxsat_solution AND export_solution
- Must call solver.solve() on the standard solver
- Always split your code into logical sections with % Item markers
- Use the export_maxsat_solution result in your export_solution call
- This pattern avoids all dictionary-related validation errors

### Correct Dictionary Usage with MaxSAT

When working with dictionaries in MaxSAT problems, it's important to use proper dictionary update patterns to avoid validation errors:

```python
# CORRECT: Update dictionary with key assignment
solution = {}
for name, var_id in var_mapping.items():
    solution[name] = (var_id in model)  # Use dict[key] = value pattern

# INCORRECT: This will trigger validation errors
solution = {name: (var_id in model) for name, var_id in var_mapping.items()}

# CORRECT: Using comprehension outside the loop and then using it
solution_values = {name: (var_id in model) for name, var_id in var_mapping.items()}
export_maxsat_solution({
    "satisfiable": True,
    "assignment": solution_values
})

# BEST PRACTICE: Using export_maxsat_solution directly with the solver
# This handles all the variable extraction internally
export_maxsat_solution(solver, var_mapping)  # Most concise and reliable
```

Using the direct `export_maxsat_solution(solver, var_mapping)` pattern is recommended as it handles all the dictionary operations correctly internally.

### Namespacing Variables

```python
# Using functions for different variable types
def state_var(state, time):
    return create_var(f"state_{state}_at_{time}")

def action_var(action, time):
    return create_var(f"action_{action}_at_{time}")

def piece_at(piece, position):
    return create_var(f"piece_{piece}_at_{position}")
```

### Integrating with Constraint Helpers

```python
# Ensure exactly one color per node
node_colors = [create_var(f"{node}_{color}") for color in colors]
for clause in exactly_one(node_colors):
    formula.append(clause)

# Example: "if user selected feature A AND feature B, then feature C must be enabled"
a_id = create_var("feature_A")
b_id = create_var("feature_B") 
c_id = create_var("feature_C")

# A ∧ B → C is equivalent to ¬A ∨ ¬B ∨ C
formula.append([-a_id, -b_id, c_id])
```

## Standard Code Pattern

```python
# 1. Create variables and formula
formula = CNF()
var_mapping = {}
var_count = 1

def create_var(name):
    nonlocal var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# 2. Add constraints using helper functions
from pysat.card import *
x1 = create_var("x1")
x2 = create_var("x2")
x3 = create_var("x3")

for clause in exactly_k([x1, x2, x3], 2):
    formula.append(clause)

# 3. Solve and process results
solver = Glucose3()
solver.append_formula(formula)
if solver.solve():
    model = solver.get_model()
    solution = {name: (vid in model) for name, vid in var_mapping.items()}
    
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })
    
# 4. Always free solver memory
solver.delete()
```

## Interpreting the Solver's Model

When a solution is found, `solver.get_model()` returns a list of literals:

- Positive numbers (e.g., 1, 5, 7) represent variables that are TRUE
- Negative numbers (e.g., -2, -3, -4) represent variables that are FALSE

```python
model = solver.get_model()
is_true = X in model  # Will be True if variable X is true in the solution

# With variable mapping
is_x1_true = var_mapping["x1"] in model

# Create a complete solution dictionary
solution = {name: (vid in model) for name, vid in var_mapping.items()}
```

## Handling Imports

Available imports:

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

# Helper functions (must be explicitly imported)
from pysat.card import *
```

## Example: Graph Coloring Problem

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *

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

solver.delete()
```

## Solution Export

`export_solution(data)`: Export data as a solution

- Requires at minimum: `{"satisfiable": True/False}`
- Structure data using custom dictionaries that reflect your problem domain
- Values will be automatically extracted into a flat "values" dictionary
- If multiple dictionaries contain the same key, values are preserved by prefixing keys with their parent dictionary name
- Keys that appear in only one dictionary won't be prefixed

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.
- **Split long code parts** into smaller items.
- **Verification:**  
  Always verify the solution after a solve operation by checking that all constraints are satisfied and justified.

Happy modeling with MCP Solver!
