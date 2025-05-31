# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to MaxSAT (Maximum Satisfiability) optimization with a simplified interface for weighted constraint optimization using WCNF (Weighted Conjunctive Normal Form).



## Overview

The MCP Solver integrates MaxSAT optimization with the Model Context Protocol, allowing you to create, modify, and solve weighted constraint optimization problems. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using the RC2 MaxSAT solver.



## Quick Start Example

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Simple MaxSAT optimization problem: 
# Find values for A, B, C that maximize satisfied soft constraints while
# respecting all hard constraints

# Create a MaxSAT formula with weighted clauses
wcnf = WCNF()

# Define variables: 1=A, 2=B, 3=C
A, B, C = 1, 2, 3

# Add hard constraints (must be satisfied)
wcnf.append([A, B])        # A OR B
wcnf.append([-A, -B])      # NOT A OR NOT B (can't both be true)

# Add soft constraints with weights (satisfy if possible)
wcnf.append([A], weight=2)       # Try to make A true (value 2)
wcnf.append([B], weight=3)       # Try to make B true (value 3)
wcnf.append([C], weight=5)       # Try to make C true (value 5)
wcnf.append([-C], weight=1)      # Try to make C false (value 1)

# Map variable IDs to meaningful names
var_mapping = {
    "A": A,
    "B": B, 
    "C": C
}

# Solve with RC2 MaxSAT solver
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model is not None:
        print(f"Solution found with cost: {solver.cost}")
        
        # Create a readable solution
        solution = {
            name: (var_id in model)
            for name, var_id in var_mapping.items()
        }
        
        # Export the MaxSAT solution
        export_maxsat_solution(solver, var_mapping)
    else:
        print("No solution found (problem is unsatisfiable)")
        export_maxsat_solution({
            "satisfiable": False,
            "status": "unsatisfiable"
        })
```

## ⚠️ Common Pitfalls

- **Incomplete Variables**: Always complete variable assignments (e.g., `item_vars = [has_item(i) for i in items]`)
- **Dictionary Updates**: Use `selected_items[item] = value` (not `selected_items = value`)
- **Export Solution**: Always use `export_maxsat_solution()` to return results
- **WCNF vs CNF**: Always use `WCNF()` for MaxSAT optimization problems, not `CNF()`
- **Variable Ranges**: MaxSAT variables must be positive integers (1, 2, 3, ...)

## Available Tools

| Tool           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `clear_model`  | Reset the MaxSAT optimization model                           |
| `add_item`     | Add Python code to the model                                 |
| `replace_item` | Replace code in the model                                    |
| `delete_item`  | Delete code from the model                                   |
| `solve_model`  | Solve the current model (requires timeout parameter between 1-30 seconds) |
| `get_model`    | Fetch the current content of the MaxSAT model                |

> **Timeout Handling:** When using `solve_model`, always specify a timeout (in seconds) to prevent long-running computations. If your model times out, you'll receive a response with `"status": "timeout"` and `"timeout": true`, but the connection will be maintained so you can modify and retry your model.



## Solving and Optimization

- **Solution Verification:**  
  After solving, verify that the returned solution satisfies all specified hard constraints. Check that the optimization objective makes sense for your problem.

## ⭐ MaxSAT Concepts

MaxSAT distinguishes between two types of constraints:

1. **Hard Constraints**: Must be satisfied for a solution to be valid
2. **Soft Constraints**: Can be violated at a cost equal to their weight

The MaxSAT solver finds an assignment that:
- Satisfies all hard constraints
- Minimizes the total weight of violated soft constraints

### Key Components

- **WCNF**: Weighted Conjunctive Normal Form formula
- **Hard Clauses**: Added with `wcnf.append([literals])` (no weight specified)
- **Soft Clauses**: Added with `wcnf.append([literals], weight=W)` (positive weight value)
- **Cost**: Sum of weights of violated soft constraints (lower is better)
- **RC2 Solver**: Modern MaxSAT solver with excellent performance

### Basic Optimization Example

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create a MaxSAT formula
wcnf = WCNF()

# Define variables with meaningful names
var_mapping = {
    "item1": 1,
    "item2": 2,
    "item3": 3,
    "item4": 4
}

# Hard constraints (must be satisfied)
wcnf.append([-1, -4])  # Items 1 and 4 are mutually exclusive

# Budget constraint: can't select all items
wcnf.append([-1, -2, -3, -4])  # At least one must be false

# Dependency: Item 3 requires Item 2
wcnf.append([-3, 2])  # NOT item3 OR item2

# Soft constraints (preferences) with weights
wcnf.append([1], weight=10)  # Value of item 1 is 10
wcnf.append([2], weight=5)   # Value of item 2 is 5  
wcnf.append([3], weight=7)   # Value of item 3 is 7
wcnf.append([4], weight=12)  # Value of item 4 is 12

# Solve the MaxSAT problem
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model is not None:
        # Calculate the total value of selected items
        total_value = sum(
            weight for var, weight in zip([1, 2, 3, 4], [10, 5, 7, 12])
            if var in model
        )
        
        # Convert model to readable selection
        selected_items = {
            f"item{i}": (i in model)
            for i in range(1, 5)
        }
        
        # Export the solution
        export_maxsat_solution({
            "satisfiable": True,
            "status": "optimal",
            "selected_items": selected_items,
            "total_value": total_value,
            "cost": solver.cost
        }, var_mapping)
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "status": "unsatisfiable"
        })
```

## Variable Mapping for Readability

```python
# Create a variable mapping dictionary
var_mapping = {}
var_count = 1

# Create variables with meaningful names
def create_var(name):
    global var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# Create variables
feature_premium = create_var("premium")
feature_cloud = create_var("cloud")

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

IMPORTANT: Always use the `global` keyword for counter variables at the module level. The `nonlocal` keyword is only appropriate when referencing variables from an outer function, not for module-level variables.

### MaxSAT Solution Export

The `export_maxsat_solution` function simplifies returning results from MaxSAT problems:

```python
# With a direct dictionary
export_maxsat_solution({
    "satisfiable": True,
    "assignment": {"item1": True, "item2": False},
    "cost": 5,
    "objective": 10  # Often the negative of cost for maximization problems
})

# Or directly with the RC2 solver
export_maxsat_solution(solver, var_mapping)
```

Key fields in MaxSAT solutions:
- `satisfiable`: Boolean indicating whether a solution was found
- `status`: Solution status ("optimal", "satisfiable", "unsatisfiable", "error")
- `cost`: Sum of weights of unsatisfied clauses (lower is better)
- `objective`: Value being optimized (often the negative of cost for maximization)
- `model`: List of true variable IDs
- `assignment`: Dictionary mapping variable names to boolean values (if var_mapping provided)

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

### Weighted Maximum Cut Example

This example illustrates using MaxSAT for the weighted maximum cut problem:

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Problem: Find a cut that maximizes the total weight of edges crossing the cut
wcnf = WCNF()

# Define graph as edge list with weights: (node1, node2, weight)
edges = [
    (1, 2, 5),
    (1, 3, 7),
    (1, 4, 2),
    (2, 3, 9),
    (2, 5, 4),
    (3, 4, 3),
    (4, 5, 8)
]

# Variable mapping: we'll use node IDs directly as variables
# Each variable represents which side of the cut a node is on
var_mapping = {f"node{i}": i for i in range(1, 6)}

# For each edge, we want to maximize the weight of edges where
# the two nodes are on opposite sides of the cut
for u, v, weight in edges:
    # The soft clause (u XOR v) represents nodes on opposite sides
    # XOR can be represented as two clauses: (u OR v) AND (NOT u OR NOT v)
    # But for MaxSAT, we use two soft clauses with the same weight:
    
    # Clause for nodes on opposite sides: NOT u OR NOT v
    wcnf.append([-u, -v], weight=weight)
    
    # Clause for nodes on opposite sides: u OR v
    wcnf.append([u, v], weight=weight)

# Solve with RC2
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Extract the cut
        side_a = [i for i in range(1, 6) if i in model]
        side_b = [i for i in range(1, 6) if i not in model]
        
        # Calculate the cut weight
        cut_weight = sum(
            weight for u, v, weight in edges
            if (u in model and v not in model) or (u not in model and v in model)
        )
        
        # Extract cut edges
        cut_edges = [
            (u, v) for u, v, _ in edges
            if (u in model and v not in model) or (u not in model and v in model)
        ]
        
        # Export the solution
        export_maxsat_solution({
            "satisfiable": True,
            "status": "optimal",
            "side_a": side_a,
            "side_b": side_b,
            "cut_weight": cut_weight,
            "cut_edges": cut_edges,
            "cost": solver.cost
        }, var_mapping)
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })
```

## Standard MaxSAT Pattern

```python
# 1. Create WCNF formula and variables
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

wcnf = WCNF()
var_mapping = {}
var_count = 1

def create_var(name):
    global var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# 2. Define variables and add constraints
# Hard constraints (must be satisfied)
x1 = create_var("x1")
x2 = create_var("x2")
wcnf.append([x1, x2])  # x1 OR x2

# Soft constraints with weights
wcnf.append([x1], weight=5)  # Prefer x1 to be true (weight 5)
wcnf.append([x2], weight=3)  # Prefer x2 to be true (weight 3)

# 3. Solve and process results
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Extract solution details
        assignment = {name: (vid in model) for name, vid in var_mapping.items()}
        objective = sum(
            weight for var, weight in [(x1, 5), (x2, 3)]
            if var in model
        )
        
        # Export the solution
        export_maxsat_solution({
            "satisfiable": True,
            "status": "optimal",
            "assignment": assignment,
            "objective": objective,
            "cost": solver.cost
        }, var_mapping)
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })
```

## Interpreting the Solver's Model

When a solution is found, `solver.compute()` returns a list of literals:

- Positive numbers (e.g., 1, 5, 7) represent variables that are TRUE
- Negative numbers (e.g., -2, -3, -4) represent variables that are FALSE (rarely in the model)
- Variables not in the model are FALSE

```python
model = solver.compute()
is_true = X in model  # Will be True if variable X is true in the solution

# With variable mapping
is_feature_enabled = var_mapping["premium"] in model

# Create a complete solution dictionary
solution = {name: (vid in model) for name, vid in var_mapping.items()}
```

## Handling Imports and Helper Functions

Available imports:

```python
# Core PySAT modules
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Standard Python modules
import math  
import random
import collections
import itertools
import re
import json
```

### Template Helpers

The MCP Solver provides built-in helper functions for common MaxSAT patterns:

#### Variable Mapping

```python
# Create a variable map
var_map = VariableMap()

# Create variables with meaningful names
x1 = var_map.create_var("x1")
x2 = var_map.create_var("x2")

# Create multiple variables at once
features = var_map.create_vars(["base", "premium", "cloud"])

# Extract solution
with RC2(wcnf) as solver:
    model = solver.compute()
    if model:
        # Convert to meaningful variable assignments
        solution = var_map.interpret_model(model)
        print(solution)  # {'x1': True, 'x2': False, ...}
```

#### Basic Templates

```python
# Add constraints
add_hard_constraint(wcnf, [x1, x2])         # x1 OR x2
add_soft_constraint(wcnf, [x1], weight=5)   # Prefer x1=True (weight 5)

# Encode common patterns
encode_binary_variable(wcnf, x1, soft_weight=3, preferred_value=True)
encode_mutual_exclusion(wcnf, [x1, x2, x3])  # At most one can be true
encode_dependency(wcnf, x1, x2)              # If x1 then x2

# Quick solving
result = solve_maxsat_problem(wcnf, var_map.get_mapping())
```

#### Complete Problem Templates

For common optimization problems, you can use high-level templates:

```python
# Feature selection
result = feature_selection_problem(
    features={"base": 0, "premium": 10, "cloud": 15},
    dependencies=[("premium", "base"), ("sync", "cloud")],
    exclusions=[("basic", "premium")],
    required_features=["base"]
)

# Maximum cut
result = weighted_max_cut(
    edges=[(1, 2, 5), (1, 3, 7), (2, 3, 9)]
)

# Set cover
result = set_cover_problem(
    universe=[1, 2, 3, 4, 5],
    sets={"A": [1, 2, 3], "B": [2, 4], "C": [3, 5]},
    costs={"A": 5, "B": 3, "C": 2}
)

# Knapsack
result = knapsack_problem(
    items={
        "item1": {"weight": 5, "value": 10},
        "item2": {"weight": 3, "value": 5}
    },
    capacity=10,
    dependencies=[("item3", "item2")]
)
```

## Solution Export

`export_maxsat_solution(data)`: Export data as a MaxSAT solution

- Requires at minimum: `{"satisfiable": True/False}`
- Structure data using custom dictionaries that reflect your problem domain
- Values will be automatically extracted into a flat "values" dictionary
- For optimization problems, include `cost` and/or `objective` fields
- For convenience, include problem-specific fields like `selected_items` or `assignment`

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.
- **Split long code parts** into smaller items.
- **Verification:**  
  Always verify the solution after a solve operation by checking that all hard constraints are satisfied and the optimization objective makes sense.

Happy optimizing with MCP Solver!