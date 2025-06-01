# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to MaxSAT (Maximum Satisfiability) optimization with a simplified interface for weighted constraint optimization using WCNF (Weighted Conjunctive Normal Form).

> **IMPORTANT**: The `export_maxsat_solution()` function is automatically available in the environment and does not need to be imported. You can call it directly in your code to return MaxSAT solutions.



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
# export_maxsat_solution is automatically available in the environment

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

# Add soft constraints with weights
# IMPORTANT: The solver minimizes the sum of weights of UNSATISFIED clauses
wcnf.append([A], weight=2)       # Cost 2 if A is FALSE (clause [A] unsatisfied)
wcnf.append([B], weight=3)       # Cost 3 if B is FALSE (clause [B] unsatisfied)
wcnf.append([C], weight=5)       # Cost 5 if C is FALSE (clause [C] unsatisfied)
wcnf.append([-C], weight=1)      # Cost 1 if C is TRUE (clause [-C] unsatisfied)

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
        # Note: cost = sum of weights of unsatisfied soft clauses
        
        # Create a readable solution
        solution = {
            name: (var_id in model)
            for name, var_id in var_mapping.items()
        }
        
        # Export the MaxSAT solution
        export_maxsat_solution(solver, var_mapping)
    else:
        export_maxsat_solution({
            "satisfiable": False,
            "status": "unsatisfiable"
        })
```

## ⚠️ Common Pitfalls

- **Using export_maxsat_solution**: This function is automatically available in the environment - no import needed
- **Incomplete Variables**: Always complete variable assignments (e.g., `item_vars = [has_item(i) for i in items]`)
- **Dictionary Updates**: Use `selected_items[item] = value` (not `selected_items = value`)
- **Export Solution**: Always use `export_maxsat_solution()` to return results
- **WCNF vs CNF**: Always use `WCNF()` for MaxSAT optimization problems, not `CNF()`
- **Variable Ranges**: MaxSAT variables must be positive integers (1, 2, 3, ...)
- **Soft Constraint Literals**: Always use brackets around literals in `wcnf.append()` calls:
  - ✅ Correct: `wcnf.append([x], weight=1)` (with square brackets)
  - ❌ Incorrect: `wcnf.append(x, weight=1)` (missing brackets)
  - ❌ Incorrect: `wcnf.append(, weight=1)` (incomplete)
- **Dictionary Summation**: When summing values in a dictionary, use `sum(dict.values())` not `sum(dict)`

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

### Understanding MaxSAT Optimization (CRITICAL)

**FUNDAMENTAL CONCEPT**: MaxSAT finds an assignment that:
1. Satisfies ALL hard constraints (clauses without weights)
2. **Minimizes the sum of weights of UNSATISFIED soft constraints**

**Note**: If the hard constraints are contradictory (unsatisfiable), the solver will return `model = None` and the export will show `satisfiable: False` with `status: "unsatisfiable"`.

This means:
- If you have a soft clause `[x]` with weight W, the solver pays cost W if x is FALSE (clause unsatisfied)
- If you have a soft clause `[-x]` with weight W, the solver pays cost W if x is TRUE (clause unsatisfied)
- The `solver.cost` represents the total weight of all unsatisfied soft clauses

### Converting Between Maximization and Minimization

Since MaxSAT minimizes the cost of unsatisfied clauses, to maximize a value:
- If setting x=TRUE gives value V, add soft clause `[x]` with weight V
- This way, NOT selecting x (x=FALSE) costs V, so the solver will try to select it

Example:
```python
# To maximize the selection of items with values
item_value = 10
wcnf.append([item_var], weight=item_value)  # Cost 10 if item NOT selected
```

### Types of Constraints

1. **Hard Constraints**: Must be satisfied for a solution to be valid
   - Added with: `wcnf.append([literals])`  (no weight parameter)
   
2. **Soft Constraints**: Can be violated at a cost equal to their weight
   - Added with: `wcnf.append([literals], weight=W)`  (positive weight value)
   - The solver minimizes the sum of weights of violated soft constraints

### Key Components

- **WCNF**: Weighted Conjunctive Normal Form formula
- **Hard Clauses**: Added with `wcnf.append([literals])` (no weight specified)
- **Soft Clauses**: Added with `wcnf.append([literals], weight=W)` (positive weight value)
- **Cost**: Sum of weights of violated soft constraints (lower is better)
- **RC2 Solver**: Modern MaxSAT solver with excellent performance

### Cost vs. Objective Value

Understanding the relationship between cost and your optimization objective is crucial:

- **Cost** (`solver.cost`): What the solver minimizes - the sum of weights of unsatisfied soft clauses
- **Objective**: Your problem's value (often different from cost)

For maximization problems:
- If you want to maximize a total value of 100, and achieve 80, then cost = 100 - 80 = 20
- The cost represents the "lost value" from not achieving the maximum

Example:
```python
# Problem: Select items to maximize total value
total_possible_value = sum(all_item_values)
achieved_value = sum(values of selected items)
# solver.cost ≈ total_possible_value - achieved_value
```

### Basic Optimization Example

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
# export_maxsat_solution is automatically available

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
        item_values = {1: 10, 2: 5, 3: 7, 4: 12}  # Map variables to their values
        total_value = sum(
            item_values[var] for var in [1, 2, 3, 4]
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

The `export_maxsat_solution` function simplifies returning results from MaxSAT problems. This function is automatically available in the environment and does not need to be imported:

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
# export_maxsat_solution is automatically available

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
# export_maxsat_solution is automatically available

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
# export_maxsat_solution is automatically available

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

# MaxSAT helper functions (automatically available)
from mcp_solver.maxsat.templates import (
    # Objective helpers
    maximize_sum, minimize_sum, optimize_net_value, calculate_objective_value,
    # Cardinality constraints
    at_least_k, at_most_k, prefer_at_least_k, prefer_at_most_k,
    # Variable mapping
    VariableMap, create_variable_map,
    # Basic helpers
    add_hard_constraint, add_soft_constraint, solve_maxsat_problem
)
```

### Template Helpers

The MCP Solver provides built-in helper functions for common MaxSAT patterns:

#### Objective Optimization

```python
# Maximize sum of values
maximize_sum(wcnf, [(var1, 10), (var2, 20), (var3, 15)])
# Creates soft clauses [var] with weight=value for each

# Minimize sum of costs  
minimize_sum(wcnf, [(var1, 5), (var2, 8), (var3, 3)])
# Creates soft clauses [-var] with weight=cost for each

# Handle items with both benefits and costs
optimize_net_value(wcnf, [
    (var1, benefit=20, cost=5),   # net value = 15 (positive)
    (var2, benefit=10, cost=15),  # net value = -5 (negative)
])

# Calculate achieved objective value from solution
objective = calculate_objective_value(model, [(var1, 10), (var2, 20)])
```

#### Cardinality Constraints

```python
# Hard constraints
at_least_k(wcnf, [var1, var2, var3, var4], k=2)  # At least 2 must be true
at_most_k(wcnf, [var1, var2, var3, var4], k=3)   # At most 3 can be true

# Soft preferences (simpler encoding for optimization)
prefer_at_least_k(wcnf, variables, k=5, penalty_per_missing=10)
prefer_at_most_k(wcnf, variables, k=3, penalty_per_extra=5)
```

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
        # solution will be {'x1': True, 'x2': False, ...}
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

## Common Problem Patterns

### Assignment Problems (Workers to Tasks)

```python
# Workers assigned to tasks with preferences
workers = ["Alice", "Bob", "Carol"]
tasks = ["Task1", "Task2", "Task3"]

# Create assignment variables
assignments = {}
for worker in workers:
    for task in tasks:
        var = create_var(f"{worker}_{task}")
        assignments[(worker, task)] = var

# Each task needs exactly one worker
for task in tasks:
    task_vars = [assignments[(w, task)] for w in workers]
    exactly_one(wcnf, task_vars)  # Hard constraint

# Workers have preferences (enthusiasm scores)
preferences = {
    ("Alice", "Task1"): 8,
    ("Alice", "Task2"): 5,
    ("Bob", "Task1"): 6,
    ("Bob", "Task3"): 9,
    ("Carol", "Task2"): 7,
    ("Carol", "Task3"): 6
}

# Maximize total enthusiasm
var_value_pairs = [(assignments[key], value) for key, value in preferences.items()]
maximize_sum(wcnf, var_value_pairs)
```

### Selection with Budget

```python
# Select items within budget constraint
items = {
    "A": {"cost": 10, "value": 25},
    "B": {"cost": 15, "value": 30},
    "C": {"cost": 8, "value": 18},
    "D": {"cost": 12, "value": 22}
}
budget = 30

# Create selection variables
item_vars = {name: create_var(name) for name in items}

# Budget constraint (simplified for small problems)
# For each subset exceeding budget, at least one must be false
for subset_size in range(2, len(items) + 1):
    for subset in itertools.combinations(items.keys(), subset_size):
        total_cost = sum(items[i]["cost"] for i in subset)
        if total_cost > budget:
            wcnf.append([-item_vars[i] for i in subset])

# Maximize value
maximize_sum(wcnf, [(item_vars[name], info["value"]) for name, info in items.items()])
```

### Scheduling with Soft Preferences

```python
# Schedule shifts with preferences and constraints
shifts = ["Morning", "Afternoon", "Night"]
workers = ["Alex", "Beth", "Carl", "Dana"]

# Create shift assignment variables
shift_vars = {}
for worker in workers:
    for shift in shifts:
        shift_vars[(worker, shift)] = create_var(f"{worker}_{shift}")

# Hard: Each shift needs between 2-3 workers
for shift in shifts:
    vars_for_shift = [shift_vars[(w, shift)] for w in workers]
    at_least_k(wcnf, vars_for_shift, k=2)
    at_most_k(wcnf, vars_for_shift, k=3)

# Soft: Workers should work 1-2 shifts (preference)
for worker in workers:
    vars_for_worker = [shift_vars[(worker, s)] for s in shifts]
    prefer_at_least_k(wcnf, vars_for_worker, k=1, penalty_per_missing=5)
    prefer_at_most_k(wcnf, vars_for_worker, k=2, penalty_per_extra=3)

# Worker preferences for shifts
preferences = {
    ("Alex", "Morning"): 8,
    ("Beth", "Afternoon"): 7,
    ("Carl", "Night"): 9,
    # ... more preferences
}
maximize_sum(wcnf, [(shift_vars[key], value) for key, value in preferences.items()])
```

## Debugging MaxSAT Solutions

When your solution has unexpectedly high cost or doesn't produce expected results:

### 1. Check Your Soft Constraint Encoding

```python
# WRONG: Thinking this "prefers" item selection
wcnf.append([item_var], weight=value)  # Actually: cost if item NOT selected

# Remember: weight is the cost of the clause being UNSATISFIED
# Clause [x] is unsatisfied when x=FALSE
# Clause [-x] is unsatisfied when x=TRUE
```

### 2. Verify Cost Interpretation

```python
# solver.cost = Sum of weights of unsatisfied soft clauses

# For maximization problems, high cost might be normal:
# If total possible value = 100 and cost = 20, you achieved value = 80
```

### 3. Debug Which Soft Clauses Are Unsatisfied

```python
# After solving, check which soft constraints are violated
if model is not None:
    # Track unsatisfied soft constraints in your solution
    unsatisfied_info = []
    
    # Example: if you have soft clause [var] with weight W
    if var not in model:  # var is FALSE
        unsatisfied_info.append({
            "clause": "[var]",
            "weight": W,
            "reason": "var is FALSE"
        })
    
    # Include this in your export for debugging
    export_maxsat_solution({
        "satisfiable": True,
        "cost": solver.cost,
        "debug_unsatisfied": unsatisfied_info,
        # ... other solution data
    })
```

### 4. Common Debugging Patterns

- **Unexpectedly high cost**: Check if you're encoding maximization correctly
- **All variables false**: You might have contradictory soft constraints
- **Wrong optimization direction**: Ensure you understand what's being minimized
- **Infeasible problems**: Check your hard constraints aren't over-constrained

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.
- **Split long code parts** into smaller items.
- **Verification:**  
  Always verify the solution after a solve operation by checking that all hard constraints are satisfied and the optimization objective makes sense.

Happy optimizing with MCP Solver!