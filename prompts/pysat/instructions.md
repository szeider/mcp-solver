# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to PySAT (Python SAT Solver) with a simplified interface for propositional constraint modeling using CNF (Conjunctive Normal Form).

**IMPORTANT**: PySAT is fully available in this environment. Always start your code with the standard imports shown in the examples below. Never attempt manual solutions - always use proper SAT encoding.

**YOU MUST USE ACTUAL PYSAT**: The environment has PySAT pre-installed and ready to use. DO NOT create mock solvers or try to simulate SAT solving manually. If you have any issues with imports, check your syntax - the environment definitely has PySAT available.

**CRITICAL**: The `export_solution` function is automatically available in the environment - you do NOT need to import it. Just use it directly as shown in the examples.



## Overview

The MCP Solver integrates PySAT solving with the Model Context Protocol, allowing you to create, modify, and solve PySAT encodings for satisfiability problems. The following tools are available:

- **clear_model**
- **add_item** (uses 0-based indexing: first item is index=0)
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using a SAT solver.


## Required Imports

**ALWAYS start your code with these imports:**

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
```

**Optional imports for advanced features:**
```python
from pysat.card import *  # For cardinality constraints (at_most_k, exactly_k, etc.)
```

**DO NOT import:**
- `export_solution` - it's automatically available
- Helper functions like `at_most_one`, `exactly_one`, etc. - they're automatically available

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

- **Incorrect Problem Parameters**: ALWAYS encode the EXACT problem specifications. If the problem asks for "9 colors", use `range(9)` not `range(10)`. If it asks for "6 queens and 5 knights", use exactly those numbers.
- **Off-by-one Errors**: Be careful with ranges. "9 colors" means colors 0-8 (using `range(9)`), NOT colors 0-9.
- **Arbitrary Problem Modifications**: NEVER change problem parameters (e.g., reducing piece counts) without explicit justification. Solve the problem as stated.
- **Incomplete Variables**: Always complete variable assignments (e.g., `node_color_vars = [has_color(node, color) for color in colors]`)
- **Dictionary Updates**: Use `node_colors[node] = color` (not `node_colors = color`)
- **Export Solution**: Always include `export_solution()` with at minimum `{"satisfiable": True/False}`
- **Memory Management**: Always call `solver.delete()` to free memory
- **Variable Ranges**: PySAT variables must be positive integers (1, 2, 3, ...)

## 🚨 CRITICAL: Problem Parameter Verification

**NEVER modify problem parameters!** The agent has been observed changing numbers, which leads to solving the wrong problem entirely.

### Before encoding ANY problem:

1. **Extract ALL numeric parameters** from the problem statement
2. **Add verification comments** to track them:
   ```python
   # Problem parameters (DO NOT CHANGE):
   # - Number of colors: 9
   # - Number of queens: 6  
   # - Number of knights: 5
   # - Grid size: 6x6
   
   # Verify encoding matches:
   colors = list(range(9))  # ✓ 9 colors [0,1,2,3,4,5,6,7,8]
   num_queens = 6          # ✓ Exactly 6 queens
   num_knights = 5         # ✓ Exactly 5 knights
   ```

3. **Include parameter checks** in your solution:
   ```python
   # Verification: Using exactly the required parameters
   assert len(colors) == 9, f"Expected 9 colors, using {len(colors)}"
   ```

### ❌ NEVER DO THIS:
```python
# Problem asks for 9 colors
colors = list(range(10))  # WRONG: This is 10 colors!

# Problem asks for 6 queens
for clause in exactly_k(queen_vars, 5):  # WRONG: Changed to 5!
```

### ✅ ALWAYS DO THIS:
```python
# Problem asks for 9 colors
colors = list(range(9))  # CORRECT: Exactly 9 colors [0-8]

# Problem asks for 6 queens  
for clause in exactly_k(queen_vars, 6):  # CORRECT: Exactly 6
```

## Counting and Ranges

### Understanding Problem Specifications:
- **"N colors"** means exactly N colors, typically `range(N)` which gives `[0,1,...,N-1]`
- **"N items"** means exactly N items, no more, no less
- **"K-coloring"** means using at most K colors (but often we want to know if exactly K suffice)

### Common Range Patterns:
```python
# For N colors:
colors = list(range(N))        # [0, 1, 2, ..., N-1]

# For grid positions:
positions = [(i,j) for i in range(rows) for j in range(cols)]

# For counting items:
assert len(selected_items) == required_count
```

### Verification is Key:
Always verify your encoding matches the problem:
```python
# Verify counts match problem specification
assert len(colors) == required_colors, f"Wrong number of colors"
assert num_pieces == required_pieces, f"Wrong number of pieces"
```

## Available Tools

| Tool           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `clear_model`  | Reset the PySAT model                                        |
| `add_item`     | Add Python code to the model                                 |
| `replace_item` | Replace code in the model                                    |
| `delete_item`  | Delete code from the model                                   |
| `solve_model`  | Solve the current model (requires timeout parameter between 1-30 seconds) |
| `get_model`    | Fetch the current content of the PySAT model                 |

> **Timeout Handling:** When using `solve_model`, always specify a timeout (in seconds) to prevent long-running computations. If your model times out, you'll receive a response with `"status": "timeout"` and `"timeout": true`, but the connection will be maintained so you can modify and retry your model.

## List Semantics for Model Operations

The model items behave like a standard Python list with 0-based indexing:

- **add_item(index, content)**: Inserts code at the specified position, shifting subsequent items to the right.
  - Example: Model has 3 items. `add_item(1, "new code")` inserts at position 1, shifting items 1+ to the right.
  - Valid indices: 0 to length (inclusive)
  
- **delete_item(index)**: Removes the item at index, shifting subsequent items to the left.
  - Example: Model has 4 items. `delete_item(1)` removes item 1, items 2→1, 3→2, etc.
  - Valid indices: 0 to length-1
  
- **replace_item(index, content)**: Replaces item at index in-place. No shifting.
  - Example: `replace_item(1, "updated code")` changes only item 1.
  - Valid indices: 0 to length-1

**Remember**: First item is at index 0, not 1!

## Important: Two Types of Indexing

1. **Model Item Indexing**: Always 0-based
   - First item is at index 0
   - Used with add_item, replace_item, delete_item
   - Example: `add_item(0, "# First code item")` adds at the beginning

2. **SAT Variable Numbering**: Must be positive integers (1, 2, 3, ...)
   - PySAT requires variables to be numbered starting from 1
   - Variable 0 is invalid in SAT solvers
   - Example: `queen_at_1_1 = 1`, `queen_at_1_2 = 2`, etc.

## Variable Mapping Pattern

When mapping problem entities to SAT variables, use this clear pattern:

```python
# Good: Clear mapping with 1-based variables
queen_vars = {}
var_num = 1
for row in range(8):  # 0-based iteration
    for col in range(8):
        queen_vars[(row, col)] = var_num  # Map position to variable
        var_num += 1

# Alternative: Using a formula
def get_var(row, col, num_cols):
    return row * num_cols + col + 1  # +1 ensures 1-based

# Usage in constraints
for row in range(8):
    # At least one queen in each row
    row_vars = [queen_vars[(row, col)] for col in range(8)]
    formula.append(row_vars)
```

**Key Points:**
- Iterate using 0-based indices (Python convention)
- Map to 1-based variable numbers (SAT requirement)
- Keep mapping consistent throughout your model

## ⚠️ WARNING: Do Not Modify Problem Parameters

### The #1 cause of PySAT failures is changing problem numbers!

❌ **NEVER** change numbers to "simplify" the problem  
❌ **NEVER** reduce piece counts thinking it will help  
❌ **NEVER** add extra colors "just in case"  

✅ **ALWAYS** encode the EXACT problem as stated  
✅ **ALWAYS** use the precise numbers given  
✅ **ALWAYS** verify your parameters match the problem  

**Remember**: If the problem specifies exact numbers, you MUST use those exact numbers. Changing them means solving a different problem entirely.

## Solving and Verification

- **Pre-Solving Verification:**  
  Before solving, verify that your encoding matches the problem specification exactly. Double-check numerical parameters, ranges, and counts.

- **Solution Verification:**  
  After solving, verify that the returned solution satisfies all specified constraints. If you get UNSAT, then check that all clauses are indeed justified from the problem description.

- **Parameter Validation Example:**
  ```python
  # Problem: "Find a coloring with 9 colors"
  # CORRECT:
  colors = list(range(9))  # [0, 1, 2, 3, 4, 5, 6, 7, 8] - exactly 9 colors
  print(f"Using {len(colors)} colors: {colors}")  # Verify before solving
  
  # INCORRECT:
  # colors = list(range(10))  # This gives 10 colors!
  ```

## ⭐ Advanced Helper Functions and Templates

The MCP Solver provides two categories of helpers to make encoding easier:

### 1. Pre-Implemented Constraint Helpers

These functions simplify encoding common logical patterns and are **automatically available without any import**:

```python
# NO IMPORT NEEDED - these are available by default
at_most_one([1, 2, 3])        # At most one variable is true
exactly_one([1, 2, 3])        # Exactly one variable is true
implies(1, 2)                 # If var1 is true, then var2 must be true
mutually_exclusive([1, 2, 3]) # Variables are mutually exclusive
if_then_else(1, 2, 3)         # If var1 then var2 else var3
```

### 2. Cardinality Constraint Templates

For more advanced cardinality constraints:

```python
# These require explicit import
from pysat.card import *

# At most k variables are true
for clause in at_most_k([1, 2, 3, 4], 2): 
    formula.append(clause)

# At least k variables are true
for clause in at_least_k([1, 2, 3, 4], 2): 
    formula.append(clause)

# Exactly k variables are true
for clause in exactly_k([1, 2, 3, 4], 2): 
    formula.append(clause)
```

### 3. Variable Mapping Helper

For simpler variable management, use the `VariableMap` helper:

```python
# Create a variable map for managing variables
var_map = VariableMap()

# Create variables with meaningful names
x1 = var_map.create_var("x1")
x2 = var_map.create_var("x2")

# Create multiple variables at once
features = var_map.create_vars(["premium", "basic", "cloud"])

# After solving, get a readable solution
if solver.solve():
    model = solver.get_model()
    solution = var_map.interpret_model(model)
    # solution = {"x1": True, "x2": False, "premium": True, ...}
```

**IMPORTANT:** These helper functions return clauses that must be added to your formula using the pattern shown above.

### Example Using Cardinality Constraints with Parameter Verification

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *

# PARAMETER VERIFICATION - Track problem requirements
# Problem parameters (example):
# - Total items: 4
# - Must select: exactly 2
# - Constraint: if first item selected, at most 1 of the others

# Verify we're using correct parameters
total_items = 4
required_selections = 2

formula = CNF()

# Create variables with meaningful names
items = list(range(1, total_items + 1))  # [1, 2, 3, 4]
assert len(items) == total_items, f"Expected {total_items} items"

a, b, c, d = items
var_names = {1: "a", 2: "b", 3: "c", 4: "d"}

# Add cardinality constraints using verified parameters
# Exactly 'required_selections' items must be selected
for clause in exactly_k(items, required_selections):  # Using verified parameter
    formula.append(clause)
    
# Additional constraint from problem
# (Example constraint logic here)
    
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

You can use either the built-in `VariableMap` class or create your own variable mapping functions:

### Option 1: Using the VariableMap Class (Recommended)

```python
# Create a variable map for managing variables
var_map = VariableMap()

# Create variables with meaningful names
x1 = var_map.create_var("x1")
edge_ab = var_map.create_var("edge_a_b")

# Create variables for all nodes
node_vars = var_map.create_vars(["node1", "node2", "node3"])

# After solving, get a readable solution
if solver.solve():
    model = solver.get_model()
    solution = var_map.interpret_model(model)
    # solution = {"x1": True, "edge_a_b": False, "node1": True, ...}
    
    # Get the variable mapping for export
    var_mapping = var_map.get_mapping()
```

### Option 2: Creating Your Own Mapping Functions

```python
# Create a variable mapping dictionary
var_mapping = {}
var_count = 1

# Create variables with meaningful names
def create_var(name):
    global var_count  # Use global for module-level variables
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

## Variable Namespacing

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

### CRITICAL: Problem Specification Validation

Always validate your encoding parameters against the problem statement:

```python
# Example: Problem asks for "6 queens and 5 knights"
NUM_QUEENS = 6  # From problem specification
NUM_KNIGHTS = 5  # From problem specification

# Add validation output
print(f"Encoding {NUM_QUEENS} queens and {NUM_KNIGHTS} knights")

# Use exact counts in constraints
for clause in exactly_k(all_queen_vars, NUM_QUEENS):
    formula.append(clause)
for clause in exactly_k(all_knight_vars, NUM_KNIGHTS):
    formula.append(clause)
```

Here are two recommended patterns for creating SAT encodings:

### Pattern 1: Using VariableMap (Recommended)

```python
# 1. Create formula and variable mapping
from pysat.formula import CNF
from pysat.solvers import Glucose3

formula = CNF()
var_map = VariableMap()

# 2. Create variables with meaningful names
x1 = var_map.create_var("x1")
x2 = var_map.create_var("x2")
x3 = var_map.create_var("x3")

# 3. Add constraints using helper functions
# Add constraint: exactly 2 of x1, x2, x3 must be true
for clause in exactly_k([x1, x2, x3], 2):
    formula.append(clause)

# Add constraint: if x1 then x2
formula.append([-x1, x2])  # NOT x1 OR x2

# 4. Solve and process results
solver = Glucose3()
solver.append_formula(formula)

if solver.solve():
    model = solver.get_model()
    solution = var_map.interpret_model(model)
    
    export_solution({
        "satisfiable": True,
        "assignment": solution
    })
else:
    export_solution({
        "satisfiable": False,
        "message": "No solution exists"
    })
    
# 5. Always free solver memory
solver.delete()
```

### Pattern 2: Traditional Approach

```python
# 1. Create variables and formula
from pysat.formula import CNF
from pysat.solvers import Glucose3
from pysat.card import *

formula = CNF()
var_mapping = {}
var_count = 1

def create_var(name):
    global var_count
    var_mapping[name] = var_count
    var_count += 1
    return var_mapping[name]

# 2. Add constraints using helper functions
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
from pysat.formula import CNF
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
    global var_count  # Use global for module-level variables
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

## Troubleshooting

### If you encounter import errors:
1. Make sure you're using the exact imports shown above
2. DO NOT create mock objects or use `types.SimpleNamespace` 
3. DO NOT try to work around missing imports - the environment has everything you need

### Common mistakes to avoid:
- Creating fake solver objects: `solver = types.SimpleNamespace()` ❌
- Theoretical reasoning without SAT encoding ❌
- Manual constraint checking without using a SAT solver ❌

**If imports fail, the problem is likely in your code syntax, not the environment.**

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.
- **Split long code parts** into smaller items.
- **Verification:**  
  Always verify the solution after a solve operation by checking that all constraints are satisfied and justified.
- **Always use actual PySAT solvers** - never create mock objects or try manual solutions.

Happy modeling with MCP Solver!