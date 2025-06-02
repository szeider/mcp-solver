# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to MaxSAT (Maximum Satisfiability) optimization with a simplified interface for weighted constraint problems using WCNF (Weighted Conjunctive Normal Form).



## Overview

The MCP Solver integrates MaxSAT solving with the Model Context Protocol, allowing you to create, modify, and solve weighted constraint optimization problems. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using a MaxSAT solver.


## Quick Start Example

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Simple optimization problem: Select items to maximize value
# Each item has a value (benefit if selected)

wcnf = WCNF()

# Define variables: 1=itemA, 2=itemB, 3=itemC
itemA, itemB, itemC = 1, 2, 3

# Hard constraints (must be satisfied)
# Example: Can't select both A and B
wcnf.append([-itemA, -itemB])

# Soft constraints with weights
# Weight = penalty if clause is FALSE
wcnf.append([itemA], weight=5)   # Penalty 5 if itemA not selected
wcnf.append([itemB], weight=3)   # Penalty 3 if itemB not selected  
wcnf.append([itemC], weight=2)   # Penalty 2 if itemC not selected

# Solve with RC2 MaxSAT solver
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Create solution
        solution = {
            "itemA": itemA in model,
            "itemB": itemB in model,
            "itemC": itemC in model
        }
        
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            "assignment": solution
        })
    else:
        export_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })
```

## ⚠️ Common Pitfalls

- **Incorrect Problem Parameters**: ALWAYS encode the EXACT problem specifications. If the problem asks for "9 colors", use `range(9)` not `range(10)`. If it asks for "6 queens and 5 knights", use exactly those numbers.
- **Off-by-one Errors**: Be careful with ranges. "9 colors" means colors 0-8 (using `range(9)`), NOT colors 0-9.
- **Arbitrary Problem Modifications**: NEVER change problem parameters (e.g., reducing piece counts) without explicit justification. Solve the problem as stated.
- **Incomplete Variables**: Always complete variable assignments (e.g., `item_vars = [has_item(i) for i in items]`)
- **Dictionary Updates**: Use `selected_items[item] = value` (not `selected_items = value`)
- **Export Solution**: Always include `export_solution()` with at minimum `{"satisfiable": True/False}`
- **Memory Management**: Always call `solver.delete()` or use `with` statement
- **Variable Ranges**: MaxSAT variables must be positive integers (1, 2, 3, ...)
- **WCNF vs CNF**: Always use `WCNF()` for MaxSAT problems, not `CNF()`

## 🚨 CRITICAL: Problem Parameter Verification

**NEVER modify problem parameters!** The agent has been observed changing numbers, which leads to solving the wrong problem entirely.

### Before encoding ANY problem:

1. **Extract ALL numeric parameters** from the problem statement
2. **Add verification comments** to track them:
   ```python
   # Problem parameters (DO NOT CHANGE):
   # - Number of items: 10
   # - Must select: exactly 3  
   # - Budget limit: 50
   
   # Verify encoding matches:
   items = list(range(10))  # ✓ 10 items [0,1,2,3,4,5,6,7,8,9]
   required_selections = 3   # ✓ Exactly 3 items
   budget = 50              # ✓ Budget limit 50
   ```

3. **Include parameter checks** in your solution:
   ```python
   # Verification: Using exactly the required parameters
   assert len(items) == 10, f"Expected 10 items, using {len(items)}"
   assert required_selections == 3, "Must select exactly 3 items"
   ```

### ❌ NEVER DO THIS:
```python
# Problem asks for 9 colors
colors = list(range(10))  # WRONG: This is 10 colors!

# Problem asks for 6 nurses per shift
exactly_k(wcnf, nurse_vars, 5)  # WRONG: Changed to 5!
```

### ✅ ALWAYS DO THIS:
```python
# Problem asks for 9 colors
colors = list(range(9))  # CORRECT: Exactly 9 colors [0-8]

# Problem asks for 6 nurses per shift  
exactly_k(wcnf, nurse_vars, 6)  # CORRECT: Exactly 6
```

## Counting and Ranges

### Understanding Problem Specifications:
- **"N items"** means exactly N items, typically `range(N)` which gives `[0,1,...,N-1]`
- **"K selections"** means exactly K items must be selected, no more, no less
- **"Maximum of M"** means at most M (could be less)
- **"Minimum of M"** means at least M (could be more)

### Common Range Patterns:
```python
# For N items:
items = list(range(N))        # [0, 1, 2, ..., N-1]

# For grid positions:
positions = [(i,j) for i in range(rows) for j in range(cols)]

# For counting selections:
assert len(selected_items) == required_count
```

### Verification is Key:
Always verify your encoding matches the problem:
```python
# Verify counts match problem specification
assert len(items) == required_items, f"Wrong number of items"
assert num_selections == required_selections, f"Wrong number of selections"
```

## ⚠️ WARNING: Do Not Modify Problem Parameters

### The #1 cause of MaxSAT failures is changing problem numbers!

❌ **NEVER** change numbers to "simplify" the problem  
❌ **NEVER** reduce constraints thinking it will help  
❌ **NEVER** add extra variables "just in case"  

✅ **ALWAYS** encode the EXACT problem as stated  
✅ **ALWAYS** use the precise numbers given  
✅ **ALWAYS** verify your parameters match the problem  

**Remember**: If the problem specifies exact numbers, you MUST use those exact numbers. Changing them means solving a different problem entirely.

## Solving and Verification

- **Pre-Solving Verification:**  
  Before solving, verify that your encoding matches the problem specification exactly. Double-check numerical parameters, ranges, and counts.

- **Solution Verification:**  
  After solving, verify that the returned solution satisfies all specified constraints. Check both hard constraints and the optimization objective.

- **Parameter Validation Example:**
  ```python
  # Problem: "Select 3 items from 10 to maximize value"
  # CORRECT:
  items = list(range(10))  # [0, 1, 2, ..., 9] - exactly 10 items
  print(f"Using {len(items)} items: {items}")  # Verify before solving
  
  # Add constraint: exactly 3 items selected
  item_vars = [i+1 for i in items]  # Variables 1-10
  exactly_k(wcnf, item_vars, 3)
  
  # INCORRECT:
  # items = list(range(11))  # This gives 11 items!
  ```

## MaxSAT Semantics

MaxSAT minimizes the sum of weights of UNSATISFIED soft clauses:
- Soft clause `[x]` with weight W costs W if x is FALSE
- Soft clause `[-x]` with weight W costs W if x is TRUE
- The solver finds an assignment that minimizes total cost

## Available Tools

| Tool           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `clear_model`  | Reset the MaxSAT model                                       |
| `add_item`     | Add Python code to the model                                 |
| `replace_item` | Replace code in the model                                    |
| `delete_item`  | Delete code from the model                                   |
| `solve_model`  | Solve the current model (requires timeout parameter between 1-30 seconds) |
| `get_model`    | Fetch the current content of the MaxSAT model                |

> **Timeout Handling:** When using `solve_model`, always specify a timeout (in seconds) to prevent long-running computations. If your model times out, you'll receive a response with `"status": "timeout"` and `"timeout": true`, but the connection will be maintained so you can modify and retry your model.

## ⭐ Cardinality Constraints

The following helper functions are available for common constraint patterns:

```python
# Import cardinality helpers (automatically available)
from mcp_solver.maxsat.templates import at_most_k, at_least_k, exactly_k

# At most k variables can be true
at_most_k(wcnf, [var1, var2, var3, var4], k=2)

# At least k variables must be true  
at_least_k(wcnf, [var1, var2, var3, var4], k=2)

# Exactly k variables must be true
exactly_k(wcnf, [var1, var2, var3, var4], k=2)
```

**IMPORTANT:** These helper functions add clauses to your WCNF formula. They handle the combinatorial encoding for you.

## Variable Creation and Mapping

For MaxSAT problems, you need to create variables as positive integers and map them to meaningful names:

### Basic Variable Creation Pattern
```python
# Simple counter approach
var_count = 0
def create_var(name):
    global var_count
    var_count += 1
    return var_count

# Create variables
x1 = create_var("x1")
x2 = create_var("x2")
```

### Variable Mapping for Complex Problems
```python
# For grid problems
cell_vars = {}
var_count = 0
for i in range(rows):
    for j in range(cols):
        var_count += 1
        cell_vars[(i, j)] = var_count

# For assignment problems  
assign_vars = {}
var_count = 0
for worker in workers:
    for task in tasks:
        var_count += 1
        assign_vars[(worker, task)] = var_count
```

### Variable Namespacing
```python
# Using functions for different variable types
var_count = 0

def item_selected(item_id):
    global var_count
    var_count += 1
    return var_count

def feature_enabled(feature_name):
    global var_count
    var_count += 1
    return var_count
```

## Standard Code Pattern

Here's the recommended pattern for creating MaxSAT models:

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# 1. Create WCNF formula
wcnf = WCNF()

# 2. Create variables with meaningful names
var_count = 0
def create_var(name):
    global var_count
    var_count += 1
    return var_count

# Create variables
x1 = create_var("x1")
x2 = create_var("x2")
x3 = create_var("x3")

# 3. Add hard constraints (no weight parameter)
wcnf.append([x1, x2])        # x1 OR x2 must be true
wcnf.append([-x1, -x2, x3])  # NOT x1 OR NOT x2 OR x3

# 4. Add soft constraints (with weight = penalty if false)
wcnf.append([x1], weight=3)   # Prefer x1 to be true (cost 3 if false)
wcnf.append([-x2], weight=2)  # Prefer x2 to be false (cost 2 if true)

# 5. Solve and export results
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Map variables back to names
        assignment = {
            "x1": x1 in model,
            "x2": x2 in model,
            "x3": x3 in model
        }
        
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            "assignment": assignment
        })
    else:
        export_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })
```

## Interpreting the Solver's Model

When a solution is found, `solver.compute()` returns a model - a list of literals:

- Positive numbers (e.g., 1, 5, 7) represent variables that are TRUE
- Negative numbers (e.g., -2, -3, -4) represent variables that are FALSE

```python
model = solver.compute()
# Example model: [1, -2, 3, -4, 5]
# This means: var1=True, var2=False, var3=True, var4=False, var5=True

# Check if a variable is true
is_true = var_id in model  # True if variable is in the model

# Create a complete solution dictionary
solution = {}
for name, var_id in variable_mapping.items():
    solution[name] = (var_id in model)
```

## Handling Imports

Available imports:

```python
# Core MaxSAT modules
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Standard Python modules
import math  
import random
import collections
import itertools
import re
import json

# Cardinality constraint helpers (automatically available)
at_most_k, at_least_k, exactly_k
```

## Example: Nurse Scheduling

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Problem: Schedule 5 nurses for 3 shifts
# Constraint: Exactly 2 nurses per shift
# Preference: Each nurse works 1-2 shifts

wcnf = WCNF()

# PARAMETER VERIFICATION
# Problem parameters (DO NOT CHANGE):
# - Number of nurses: 5
# - Number of shifts: 3
# - Nurses per shift: exactly 2
# - Preferred shifts per nurse: 1-2

num_nurses = 5
num_shifts = 3
nurses_per_shift = 2

# Create variables: nurse_shift[n][s] = nurse n works shift s
var_count = 0
nurse_shift = {}
for n in range(num_nurses):
    for s in range(num_shifts):
        var_count += 1
        nurse_shift[(n, s)] = var_count

# Verify parameters
assert len(nurse_shift) == num_nurses * num_shifts, "Wrong number of variables"

# Hard constraint: Exactly 2 nurses per shift
for shift in range(num_shifts):
    shift_vars = [nurse_shift[(n, shift)] for n in range(num_nurses)]
    exactly_k(wcnf, shift_vars, nurses_per_shift)

# Soft preference: Prefer each nurse works at least 1 shift
for nurse in range(num_nurses):
    nurse_vars = [nurse_shift[(nurse, s)] for s in range(num_shifts)]
    # At least one shift for this nurse
    wcnf.append(nurse_vars, weight=10)

# Soft preference: Discourage nurses from working all 3 shifts
for nurse in range(num_nurses):
    # Penalize if nurse works shift 0 AND shift 1 AND shift 2
    wcnf.append([
        -nurse_shift[(nurse, 0)],
        -nurse_shift[(nurse, 1)], 
        -nurse_shift[(nurse, 2)]
    ], weight=5)

# Solve
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        schedule = {}
        for n in range(num_nurses):
            schedule[f"nurse_{n}"] = []
            for s in range(num_shifts):
                if nurse_shift[(n, s)] in model:
                    schedule[f"nurse_{n}"].append(f"shift_{s}")
        
        # Verify solution
        for s in range(num_shifts):
            assigned = sum(1 for n in range(num_nurses) 
                         if nurse_shift[(n, s)] in model)
            assert assigned == nurses_per_shift, f"Shift {s} has {assigned} nurses"
        
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            "schedule": schedule
        })
    else:
        export_solution({
            "satisfiable": False,
            "message": "No valid schedule exists"
        })
```

## Solution Export

`export_solution(data)`: Export data as a solution

- Requires at minimum: `{"satisfiable": True/False}`
- Include relevant data like cost, assignment, or custom fields
- The solution will be automatically extracted and returned

## Final Notes

- **Review Return Information:**  
  Carefully review the confirmation messages and the current model after each tool call.
- **Direct Approach**: Build constraints explicitly rather than using abstractions
- **Weight Semantics**: Remember weights are penalties for UNSATISFIED clauses
- **Variable Creation**: Use simple counter pattern for variable management
- **Verification**: Always verify the solution after a solve operation by checking that all constraints are satisfied and justified.

Happy modeling with MCP Solver!