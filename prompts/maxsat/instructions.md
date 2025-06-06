# MCP Solver – Quick Start Guide

Welcome to the MCP Solver. This document provides concise guidelines on how to interact with the MCP Solver.

This service provides access to MaxSAT (Maximum Satisfiability) optimization with a simplified interface for weighted constraint problems using WCNF (Weighted Conjunctive Normal Form).

**IMPORTANT**: MaxSAT/PySAT is fully available in this environment. Always use the actual RC2 solver as shown in the examples. Never attempt manual optimization or create mock solvers.

**YOU MUST USE ACTUAL RC2**: The environment has PySAT and RC2 pre-installed and ready to use. DO NOT create mock solvers or try to simulate optimization manually. If you have any issues with imports, check your syntax - the environment definitely has MaxSAT available.

**CRITICAL**: The `export_solution` function is automatically available in the environment - you do NOT need to import it. Just use it directly as shown in the examples.

## Overview

The MCP Solver integrates MaxSAT solving with the Model Context Protocol, allowing you to create, modify, and solve weighted constraint optimization problems. The following tools are available:

- **clear_model**
- **add_item**
- **replace_item**
- **delete_item**
- **solve_model**

These tools let you construct your model incrementally and solve it using a MaxSAT solver.

## List Semantics for Model Operations

The model uses 0-based indexing and standard list semantics:

- **add_item(index, content)**: Inserts code at position, shifting items at index and after to the right
  - Example: Items [0, 1, 2] → add_item(1, X) → [0, X, 1, 2]
  - Valid indices: 0 to length (can append at end)
  
- **delete_item(index)**: Removes item, shifting subsequent items left
  - Example: Items [0, 1, 2, 3] → delete_item(1) → [0, 2, 3]
  - Valid indices: 0 to length-1
  
- **replace_item(index, content)**: Replaces item at index (no shifting)
  - Example: Items [0, 1, 2] → replace_item(1, X) → [0, X, 2]
  - Valid indices: 0 to length-1

**Critical**: Indices start at 0! First item is index 0, second is index 1, etc.

## 📋 MANDATORY Blueprint Structure

**EVERY MaxSAT model MUST follow this exact 5-item structure (0-indexed):**

### Blueprint Template:
```python
# Item 0: Imports and WCNF initialization
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
# Note: Helper functions (exactly_k, at_most_k, etc.) are already available - no import needed
wcnf = WCNF()

# Item 1: Problem parameters and variables
# Define all variables with clear comments
# var = number

# Item 2: Hard constraints
# Add all hard constraints (must be satisfied)
# wcnf.append([literals])

# Item 3: Soft constraints
# Add all soft constraints with weights
# wcnf.append([literals], weight=value)

# Item 4: Solve and export solution (MANDATORY)
# MUST instantiate RC2, call compute(), and export results
```

## 📋 Quick Reference Card

### MaxSAT Optimization Patterns
```python
# MAXIMIZE value → Penalize NOT selecting
wcnf.append([item], weight=value)        # Penalty if item=FALSE

# MINIMIZE cost → Penalize selecting  
wcnf.append([-item], weight=cost)        # Penalty if item=TRUE

# Exactly k items
for clause in exactly_k(items, k):
    wcnf.append(clause)

# Handle UNSAT
if not solver.compute():
    export_solution({"satisfiable": False})
```

**Remember**: You penalize what you DON'T want!

## Quick Start Example

### Item 0: Imports and initialization
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
# Note: Helper functions (exactly_k, at_most_k, etc.) are already available

wcnf = WCNF()
```

### Item 1: Variables
```python
# Problem: Select items to maximize value
# Variables: 1=itemA, 2=itemB, 3=itemC
itemA = 1
itemB = 2  
itemC = 3
```

### Item 2: Hard constraints
```python
# Hard constraint: Can't select both A and B
wcnf.append([-itemA, -itemB])
```

### Item 3: Soft constraints
```python
# Soft constraints: We want to MAXIMIZE total value
# Remember: MaxSAT MINIMIZES penalties, so we penalize NOT selecting valuable items

wcnf.append([itemA], weight=5)   # Lose 5 points if itemA=FALSE (not selected)
wcnf.append([itemB], weight=3)   # Lose 3 points if itemB=FALSE (not selected)  
wcnf.append([itemC], weight=2)   # Lose 2 points if itemC=FALSE (not selected)

# MaxSAT will select items to minimize penalties → maximizes value!
```

### Item 4: Solve and export
```python
# MANDATORY: You MUST instantiate and use the RC2 solver
with RC2(wcnf) as solver:
    # MANDATORY: Call compute() to solve the problem
    model = solver.compute()  # Returns variable assignments or None
    
    if model:
        # Solution found - model contains true variables
        # Create solution dictionary
        solution = {
            "itemA": itemA in model,
            "itemB": itemB in model,
            "itemC": itemC in model
        }
        
        # MANDATORY: Call export_solution with results
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,  # Total weight of unsatisfied soft clauses
            "assignment": solution
        })
    else:
        # No solution exists (hard constraints cannot be satisfied)
        export_solution({
            "satisfiable": False,
            "message": "No solution exists"
        })
```

## ⚠️ Common Pitfalls

- **Blueprint Violations**: ALWAYS use the 5-item structure (0-indexed). Do NOT combine or skip items!
- **Missing RC2 Solver**: MUST instantiate RC2 solver and call compute() - don't just build the WCNF!
- **Incorrect Problem Parameters**: ALWAYS encode the EXACT problem specifications
- **Export Solution**: Always include `export_solution()` - it's automatically available
- **Variable Ranges**: MaxSAT variables must be positive integers (1, 2, 3, ...)
- **WCNF vs CNF**: Always use `WCNF()` for MaxSAT problems, not `CNF()`

## ❌ Common Mistakes to Avoid

### 1. Trying to Import Helper Functions
```python
# ❌ WRONG - Don't import helpers
from mcp_solver.maxsat.constraints import exactly_k
from pysat.card import *

# ✅ CORRECT - Helpers are already available
for clause in exactly_k(vars, 3):
    wcnf.append(clause)
```

### 2. Wrong Optimization Direction
```python
# Problem: Maximize total value
# ❌ WRONG - This minimizes value
wcnf.append([-item], weight=value)

# ✅ CORRECT - Penalty if not selected
wcnf.append([item], weight=value)
```

### 3. Forgetting to Add All Clauses from Helpers
```python
# ❌ WRONG - Only adds one clause
wcnf.append(exactly_k(vars, 2))

# ✅ CORRECT - Add all clauses returned
for clause in exactly_k(vars, 2):
    wcnf.append(clause)
```

### 4. Missing RC2 Solver Instantiation
```python
# ❌ WRONG - Forgetting to use RC2 solver
# Just building wcnf without solving
wcnf = WCNF()
# ... add constraints ...
# No solver usage!

# ✅ CORRECT - MUST instantiate and use RC2
with RC2(wcnf) as solver:
    model = solver.compute()
    if model:
        export_solution({...})
```

### 5. Missing Solution Export
```python
# ❌ WRONG - No export
model = solver.compute()
if model:
    print(model)  # Just printing, not exporting

# ✅ CORRECT - Always export
model = solver.compute()
if model:
    export_solution({
        "satisfiable": True,
        "cost": solver.cost,
        "solution": model
    })
```

## 🚨 CRITICAL: Problem Parameter Verification

**NEVER modify problem parameters!**

### Before encoding ANY problem:

1. **Extract ALL numeric parameters** from the problem statement
2. **Add verification comments** in Item 1:
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

## Weight Semantics in MaxSAT

### 🚨 CRITICAL PATTERN - GET THIS RIGHT!

```python
# Problem: Maximize total value of selected items

# ❌ WRONG - This MINIMIZES value (opposite of what you want!)
for item, value in items.items():
    wcnf.append([-item], weight=value)  # NO! Penalizes selecting valuable items!

# ✅ CORRECT - This MAXIMIZES value  
for item, value in items.items():
    wcnf.append([item], weight=value)  # YES! Penalizes NOT selecting valuable items!
```

**The #1 error in MaxSAT**: Getting the polarity backwards!

### Understanding Soft Constraints
- **Soft clause weight** = penalty for NOT satisfying the clause
- MaxSAT minimizes the total weight of unsatisfied soft clauses
- Think of weights as "costs to pay" when a preference is violated

### ⚠️ CRITICAL: MaxSAT Minimizes UNSATISFIED Clause Weights!

**This is COUNTERINTUITIVE and the #1 source of errors!**

MaxSAT finds assignments that minimize the total weight of **violated** (unsatisfied) clauses. This means:
- To **maximize** selecting valuable items → penalize **NOT** selecting them
- To **minimize** selecting costly items → penalize selecting them

```python
# 🚨 OPPOSITE OF INTUITION - We penalize NOT doing what we want!

# Want to MAXIMIZE value? Penalize NOT selecting:
wcnf.append([item], weight=value)      # Penalty if item=FALSE

# Want to MINIMIZE cost? Penalize selecting:
wcnf.append([-item], weight=cost)      # Penalty if item=TRUE
```

**Remember**: You're defining penalties for **bad** outcomes, not rewards for good ones!

### 🎯 Optimization Direction Quick Reference

**Remember: MaxSAT always MINIMIZES the total penalty (cost)**

1. **To MAXIMIZE value/benefit** (want variable to be TRUE):
   ```python
   # If you want to maximize selecting items with values
   wcnf.append([var], weight=value)  # Penalty = value if var is FALSE
   ```

2. **To MINIMIZE cost** (want variable to be FALSE):
   ```python
   # If you want to minimize selecting items with costs
   wcnf.append([-var], weight=cost)  # Penalty = cost if var is TRUE
   ```

### Examples:
```python
# Problem: Maximize total value of selected features
# Feature A has value 10
wcnf.append([featureA], weight=10)  # Lose 10 points if A not selected

# Problem: Minimize total cost of selected items  
# Item B has cost 5
wcnf.append([-itemB], weight=5)  # Pay 5 points if B is selected
```

### Choosing Soft Constraint Weights

When balancing multiple objectives (e.g., minimize cost while maximizing value):
- **Start with raw values** from the problem statement
- **Keep weights in similar ranges** for balanced optimization
- **Only adjust if needed** - if the solution heavily favors one objective
- **Example**: If costs are 1-5 and values are 10-50, consider scaling to similar ranges

### Encoding "At Least K" Requirements

When a problem requires "select at least k items":
- **If it's a hard requirement**: Use `at_least_k(variables, k)` as hard constraints
- **If it's a soft preference with other objectives**:
  - Option 1: Add at_least_k as hard constraint if the requirement is explicit
  - Option 2: For true optimization, penalize shortfall:
    ```python
    # Penalize each missing item below k
    shortfall_vars = []
    for i in range(1, k):
        var = wcnf.nv + 1
        wcnf.nv = var
        shortfall_vars.append(var)
        # var is true if we have exactly i items (less than k)
        for clause in exactly_k(selected_vars, i):
            wcnf.append(clause + [-var])
        wcnf.append([var], weight=penalty_per_missing * (k - i))
    ```

## Available Helper Functions

The following helper functions are automatically available in your code (no import needed):

### Cardinality Constraints:
- `exactly_k(variables, k)` - Returns clauses for exactly k variables true
- `at_most_k(variables, k)` - Returns clauses for at most k variables true
- `at_least_k(variables, k)` - Returns clauses for at least k variables true
- `exactly_one(variables)` - Returns clauses for exactly one variable true
- `at_most_one(variables)` - Returns clauses for at most one variable true

### Logical Constraints:
- `implies(a, b)` - Returns clause for: if a then b
- `mutually_exclusive(variables)` - Same as at_most_one (mutually exclusive)
- `if_then_else(condition, then_var, else_var)` - If-then-else construct

### ⚠️ IMPORTANT: No Import Needed!
These functions are **embedded** in the environment. Just use them directly:

```python
# Item 2: Hard constraints
# CORRECT - Just use the function directly
for clause in exactly_k([item1, item2, item3], 2):
    wcnf.append(clause)

# WRONG - Do NOT import anything
# from mcp_solver.maxsat.constraints import exactly_k  # ❌ NO!
```

## Example: Feature Selection

### Item 0: Imports
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
# Note: Helper functions are already available

wcnf = WCNF()
```

### Item 1: Variables
```python
# PARAMETER VERIFICATION
# Problem parameters (DO NOT CHANGE):
# - Number of features: 6
# - Feature values: cloud=10, ai=8, security=6, mobile=7, analytics=9, integration=5

# Variables: 1=cloud, 2=ai, 3=security, 4=mobile, 5=analytics, 6=integration
cloud = 1
ai = 2
security = 3
mobile = 4
analytics = 5
integration = 6

# Feature values for soft constraints
values = {
    cloud: 10,
    ai: 8,
    security: 6,
    mobile: 7,
    analytics: 9,
    integration: 5
}
```

### Item 2: Hard constraints
```python
# Hard constraints: Dependencies
# If AI selected, must have cloud
wcnf.append([-ai, cloud])  # NOT ai OR cloud

# If analytics selected, must have integration
wcnf.append([-analytics, integration])
```

### Item 3: Soft constraints
```python
# Goal: Maximize total value of selected features
# Strategy: Penalize NOT selecting each feature by its value
# This is COUNTERINTUITIVE but correct!

for feature, value in values.items():
    wcnf.append([feature], weight=value)  # Penalty = value if feature=FALSE
    
# Example: cloud (value=10) not selected → penalty of 10
# MaxSAT minimizes penalties → selects high-value features
```

### Item 4: Solve and export
```python
with RC2(wcnf) as solver:
    model = solver.compute()
    if model:
        # Extract selected features
        selected_features = {
            "cloud": cloud in model,
            "ai": ai in model,
            "security": security in model,
            "mobile": mobile in model,
            "analytics": analytics in model,
            "integration": integration in model
        }
        
        # Calculate total value
        total_value = sum(values[f] for f in model if f in values)
        
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            "selected_features": selected_features,
            "total_value": total_value
        })
    else:
        export_solution({
            "satisfiable": False,
            "message": "No valid feature selection exists"
        })
```

## Example: Nurse Scheduling

### Item 0: Imports
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
# Note: Helper functions are already available

wcnf = WCNF()
```

### Item 1: Variables
```python
# PARAMETER VERIFICATION
# Problem parameters (DO NOT CHANGE):
# - Number of nurses: 5
# - Number of shifts: 3
# - Nurses per shift: exactly 2
# - Preferred shifts per nurse: 1-2

num_nurses = 5
num_shifts = 3
nurses_per_shift = 2

# Variables: nurse_shift[n][s] = nurse n works shift s
var_count = 0
nurse_shift = {}
for n in range(num_nurses):
    for s in range(num_shifts):
        var_count += 1
        nurse_shift[(n, s)] = var_count

# Verify parameters
assert len(nurse_shift) == num_nurses * num_shifts, "Wrong number of variables"
```

### Item 2: Hard constraints
```python
# Hard constraint: Exactly 2 nurses per shift
for shift in range(num_shifts):
    shift_vars = [nurse_shift[(n, shift)] for n in range(num_nurses)]
    for clause in exactly_k(shift_vars, nurses_per_shift):
        wcnf.append(clause)
```

### Item 3: Soft constraints
```python
# Soft preference: Each nurse works at least 1 shift (weight 10)
for nurse in range(num_nurses):
    nurse_vars = [nurse_shift[(nurse, s)] for s in range(num_shifts)]
    wcnf.append(nurse_vars, weight=10)  # At least one shift

# Soft preference: Discourage working all 3 shifts (weight 5)
for nurse in range(num_nurses):
    # Penalize if nurse works all shifts
    wcnf.append([
        -nurse_shift[(nurse, 0)],
        -nurse_shift[(nurse, 1)], 
        -nurse_shift[(nurse, 2)]
    ], weight=5)
```

### Item 4: Solve and export
```python
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        schedule = {}
        for n in range(num_nurses):
            schedule[f"nurse_{n}"] = []
            for s in range(num_shifts):
                if nurse_shift[(n, s)] in model:
                    schedule[f"nurse_{n}"].append(f"shift_{s}")
        
        # Verify solution meets requirements
        for s in range(num_shifts):
            assigned = sum(1 for n in range(num_nurses) 
                         if nurse_shift[(n, s)] in model)
            assert assigned == nurses_per_shift
        
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

## Common Encoding Patterns

### Pattern 1: Maximizing Value Selection
When you want to maximize the total value of selected items:

```python
# WRONG ❌ - This MINIMIZES value!
for item, value in item_values.items():
    wcnf.append([item], weight=value)  # Penalizes selecting valuable items

# CORRECT ✅ - Penalize NOT selecting valuable items
for item, value in item_values.items():
    wcnf.append([item], weight=value)  # Penalty = value if item NOT selected
```

**Why this works**: MaxSAT minimizes penalties. By penalizing NOT selecting an item by its value, MaxSAT will select high-value items to avoid large penalties.

### Pattern 2: Minimizing Cost Selection
When you want to minimize the total cost of selected items:

```python
# WRONG ❌ - This MAXIMIZES cost!
for item, cost in item_costs.items():
    wcnf.append([item], weight=cost)  # Penalizes NOT selecting costly items

# CORRECT ✅ - Penalize selecting costly items
for item, cost in item_costs.items():
    wcnf.append([-item], weight=cost)  # Penalty = cost if item IS selected
```

**Why this works**: By penalizing the selection of an item by its cost, MaxSAT will avoid selecting high-cost items.

### Pattern 3: Bonus for Relationship (Auxiliary Variables)
When you want to give a bonus if two things happen together:

```python
# Problem: Give bonus of 5 if both personA and personB are selected

# Step 1: Create auxiliary variable
aux_var = max(all_variables) + 1  # Next available variable number

# Step 2: Link auxiliary to the relationship
# aux is true IFF both persons are selected
wcnf.append([-personA, -personB, aux_var])   # personA AND personB => aux
wcnf.append([personA, -aux_var])             # aux => personA
wcnf.append([personB, -aux_var])             # aux => personB

# Step 3: Give bonus when auxiliary is true
wcnf.append([aux_var], weight=5)  # Penalty of 5 if aux is FALSE (bonus not given)
```

**Why this works**: The auxiliary variable tracks whether both conditions are met. We penalize NOT having the auxiliary true, which encourages satisfying both conditions.

### Pattern 4: Exactly One Relationship Bonus
When multiple pairs could get a bonus, but you want to count it only once:

```python
# Problem: Carol and Emma get bonus 3 if in same room (count once, not per room)

# Create one auxiliary for the relationship
same_room_aux = max_var + 1

# For each room, if both are in it, auxiliary must be true
for room in rooms:
    carol_in_room = person_room[(carol_id, room)]
    emma_in_room = person_room[(emma_id, room)]
    wcnf.append([-carol_in_room, -emma_in_room, same_room_aux])

# If auxiliary is true, they must be in some room together
room_clauses = []
for room in rooms:
    room_clauses.extend([-same_room_aux, person_room[(carol_id, room)]])
    room_clauses.extend([-same_room_aux, person_room[(emma_id, room)]])
wcnf.append(room_clauses)

# Give bonus once
wcnf.append([same_room_aux], weight=3)  # Penalty if NOT in same room
```

### Pattern 5: Conditional Penalties
When you want to penalize something only if a condition is met:

```python
# Problem: If itemA is selected, penalize selecting itemB by 10

# Create auxiliary: aux = (itemA AND itemB)
aux = max_var + 1
wcnf.append([-itemA, -itemB, aux])    # itemA AND itemB => aux
wcnf.append([itemA, -aux])            # aux => itemA  
wcnf.append([itemB, -aux])            # aux => itemB

# Penalize the bad combination
wcnf.append([-aux], weight=10)  # Penalty if both are selected
```

## Solution Export

### CRITICAL: Always Export Your Solution

**You MUST call `export_solution()` to return results.** This function is automatically available in the environment.

`export_solution(data)`: Extract and format solutions from a MaxSAT solver or solution data

This function processes MaxSAT solution data and creates a standardized output format. It supports both direct dictionary input and RC2 MaxSAT solver objects.

### Basic Usage:

```python
# Method 1: Direct dictionary (recommended)
export_solution({
    "satisfiable": True,
    "cost": solver.cost,
    "selected_items": selected,
    "total_value": total
})

# Method 2: Pass RC2 solver directly
export_solution(solver, variables=var_mapping, objective=total_value)
```

### Important Notes:

- Requires at minimum: `{"satisfiable": True/False}`
- Structure data using custom dictionaries that reflect your problem domain
- For optimization problems, include `cost` from the solver
- The function marks solutions with `_is_maxsat_solution` for identification

## Handling Unsatisfiable Problems

### When MaxSAT Returns UNSATISFIABLE

If the RC2 solver cannot find a solution:

```python
with RC2(wcnf) as solver:
    if solver.compute():
        # Process solution as normal
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            # ... solution details ...
        })
    else:
        # Problem is UNSATISFIABLE
        export_solution({
            "satisfiable": False,
            "message": "No solution exists - hard constraints cannot all be satisfied"
        })
```

### ⚠️ CRITICAL: Never Remove Constraints!

When you encounter unsatisfiability:
1. **DO NOT** remove constraints to make the problem satisfiable
2. **DO NOT** modify the problem to find "some" solution
3. **DO** report that the problem is unsatisfiable
4. **DO** explain which constraints likely conflict if possible

### Example: Detecting Conflicting Constraints

```python
# Problem: Schedule 3 items in 2 slots, each slot holds at most 1 item
# This is UNSATISFIABLE (3 items can't fit in 2 single-item slots)

# After adding constraints and solving:
with RC2(wcnf) as solver:
    if not solver.compute():
        export_solution({
            "satisfiable": False,
            "message": "Cannot fit 3 items in 2 slots with capacity 1 each",
            "conflict": "3 items require 3 slots, but only 2 available"
        })
```

## ✅ Final Checklist

Before submitting your solution, verify:

- [ ] **Item 0**: Imports WCNF and RC2, creates wcnf = WCNF()
- [ ] **Item 1**: Defines all variables with clear comments and parameter verification
- [ ] **Item 2**: Adds all hard constraints using wcnf.append()
- [ ] **Item 3**: Adds all soft constraints with appropriate weights
- [ ] **Item 4**: MANDATORY - Instantiates RC2(wcnf), calls solver.compute(), and exports solution
- [ ] **Parameters**: All problem parameters match exactly (no modifications!)
- [ ] **Variables**: All variables are positive integers
- [ ] **Export**: Solution includes satisfiable, cost, and problem-specific data

## Final Notes

- **Blueprint Compliance**: ALWAYS follow the 5-item structure (0-indexed)
- **Direct Approach**: Build constraints explicitly without abstractions
- **Weight Semantics**: Remember weights are penalties for UNSATISFIED clauses
- **Review Output**: Check confirmation messages after each tool call

## Important: Two Types of Indexing

1. **Model Item Indexing**: Always 0-based
   - First item is at index 0 (imports)
   - Used with add_item, replace_item, delete_item
   - Example: `add_item(0, "# Import statements")` adds at the beginning

2. **SAT Variable Numbering**: Must be positive integers (1, 2, 3, ...)
   - MaxSAT requires variables to be numbered starting from 1
   - Variable 0 is invalid in SAT solvers
   - Example: `item1 = 1`, `item2 = 2`, etc.

## Variable Mapping Pattern

When mapping problem entities to SAT variables:

```python
# Good: Clear mapping with 1-based variables
var_count = 0
nurse_shift = {}
for n in range(num_nurses):      # 0-based iteration
    for s in range(num_shifts):
        var_count += 1
        nurse_shift[(n, s)] = var_count  # 1-based variable

# Alternative: Direct formula
def get_var(nurse, shift, num_shifts):
    return nurse * num_shifts + shift + 1  # +1 for 1-based
```

**Remember:** Iterate with 0-based indices, map to 1-based variables!

## Troubleshooting

### "NameError" or Import Errors

If you see errors like `NameError: name 'RC2' is not defined` or `__build_class__ not found`, the issue is likely a **syntax error** in your code, not a missing import. The environment has all necessary imports available.

**Common causes:**
- Missing closing parenthesis, bracket, or quote
- Incorrect indentation
- Typo in variable or function names

**Solution:** Carefully check your syntax. The error message will usually point to the line with the issue.

### Common mistakes to avoid:

1. **Creating fake solver objects** ❌
```python
# NEVER DO THIS - The environment has real RC2!
import types
solver = types.SimpleNamespace()
solver.compute = lambda: [1, 2, 3]  # compute() returns the model directly
```

2. **Trying to mock MaxSAT optimization** ❌
```python
# NEVER DO THIS - Use actual RC2!
def fake_optimize(wcnf):
    # Manual "optimization" logic
    return some_solution
```

3. **Assuming the environment doesn't have MaxSAT** ❌
```python
# NEVER DO THIS
print("MaxSAT not available, using heuristic instead")
```

**REMEMBER**: The environment has PySAT and RC2 fully installed and ready to use. Always use the actual MaxSAT solver as shown in the examples. If you encounter any import errors, it's likely a syntax error in your code, not a missing module.