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

## 📋 MANDATORY Blueprint Structure

**EVERY MaxSAT model MUST follow this exact structure with these item numbers:**

### Blueprint Template:
```python
# Item 1: Imports and WCNF initialization
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
wcnf = WCNF()

# Item 2: Problem parameters and variables
# Define all variables with clear comments
# var = number

# Item 3: Hard constraints
# Add all hard constraints (must be satisfied)
# wcnf.append([literals])

# Item 4: Soft constraints
# Add all soft constraints with weights
# wcnf.append([literals], weight=value)

# Item 5: Solve and export solution
# Use RC2 solver and export results
```

## Quick Start Example

### Item 1: Imports and initialization
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

wcnf = WCNF()
```

### Item 2: Variables
```python
# Problem: Select items to maximize value
# Variables: 1=itemA, 2=itemB, 3=itemC
itemA = 1
itemB = 2  
itemC = 3
```

### Item 3: Hard constraints
```python
# Hard constraint: Can't select both A and B
wcnf.append([-itemA, -itemB])
```

### Item 4: Soft constraints
```python
# Soft constraints (weight = penalty if FALSE)
wcnf.append([itemA], weight=5)   # Penalty 5 if itemA not selected
wcnf.append([itemB], weight=3)   # Penalty 3 if itemB not selected  
wcnf.append([itemC], weight=2)   # Penalty 2 if itemC not selected
```

### Item 5: Solve and export
```python
# Solve with RC2 MaxSAT solver
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model:
        # Create solution dictionary
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

- **Blueprint Violations**: ALWAYS use the 5-item structure. Do NOT combine or skip items!
- **Incorrect Problem Parameters**: ALWAYS encode the EXACT problem specifications
- **Export Solution**: Always include `export_solution()` - it's automatically available
- **Variable Ranges**: MaxSAT variables must be positive integers (1, 2, 3, ...)
- **WCNF vs CNF**: Always use `WCNF()` for MaxSAT problems, not `CNF()`

## 🚨 CRITICAL: Problem Parameter Verification

**NEVER modify problem parameters!**

### Before encoding ANY problem:

1. **Extract ALL numeric parameters** from the problem statement
2. **Add verification comments** in Item 2:
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

### Understanding Soft Constraints
- **Soft clause weight** = penalty for NOT satisfying the clause
- MaxSAT minimizes the total weight of unsatisfied soft clauses
- Think of weights as "costs to pay" when a preference is violated

### Common Patterns:

1. **Maximize selections** (want variable to be true):
   ```python
   wcnf.append([var], weight=benefit)  # Penalty if var is FALSE
   ```

2. **Minimize selections** (want variable to be false):
   ```python
   wcnf.append([-var], weight=cost)  # Penalty if var is TRUE
   ```

## Available Helper Functions

The following functions are automatically available:

### Cardinality Constraints:
- `exactly_k(variables, k)` - Returns clauses for exactly k variables true
- `at_most_k(variables, k)` - Returns clauses for at most k variables true
- `at_least_k(variables, k)` - Returns clauses for at least k variables true

### Basic Constraints:
- `exactly_one(variables)` - Exactly one variable must be true
- `at_most_one(variables)` - At most one variable can be true
- `implies(a, b)` - If a then b

### Usage in Item 3 (hard constraints):
```python
# Item 3: Hard constraints
# Exactly 2 items must be selected
for clause in exactly_k([item1, item2, item3], 2):
    wcnf.append(clause)
```

## Example: Feature Selection

### Item 1: Imports
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

wcnf = WCNF()
```

### Item 2: Variables
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

### Item 3: Hard constraints
```python
# Hard constraints: Dependencies
# If AI selected, must have cloud
wcnf.append([-ai, cloud])  # NOT ai OR cloud

# If analytics selected, must have integration
wcnf.append([-analytics, integration])
```

### Item 4: Soft constraints
```python
# Soft constraints: Maximize total value
# For each feature, add penalty equal to its value if not selected
for feature, value in values.items():
    wcnf.append([feature], weight=value)
```

### Item 5: Solve and export
```python
with RC2(wcnf) as solver:
    if solver.compute():
        model = solver.model
        
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

### Item 1: Imports
```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

wcnf = WCNF()
```

### Item 2: Variables
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

### Item 3: Hard constraints
```python
# Hard constraint: Exactly 2 nurses per shift
for shift in range(num_shifts):
    shift_vars = [nurse_shift[(n, shift)] for n in range(num_nurses)]
    for clause in exactly_k(shift_vars, nurses_per_shift):
        wcnf.append(clause)
```

### Item 4: Soft constraints
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

### Item 5: Solve and export
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

## ✅ Final Checklist

Before submitting your solution, verify:

- [ ] **Item 1**: Imports WCNF and RC2, creates wcnf = WCNF()
- [ ] **Item 2**: Defines all variables with clear comments and parameter verification
- [ ] **Item 3**: Adds all hard constraints using wcnf.append()
- [ ] **Item 4**: Adds all soft constraints with appropriate weights
- [ ] **Item 5**: Uses RC2 solver with compute() and calls export_solution()
- [ ] **Parameters**: All problem parameters match exactly (no modifications!)
- [ ] **Variables**: All variables are positive integers
- [ ] **Export**: Solution includes satisfiable, cost, and problem-specific data

## Final Notes

- **Blueprint Compliance**: ALWAYS follow the 5-item structure
- **Direct Approach**: Build constraints explicitly without abstractions
- **Weight Semantics**: Remember weights are penalties for UNSATISFIED clauses
- **Review Output**: Check confirmation messages after each tool call