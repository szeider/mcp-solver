# MCP Solver - Z3 Mode

This document provides information about using MCP Solver with the Z3 SMT Solver backend.

## ⚠️ IMPORTANT: Solution Export Requirement ⚠️

All Z3 models MUST call `export_solution(solver=solver, variables=variables)` to properly extract solutions. Without this call, results will not be captured, even if your solver finds a solution!

```python
# Always include this import
from mcp_solver.z3 import export_solution

# After solving, always call export_solution with BOTH parameters
if solver.check() == sat:
    export_solution(solver=solver, variables=variables)
else:
    print("No solution found")
```

## Configuration

To run the MCP Solver in Z3 mode:

- Use `mcp-solver-z3` command (instead of `mcp-solver`)

## Core Features

Z3 mode provides SMT (Satisfiability Modulo Theories) solving capabilities:

- **Rich type system**: Booleans, integers, reals, bitvectors, arrays, and more
- **Constraint solving**: Solve complex constraint satisfaction problems
- **Optimization**: Optimize with respect to objective functions
- **Quantifiers**: Express constraints with universal and existential quantifiers

## Best Practices for Problem Modeling

1. **Translate all constraints correctly**:

   - Consider edge cases and implicit constraints
   - Verify that each constraint's logical formulation matches the intended meaning
   - Be explicit about ranges, domains, and special case handling

2. **Structure your model clearly**:

   - Use descriptive variable names for readability
   - Group related constraints together
   - Comment complex constraint logic

3. **Use the correct export_solution call**:

   ```python
   export_solution(solver=solver, variables=variables)
   ```

   - Always provide both parameters
   - Always check if a solution exists before exporting

4. **For complex problems, use incremental development**:

   - Build and test constraints one at a time
   - Verify each constraint independently before combining them
   - Use intermediate assertions to check state

5. **For difficult problems, include verification code**:

   - Add checks that verify all constraints are satisfied
   - Output detailed information for debugging purposes
   - Test edge cases explicitly

## Template Library

Z3 mode includes templates for common modeling patterns:

```python
from mcp_solver.z3.templates import (
    # Array and sequence properties
    array_is_sorted,    # Array elements are in non-decreasing order
    all_distinct,       # All elements in array are different
    
    # Cardinality constraints
    exactly_k,          # Exactly k elements equal to value
    at_most_k,          # At most k elements equal to value
    
    # Optimization templates
    smallest_subset_with_property  # Find minimal subset with a property
)
```

## Example Model: Sudoku Solver

```python
from z3 import *
from mcp_solver.z3 import export_solution  # Import the export_solution function

def build_model():
    # Sudoku puzzle (3x3 for simplicity)
    # 0 represents empty cells
    puzzle = [
        [5, 3, 0],
        [6, 0, 0],
        [0, 9, 8]
    ]
    
    # Create 3x3 matrix of integer variables
    cells = [[Int(f"cell_{i}_{j}") for j in range(3)] for i in range(3)]
    
    # Create solver
    s = Solver()
    
    # Add constraints
    for i in range(3):
        for j in range(3):
            # Domain constraints
            s.add(cells[i][j] >= 1, cells[i][j] <= 9)
            
            # Fixed cells
            if puzzle[i][j] != 0:
                s.add(cells[i][j] == puzzle[i][j])
    
    # All distinct in rows, columns
    for i in range(3):
        s.add(all_distinct([cells[i][j] for j in range(3)]))  # Rows
        s.add(all_distinct([cells[j][i] for j in range(3)]))  # Columns
    
    variables = {f"cell_{i}_{j}": cells[i][j] for i in range(3) for j in range(3)}
    return s, variables

# Solve the model and export the solution
solver, variables = build_model()

# Always check if a solution exists before exporting
if solver.check() == sat:
    # This line is REQUIRED to extract the solution
    export_solution(solver=solver, variables=variables)
else:
    print("No solution found")
```

## Function Scope and Variable Access

When working with Z3 models, variable scope is automatically managed to ensure variables are accessible when needed:

```python
# RECOMMENDED APPROACH:
# Define a function to build your model that returns solver and variables
def build_model():
    x = Int('x')
    y = Int('y')
    
    solver = Solver()
    solver.add(x > 0, y > 0, x + y == 10)
    
    # Return all necessary context
    return solver, {"x": x, "y": y}

# Call the function and use its return values
solver, variables = build_model()

# Call export_solution OUTSIDE the function
if solver.check() == sat:
    # CORRECT way to call export_solution
    export_solution(solver=solver, variables=variables)
else:
    print("No solution")
```

### Nested Functions and Complex Scope Management

MCP Solver now supports variables defined in nested function scopes. This is particularly useful for complex models:

```python
def build_complex_model():
    # Outer function that defines variables
    def define_variables():
        x = Int('x')
        y = Int('y')
        z = Int('z')
        return x, y, z
    
    # Inner function that adds constraints
    def add_constraints(solver, variables):
        x, y, z = variables
        solver.add(x > 0)
        solver.add(y > x)
        solver.add(z > y)
        solver.add(x + y + z == 10)
        return solver
    
    # Create solver
    s = Solver()
    
    # Get variables and add constraints using nested functions
    x, y, z = define_variables()
    s = add_constraints(s, (x, y, z))
    
    # Return solver and variables dictionary
    return s, {"x": x, "y": y, "z": z}

# Variables from nested functions are properly accessible
solver, variables = build_complex_model()

if solver.check() == sat:
    # This call is REQUIRED to extract the solution
    export_solution(solver=solver, variables=variables)
else:
    print("No solution found")
```

### Troubleshooting Variable Scope Issues

If you encounter scope-related errors:

1. Always return variables from inner functions to outer scopes
2. Create a dictionary mapping variable names to Z3 variables
3. Pass both solver and variables to `export_solution`
4. Prefer the function-based approach shown above

## Solution Export

```python
# CORRECT way to export a solution - both parameters required
export_solution(solver=solver, variables=variables)

# INCORRECT - missing parameters
# export_solution()  # This will fail
# export_solution(solver=solver)  # This will fail
# export_solution(variables=variables)  # This will fail

# INCORRECT - unsupported parameters
# export_solution(solver=solver, variables=variables, solution_data={})  # This will fail
```

## Debugging Checklist When Solutions Aren't Properly Extracted

If your solution isn't being properly captured:

1. ✅ Did you import the export_solution function?

   ```python
   from mcp_solver.z3 import export_solution
   ```

2. ✅ Did you call export_solution with both required parameters?

   ```python
   export_solution(solver=solver, variables=variables)
   ```

3. ✅ Did you check if the solver found a solution before calling export_solution?

   ```python
   if solver.check() == sat:
       export_solution(solver=solver, variables=variables)
   ```

4. ✅ Did you collect all variables in a dictionary and pass them correctly?

   ```python
   variables = {"x": x, "y": y}
   ```

5. ✅ Are you using a scope where the variables are still accessible?

   - Variables inside functions without a return may be inaccessible

