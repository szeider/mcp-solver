# MCP Solver - Z3 Mode

This document provides information about using MCP Solver with the Z3 SMT Solver backend.

## Configuration

To run the MCP Solver in Z3 mode:

- Use `mcp-solver-z3` command (instead of `mcp-solver`)
- Include the `--lite` flag as Z3 currently only supports lite mode

## Core Features

Z3 mode provides SMT (Satisfiability Modulo Theories) solving capabilities:

- **Rich type system**: Booleans, integers, reals, bitvectors, arrays, and more
- **Constraint solving**: Solve complex constraint satisfaction problems
- **Optimization**: Optimize with respect to objective functions
- **Quantifiers**: Express constraints with universal and existential quantifiers

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
from mcp_solver.z3.templates import all_distinct

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
export_solution(solver=solver, variables=variables)
```

## Best Practices

1. Always call `export_solution()` at the end of your model to extract the solution

2. Wrap complex models in functions to avoid variable scope issues

3. For optimization problems, use specialized templates:

   ```python
   from mcp_solver.z3.templates import optimization_template
   ```

4. Unlike PySAT, Z3 handles memory management automatically (no need to call delete)

5. Use descriptive variable names to make solutions easier to interpret

6. For complex constraints, break them down into smaller, more manageable parts

## Function Scope and Variable Access

When working with Z3 models, be careful with variable scope and how you call `export_solution()`:

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
    export_solution(solver=solver, variables=variables)
else:
    print("No solution")
```

### Avoiding Common Scope Issues

If you need to call `export_solution()` inside a function, make sure to pass complete context:

```python
# When calling from inside a function:
def solve_problem():
    x = Int('x')
    y = Int('y')
    
    solver = Solver()
    solver.add(x > 0, y > 0, x + y == 10)
    
    if solver.check() == sat:
        # Must pass BOTH solver AND variables to export_solution
        export_solution(
            solver=solver,
            variables={"x": x, "y": y}
        )
        return True
    else:
        return False
```

## Solution Export

```python
# Basic solution export
export_solution(solver=solver, variables=variables)

# With additional structure
export_solution(
    solver=solver, 
    variables=variables,
    solution_data={
        "puzzle": puzzle_representation,
        "statistics": solving_statistics
    }
)
```

For more information on Z3, visit the [Z3 documentation](https://z3prover.github.io/api/html/namespacez3py.html).