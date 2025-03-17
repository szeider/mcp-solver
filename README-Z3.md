# MCP Solver - Z3 Mode

This document provides information about using MCP Solver with the Z3 SMT Solver backend. For general information about the MCP Solver, see the [main README](https://claude.ai/chat/README.md).

## Configuration

To run the MCP Solver in Z3 mode, use the command `mcp-solver-z3` instead of `mcp-solver` in your client configuration. Currently, PySAT mode only supports the lite mode, so also include the `--lite` flag.

## Core Features

Z3 mode provides access to SMT (Satisfiability Modulo Theories) solving capabilities using Python syntax:

- **Rich type system**: Support for booleans, integers, reals, bitvectors, arrays, and more
- **Constraint solving**: Solve complex constraint satisfaction problems
- **Optimization**: Optimize with respect to objective functions
- **Quantifiers**: Express constraints with universal and existential quantifiers
- **Secure execution**: Models run in a restricted environment with proper memory management

## Enhanced Error Handling

The system provides robust error detection and reporting to help diagnose problems in your Z3 models:

1. **Descriptive Error Messages**: Errors are intercepted and translated into user-friendly messages with context about what went wrong
2. **Input Validation**: The system automatically checks variable names, types, and values before execution
3. **Code Safety Analysis**: Validates Python code for potentially unsafe patterns before execution
4. **Context-Aware Diagnostics**: Error messages include details about the current state of your model, making it easier to identify issues
5. **Solution Structure Validation**: The system validates solution structures before exporting, preventing common serialization errors

These enhancements help you identify and fix issues more quickly, with clear guidance on what might be causing problems in your models.

## Template Library

Z3 mode includes a comprehensive template library for common modeling patterns:

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

## Example Model

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

3. For complex problems, use the function-based templates:

   ```python
   from mcp_solver.z3.templates import optimization_template
   ```

4. Use descriptive variable names to make solutions easier to interpret

5. Z3 handles memory management automatically (unlike PySAT)

6. When debugging, pay attention to the error messages which provide context on what went wrong

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