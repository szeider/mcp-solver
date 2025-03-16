# MCP Solver - PySAT Mode

This document provides information about using MCP Solver with the PySAT backend. For general information about the MCP Solver, see the [main README](README.md).

## Installation

Install the *MCP Solver* with PySAT support:

```bash
uv pip install -e ".[pysat]"
```

The setup can be tested with:

```bash
uv run test-setup-pysat
```

## Configuration

To run the MCP Solver in PySAT mode, use the command `mcp-solver-pysat` instead of `mcp-solver` in your client configuration. Currently, PySAT mode only supports the lite mode, so also include the `--lite` flag.

## Core Features

PySAT mode provides access to SAT solving capabilities using Python syntax:

- **Standard SAT solving**: Create and solve boolean satisfiability problems
- **MaxSAT optimization**: Optimize problems with hard and soft constraints
- **Cardinality constraints**: Efficiently express constraints on the number of true variables
- **Secure execution**: Models run in a restricted environment with proper memory management

## Helper Functions

PySAT mode comes with built-in helper functions for common constraints:

```python
# Cardinality constraints
at_most_k([1, 2, 3, 4], 2)  # At most 2 variables can be true
at_least_k([1, 2, 3, 4], 1)  # At least 1 variable must be true
exactly_k([1, 2, 3, 4], 1)   # Exactly 1 variable must be true

# Logical relationships
implies(1, 2)               # If variable 1 is true, variable 2 must be true
mutually_exclusive([1, 2, 3]) # At most one variable can be true
if_then_else(condition, x, y) # If condition then x else y
```

## Improved MaxSAT Support

PySAT mode now includes enhanced support for MaxSAT problems with a simpler API:

```python
# Initialize a MaxSAT problem
initialize_maxsat()

# Add hard and soft constraints
add_hard_clause([1, 2])          # Hard constraint (must be satisfied)
add_soft_clause([-1], weight=2)  # Soft constraint with weight

# Soft cardinality constraints
add_at_most_k_soft([1, 2, 3], 1, weight=2)  # Try to have at most 1 true
add_at_least_k_soft([4, 5, 6], 2, weight=3) # Try to have at least 2 true

# Solve the MaxSAT problem
model, cost = solve_maxsat(timeout=5.0)  # Returns solution and penalty
```

These helper functions make it easier to model complex optimization problems where some constraints can be violated at a cost.

## Example Model

```python
from pysat.formula import CNF
from pysat.solvers import Glucose3
from mcp_solver.pysat import exactly_one

# Create a CNF formula for a simple scheduling problem
formula = CNF()

# Task 1-3 must be assigned to exactly one day (Mon, Tue, Wed)
formula.extend(exactly_one([1, 2, 3]))  # Task 1 assignment
formula.extend(exactly_one([4, 5, 6]))  # Task 2 assignment
formula.extend(exactly_one([7, 8, 9]))  # Task 3 assignment

# Create solver and solve
solver = Glucose3()
solver.append_formula(formula)
satisfiable = solver.solve()

# Extract and export the solution
if satisfiable:
    model = solver.get_model()
    export_solution({
        "satisfiable": True,
        "model": model
    })
else:
    export_solution({"satisfiable": False})

# Always free memory
solver.delete()
```

## Important Notes

1. Always call `solver.delete()` after using a solver to prevent memory leaks
2. Always call `export_solution()` at the end of your model to extract the solution
3. PySAT uses integer IDs for variables (positive for variables, negative for negated variables)

## Handling Solver Results

When using PySAT to solve boolean satisfaction problems:

1. Always store the solver's return value in a variable and use that variable consistently in conditional logic
2. Don't hardcode expected results in print statements
3. The solver returns `True` if satisfiable and `False` if unsatisfiable
4. Only process the model/solution when the solver returns `True`

Example of the correct pattern:
```python
# Correct pattern
is_sat = solver.solve()
if is_sat:  # Use the actual return value
    model = solver.get_model()
    # Process solution
    export_solution({
        "satisfiable": True,
        "model": model
    })
else:
    # Handle unsatisfiable case
    export_solution({"satisfiable": False})
```

This pattern ensures that your code correctly handles both satisfiable and unsatisfiable results, and that the JSON output matches your printed output. 