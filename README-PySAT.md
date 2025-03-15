# MCP Solver - PySAT Mode

This document provides information about using MCP Solver with the PySAT backend. For general information about the MCP Solver, see the [main README](README.md).

## Installation

To install MCP Solver with PySAT support:

```bash
uv pip install -e ".[pysat]"
```

## Configuration

There are two ways to use the PySAT mode:

### Option 1: Dedicated Command (Recommended)

Use the dedicated PySAT entry point in your client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "MCP Solver PySAT": { 
      "command": "uv", 
      "args": [
        "--directory", 
        "/path/to/mcp-solver", 
        "run", 
        "mcp-solver-pysat"
      ] 
    }
  }
}
```

### Option 2: Command Line Flags

Alternatively, you can use command line flags with the main entry point:

```json
{
  "server_args": "--pysat --lite"
}
```

The PySAT mode is enabled by the `--pysat` flag. Currently, PySAT mode only supports the lite mode.

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
4. For complex problems, use the template library:
   ```python
   from mcp_solver.pysat.templates import weighted_maxsat_template
   ``` 