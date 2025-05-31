# MaxSAT Mode Documentation

## Overview

The MaxSAT mode is a specialized mode in MCP Solver designed for optimization problems. It uses the Weighted Conjunctive Normal Form (WCNF) formulation and the RC2 solver from the PySAT library to find optimal solutions that maximize the satisfaction of weighted soft constraints while respecting hard constraints.

## Key Features

- **Optimization Focus**: Specifically designed for optimization problems, not just satisfiability
- **Weighted Constraints**: Support for both hard constraints (must be satisfied) and soft constraints (with weights)
- **Cost Minimization**: Finds assignments that minimize the total weight of violated soft constraints
- **Objective Maximization**: Supports expressing problems in terms of maximizing an objective function
- **Solution Details**: Provides detailed optimization results including cost, objective values, and assignment details

## Usage

### Starting the Server

```bash
uv run mcp-solver-maxsat
```

### Test Client

```bash
uv run test-client --mode maxsat --query path/to/problem.md
```

### Verify Setup

```bash
uv run test-setup-maxsat
```

## Core Components

1. **WCNF Formula**: Represents constraints with weights
   - Hard constraints: Added with `wcnf.append([literals])`
   - Soft constraints: Added with `wcnf.append([literals], weight=W)`

2. **RC2 Solver**: Specialized MaxSAT solver for optimization
   ```python
   with RC2(wcnf) as solver:
       model = solver.compute()
       # Access optimization results
       cost = solver.cost
   ```

3. **Solution Export**: Specialized function for MaxSAT results
   ```python
   export_maxsat_solution({
       "satisfiable": True,
       "status": "optimal",
       "assignment": assignment,
       "cost": cost,
       "objective": objective_value
   }, var_mapping)
   ```

## Example Problem

Here's a simple MaxSAT problem solving:

```python
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Create a WCNF formula
wcnf = WCNF()

# Define variables: 1=A, 2=B, 3=C
A, B, C = 1, 2, 3

# Hard constraint: A and B cannot both be true
wcnf.append([-A, -B])

# Soft constraints with weights
wcnf.append([A], weight=2)  # Prefer A to be true (weight 2)
wcnf.append([B], weight=3)  # Prefer B to be true (weight 3)
wcnf.append([C], weight=1)  # Prefer C to be true (weight 1)

# Map variables to names
var_mapping = {"A": A, "B": B, "C": C}

# Solve with RC2
with RC2(wcnf) as solver:
    model = solver.compute()
    
    if model is not None:
        # Create solution
        assignment = {name: (var_id in model) for name, var_id in var_mapping.items()}
        
        # Export the solution
        export_maxsat_solution({
            "satisfiable": True,
            "status": "optimal",
            "assignment": assignment,
            "cost": solver.cost
        }, var_mapping)
```

## Use Cases

MaxSAT mode is ideal for problems involving:

1. **Feature Selection**: Choosing optimal subset of features under constraints
2. **Resource Allocation**: Assigning limited resources to maximize utility
3. **Scheduling**: Finding optimal schedules that minimize conflicts
4. **Maximum Cut**: Finding optimal cuts in graphs to maximize edge weight
5. **Weighted Maximum Satisfiability**: General optimization problems

## Difference from PySAT Mode

While the PySAT mode focuses on satisfiability (finding any solution that satisfies all constraints), the MaxSAT mode focuses on optimization (finding the best solution according to an objective function). Key differences:

- **Formula Type**: WCNF vs CNF
- **Solver Type**: RC2 vs Glucose3/Cadical
- **Result Focus**: Optimization metrics vs Satisfiability only
- **Solution Format**: Includes cost/objective values and optimization status

## Implementation Notes

The MaxSAT mode was implemented as a separate mode from PySAT to provide specialized functionality for optimization problems, while keeping the PySAT mode focused on satisfiability problems. The implementation shares core utilities with the PySAT mode but provides optimized and specialized handling for weighted constraints and optimization problems.