# Z3 SMT Solver (Lite Mode)

This MCP server provides access to the Z3 SMT solver through a Python interface. Z3 is a high-performance theorem prover developed at Microsoft Research, capable of solving complex constraint satisfaction and optimization problems.

## Available Tools

| Tool | Description |
|------|-------------|
| `clear_model` | Reset the Z3 model |
| `add_item` | Add Python code to the model |
| `replace_item` | Replace code in the model |
| `delete_item` | Delete code from the model |
| `solve_model` | Solve the Z3 model |
| `get_solution` | Get the complete solution |
| `get_variable_value` | Get the value of a specific variable |
| `get_solve_time` | Get the time taken to solve the model |

## Required Format

Write Z3 Python code with these guidelines:

- Always import Z3: `from z3 import *`
- Create variables as needed: `x = Int('x')`, `y = Real('y')`, `b = Bool('b')`
- Create a solver: `s = Solver()` or `opt = Optimize()`
- Add constraints: `s.add(x > 0)`, `s.add(x + y <= 10)`
- Always end with export_solution() to return results:
  `export_solution(solver=s, variables={'x': x, 'y': y})`

## Examples

### Basic Constraint Solving

```python
from z3 import *

# Define variables
x = Int('x')
y = Int('y')

# Create solver
s = Solver()

# Add constraints
s.add(x > 0)
s.add(y > 0)
s.add(x + y <= 10)
s.add(x * 2 >= y)

# Export solution
export_solution(solver=s, variables={'x': x, 'y': y})
```

### Optimization Problem

```python
from z3 import *

# Define variables
x = Int('x')
y = Int('y')

# Create optimizer
opt = Optimize()

# Add constraints
opt.add(x >= 0)
opt.add(y >= 0)
opt.add(x + y <= 10)

# Define objective
objective = x + 2*y
opt.maximize(objective)

# Export solution with objective
export_solution(solver=opt, variables={'x': x, 'y': y}, objective=objective)
```

### Boolean Satisfiability

```python
from z3 import *

# Define boolean variables
a = Bool('a')
b = Bool('b')
c = Bool('c')

# Create solver
s = Solver()

# Add constraints
s.add(Or(a, b, c))
s.add(Implies(a, Not(b)))
s.add(Implies(b, Not(c)))

# Export solution
export_solution(solver=s, variables={'a': a, 'b': b, 'c': c})
```

### Working with Arrays

```python
from z3 import *

# Define array
arr = Array('arr', IntSort(), IntSort())
i = Int('i')
j = Int('j')

# Create solver
s = Solver()

# Add constraints
s.add(arr[1] == 10)
s.add(arr[2] == 20)
s.add(arr[3] == 30)
s.add(i >= 1, i <= 3)
s.add(arr[i] > 15)

# Export solution
export_solution(solver=s, variables={'i': i, 'arr[1]': Select(arr, 1), 'arr[2]': Select(arr, 2), 'arr[3]': Select(arr, 3)})
```

## Tips

1. Always call `export_solution()` at the end of your code
2. Use descriptive variable names in the `variables` dictionary
3. For optimization problems, include the objective in the `export_solution()` call
4. Z3 supports integers, reals, booleans, bit-vectors, arrays, and more
5. Complex constraints can be built using Z3's operators and functions
6. The solver has a timeout of 10 seconds by default 