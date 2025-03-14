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

## Supported Z3 Features

MCP Solver includes a subset of Z3's functionality. The following features are available:

### Core Z3 Types and Functions
- **Basic Types**: `Int`, `Real`, `Bool`, `Array`, `BitVec`
- **Solvers**: `Solver`, `Optimize`
- **Core Functions**: `And`, `Or`, `Not`, `Implies`, `If`, `Distinct`, `Sum`
- **Operators**: `==`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `/`, `%`
- **Quantifiers**: `ForAll`, `Exists`

### Supported Z3 Libraries
- **Pseudo-Boolean**: `PbEq`, `PbLe`, `PbGe` (for cardinality constraints)
- **Arrays**: `Store`, `Select` (for array manipulation)
- **BitVectors**: Core bit-vector operations

### Available Templates
MCP Solver provides the following templates to simplify common patterns:
- `array_is_sorted`, `all_distinct`, `array_contains`
- `exactly_k`, `at_most_k`, `at_least_k`
- `function_is_injective`, `function_is_surjective`

### Not Supported
The following Z3 features are NOT available in MCP Solver:
- External SMT-LIB file loading
- Custom tactics and solvers
- Low-level proof generation
- `FixedPoint` solver
- External process communication
- File I/O operations

If you need functionality that is not supported, you may need to implement it using the supported features or break down your problem differently.

## Working with Multiple Items

Break your Z3 models into modular components rather than writing everything in a single block. This improves:
- **Readability**: Smaller, focused code blocks are easier to understand
- **Maintainability**: Edit specific parts without rewriting everything
- **Flexibility**: Try different variations of constraints
- **Debugging**: Isolate and fix issues in specific components

### Example of a Fine-Grained Multi-Item Approach:

**Item 1**: Imports and global definitions
```python
from z3 import *

# Global constants
N = 8  # Problem size
MAX_VALUE = 100
```

**Item 2**: Variable declarations
```python
# Define variables
x = Int('x')
y = Int('y')
z = Int('z')

# Define arrays
values = Array('values', IntSort(), IntSort())
assignments = Array('assignments', IntSort(), IntSort())
```

**Item 3**: Solver initialization and domain constraints
```python
# Create solver
s = Solver()

# Basic domain constraints
s.add(x >= 0, x <= MAX_VALUE)
s.add(y >= 0, y <= MAX_VALUE)
s.add(z >= 0, z <= MAX_VALUE)

# Array domain constraints
for i in range(N):
    s.add(values[i] >= 0, values[i] <= MAX_VALUE)
    s.add(Or(assignments[i] == 0, assignments[i] == 1))
```

**Item 4**: Core problem constraints
```python
# Main problem constraints
s.add(x + y + z <= 15)
s.add(x >= y)
s.add(z == x + y - 5)

# Relationship between arrays
s.add(values[0] == x)
s.add(values[1] == y)
s.add(values[2] == z)
```

**Item 5**: Advanced constraints (e.g., quantifiers)
```python
# Define quantifier variables
i = Int('i')
j = Int('j')

# Add quantified constraints
s.add(ForAll([i], Implies(
    And(0 <= i, i < N),
    values[i] <= MAX_VALUE
)))

s.add(ForAll([i, j], Implies(
    And(0 <= i, i < j, j < N),
    Implies(assignments[i] == 1, assignments[j] == 0)
)))
```

**Item 6**: Objective and solution export
```python
# Set up objective
objective = x*2 + y*3 - z

# Create optimizer and transfer constraints
opt = Optimize()
opt.add(s.assertions())
opt.maximize(objective)

# Export solution with all relevant variables
export_solution(
    solver=opt, 
    variables={
        'x': x, 
        'y': y, 
        'z': z,
        'objective': objective,
        **{f'values[{i}]': values[i] for i in range(N)},
        **{f'assignments[{i}]': assignments[i] for i in range(N)}
    }
)
```

This fine-grained approach makes it easier to:
- Modify specific aspects of your model
- Test different constraint combinations
- Reuse components across problems
- Identify performance bottlenecks

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

## Working with Quantifiers

Z3 supports powerful universal (∀) and existential (∃) quantifiers for expressing properties over entire domains:

### Common Quantifier Patterns

**Array is sorted (ascending order)**:
```python
i, j = Ints('i j')
n = 10  # Array size
sorted_constraint = ForAll([i, j], Implies(
    And(0 <= i, i < j, j < n),
    arr[i] <= arr[j]
))
s.add(sorted_constraint)
```

**All elements are distinct**:
```python
i, j = Ints('i j')
n = 10  # Array size
distinct_constraint = ForAll([i, j], Implies(
    And(0 <= i, i < j, j < n),
    arr[i] != arr[j]
))
s.add(distinct_constraint)
```

**Array contains a specific value**:
```python
i = Int('i')
n = 10  # Array size
target = Int('target')
contains_constraint = Exists([i], And(
    0 <= i, i < n, 
    arr[i] == target
))
s.add(contains_constraint)
```

### Combining Quantifiers

Example of combining universal and existential quantifiers:

```python
# For any item i not assigned to person A, there exists an item j
# assigned to A that A values at least as much
i_var = Int('i_var')
j_var = Int('j_var')

exists_better_item = ForAll([i_var], Implies(
    And(0 <= i_var, i_var < n, assignment[i_var] == 1),  # If item i goes to B
    Exists([j_var], And(
        0 <= j_var, j_var < n,
        assignment[j_var] == 0,   # There exists an item j that goes to A
        value_A[j_var] >= value_A[i_var]  # A values j at least as much as i
    ))
))

s.add(exists_better_item)
```

## Using Quantifier Templates

To simplify working with quantifiers, MCP Solver provides a library of pre-built quantifier patterns that you can import and use directly:

```python
from z3 import *
from z3_templates import array_is_sorted, all_distinct, exactly_k

# Using templates instead of writing complex quantifiers
n = 10  # Size of array
arr = Array('arr', IntSort(), IntSort())
s = Solver()

# This single line replaces a complex ForAll quantifier
s.add(array_is_sorted(arr, n))

# Ensure all elements are different
s.add(all_distinct(arr, n))

# Cardinality constraints for boolean variables
bool_vars = [Bool(f'b_{i}') for i in range(5)]
s.add(exactly_k(bool_vars, 2))  # Exactly 2 must be true

# Export solution
export_solution(solver=s, variables={
    **{f'arr[{i}]': arr[i] for i in range(n)},
    **{f'b_{i}': bool_vars[i] for i in range(5)}
})
```

### Available Template Functions

| Function | Description |
|----------|-------------|
| `array_is_sorted(arr, size, strict=False)` | Array elements are in ascending order |
| `all_distinct(arr, size)` | All elements in the array are different |
| `array_contains(arr, size, value)` | Array contains a specific value |
| `exactly_k(bool_vars, k)` | Exactly k boolean variables are true |
| `at_most_k(bool_vars, k)` | At most k boolean variables are true |
| `at_least_k(bool_vars, k)` | At least k boolean variables are true |
| `function_is_injective(func, domain_size)` | Function is one-to-one (injective) |
| `function_is_surjective(func, domain_size, range_size)` | Function maps onto the entire range (surjective) |

Using these templates can significantly simplify your code and help avoid common errors in quantifier patterns. The template functions are optimized for performance and provide clear semantics.

## Tips

1. Always call `export_solution()` at the end of your code
2. Use descriptive variable names in the `variables` dictionary
3. For optimization problems, include the objective in the `export_solution()` call
4. Z3 supports integers, reals, booleans, bit-vectors, arrays, and more
5. Complex constraints can be built using Z3's operators and functions
6. The solver has a timeout of 10 seconds by default
7. Break large models into multiple items for better organization
8. Name quantifier variables descriptively to avoid confusion (e.g., `row_idx` instead of `i`)
9. Quantified formulas can be slow to solve - use them only when necessary
10. Include bounds for quantifier variables to restrict their domains

## Quantifier Best Practices

1. **Bounded Domains**: Always include explicit bounds for quantifier variables
   ```python
   # Good: Bounded domain
   ForAll([i], Implies(And(0 <= i, i < n), arr[i] > 0))
   
   # Bad: Unbounded domain can cause performance problems
   ForAll([i], arr[i] > 0)
   ```

2. **Avoid Nested Quantifiers** where possible as they can significantly slow down solving

3. **Start Simple**: Begin with non-quantified constraints, then add quantifiers incrementally

4. **Debug Quantifiers**: If your model is unsatisfiable or slow, try removing quantifiers to isolate issues