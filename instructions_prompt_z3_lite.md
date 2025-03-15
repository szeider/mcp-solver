# Z3 SMT Solver (Lite Mode)

This MCP server provides access to the Z3 SMT solver through a Python interface. Z3 is a high-performance theorem prover developed at Microsoft Research, capable of solving complex constraint satisfaction and optimization problems.

## Available Tools

| Tool | Description |
|------|-------------|
| `clear_model` | Reset the Z3 model |
| `add_item` | Add Python code to the model |
| `replace_item` | Replace code in the model |
| `delete_item` | Delete code from the model |
| `solve_model` | Solve the Z3 model (requires timeout parameter between 1-10 seconds) |
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
MCP Solver provides the following templates to simplify common constraint patterns:

```python
# Import templates
from mcp_solver.z3.templates import (
    # Quantifier pattern templates
    array_is_sorted,
    all_distinct,
    array_contains,
    exactly_k,
    at_most_k,
    at_least_k,
    function_is_injective,
    function_is_surjective,
    
    # Function templates
    constraint_satisfaction_template,
    optimization_template,
    array_template,
    quantifier_template,
    
    # Subset templates
    smallest_subset_with_property
)
```

These templates are covered in more detail in later sections.

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

Z3 models can be developed using three approaches in MCP Solver, each with different trade-offs:

1. **Multi-Item Approach**: Breaking models into separate items (shown below)
2. **Function-Based Approach**: Encapsulating all logic in a custom function
3. **Function Template Approach**: Using pre-built templates for common patterns (recommended)

The multi-item approach breaks your Z3 models into modular components. This improves:
- **Readability**: Smaller, focused code blocks are easier to understand
- **Maintainability**: Edit specific parts without rewriting everything
- **Flexibility**: Try different variations of constraints
- **Debugging**: Isolate and fix issues in specific components

However, this approach can lead to variable scope issues and import problems. For most users, the **Function Template Approach** (described in the "Function Templates" section) is recommended as it provides the benefits of modularity while avoiding common issues.

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

## Best Practices for Z3 Models

Following these practices will make your Z3 models more robust and less prone to errors:

### Model Structure and Organization

#### Function-Based Approach
For complex models, wrap your logic in a function to avoid scope issues. There are two ways to do this:

1. **Basic Function-Based Approach** - Create your own function with custom internal structure:

```python
from z3 import *

def build_model():
    # Import any additional libraries inside the function
    from itertools import combinations
    
    # Define variables, constraints, objective here in any order
    x = Int('x')
    solver = Solver()  # or Optimize()
    solver.add(x > 0)
    
    # Return solver and variables dictionary
    return solver, {'x': x}

# Call your function and export solution
solver, variables = build_model()
export_solution(solver=solver, variables=variables)
```

2. **Template Function Approach (Recommended)** - Use the pre-built template functions with structured sections:

```python
from z3 import *
from mcp_solver.z3.templates import constraint_satisfaction_template

def build_model():
    # The template provides a standardized structure with clear sections
    # [SECTION: VARIABLE DEFINITION]
    x = Int('x')
    
    # [SECTION: SOLVER CREATION] 
    # The solver is already created by the template
    
    # [SECTION: CONSTRAINTS]
    # Add your constraints here following the template structure
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {'x': x}
    
    return solver, variables

# Execute the model as usual
solver, variables = build_model()
export_solution(solver=solver, variables=variables)
```

The template function approach is strongly recommended as it provides consistent structure and helps avoid common errors.

### Choosing the Right Template

| Template | When to Use | Key Features |
|----------|-------------|--------------|
| `constraint_satisfaction_template()` | For problems focused on finding a valid solution that satisfies all constraints | Uses `Solver()`, emphasizes constraint definition |
| `optimization_template()` | For problems that maximize or minimize an objective function | Uses `Optimize()`, includes objective section |
| `array_template()` | For problems involving sequences, lists, or grids | Includes array creation helper functions |
| `quantifier_template()` | For problems with "for all" or "there exists" constraints | Includes helper functions for common quantifier patterns |

### Using Function Templates

```python
from z3 import *
from mcp_solver.z3.templates import optimization_template  # Choose the appropriate template

def build_model():
    # [SECTION: VARIABLE DEFINITION]
    # Define all variables, constants, and parameters here
    x = Int('x')
    y = Int('y')
    
    # [SECTION: SOLVER CREATION]
    # This is created automatically by the template, but you can customize it
    optimizer = Optimize()  # Will be Solver() for constraint_satisfaction_template
    
    # [SECTION: CONSTRAINTS]
    # Add all your constraints here
    optimizer.add(x >= 0)
    optimizer.add(y >= 0)
    optimizer.add(x + y <= 10)
    
    # [SECTION: OBJECTIVE]  # Only in optimization_template
    # Define and set your objective function
    objective = x + 2*y
    optimizer.maximize(objective)
    
    # [SECTION: VARIABLES TO EXPORT]
    # List all variables you want to see in the solution
    variables = {'x': x, 'y': y, 'objective': objective}
    
    return optimizer, variables

# Execute the model
solver, variables = build_model()
export_solution(solver=solver, variables=variables)
```

### Template Structure Details

Each template provides a different structure optimized for its purpose:

**Constraint Satisfaction Template:**
```python
def constraint_satisfaction_template():
    # [SECTION: VARIABLE DEFINITION]
    # Define variables here
    
    # [SECTION: SOLVER CREATION]
    solver = Solver()
    
    # [SECTION: CONSTRAINTS]
    # Add constraints here
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {}
    
    return solver, variables
```

**Optimization Template:**
```python
def optimization_template():
    # [SECTION: VARIABLE DEFINITION]
    # Define variables here
    
    # [SECTION: SOLVER CREATION]
    optimizer = Optimize()
    
    # [SECTION: CONSTRAINTS]
    # Add constraints here
    
    # [SECTION: OBJECTIVE]
    # Define objective and call maximize() or minimize()
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {}
    
    return optimizer, variables
```

**Benefits of Using Function Templates:**
- **Scope Management**: Avoids variable scope issues between items
- **Import Organization**: Ensures imports are properly scoped
- **Consistent Structure**: Provides a standardized model structure
- **Solution Export**: Encourages proper export of solution variables
- **Readability**: Includes clear section comments for better organization
- **Reusability**: Makes it easier to adapt and modify models

Function templates are the recommended approach for almost all Z3 models in MCP Solver.

#### Model Organization Recommendations
- **For Simple Models**: Use a single comprehensive item with clear section comments
- **For Complex Models**: Split into multiple items, but follow this structure:
  1. Define ALL constants and imports in the first item
  2. Define ALL variables in the second item
  3. Group related constraints in subsequent items
  4. Include the objective function and export in the final item

### Variable Management

- Use descriptive variable names that reflect the problem domain (e.g., `start_time` instead of `x`)
- Prefer explicit variable names rather than list indexing where possible
- When using arrays of variables, create meaningful indexing schemes
- Document the meaning and units of variables in comments

### Incremental Development

- Build and test your model incrementally
- Start with basic constraints before adding complex ones
- Verify satisfiability at checkpoints as you add constraints
- Use small test cases with known solutions first

### Debugging Tips

- Include comments explaining the purpose of complex constraints
- Check satisfiability of subcombinations of constraints to isolate issues
- Add temporary debug constraints (e.g., `s.add(x < 10)`) to narrow down solution space
- Use Z3's `unsat_core()` when available to identify conflicting constraints

### Common Pitfalls to Avoid

- **Variable Scoping**: Define variables where they'll be visible across all items
- **Mixing Solvers**: Don't mix constraints between different solver instances
- **Forgetting Conversions**: Be explicit with type conversions (e.g., `ToReal(x)`)
- **Quantifier Abuse**: Avoid nested quantifiers when possible—they can lead to performance issues
- **Empty Arrays**: Initialize arrays with at least one value to avoid unexpected behavior

## Using Quantifier Templates

MCP Solver provides templates to simplify common quantified constraints. These templates offer a more readable and efficient way to express complex constraints without writing ForAll/Exists expressions manually.

### Using Quantifier Pattern Templates

```python
from z3 import *
from mcp_solver.z3.templates import array_is_sorted, all_distinct, array_contains

# Create array variables
arr = Array('arr', IntSort(), IntSort())
n = 5

# Basic domain constraints
s = Solver()
for i in range(n):
    s.add(arr[i] >= 0, arr[i] < n)

# Use templates instead of complex quantifiers
s.add(all_distinct(arr, n))
s.add(array_is_sorted(arr, n))

# These are equivalent to the manual quantifier expressions:
# i, j = Ints('i j')
# s.add(ForAll([i, j], Implies(And(0 <= i, i < j, j < n), arr[i] != arr[j])))  # all_distinct
# s.add(ForAll([i, j], Implies(And(0 <= i, i < j, j < n), arr[i] <= arr[j])))  # array_is_sorted

export_solution(solver=s, variables={f'arr[{i}]': arr[i] for i in range(n)})
```

### Quantifier Template Best Practices

- Templates often provide better performance than equivalent manually written quantifiers
- Use templates for common patterns to improve readability
- Templates can be composed to express more complex constraints
- When debugging, replace templates with explicit constraints to isolate issues

## Common Challenges and Solutions

When implementing complex optimization models in Z3, you may encounter several common challenges. Here are practical solutions based on real-world experience:

### Variable Scope Issues

**Challenge**: Variables defined in one item may not be accessible in another item, causing errors like `NameError: name 'X' is not defined`.

**Solution**:
```python
# AVOID this pattern (splitting variables across items):
# Item 1
REGIONS = ["A", "B", "C"]

# Item 2
for region in REGIONS:  # Error: REGIONS not defined!
    # ...

# INSTEAD use this pattern (function-based approach):
def build_model():
    REGIONS = ["A", "B", "C"]
    # All model code using REGIONS here
    return solver, variables

solver, variables = build_model()
export_solution(solver=solver, variables=variables)
```

### Import Handling

**Challenge**: Imports in one item may not be available in subsequent items, causing errors like `NameError: name 'combinations' is not defined`.

**Solution**:

#### IMPORTANT: Never Mix Import Strategies!

Choose ONE of these approaches and stick with it for your entire model:

1. **Function-Based Approach (Recommended):** 
   ```python
   from z3 import *  # Only Z3 at the top level
   
   def build_model():
       # Place ALL other imports inside the function
       from itertools import combinations
       import numpy as np
       import random
       
       # Now use these imports safely within the function scope
       for combo in combinations(range(5), 2):
           # ...
       
       return solver, variables

   # Build and solve the model
   solver, variables = build_model()
   export_solution(solver=solver, variables=variables)
   ```

2. **Multi-Item Approach (Not Recommended):**
   ```python
   # Item 1 - Place ALL imports here
   from z3 import *
   from itertools import combinations
   import numpy as np
   import random
   
   # Item 2 - NO IMPORTS here!
   x = Int('x')
   
   # Item 3 - NO IMPORTS here!
   s = Solver()
   s.add(x > 0)
   ```

#### Common Import Errors and Solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `NameError: name 'combinations' is not defined` | Import in wrong place | Move import inside function or to first item |
| `ModuleNotFoundError: No module named 'X'` | Missing dependency | Check available modules in MCP Solver |
| `ImportError: cannot import name 'X'` | Incorrect import name | Check correct template names and import paths |

#### Best Practices for Imports:

1. **Always Import Z3 First**: `from z3 import *` should always be your first import
2. **Avoid Duplicate Imports**: Don't import the same module in multiple places
3. **Use Function Templates**: Templates handle import scoping automatically
4. **Test Imports First**: When developing complex models, test imports work before adding logic

The most common error in complex Z3 models is import problems. The function template approach eliminates almost all import issues and is strongly recommended.

### Handling Rational Numbers

**Challenge**: Z3 results may include question marks for rational numbers (like `0.33333?` or `0.1666666666?`), causing parsing errors or conversion issues.

**Solution**:
```python
# Option 2: If precision matters, use integer variables with scaling (RECOMMENDED)
# Instead of:
allocation = Real('allocation')  # Might result in fractions like 0.1666666666?

# Consider:
allocation_pct = Int('allocation_pct')  # Represents percentage (0-100)
s.add(allocation_pct >= 0, allocation_pct <= 100)
# Then work with allocation_pct/100.0 in your constraints

# This approach works best for:
# - Percentages (scale by 100)
# - Money values (scale by 100 for dollars/cents)
# - Probabilities (scale by 1000 for three decimal places)

# Example: Supply chain allocation with integer percentages
# Define allocation variables as integer percentages (0-100)
allocations = {}
for region in regions:
    allocations[region] = Int(f"allocation_{region}")

# Allocations must sum to 100%
optimizer.add(Sum([allocations[region] for region in regions]) == 100)

# Each region gets between 5% and 30%
for region in regions:
    optimizer.add(allocations[region] >= 5)   # Min 5%
    optimizer.add(allocations[region] <= 30)  # Max 30%

# Option 3: Accept that some results will be displayed as fractions
# This is often fine for understanding the model's solution
```

**Remember**: Z3's rational numbers (like `0.1666666666?`) represent exact fractions (e.g., 1/6) rather than floating-point approximations. The question mark indicates that it's an exact value that can't be precisely represented in decimal.

### Solution Verification

**Challenge**: Z3 may return a solution that looks correct but doesn't fully satisfy all constraints.

**Solution**:
```python
# Always verify critical constraints after solving
def verify_solution(model, variables):
    # Extract values from model
    values = {name: model.eval(var) for name, var in variables.items()}
    
    # Check key constraints
    capacity_satisfied = check_capacity_constraint(values)
    allocation_satisfied = check_allocation_constraints(values)
    
    if not (capacity_satisfied and allocation_satisfied):
        print("Warning: Solution may not satisfy all constraints!")
        print(f"Capacity constraint satisfied: {capacity_satisfied}")
        print(f"Allocation constraints satisfied: {allocation_satisfied}")
    
    return values
```

### Comprehensive Solution Verification for Complex Problems

For complex optimization problems (like supply chain optimization), it's essential to verify that the solution satisfies all constraints. This is particularly important for problems with many constraints or where the solver might return approximate solutions.

```python
def build_supply_chain_model():
    # ... model definition ...
    
    # Return the optimizer and variables
    return optimizer, export_variables

# Build and solve the model
optimizer, variables = build_supply_chain_model()

# Get the solution
if optimizer.check() == sat:
    model = optimizer.model()
    
    # Verify the solution
    def verify_constraints(model):
        # Extract actual values from the model
        allocations = {region: model.eval(variables[region]).as_long() 
                      for region in regions}
        
        print("Verifying solution constraints:")
        
        # 1. Check total allocation = 100%
        total = sum(allocations.values())
        print(f"Total allocation: {total}% (should be 100%)")
        
        # 2. Check min/max constraints
        min_ok = all(alloc >= 5 for alloc in allocations.values())
        max_ok = all(alloc <= 30 for alloc in allocations.values())
        print(f"Min/max constraints: {min_ok and max_ok}")
        
        # 3. Check resilience constraints
        worst_case_capacity = float('inf')
        for scenario in all_disruption_scenarios:
            # Calculate capacity with this disruption
            capacity = calculate_scenario_capacity(scenario, allocations)
            worst_case_capacity = min(worst_case_capacity, capacity)
        
        print(f"Worst-case capacity: {worst_case_capacity}% (minimum: 70%)")
        resilience_ok = worst_case_capacity >= 70
        
        # Overall verification
        all_constraints_satisfied = (total == 100 and min_ok and max_ok and resilience_ok)
        print(f"All constraints satisfied: {all_constraints_satisfied}")
        
        return all_constraints_satisfied
    
    # Run verification
    verification_result = verify_constraints(model)
    
    # Export the solution
    export_solution(solver=optimizer, variables=variables)
else:
    print("No solution found")
```

Key benefits of comprehensive verification:
- Confirms that all constraints are actually satisfied
- Helps identify which specific constraints might be violated
- Provides confidence in the solution before implementing it
- Acts as documentation of what the model is supposed to satisfy

Verifying solutions is especially important when:
- Working with complex multi-constraint problems
- Using Real variables that may have precision issues
- Implementing critical business decisions based on the model
- Dealing with potentially conflicting constraints

### Complex Constraints Generation

**Challenge**: Generating all combinations for complex constraints can lead to scope issues.

**Solution**:
```python
# For complex combination constraints, define them explicitly
constraints = []

# Pre-generate all constraint data
scenarios = [
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    # etc.
]

# Then add constraints in a loop
for scenario in scenarios:
    capacity = calculate_capacity(scenario)
    constraints.append(capacity >= MIN_CAPACITY)

# Add all constraints at once
for c in constraints:
    optimizer.add(c)
```

This pattern is particularly useful for complex problems like supply chain optimization with multiple disruption scenarios:

```python
# Pre-generate all disruption scenarios
all_disruption_scenarios = []
for k in range(1, MAX_DISRUPTED_REGIONS + 1):
    all_disruption_scenarios.extend(list(combinations(regions, k)))

# Then use these scenarios for constraints
for disruption_scenario in all_disruption_scenarios:
    capacity = calculate_capacity(disruption_scenario)
    optimizer.add(capacity >= MIN_TOTAL_CAPACITY * 100)
```

## Optimization Problems Best Practices

When working with optimization problems in Z3, follow these specific guidelines to ensure efficient and correct models:

### Structuring Optimization Models

1. **Start with feasibility**: First build a model that satisfies all constraints without any objective:
   ```python
   # Step 1: Build and check feasibility
   s = Solver()
   # Add all constraints
   if s.check() == sat:
       # Model is feasible, proceed to optimization
       opt = Optimize()
       # Transfer constraints and add objective
   else:
       print("Model is infeasible - fix constraints first")
   ```

2. **Add the objective function last**: Once you know your model is feasible, add the objective:
   ```python
   # After ensuring feasibility
   opt = Optimize()
   opt.add(s.assertions())  # Transfer all constraints
   opt.maximize(objective)  # Add objective last
   ```

3. **Use the optimization template** for complex optimization problems:
   ```python
   from mcp_solver.z3.templates import optimization_template
   
   def build_model():
       # [SECTION: VARIABLE DEFINITION]
       # Define variables
       
       # [SECTION: SOLVER CREATION]
       optimizer = Optimize()
       
       # [SECTION: CONSTRAINTS]
       # Add all constraints
       
       # [SECTION: OBJECTIVE]
       # Define your objective and set maximize/minimize
       objective = Sum([...])
       optimizer.maximize(objective)
       
       # [SECTION: VARIABLES TO EXPORT]
       variables = {..., 'objective': objective}
       
       return optimizer, variables
   ```

### Handling Multiple Objectives

For problems with multiple objectives, set priorities:

```python
# Primary objective
opt.maximize(primary_objective)

# Secondary objective (considered only after primary is optimized)
opt.maximize(secondary_objective)
```

### Pre-generate Complex Constraints

For optimization problems with many similar constraints (like the disruption scenarios in supply chain problems):

```python
# Pre-generate all constraint data
all_scenarios = []
for k in range(1, MAX_DISRUPTED_REGIONS + 1):
    all_scenarios.extend(list(combinations(regions, k)))

# Build constraints list
constraints = []
for scenario in scenarios:
    capacity = calculate_capacity(scenario)
    constraints.append(capacity >= MIN_CAPACITY)

# Add all constraints at once
for c in constraints:
    optimizer.add(c)
```

### Performance Tips for Optimization

1. **Set timeout for complex problems**: 
   ```python
   optimizer.set("timeout", 10000)  # 10 second timeout
   ```

2. **Try different objective structures**: Sometimes reformulating the objective function can make solving easier:
   ```python
   # Instead of:
   objective = Sum([complex_expression(i) for i in range(n)])
   
   # Try:
   objective_parts = []
   for i in range(n):
       part = complex_expression(i)
       objective_parts.append(part)
   objective = Sum(objective_parts)
   ```

3. **Use incremental solving** for complex problems:
   ```python
   # Start with core constraints only
   opt.add(core_constraints)
   
   # Get an initial solution
   if opt.check() == sat:
       model = opt.model()
       # Extract initial values for variables
       
       # Now add harder constraints
       opt.add(harder_constraints)
       # Solve again, potentially using initial values as hints
   ```

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