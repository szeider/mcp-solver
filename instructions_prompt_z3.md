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

## Z3 Variable Types and Common Errors

### Python Primitives vs. Z3 Variables

Z3 requires symbolic variables, not Python primitives. This is a common source of errors:

```python
# INCORRECT - Will cause "'int' object has no attribute 'as_ast'" error
x = 5  # Python primitive
y = Int('y')  # Z3 variable
solver.add(x + y == 10)  # Error when used in constraint
variables = {"x": x, "y": y}  # Error when exported

# CORRECT - Always use Z3 type constructors
x = Int('x')  # Z3 variable
y = Int('y')  # Z3 variable
solver.add(x == 5)  # Set value through constraint
solver.add(x + y == 10)  # Now works correctly
variables = {"x": x, "y": y}  # Now works correctly
```

### Exporting Boolean Results

When exporting boolean verification results, you must create Z3 boolean variables:

```python
# INCORRECT - Will cause "'bool' object has no attribute 'as_ast'" error
result = verify_formula()  # Returns a Python boolean
export_solution(solver=solver, variables={"verified": result})  # Error

# CORRECT - Connect Python boolean to Z3 boolean
result = verify_formula()  # Returns a Python boolean
verified = Bool('verified')
solver = Solver()
solver.add(verified == result)  # Connect Python result to Z3 variable
export_solution(solver=solver, variables={"verified": verified})  # Works
```

## Mathematical Proofs with Z3

Z3 can be used for mathematical proofs in addition to constraint solving. Here are patterns for common proof techniques:

### Proving Algebraic Identities

```python
# To prove expressions e1 and e2 are equal for all values in the domain:
from z3 import *

def prove_equality(e1, e2):
    # Create a solver that attempts to find a counterexample
    s = Solver()
    x = Int('x')
    # Add domain constraints if necessary
    s.add(x > 0)
    # Try to find a case where expressions are NOT equal
    s.add(simplify(e1) != simplify(e2))
    
    # If check() returns unsat, no counterexample exists, proving equality
    if s.check() == unsat:
        print("Expressions are equal for all valid inputs")
        return True
    else:
        print("Found counterexample:", s.model())
        return False
```

### Mathematical Induction - Complete Example

Here's a complete example for proving a formula by induction:

```python
from z3 import *
from mcp_solver.z3 import export_solution

def prove_sum_of_squares():
    """Prove that 1² + 2² + ... + n² = n(n+1)(2n+1)/6"""
    
    # 1. Verify base case (n=1)
    base_value = 1
    base_sum = base_value**2  # 1² = 1
    base_formula = base_value*(base_value+1)*(2*base_value+1)//6  # 1*2*3/6 = 1
    
    print(f"Base case (n=1): {base_sum} = {base_formula}")
    
    # 2. Symbolic verification of inductive step
    k = Int('k')
    
    # Inductive hypothesis: formula holds for n=k
    sum_k = k*(k+1)*(2*k+1)//6
    
    # We need to prove it holds for n=k+1
    next_term = (k+1)**2
    sum_k_plus_1 = sum_k + next_term
    formula_k_plus_1 = (k+1)*(k+2)*(2*(k+1)+1)//6
    
    # Create solver to check if adding (k+1)² to sum_k equals formula for k+1
    solver = Solver()
    solver.add(k > 0)
    
    # Try to find a counterexample where the formula doesn't hold
    solver.add(simplify(sum_k_plus_1) != simplify(formula_k_plus_1))
    
    if solver.check() == unsat:
        print("Inductive step verified: No counterexample exists")
        proof_result = True
    else:
        print(f"Found counterexample: k = {solver.model()[k]}")
        proof_result = False
    
    # Create a Z3 boolean variable for the result
    verified = Bool('verified')
    result_solver = Solver()
    result_solver.add(verified == proof_result)
    
    return result_solver, {"verified": verified}

# Solve and export the proof result
solver, variables = prove_sum_of_squares()
export_solution(solver=solver, variables=variables)
```

### Algebraic Manipulation with Z3

When working with algebraic transformations, use `simplify()` to verify steps:

```python
from z3 import *

# Example: Verify an algebraic transformation
k = Int('k')

# Starting expression
expr1 = k**2 * (k+1)**2 / 4 + (k+1)**3

# Target expression after transformation
expr2 = (k+1)**2 * (k+2)**2 / 4

# Expand both expressions for comparison
expanded1 = simplify(expr1)
expanded2 = simplify(expr2)

print("Expanded expr1:", expanded1)
print("Expanded expr2:", expanded2)

# Verify they are equal
if simplify(expanded1 - expanded2) == 0:
    print("Expressions are equivalent")
else:
    print("Expressions differ")
```

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

## Export Solution Formats

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

The `export_solution` function formats solutions differently depending on the scenario:

1. **For satisfiable constraint problems**: Returns variable assignments that satisfy all constraints
2. **For optimization problems**: Returns the optimal solution values
3. **For proofs by contradiction (unsat results)**: When unsat is expected and proves the theorem, you may need to state explicitly in your output that the proof has succeeded

## Working with Rational Arithmetic

When dealing with divisions and fractions, use caution with integer vs. real types:

```python
# For formula like n²(n+1)²/4 where division is involved:

# Option 1: Use integer division and ensure divisibility
n = Int('n')
formula_expr = (n * n * (n + 1) * (n + 1)) / 4  # Z3 will use integer division

# Option 2: Use Real type for exact rational arithmetic
n = Real('n')
formula_expr = (n * n * (n + 1) * (n + 1)) / 4  # Exact rational result
```

## Multi-part Proofs

For models that combine specific case verification with general proofs:

```python
# RECOMMENDED APPROACH - Separate solvers for different parts of proof
# First solver for specific case (e.g., n=5)
solver1 = Solver()
n5 = Int('n5')
solver1.add(n5 == 5)
# Add specific case constraints...

# Second solver for general proof
solver2 = Solver()
n = Int('n')
# Add general proof constraints...

# Check and export each part separately
if solver1.check() == sat:
    export_solution(solver=solver1, variables={"n5": n5})

# For general proofs that prove by contradiction (expecting UNSAT)
if solver2.check() == unsat:
    print("General proof succeeded")
else:
    # Export counterexample if found
    export_solution(solver=solver2, variables={"n": n})
```

## End-to-End Workflow Example: Sum of Cubes

Here's a complete example showing how to prove that the sum of cubes from 1 to n equals n²(n+1)²/4:

```python
from z3 import *
from mcp_solver.z3 import export_solution

def prove_sum_of_cubes():
    # Create solver
    solver = Solver()
    
    # 1. Verify concrete examples first
    print("CONCRETE EXAMPLES VERIFICATION:")
    for n_val in range(1, 6):
        # Calculate sum of cubes
        sum_cubes = sum(i**3 for i in range(1, n_val+1))
        # Calculate formula result
        formula = (n_val**2 * (n_val+1)**2) // 4
        
        print(f"n = {n_val}: sum = {sum_cubes}, formula = {formula}")
        if sum_cubes != formula:
            print(f"MISMATCH at n = {n_val}")
    
    # 2. Base case verification (n=1)
    print("\nBASE CASE (n=1):")
    base_sum = 1  # 1³ = 1
    base_formula = (1**2 * (1+1)**2) // 4  # 1²*2²/4 = 1
    print(f"Base case: {base_sum} = {base_formula}")
    
    # 3. Inductive step - symbolic verification
    print("\nINDUCTIVE STEP:")
    
    # Symbolic variables
    k = Int('k')
    
    # Inductive hypothesis: sum of cubes from 1 to k = k²(k+1)²/4
    sum_k = k*k*(k+1)*(k+1) / 4
    
    # Next term (k+1)³
    next_term = (k+1)*(k+1)*(k+1)
    
    # Sum for k+1: sum for k + (k+1)³ 
    sum_k_plus_1 = sum_k + next_term
    
    # Formula for k+1: (k+1)²(k+2)²/4
    formula_k_plus_1 = (k+1)*(k+1)*(k+2)*(k+2) / 4
    
    # Check if they are equal by trying to find a counterexample
    ind_solver = Solver()
    ind_solver.add(k > 0)
    ind_solver.add(simplify(sum_k_plus_1) != simplify(formula_k_plus_1))
    
    if ind_solver.check() == unsat:
        print("Inductive step verified: No counterexample found")
        
        # 4. Show algebraic expansion for clarity
        print("\nALGEBRAIC VERIFICATION:")
        print("Starting with sum_k + (k+1)³ where sum_k = k²(k+1)²/4")
        print("= k²(k+1)²/4 + (k+1)³")
        print("= k²(k+1)²/4 + 4(k+1)³/4")
        print("= [k²(k+1)² + 4(k+1)³]/4")
        
        print("\nExpanding k²(k+1)²:")
        print("= k²(k² + 2k + 1) = k⁴ + 2k³ + k²")
        
        print("\nExpanding 4(k+1)³:")
        print("= 4(k³ + 3k²k + 3k + 1) = 4k³ + 12k² + 12k + 4")
        
        print("\nCombining terms in the numerator:")
        print("= [k⁴ + 2k³ + k² + 4k³ + 12k² + 12k + 4]/4")
        print("= [k⁴ + 6k³ + 13k² + 12k + 4]/4")
        
        print("\nFactoring to match (k+1)²(k+2)²/4:")
        print("= [(k+1)²(k+2)²]/4")
        print("= (k+1)²(k+2)²/4")
        
        # Create Z3 boolean for the result
        proof_verified = Bool('proof_verified')
        result_solver = Solver()
        result_solver.add(proof_verified == True)
        
        return result_solver, {"proof_verified": proof_verified}
    else:
        print(f"Counterexample found: k = {ind_solver.model()[k]}")
        
        # Create Z3 boolean for the result
        proof_verified = Bool('proof_verified')
        result_solver = Solver()
        result_solver.add(proof_verified == False)
        
        return result_solver, {"proof_verified": proof_verified}

# Call the proof function
solver, variables = prove_sum_of_cubes()

# Export the result
export_solution(solver=solver, variables=variables)
```

## Model Tracking

When working with the model tracking system:

- Always verify your model state after adding or modifying items
- Use `get_model()` to check the current state
- Make sure items are properly indexed (starting at 1)
- If model appears empty after adding items, try refreshing or re-checking

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

6. ✅ Are all variables in your dictionary Z3 variables (not Python primitives)?

   ```python
   # Check that all are Z3 variables
   for var_name, var in variables.items():
       if not isinstance(var, z3.ExprRef):
           print(f"Warning: {var_name} is not a Z3 variable!")
   ```

## Common Error Patterns and Solutions

1. **"'bool' object has no attribute 'as_ast'"**:

   - **Problem**: Using Python boolean instead of Z3 Bool variable
   - **Solution**: Create a Z3 Bool variable and connect it to your Python result

   ```python
   verified = Bool('verified')
   solver.add(verified == your_python_boolean)
   ```

2. **"'int' object has no attribute 'as_ast'"**:

   - **Problem**: Using Python integer instead of Z3 Int variable
   - **Solution**: Create Z3 Int variables and use constraints to set values

   ```python
   x = Int('x')
   solver.add(x == 5)  # Instead of x = 5
   ```

3. **"Solver returned unknown"**:

   - **Problem**: Problem may be too complex or timeouts
   - **Solution**: Simplify constraints, increase timeout, or check for inconsistent constraints

4. **Empty results even with sat solution**:

   - **Problem**: Missing export_solution call or wrong variables dictionary
   - **Solution**: Ensure export_solution is called with the correct solver and variables
