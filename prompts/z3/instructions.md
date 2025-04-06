# MCP Solver - Z3 Mode

This document provides information about using MCP Solver with the Z3 SMT Solver backend.

## ⚠️ IMPORTANT: Solution Export Requirement ⚠️

All Z3 models MUST call `export_solution` to properly extract solutions. Without this call, results will not be captured, even if your solver finds a solution!

```python
# Always include this import
from mcp_solver.z3 import export_solution

# After solving, always call export_solution with the appropriate parameters

## For standard constraint solving:
if solver.check() == sat:
    # For satisfiable solutions, provide the solver and variables
    export_solution(solver=solver, variables=variables)
else:
    # For unsatisfiable problems, explicitly set satisfiable=False
    print("No solution exists that satisfies all constraints.")
    export_solution(satisfiable=False, variables=variables)

## For property verification (theorem proving):
# When verifying a property, we look for a counterexample (negate the property)
# solver.add(Not(property_to_verify))

if solver.check() == sat:
    # Counterexample found (property doesn't hold)
    print("Property verification failed. Counterexample found.")
    export_solution(
        solver=solver, 
        variables=variables,
        is_property_verification=True
    )
else:
    # No counterexample found (property holds for all cases)
    print("Property verified successfully.")
    export_solution(
        satisfiable=False,
        variables=variables,
        is_property_verification=True
    )
```

## Export Solution Parameters

The `export_solution` function supports the following parameters:

- `solver`: The Z3 solver object containing constraints and status
- `variables`: Dictionary mapping variable names to Z3 variables
- `satisfiable`: Optional boolean to explicitly set satisfiability status (useful for unsatisfiable problems)
- `objective`: Optional objective expression for optimization problems
- `is_property_verification`: Boolean flag indicating if this is a property verification problem

## Core Features

Z3 mode provides SMT (Satisfiability Modulo Theories) solving capabilities:

- **Rich type system**: Booleans, integers, reals, bitvectors, arrays, and more
- **Constraint solving**: Solve complex constraint satisfaction problems
- **Optimization**: Optimize with respect to objective functions
- **Quantifiers**: Express constraints with universal and existential quantifiers

## Incremental Model Building

The MCP Solver is designed for incremental model building with separate "items". This approach is **strongly recommended** as it allows you to:

1. **Receive early feedback** on syntax errors or logic problems
2. **Modify specific parts** of your model later without resubmitting everything
3. **Debug more effectively** by identifying exactly which part of your model has issues
4. **Build complex models** gradually and verify each part before moving to the next

To use this approach:

1. **Split your code into logical sections** (e.g., imports, variable declarations, constraints, solution export)
2. **Submit each section separately** using the `add_item` tool with incrementing index values

For example, instead of submitting all your code at once, break it into logical components:

```python
# The following example shows how to split your model into separate items
# The "# Item X:" comments below are for illustration only and show where 
# you would split the code for separate add_item calls

# Item 1: Setup and imports
from z3 import *
from mcp_solver.z3 import export_solution

# Item 2: Variable declarations
x = Int('x')
y = Int('y')
variables = {'x': x, 'y': y}

# Item 3: Constraints
solver = Solver()
solver.add(x > 0)
solver.add(y > 0)
solver.add(x + y == 10)

# Item 4: Solve and export solution
if solver.check() == sat:
    export_solution(solver=solver, variables=variables)
else:
    export_solution(satisfiable=False, variables=variables)
```

Submit each section separately using `add_item` with the corresponding index (1, 2, 3, 4). The solver will automatically combine these items into a complete model. 

**Note:** The "# Item X:" comments in the examples are for illustration only to show where you would split the code for separate submissions. You don't need to include these comments in your actual code - they're just to help you understand the incremental building approach.

## Best Practices for Problem Modeling

1. **Translate all constraints correctly**:

   - Consider edge cases and implicit constraints
   - Verify that each constraint's logical formulation matches the intended meaning
   - Be explicit about ranges, domains, and special case handling

2. **Structure your model clearly**:

   - Use descriptive variable names for readability
   - Group related constraints together
   - Build long code incrementally by splitting it into separate items (as described above)
   - Comment complex constraint logic

3. **Use the correct export_solution call**:

   ```python
   # For satisfiable results
   export_solution(solver=solver, variables=variables)
   
   # For unsatisfiable results
   export_solution(satisfiable=False, variables=variables)
   
   # For direct dictionary input (advanced usage)
   export_solution(data={"satisfiable": False, "values": {}})
   ```

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

# ALTERNATIVE - Use the explicit satisfiable parameter
export_solution(satisfiable=result, variables={})  # Also works
```

## Handling Unsatisfiable Problems and Theorem Proving

When working with problems that have no solution or proving theorems by contradiction:

```python
# Check if a solution exists for a constraint satisfaction problem
result = solver.check()

# Handle the result appropriately
if result == sat:
    print("Solution found!")
    export_solution(solver=solver, variables=variables)
else:
    print("No solution exists that satisfies all constraints.")
    # For standard constraint problems that are unsatisfiable:
    export_solution(satisfiable=False, variables=variables)
```

### Theorem Proving Considerations

When proving theorems, the relationship between solver results and theorem validity is important:

1. **When searching for counterexamples**:
   - If `solver.check() == sat` → Found a counterexample → Theorem is false
   - If `solver.check() == unsat` → No counterexample exists → Theorem is true
2. **When directly modeling a theorem**:
   - If `solver.check() == sat` → Found a valid case → Theorem is true (for that case)
   - If `solver.check() == unsat` → No valid case exists → Theorem is false

Always make sure your solution export matches the semantic meaning of your result, not just the technical satisfiability status of your solver.

## Bitvector Verification Example

Here's a modular example that demonstrates using bitvectors and arrays for verification:

```python
# Item 1: Setup and imports
from z3 import *
from mcp_solver.z3 import export_solution

# Item 2: Define the bitvector verification model
def build_verification_model():
    # Create an 8-bit input X
    X = BitVec('X', 8)
    
    # Create a lookup table (array indexed by 3-bit values)
    table = Array('table', BitVecSort(3), BitVecSort(8))
    
    # Step 1: Extract the lower 3 bits of X as the index
    index = Extract(2, 0, X)
    
    # Step 2: Look up a value Y from the table
    Y = Select(table, index)
    
    # Step 3: Compute Z = (X XOR Y) & 1
    Z = Extract(0, 0, X ^ Y)
    
    # Calculate the parity of X (XOR of all bits)
    parity = BitVecVal(0, 1)
    for i in range(8):
        parity = parity ^ Extract(i, i, X)
    
    # Check if property holds: Z == parity(X)
    property_holds = (Z == parity)
    
    # Create solver and look for counterexamples
    s = Solver()
    s.add(Not(property_holds))
    
    return s, X, Y, Z, parity, index

# Item 3: Execute the verification
# Build and get our verification model components
solver, X, Y, Z, parity, index = build_verification_model()

# Check if a counterexample exists (property violation)
result = solver.check()

# Item 4: Process results and export the solution
# Create Z3 boolean for the verification result
property_verified = Bool('property_verified')

if result == sat:
    # Found a counterexample - the solver successfully found a case where the property fails
    # This means the counterexample search was satisfiable (but the property is false)
    print(f"Input X = {solver.model().evaluate(X)}, parity = {solver.model().evaluate(parity)}")
    print(f"Z = {solver.model().evaluate(Z)}, which differs from parity")
    # Export with solver result (sat) and the property_verified value (false)
    solver.add(property_verified == False)
    export_solution(solver=solver, variables={"property_verified": property_verified})
else:
    # No counterexample found - the property holds for all inputs
    # The counterexample search was unsatisfiable (meaning the property is true)
    print("Property verified: Z always equals parity of X")
    # Create a new solver to express that the property is verified
    result_solver = Solver()
    result_solver.add(property_verified == True)
    export_solution(solver=result_solver, variables={"property_verified": property_verified})
```

This example verifies whether a computed value Z (the lowest bit of X XOR table[X&7]) always equals the parity of X. The structured approach makes it easy to build, verify, and analyze complex properties using bitvectors and arrays.

## Mathematical Proofs with Z3

Z3 can be used for mathematical proofs in addition to constraint solving.

NOTE:

- Use proper Z3 syntax for mathematics and use strict JSON format.
- Unusual characters, even in a comment, can cause the item you want to add being empty, causing an error.

Here are patterns for common proof techniques:

### Proving Algebraic Identities

```python
# Item 1: Setup and imports
from z3 import *
from mcp_solver.z3 import export_solution

# Setup variables for our test
x = Int('x')

# Item 2: Define the expressions and create a solver to check equality
# Test expressions: x² + 2x ≡ x(x+2)
expr1 = x**2 + 2*x
expr2 = x*(x+2)

# Create a solver that attempts to find a counterexample
solver = Solver()

# Add domain constraints if necessary
solver.add(x > 0)

# Try to find a case where expressions are NOT equal
solver.add(simplify(expr1) != simplify(expr2))

# Item 3: Check results and export the solution
# Create Z3 boolean for the verification result
equality_proven = Bool('equality_proven')

# Check if a counterexample exists
result = solver.check()
if result == unsat:
    # No counterexample found, expressions are equal
    print("Expressions are equal for all valid inputs")
    # The counterexample search was unsatisfiable, meaning the equality is proven
    result_solver = Solver()
    result_solver.add(equality_proven == True)
    export_solution(solver=result_solver, variables={"equality_proven": equality_proven})
else:
    # Found a counterexample - the solver successfully found inputs where expressions differ
    # The counterexample search was satisfiable, meaning the equality is disproven
    model = solver.model()
    print("Found counterexample:", model)
    solver.add(equality_proven == False)
    export_solution(solver=solver, variables={"equality_proven": equality_proven})
```

## Export Solution Formats

```python
# For satisfiable constraint problems
export_solution(solver=solver, variables=variables)

# For unsatisfiable problems
export_solution(satisfiable=False, variables=variables)

# For direct dictionary input (advanced usage)
export_solution(data={"satisfiable": False, "values": {}})

# INCORRECT - missing parameters
# export_solution()  # This will fail

# INCORRECT - unsupported parameters
# export_solution(solver=solver, variables=variables, solution_data={})  # This will fail
```

The `export_solution` function formats solutions differently depending on the scenario:

1. **For satisfiable constraint problems**: Returns variable assignments that satisfy all constraints
2. **For optimization problems**: Returns the optimal solution values
3. **For proofs by contradiction (unsat results)**: Use the `satisfiable` parameter to explicitly specify the result

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
    export_solution(satisfiable=False, variables={"n": n})
else:
    # Export counterexample if found
    export_solution(solver=solver2, variables={"n": n})
```

## End-to-End Workflow Example: Sum of Cubes

Here's a complete example showing how to prove that the sum of cubes from 1 to n equals n²(n+1)²/4:

```python
# Item 1: Setup and imports
from z3 import *
from mcp_solver.z3 import export_solution

# Item 2: Verify concrete examples
print("CONCRETE EXAMPLES VERIFICATION:")
all_examples_match = True

for n_val in range(1, 6):
    # Calculate sum of cubes
    sum_cubes = sum(i**3 for i in range(1, n_val+1))
    
    # Calculate formula result
    formula = (n_val**2 * (n_val+1)**2) // 4
    
    print(f"n = {n_val}: sum = {sum_cubes}, formula = {formula}")
    
    if sum_cubes != formula:
        print(f"MISMATCH at n = {n_val}")
        all_examples_match = False

# Item 3: Verify the base case (n=1)
print("\nBASE CASE (n=1):")
base_sum = 1  # 1³ = 1
base_formula = (1**2 * (1+1)**2) // 4  # 1²*2²/4 = 1
print(f"Base case: {base_sum} = {base_formula}")

base_case_verified = (base_sum == base_formula)

# Item 4: Inductive step verification
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

# Create solver to find a counterexample
ind_solver = Solver()
ind_solver.add(k > 0)  # Domain constraint: k is positive
ind_solver.add(simplify(sum_k_plus_1) != simplify(formula_k_plus_1))

# Check if they are equal by trying to find a counterexample
induction_result = ind_solver.check()
if induction_result == unsat:
    print("Inductive step verified: No counterexample found")
    induction_verified = True
else:
    counterexample = ind_solver.model()[k]
    print(f"Counterexample found: k = {counterexample}")
    induction_verified = False

# Item 5: Consolidate results and export the solution
# Create Z3 boolean for the final verification result
proof_verified = Bool('proof_verified')
result_solver = Solver()

# The proof is verified if all three parts succeeded
result_solver.add(proof_verified == (all_examples_match and base_case_verified and induction_verified))

# Print final conclusion
if all_examples_match and base_case_verified and induction_verified:
    print("\nPROOF COMPLETED SUCCESSFULLY:")
    print("✓ All concrete examples match")
    print("✓ Base case verified")
    print("✓ Inductive step verified")
    print("\nTherefore, the formula 1³ + 2³ + ... + n³ = [n²(n+1)²]/4 is proven.")
else:
    print("\nPROOF FAILED:")
    if not all_examples_match:
        print("✗ Concrete examples don't match")
    if not base_case_verified:
        print("✗ Base case failed")
    if not induction_verified:
        print("✗ Inductive step failed")

# Export the solution
export_solution(solver=result_solver, variables={"proof_verified": proof_verified})
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
    export_solution(satisfiable=False, variables=variables)
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
    export_solution(satisfiable=False, variables=variables)
```

### Troubleshooting Variable Scope Issues

If you encounter scope-related errors:

1. Always return variables from inner functions to outer scopes
2. Create a dictionary mapping variable names to Z3 variables
3. Pass both solver and variables to `export_solution`
4. Prefer the function-based approach shown above

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

2. ✅ Did you call export_solution with the appropriate parameters?

   ```python
   # For satisfiable results
   export_solution(solver=solver, variables=variables)
   
   # For unsatisfiable results
   export_solution(satisfiable=False, variables=variables)
   ```

3. ✅ Did you check if the solver found a solution before calling export_solution?

   ```python
   if solver.check() == sat:
       export_solution(solver=solver, variables=variables)
   else:
       export_solution(satisfiable=False, variables=variables)
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

7. ✅ If you get "Missing required parameter 'content'" errors:

   - **Problem**: Your content might be too large or contain special characters
   - **Solution**: Split your code into smaller, logical items using multiple add_item calls

   ```python
   # Instead of one large item:
   # add_item(index=1, content=very_large_code)
   
   # Use multiple smaller items:
   add_item(index=1, content="""
   from z3 import *
   from mcp_solver.z3 import export_solution
   """)
   
   add_item(index=2, content="""
   x = Int('x')
   y = Int('y')
   variables = {'x': x, 'y': y}
   """)
   
   add_item(index=3, content="""
   solver = Solver()
   solver.add(x > 0, y > 0, x + y == 10)
   """)
   
   add_item(index=4, content="""
   if solver.check() == sat:
       export_solution(solver=solver, variables=variables)
   else:
       export_solution(satisfiable=False, variables=variables)
   """)
   ```
   
   Note: The "# Item X:" comments in examples are for illustration only to show where you would split the code. You don't need to include these comments in your actual submissions.

## Working with Arrays in Z3

Z3 offers two approaches for modeling array problems. Choose the approach that best fits your problem:

### 1. Python Lists of Z3 Variables (recommended for fixed-size arrays)

This approach uses individual Z3 variables for each array element and a Python list to organize them:

```python
# Approach 1: Create individual Z3 variables
a0 = Int("a0")  # First element
a1 = Int("a1")  # Second element
a2 = Int("a2")  # Third element

# Create a Python list for easier access (not a Z3 array)
array_vars = [a0, a1, a2]

# Apply array constraints using list indexing
solver.add(array_vars[0] <= array_vars[1])  # Ascending order constraint
solver.add(array_vars[1] <= array_vars[2])  
solver.add(Sum(array_vars) == 15)           # Sum constraint

# IMPORTANT: Export each variable separately
variables = {
    "a0": a0,  # Include each variable individually
    "a1": a1,
    "a2": a2
}
# Alternative dictionary comprehension for larger arrays:
# variables = {f"a{i}": array_vars[i] for i in range(len(array_vars))}

# Solution extraction using a list comprehension
if solver.check() == sat:
    model = solver.model()
    # Use a descriptive name like result_array (not solution or array)
    result_array = [model.evaluate(var).as_long() for var in array_vars]
    print(f"Array values: {result_array}")
```

### 2. Z3's Native Array Type (for symbolic indexing and unbounded arrays)

This approach uses Z3's built-in Array theory:

```python
# Create a Z3 array from integers to integers
arr = Array('arr', IntSort(), IntSort())

# Apply constraints using Select(arr, index) to access elements
for i in range(3):
    # Set bounds for each element
    solver.add(Select(arr, i) >= 0)
    solver.add(Select(arr, i) <= 10)
    
# Add ordering constraints
solver.add(Select(arr, 0) <= Select(arr, 1))
solver.add(Select(arr, 1) <= Select(arr, 2))

# Add a sum constraint
solver.add(Select(arr, 0) + Select(arr, 1) + Select(arr, 2) == 15)

# Variable export - include the array itself
variables = {'arr': arr}

# Solution extraction for Z3 arrays
if solver.check() == sat:
    model = solver.model()
    # Extract values into a Python list
    array_values = []
    for i in range(3):
        # Use Select to get each element from the array
        value = model.evaluate(Select(arr, i)).as_long()
        array_values.append(value)
    print(f"Array values: {array_values}")
```

### Common Array Operations and Constraints

Here are examples of common array operations:

```python
# Sum of array elements (using Python list of Z3 variables)
total = Sum(array_vars)
solver.add(total == 100)

# Product of array elements
from operator import mul
from functools import reduce
product = reduce(mul, array_vars)
solver.add(product == 24)

# Maximum/minimum element
max_element = array_vars[0]
for var in array_vars[1:]:
    # Use If for max comparison
    max_element = If(var > max_element, var, max_element)
solver.add(max_element <= 10)

# Finding elements matching a condition
count_evens = Sum([If(var % 2 == 0, 1, 0) for var in array_vars])
solver.add(count_evens == 2)  # Exactly 2 even numbers

# No duplicate values
solver.add(Distinct(array_vars))
```

### Best Practices and Error Prevention

To avoid common errors when working with arrays:

1. **Use distinct variable names for different concepts**:
   ```python
   # GOOD: clear, separate names
   array_vars = [Int(f"a{i}") for i in range(5)]  # List of Z3 variables
   result_values = []  # List for storing results
   
   # BAD: reusing names causes confusion
   array = [Int(f"a{i}") for i in range(5)]  # Don't name variables same as Z3 types
   solution = []  # Don't use "solution" for arrays as it might trigger dictionary warnings
   ```

2. **Keep code organized with clear variable types**:
   ```python
   # For fixed-size arrays (approach 1)
   a0, a1, a2, a3 = Int("a0"), Int("a1"), Int("a2"), Int("a3")
   array_elements = [a0, a1, a2, a3]  # Python list of Z3 Int variables
   
   # For Z3 arrays (approach 2)
   z3_array = Array('array', IntSort(), IntSort())  # Z3 Array type
   ```

3. **Extract array values appropriately**:
   ```python
   # For Python lists of Z3 variables
   model = solver.model()
   values = [model.evaluate(var).as_long() for var in array_elements]
   
   # For Z3 arrays
   values = [model.evaluate(Select(z3_array, i)).as_long() for i in range(size)]
   ```

4. **Use separate names for solution containers**:
   ```python
   # Avoid false positives in dictionary misuse detector
   result_list = []  # For storing array values
   output_data = {}  # For storing dictionary data
   ```

By following these patterns, you'll avoid common errors and ensure your array-based constraints work correctly with the export_solution system.

## Common Error Patterns and Solutions

1. **"'bool' object has no attribute 'as_ast'"**:

   - **Problem**: Using Python boolean instead of Z3 Bool variable
   - **Solution**: Create a Z3 Bool variable and connect it to your Python result or use the explicit satisfiable parameter

   ```python
   # Option 1: Use Z3 Bool variable
   verified = Bool('verified')
   solver.add(verified == your_python_boolean)
   export_solution(solver=solver, variables={"verified": verified})
   ```

2. **"'int' object has no attribute 'as_ast'"**:

   - **Problem**: Using Python integer instead of Z3 Int variable
   - **Solution**: Create Z3 Int variables and use constraints to set values

   ```python
   x = Int('x')
   solver.add(x == 5)  # Instead of x = 5
   ```

<<<<<<< HEAD
3. **Contradictory solution output (satisfiable=True but message says "No solution exists")**:

   - **Problem**: Using a secondary solver to express impossibility
   - **Solution**: Use the explicit satisfiable parameter instead

   ```python
   # INCORRECT approach
   is_possible = Bool('is_possible')
   result_solver = Solver()
   result_solver.add(is_possible == False)
   export_solution(solver=result_solver, variables={"is_possible": is_possible})
   
=======
3. **"'list' object has no attribute 'get'"**:

   - **Problem**: Using a Python list where a dictionary is expected
   - **Solution**: Use different names for different data types

   ```python
   variables = {...}           # Dictionary for export_solution
   result_array = [...]        # Different name for list results
   ```

4. **Contradictory solution output**:

   - **Problem**: Using a secondary solver to express impossibility
   - **Solution**: Use the explicit satisfiable parameter instead

   ```python
>>>>>>> dev
   # CORRECT approach
   export_solution(satisfiable=False, variables=variables)
   ```

<<<<<<< HEAD
4. **"NameError: name 'original_export_solution' is not defined"**:

   - **Problem**: Internal wrapper issue with export_solution
   - **Solution**: Always explicitly import export_solution at the top of your code
=======
5. **"NameError: name 'original_export_solution' is not defined"**:

   - **Problem**: Internal wrapper issue with export_solution
   - **Solution**: Always explicitly import export_solution at the top
>>>>>>> dev

   ```python
   from mcp_solver.z3 import export_solution
   ```

<<<<<<< HEAD
5. **Empty results even with sat solution**:

   - **Problem**: Missing export_solution call or wrong variables dictionary
   - **Solution**: Ensure export_solution is called with the correct parameters
=======
6. **Empty results even with sat solution**:

   - **Problem**: Missing export_solution call or wrong variables dictionary
   - **Solution**: Ensure export_solution is called with correct parameters
>>>>>>> dev
