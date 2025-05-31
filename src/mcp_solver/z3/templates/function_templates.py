"""
Core function templates for Z3 models.

This module provides standardized function templates for common Z3 modeling patterns.
These templates help address common issues like variable scope problems, import availability,
and structural inconsistency by encouraging a function-based approach where all model
components are encapsulated within a single function.
"""

import os
import sys


# IMPORTANT: Properly import the Z3 library (not our local package)
# First, remove the current directory from the path to avoid importing ourselves
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if current_dir in sys.path:
    sys.path.remove(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if parent_parent_dir in sys.path:
    sys.path.remove(parent_parent_dir)

# Add site-packages to the front of the path
import site


site_packages = site.getsitepackages()
for p in reversed(site_packages):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now try to import Z3
try:
    from z3 import (
        And,
        Array,
        ArrayRef,
        BitVec,
        BitVecSort,
        Bool,
        BoolRef,
        BoolSort,
        Distinct,
        Exists,
        ExprRef,
        ForAll,
        If,
        Implies,
        Int,
        Ints,
        IntSort,
        Not,
        Optimize,
        Or,
        Real,
        RealSort,
        Solver,
        Sum,
        sat,
        unknown,
        unsat,
    )
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)


# Template for basic constraint satisfaction problems
def constraint_satisfaction_template():
    """
    A template for basic constraint satisfaction problems.

    This template follows the recommended structure:
    1. Variable definition
    2. Solver creation
    3. Constraints definition
    4. Solution export

    Returns:
        A tuple containing:
        - The solver object
        - A dictionary mapping variable names to Z3 variables

    Usage:
        solver, variables = constraint_satisfaction_template()
        export_solution(solver=solver, variables=variables)
    """

    # [SECTION: VARIABLE DEFINITION]
    # Define your variables here
    x = Int("x")
    y = Int("y")

    # [SECTION: SOLVER CREATION]
    # Create solver
    s = Solver()

    # [SECTION: CONSTRAINTS]
    # Add your constraints here
    s.add(x > 0)
    s.add(y > 0)
    s.add(x + y <= 10)

    # [SECTION: VARIABLES TO EXPORT]
    # Define the variables you want to see in the solution
    variables = {"x": x, "y": y}

    return s, variables


# Template for optimization problems
def optimization_template():
    """
    A template for optimization problems.

    This template follows the recommended structure:
    1. Variable definition
    2. Optimizer creation
    3. Constraints definition
    4. Objective function definition
    5. Solution export

    Returns:
        A tuple containing:
        - The optimizer object
        - A dictionary mapping variable names to Z3 variables

    Usage:
        optimizer, variables = optimization_template()
        export_solution(solver=optimizer, variables=variables)
    """

    # [SECTION: VARIABLE DEFINITION]
    # Define your variables here
    x = Int("x")
    y = Int("y")

    # [SECTION: OPTIMIZER CREATION]
    # Create optimizer
    opt = Optimize()

    # [SECTION: CONSTRAINTS]
    # Add your constraints here
    opt.add(x >= 0)
    opt.add(y >= 0)
    opt.add(x + y <= 10)

    # [SECTION: OBJECTIVE FUNCTION]
    # Define your objective function
    objective = x + 2 * y

    # Set optimization direction
    opt.maximize(objective)  # or opt.minimize(objective)

    # [SECTION: VARIABLES TO EXPORT]
    # Define the variables you want to see in the solution
    variables = {"x": x, "y": y, "objective_value": objective}

    return opt, variables


# Template for array handling
def array_template():
    """
    A template for problems involving arrays.

    This template follows the recommended structure:
    1. Problem size definition
    2. Array variable definition
    3. Solver creation
    4. Array initialization and constraints
    5. Solution export

    Returns:
        A tuple containing:
        - The solver object
        - A dictionary mapping variable names to Z3 variables

    Usage:
        solver, variables = array_template()
        export_solution(solver=solver, variables=variables)
    """

    # [SECTION: PROBLEM SIZE]
    n = 5  # Size of your arrays

    # [SECTION: VARIABLE DEFINITION]
    # Define array variables
    arr = Array("arr", IntSort(), IntSort())

    # [SECTION: SOLVER CREATION]
    s = Solver()

    # [SECTION: ARRAY INITIALIZATION]
    # Initialize array elements (if needed)
    for i in range(n):
        s.add(arr[i] >= 0)
        s.add(arr[i] < 100)

    # [SECTION: CONSTRAINTS]
    # Add array-specific constraints
    for i in range(n - 1):
        s.add(arr[i] <= arr[i + 1])  # Ensure array is sorted

    # [SECTION: VARIABLES TO EXPORT]
    # Export array elements
    variables = {f"arr[{i}]": arr[i] for i in range(n)}

    return s, variables


# Template for problems with quantifiers
def quantifier_template():
    """
    A template for problems involving quantifiers.

    This template follows the recommended structure:
    1. Problem parameters
    2. Variable definition
    3. Solver creation
    4. Domain constraints
    5. Quantifier constraints
    6. Solution export

    Returns:
        A tuple containing:
        - The solver object
        - A dictionary mapping variable names to Z3 variables

    Usage:
        solver, variables = quantifier_template()
        export_solution(solver=solver, variables=variables)
    """

    # Import Z3 template helpers
    try:
        from z3_templates import all_distinct, array_is_sorted
    except ImportError:
        # Fallback if templates not available
        pass

    # [SECTION: PROBLEM PARAMETERS]
    n = 5  # Problem size

    # [SECTION: VARIABLE DEFINITION]
    # Define your array variables
    arr = Array("arr", IntSort(), IntSort())

    # [SECTION: SOLVER CREATION]
    s = Solver()

    # [SECTION: DOMAIN CONSTRAINTS]
    for i in range(n):
        s.add(arr[i] >= 0, arr[i] < n)

    # [SECTION: QUANTIFIER CONSTRAINTS]
    # Option 1: Using template functions (if available)
    try:
        s.add(all_distinct(arr, n))
        s.add(array_is_sorted(arr, n))
    except NameError:
        # Option 2: Using explicit quantifiers
        i = Int("i")
        j = Int("j")
        unique_values = ForAll(
            [i, j], Implies(And(i >= 0, i < j, j < n), arr[i] != arr[j])
        )
        sorted_values = ForAll(
            [i, j], Implies(And(i >= 0, i < j, j < n), arr[i] <= arr[j])
        )
        s.add(unique_values)
        s.add(sorted_values)

    # [SECTION: VARIABLES TO EXPORT]
    variables = {f"arr[{i}]": arr[i] for i in range(n)}

    return s, variables


# Combined demo template with sample usage
def demo_template():
    """
    A demo template showing how to use the function-based template pattern.
    This demonstrates a simple constraint satisfaction problem.

    Returns:
        A tuple containing:
        - The solver object
        - A dictionary mapping variable names to Z3 variables
    """

    # [SECTION: IMPORTS]
    # Import everything you need at the top of your function
    from z3 import Bool, Int, Solver

    # [SECTION: PROBLEM PARAMETERS]
    n_items = 5
    max_weight = 10

    # [SECTION: VARIABLE DEFINITION]
    # Define your variables
    weights = [Int(f"weight_{i}") for i in range(n_items)]
    selected = [Bool(f"selected_{i}") for i in range(n_items)]
    total_weight = Int("total_weight")

    # [SECTION: SOLVER CREATION]
    s = Solver()

    # [SECTION: DOMAIN CONSTRAINTS]
    # Set up domain constraints
    for i in range(n_items):
        s.add(weights[i] > 0)
        s.add(weights[i] <= max_weight)

    # [SECTION: CORE CONSTRAINTS]
    # Define the relationship between selection and weight
    weight_sum = 0
    for i in range(n_items):
        # If selected[i] is true, add weights[i] to total
        weight_sum += If(selected[i], weights[i], 0)

    s.add(total_weight == weight_sum)
    s.add(total_weight <= max_weight)

    # Must select at least 2 items
    s.add(Sum([If(selected[i], 1, 0) for i in range(n_items)]) >= 2)

    # [SECTION: VARIABLES TO EXPORT]
    variables = {
        "total_weight": total_weight,
        **{f"weight_{i}": weights[i] for i in range(n_items)},
        **{f"selected_{i}": selected[i] for i in range(n_items)},
    }

    return s, variables


# If this module is run directly, demonstrate its usage
if __name__ == "__main__":
    try:
        from z3 import export_solution
    except ImportError:
        # Define a simple export function for testing
        def export_solution(solver, variables):
            if solver.check() == sat:
                model = solver.model()
                print("Solution found:")
                for name, var in variables.items():
                    print(f"{name} = {model.eval(var)}")
            else:
                print("No solution found")

    print("Running constraint satisfaction template...")
    solver, variables = constraint_satisfaction_template()
    export_solution(solver=solver, variables=variables)
